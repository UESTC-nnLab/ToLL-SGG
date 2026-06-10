import os
import copy
import datetime
import torch
import torch.optim as optim
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.model.diff_trans.models.clustering import cluster_and_visualize, visualize_with_gt, evaluate_and_plot_clustering
from src.dataset.DataLoader import (CustomDataLoader, collate_fn_mmg_diff, collate_fn_mmg)
from src.dataset.dataset_builder import build_dataset_for_clustering, build_pretrain_dataset
from src.model.diff_trans.models.PointDif_dino import PointDif
from src.model.optimizer.scheduler import get_warmup_cosine_scheduler, get_freeze_warmup_scheduler
from src.model.diff_trans.models.monitor import EpochCollapseMonitor
from src.model.diff_trans.models.clustering import cluster_and_visualize, visualize_with_gt, analyze_kmeans_clusters

# [DDP] 辅助函数：判断是否为主进程
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def normalize_state_dict_keys(state_dict):
    return {
        (k[7:] if k.startswith('module.') else k): v
        for k, v in state_dict.items()
    }

def extract_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        if 'pointdif' in checkpoint:
            return checkpoint['pointdif']
    return checkpoint

def get_param_groups(module, base_lr, weight_decay, amsgrad):
    """
    自动将模块内的参数分为两组：
    1. decay_group: 权重 (Weights) -> 使用 weight_decay
    2. no_decay_group: 偏置 (Bias) 和 Norm层参数 -> weight_decay = 0.0
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        
        if param.ndim <= 1 or "bias" in name or "norm" in name or "bn" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    groups = []
    if len(decay_params) > 0:
        groups.append({
            'params': decay_params, 
            'lr': base_lr, 
            'weight_decay': weight_decay, 
            'amsgrad': amsgrad
        })
    if len(no_decay_params) > 0:
        groups.append({
            'params': no_decay_params, 
            'lr': base_lr, 
            'weight_decay': 0.0, 
            'amsgrad': amsgrad
        })
    
    return groups

class Pdiff4SSG_Pretraining_ddp():
    def __init__(self, config, val_cls_mode=False):
        self.config = config
        self.model_name = 'pdiff_SGG'
        
        self.save_dir = self.config.PATH # for saving models weights
        os.makedirs(self.save_dir, exist_ok=True)
        
        # [DDP] 1. 初始化进程组
        # 从环境变量获取 local_rank (torchrun 会自动设置)
        # 1. 判断是否处于 DDP 环境（检查是否有 RANK 变量）
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # 这种情况通常是 torchrun 启动的
            self.local_rank = int(os.environ['LOCAL_RANK'])
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            print(f"[Info] Running in DDP mode. Rank: {dist.get_rank()}")
        else:
            # 2. 如果没有环境变量（直接 python main.py 启动），则手动设置为“伪分布式”
            print("[Warning] No DDP environment found. Fallback to Single-GPU mode.")
            
            # 手动设置环境变量，骗过 init_process_group
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355' # 随便给个端口
            self.local_rank = 0
        
        # 初始化 backend，通常 GPU 使用 nccl
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
            
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device('cuda', self.local_rank)
        
        # 只在主进程创建目录
        if is_main_process():
            dataset_cfg = getattr(self.config, 'dataset', None)
            val_root = getattr(dataset_cfg, 'root', None) if dataset_cfg is not None else None
            val_root_3rscan = getattr(dataset_cfg, 'root_3rscan', None) if dataset_cfg is not None else None
            print(f"[Config] Validation dataset roots: dataset.root={val_root}, dataset.root_3rscan={val_root_3rscan}")

        ''' Build dataset ''' 
        # 注意：多卡训练时，每个卡只看到总数据的一部分，所以这里 len 计算的只是总数
        if val_cls_mode:      
            self.dataset_train = build_dataset_for_clustering(self.config)
        else:
            self.dataset_train = build_pretrain_dataset(self.config, for_train=True)
        
        # [DDP] TensorBoard 只在 Rank 0 初始化
        self.writer = None
         
        if is_main_process():
            if not os.path.exists(os.path.join(self.config.analysis_save_dir, "log_runs")):
                os.makedirs(os.path.join(self.config.analysis_save_dir, "log_runs"))
            log_dir = os.path.join(self.config.analysis_save_dir, "log_runs", "experiment_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            self.writer = SummaryWriter(log_dir=log_dir)
                
        ''' Build Model '''
        # [DDP] 模型先移动到对应的 GPU
        self.model = PointDif(self.config).to(self.device)
        
        # [DDP] 转换为 SyncBatchNorm (推荐在多卡小 Batch 时使用，提升性能)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # [DDP] 配置优化器 (在 DDP 包装之前配置参数组，这样 param 名称不会带 module. 前缀，保持原逻辑)
        param_groups = []
        self.model.mask_encoder.requires_grad_(False)
        # 1. 常规模块
        # param_groups.extend(get_param_groups(self.model.mask_encoder, float(config.LR)/5, self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.rel_encoder_3d, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.ca_net, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.mlp_3d, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.point_diffusion.net, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.predictor_triplet, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.predictor_edge, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.predictor_obj, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        
        # 3. 特殊模块: MMG
        param_groups.extend(get_param_groups(self.model.mmg, float(config.LR)/5, self.config.W_DECAY, self.config.AMSGRAD))

        # 4. 特殊参数: Mask Token
        param_groups.append({'params': self.model.mask_token, 'lr': float(config.LR), 'weight_decay': self.config.W_DECAY, 'amsgrad': self.config.AMSGRAD})
        param_groups.append({'params': self.model.edge_mask_token.parameters(), 'lr': float(config.LR), 'weight_decay': self.config.W_DECAY, 'amsgrad': self.config.AMSGRAD})

        self.optimizer = optim.AdamW(param_groups)

        # [DDP] 保存原始模型的引用 (raw_model)，用于保存权重和访问非DDP属性 (如 self.model.epoch)
        # 包装后 self.model 变成了 DDP 对象，访问原属性需要 self.model.module
        self.raw_model = self.model 
        
        # [DDP] 包装 DDP 模型
        self.model = DDP(self.raw_model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        # 计算 total 和 scheduler (基于数据集大小，这里简单估算，DDP下每个epoch迭代次数会变少)
        # 注意：在DDP下，len(loader) 会自动变成 total_samples / (batch_size * world_size)
        # 所以这里的计算我们留到 train 函数里根据 loader 动态获取更准确
        
        self.optimizer.zero_grad()
    
    def load_pretrained_mask_encoder(self, checkpoint_path):
        if is_main_process():
            print(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu') # 建议先 load 到 cpu 再转
        raw_state_dict = extract_checkpoint_state_dict(checkpoint)
        raw_state_dict = normalize_state_dict_keys(raw_state_dict)

        mask_encoder_dict = {}
        for k, v in raw_state_dict.items():
            if k.startswith('module.'):
                name = k[7:]
            else:
                name = k

            if name.startswith('mask_encoder.'):
                new_key = name.replace('mask_encoder.', '', 1)
                mask_encoder_dict[new_key] = v

        if len(mask_encoder_dict) > 0:
            # 注意：这里加载到 self.raw_model (DDP内部的原始模型)
            missing, unexpected = self.raw_model.mask_encoder.load_state_dict(mask_encoder_dict, strict=True)
            if is_main_process():
                print(f"Success! Loaded {len(mask_encoder_dict)} keys.")
        else:
            if is_main_process():
                print("Error: No keys starting with 'mask_encoder' found!")
    
    @torch.no_grad()
    def data_processing_train_pdiff(self, items):
        if len(items) == 10:
            obj_points, obj_points_spatial, descriptor, edge_indices, \
            anchor_ids, _obj_points_view2, _descriptor_view2, cur_obj_texts, batch_ids, obj_labels = items
        elif len(items) == 9:
            obj_points, obj_points_spatial, descriptor, edge_indices, \
            anchor_ids, _obj_points_view2, _descriptor_view2, cur_obj_texts, batch_ids = items
            obj_labels = None
        elif len(items) == 8:
            obj_points, obj_points_spatial, descriptor, edge_indices, \
            anchor_ids, cur_obj_texts, batch_ids, obj_labels = items
        else:
            obj_points, obj_points_spatial, descriptor, edge_indices, \
            anchor_ids, cur_obj_texts, batch_ids = items
            obj_labels = None
        
        obj_points = obj_points.permute(0, 2, 1).contiguous()
        if obj_labels is not None:
            obj_points, edge_indices, descriptor, batch_ids, obj_points_spatial, obj_labels = \
                self.cuda(obj_points, edge_indices, descriptor, batch_ids, obj_points_spatial, obj_labels)
        else:
            obj_points, edge_indices, descriptor, batch_ids, obj_points_spatial = \
                self.cuda(obj_points, edge_indices, descriptor, batch_ids, obj_points_spatial)

        edge_indices = edge_indices.long()
        batch_ids = batch_ids.long()

        return obj_points, descriptor, edge_indices, anchor_ids, batch_ids,\
            obj_points_spatial, cur_obj_texts, obj_labels
    
    @torch.no_grad()
    def data_processing_train(self, items):
        obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids,_ = items
        obj_points = obj_points.permute(0, 2, 1).contiguous()
        obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = \
            self.cuda(obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids)
        edge_indices = edge_indices.long()
        batch_ids = batch_ids.long()
        return obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids
       
    def train(self):
        # [DDP] 2. 创建 DistributedSampler
        train_sampler = DistributedSampler(self.dataset_train, shuffle=True)
        
        drop_last = True
        train_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_train,
            batch_size=self.config.Batch_Size, # [DDP] 这里的 Batch Size 是单卡的 Batch Size
            num_workers=4, 
            drop_last=drop_last,
            shuffle=False, # [DDP] 使用 Sampler 时，shuffle 必须为 False
            collate_fn=collate_fn_mmg_diff,
            sampler=train_sampler # [DDP] 传入 Sampler
        )
        
        # 更新 total 和 max_iteration
        self.total = len(train_loader)
        self.max_iteration = self.config.max_iteration = int(self.config.MAX_EPOCHES * self.total)
        
        # 初始化 Scheduler (放在这里是为了确保 total 正确)
        self.lr_scheduler = get_warmup_cosine_scheduler(self.optimizer, self.total*5, self.max_iteration)
                
        keep_training = True
        start_epoch = 0
        
        init_weights_path = getattr(self.config, 'INIT_WEIGHTS_PATH', None)
        resume_path = getattr(self.config, 'RESUME_PATH', None)

        init_weights_exists = bool(init_weights_path and os.path.exists(init_weights_path))
        resume_exists = bool(resume_path and os.path.exists(resume_path))

        if init_weights_exists:
            if is_main_process():
                print(f"Initializing model weights from checkpoint: {init_weights_path}")

            checkpoint = torch.load(init_weights_path, map_location=self.device)
            state_dict = extract_checkpoint_state_dict(checkpoint)
            normalized_state_dict = normalize_state_dict_keys(state_dict)

            try:
                self.raw_model.load_state_dict(normalized_state_dict, strict=True)
                if is_main_process():
                    print("[Info] Loaded initialization weights into raw_model with strict=True.")
            except Exception as e:
                if is_main_process():
                    print(f"[Warning] strict=True init load into raw_model failed, will try strict=False. Error: {e}")
                incompatible = self.raw_model.load_state_dict(normalized_state_dict, strict=False)
                if is_main_process():
                    print(f"[Warning] Loaded initialization weights into raw_model with strict=False. Missing: {len(incompatible.missing_keys)}, Unexpected: {len(incompatible.unexpected_keys)}")

            if is_main_process():
                print("[Info] Training will start from epoch 0 with fresh optimizer and scheduler state.")

        elif init_weights_path:
            if is_main_process():
                print(f"Warning: INIT_WEIGHTS_PATH '{init_weights_path}' not found. Falling back to resume logic.")

        if (not init_weights_exists) and resume_exists:
            if is_main_process():
                print(f"Resuming training from checkpoint: {resume_path}")
            
            # [DDP] map_location 确保加载到当前 GPU
            checkpoint = torch.load(resume_path, map_location=self.device)
            state_dict = extract_checkpoint_state_dict(checkpoint)
            normalized_state_dict = normalize_state_dict_keys(state_dict)
            
            # [DDP] 始终加载到 raw_model，并统一去掉 module. 前缀，
            # 否则 strict=False 可能“成功”返回但实际上几乎没有真正加载参数。
            try:
                self.raw_model.load_state_dict(normalized_state_dict, strict=True)
                if is_main_process():
                    print("[Info] Loaded model_state_dict into raw_model with strict=True.")
            except Exception as e:
                if is_main_process():
                    print(f"[Warning] strict=True load into raw_model failed, will try strict=False. Error: {e}")
                incompatible = self.raw_model.load_state_dict(normalized_state_dict, strict=False)
                if is_main_process():
                    print(f"[Warning] Loaded model_state_dict into raw_model with strict=False. Missing: {len(incompatible.missing_keys)}, Unexpected: {len(incompatible.unexpected_keys)}")

            optimizer_loaded = False
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                optimizer_loaded = True
            except Exception as e:
                if is_main_process():
                    print(f"[Warning] Failed to load optimizer_state_dict (param_groups may have changed). Will continue with fresh optimizer. Error: {e}")

            if optimizer_loaded and 'scheduler_state_dict' in checkpoint:
                try:
                    self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    if is_main_process():
                        print(f"[Warning] Failed to load scheduler_state_dict. Will continue with fresh scheduler state. Error: {e}")
            
            start_epoch = checkpoint['epoch'] + 1
        
        elif not init_weights_exists:
            if resume_path and is_main_process():
                print(f"Warning: RESUME_PATH not found. Starting from scratch.")
                self.load_pretrained_mask_encoder("/home/hyc/hyc_work/sceneGraph/SGG_pretrain/ckpt-epoch-300.pth")
            elif is_main_process():
                print("No RESUME_PATH. Starting from scratch.")

        # [DDP] epoch 记录在 raw_model 上
        self.raw_model.epoch = start_epoch
        
        if self.total == 0:
            if is_main_process(): print('No training data provided!')
            return

        ''' Train '''
        self.model.train()


        while(keep_training):
            # [DDP] 3. 关键：每个 epoch 开始前设置 sampler 的 epoch，保证 shuffle 的随机性
            train_sampler.set_epoch(self.raw_model.epoch)
            
            if self.raw_model.epoch > self.config.MAX_EPOCHES:
                break

            # [DDP] 只有主进程显示进度条
            if is_main_process():
                loader_iter = tqdm(train_loader, total=self.total, desc=f'Epoch {self.raw_model.epoch}', dynamic_ncols=True)
            else:
                loader_iter = train_loader

            for batch_idx, items in enumerate(loader_iter):
                obj_points, descriptor, edge_indices, anchor_ids, batch_ids, \
                    obj_points_spatial, cur_obj_texts, obj_labels = \
                    self.data_processing_train_pdiff(items)
                
                ''' forward '''
                # DDP forward
                total_loss, diff_loss, triplet_loss, edge_loss, obj_loss, contrastive_loss, obj_label_contrastive_loss, total_metric = self.model(
                    obj_points.permute(0,2,1).contiguous(), edge_indices, obj_points_spatial,
                    descriptor=descriptor, batch_ids=batch_ids, anchor_id=anchor_ids, 
                    istrain=True, cur_obj_texts=cur_obj_texts, obj_labels=obj_labels
                )
                
                # TensorBoard Log (仅 Rank 0)
                global_step = self.raw_model.epoch * self.total + batch_idx
                
                base_momentum = 0.99  # 通常自监督起点是 0.99 或 0.996，0.9 可能太低了，你可以自己定
                final_momentum = 1.0
                current_momentum = self.get_current_momentum(global_step, self.max_iteration, base_momentum, final_momentum)
                
                if is_main_process() and self.writer is not None:
                    self.writer.add_scalar('Train/Total_Loss', total_loss.item(), global_step)
                    self.writer.add_scalar('Train/Diff_Loss', diff_loss.item(), global_step)
                    self.writer.add_scalar('Train/Triplet_Loss', triplet_loss.item(), global_step)
                    self.writer.add_scalar('Train/Edge_Loss', edge_loss.item(), global_step)
                    self.writer.add_scalar('Train/Obj_Loss', obj_loss.item(), global_step)
                    self.writer.add_scalar('Train/ctr_loss', contrastive_loss.item(), global_step)
                    self.writer.add_scalar('Train/obj_label_ctr_loss', obj_label_contrastive_loss.item(), global_step)
                    self.writer.add_scalar('Train/total_metric', total_metric, global_step)
                
                current_lr = self.optimizer.param_groups[1]['lr']

                # 进度条刷新 (仅 Rank 0)
                if is_main_process():
                    loader_iter.set_postfix({
                        'dif': f'{diff_loss.item():.4f}',
                        'tri_ls': f'{triplet_loss.item():.4f}',
                        'rel_ls': f'{edge_loss.item():.4f}',
                        'obj_ls': f'{obj_loss.item():.4f}',
                        'ctr_ls': f'{contrastive_loss.item():.4f}',
                        'obj_ctr': f'{obj_label_contrastive_loss.item():.4f}',
                        'met': f'{total_metric:.1f}',
                        'lr': f'{current_lr:.6f}'
                    })
                
                self.backward(total_loss,current_momentum)
            
            # ==========================================
            # [新增/修改] 模型保存与验证逻辑
            # ==========================================
            current_epoch = self.raw_model.epoch
            
            # 1. 保存模型 (仅 Rank 0)
            if current_epoch % 10 == 0 and is_main_process():
                if hasattr(self, 'save_dir'):
                    save_path = os.path.join(self.save_dir, f'model_epoch_{current_epoch}.pth')
                    state_dict = self.raw_model.state_dict()
                    checkpoint = {
                        'epoch': current_epoch,
                        'model_state_dict': state_dict,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.lr_scheduler.state_dict(),
                        'loss': total_loss.item()
                    }
                    print(f'\n[Epoch {current_epoch}] Saving Checkpoint: {save_path}')
                    torch.save(checkpoint, save_path)

            # 2. 执行验证 (仅 Rank 0，每 10 epoch 且 > 0)
            # 注意：在 DDP 中，如果只让 Rank 0 做耗时操作，其他进程可能会超时。
            # 建议加上 barrier 确保同步，或者确保验证时间在 nccl timeout 范围内。
            validate_every_n_epoch = bool(getattr(self.config, 'VALIDATE_EVERY_N_EPOCH', True))
            valid_interval = int(getattr(self.config, 'VALID_INTERVAL', 10))
            if validate_every_n_epoch and valid_interval > 0 and current_epoch > 0 and current_epoch % valid_interval == 0:
                if is_main_process():
                    print(f"\n[Epoch {current_epoch}] Starting Validation...")
                    self.validation_for_cls(epoch=current_epoch)
                    print(f"[Epoch {current_epoch}] Validation Finished.")
                
                # [DDP] 关键：让其他 GPU 等待 Rank 0 跑完验证，防止不同步进入下一个 Epoch
                dist.barrier()
            
            # 3. 恢复训练模式 (验证会切换到 eval，必须切回来)
            self.model.train()
            
            # if is_main_process():
            #     self.swav_monitor.report(epoch_idx=self.raw_model.epoch)
            
            self.raw_model.epoch += 1
            # 这里的 loader = iter(train_loader) 不需要手动重置，外层 while 循环配合 for loop 即可，
            # 如果需要无限循环的 iterator 逻辑，在 DDP 下比较麻烦，建议标准的 epoch 循环结构。
            # 当前代码结构是 while(keep_training) -> for enumerate，这是标准的 epoch 结构。
    
    def cuda(self, *args):
        # [DDP] 移动到 self.device (根据 rank 确定的 device)
        return [item.to(self.device, non_blocking=True) for item in args]

    def get_current_momentum(self, current_step, max_steps, base_tau=0.99, final_tau=1.0):
        # 简单的线性增长
        return base_tau + (final_tau - base_tau) * (current_step / max_steps)
    
    def backward(self, loss, current_momentum):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        # =========================================================
        # [修改] 核心修正点
        # =========================================================
        # 检查 self.model 是否被 DDP 包装过
        if hasattr(self.model, 'module'):
            # 如果是 DDP，访问 .module 来调用自定义方法
            self.model.module._update_target(momentum=current_momentum)
        else:
            # 如果是单卡或者没用 DDP，直接调用
            self.model._update_target(momentum=current_momentum)
        # =========================================================
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        
    # Validation 部分通常只需要一个 GPU 跑，或者 DDP 需要额外的 gather 操作
    # 简单的做法是：只在 rank 0 运行 validation，或者不管它（如果是离线评测）
    # 如果要在训练中运行 validation，需要类似 train 的处理，但 shuffle=False
    @torch.no_grad()
    def validation_for_cls(self, epoch):
        """
        在线验证函数：提取特征并计算聚类指标，保存结果到 txt
        注意：此函数仅在 is_main_process() == True 时被调用
        """
        print(f"--- Running Validation for Epoch {epoch} ---")
        
        # 1. 创建验证集 DataLoader
        # 注意：验证集不需要 DistributedSampler，因为只在 Rank 0 跑
        val_dataset = build_dataset_for_clustering(self.config) # 确保这里构建的是验证集/全集
        val_loader = CustomDataLoader(
            config = self.config,
            dataset=val_dataset,
            batch_size=self.config.Batch_Size, # 可以稍微大一点，因为不存梯度
            num_workers=4,
            drop_last=False,
            shuffle=False,
            collate_fn=collate_fn_mmg,
        )

        # 2. 切换模型模式
        self.model.eval()
        
        # [DDP] 获取底层的 model，以便调用 forward_cls
        # 如果 self.model 是 DDP 包装的，它没有 forward_cls 方法，必须通过 .module 访问
        infer_model = self.model.module if isinstance(self.model, DDP) else self.model

        all_edge_feats = []
        all_obj_feats = []
        all_gt_rel_cls = []
        all_gt_obj_cls = []
        
        # 使用 tqdm 显示验证进度
        loader = tqdm(val_loader, desc=f'Validating Ep {epoch}', dynamic_ncols=True)
        
        for batch_idx, items in enumerate(loader):
            # 获取数据
            obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = self.data_processing_train(items)
        
            # 前向传播 (使用 infer_model)
            gcn_edge_feature_3d, gcn_obj_feature_3d = infer_model.forward_cls(
                obj_points.permute(0,2,1).contiguous(), 
                edge_indices,
                descriptor=descriptor, 
                batch_ids=batch_ids, 
                istrain=False
            )        
            
            # [显存优化] 立即转移到 CPU，否则大量数据积压在 GPU 会 OOM
            all_edge_feats.append(gcn_edge_feature_3d.cpu())
            all_obj_feats.append(gcn_obj_feature_3d.cpu())
            all_gt_obj_cls.append(gt_class.cpu())
            all_gt_rel_cls.append(gt_rel_cls.cpu())
        
        # 3. 拼接所有数据
        if len(all_edge_feats) > 0:
            all_edge_feats = torch.cat(all_edge_feats, dim=0)
            all_obj_feats = torch.cat(all_obj_feats, dim=0)
            all_gt_obj_cls = torch.cat(all_gt_obj_cls, dim=0)
            all_gt_rel_cls = torch.cat(all_gt_rel_cls, dim=0)
        else:
            print("Warning: Validation set is empty!")
            return

        print(f"Collected {len(all_obj_feats)} object features for evaluation.")

        # 4. 计算指标并画图
        # 确保目录存在
        cm_save_dir = os.path.join(self.config.analysis_save_dir, "cm_save")
        if not os.path.exists(cm_save_dir):
            os.makedirs(cm_save_dir)

        # 评估 Object
        metrics_obj = evaluate_and_plot_clustering(
            all_obj_feats, 
            all_gt_obj_cls, 
            save_path=os.path.join(cm_save_dir, f"cls_obj_{epoch}.png"),
            metric_prefix="val_obj"
        )
        
        # 评估 Edge (Relation)
        metrics_edge = evaluate_and_plot_clustering(
            all_edge_feats, 
            all_gt_rel_cls, 
            save_path=os.path.join(cm_save_dir, f"cls_edge_{epoch}.png"),
            metric_prefix="val_edge"
        )

        # 5. 保存指标到 TXT 文件
        self.log_metrics_to_txt(metrics_obj, "metrics_obj.txt", epoch)
        self.log_metrics_to_txt(metrics_edge, "metrics_edge.txt", epoch)
    
    def log_metrics_to_txt(self, metrics, filename, epoch):
        """
        将 metrics 字典按行追加写入 txt 文件
        格式: Epoch: 10 | key1: val1 | key2: val2 ...
        """
        if not is_main_process():
            return

        save_dir = os.path.join(self.config.analysis_save_dir, "logs_metrics")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        file_path = os.path.join(save_dir, filename)
        
        # 构造记录字符串
        log_str = f"Epoch: {epoch}"
        for k, v in metrics.items():
            # 格式化数值，保留4位小数
            val_str = f"{v:.4f}" if isinstance(v, (float, np.float32, np.float64)) else str(v)
            log_str += f" | {k}: {val_str}"
        log_str += "\n"

        # 追加写入
        with open(file_path, "a") as f:
            f.write(log_str)
        
        print(f"[Validation] Metrics saved to {file_path}")
