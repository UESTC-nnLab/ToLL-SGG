from tqdm import tqdm
import copy
import os
import torch.optim as optim
import numpy as np
import torch
from src.dataset.DataLoader import (CustomDataLoader, collate_fn_mmg_diff, collate_fn_mmg)
from src.dataset.dataset_builder import build_dataset, build_dataset_for_clustering
from src.model.diff_trans.models.PointDif_dino import PointDif
from src.model.optimizer.scheduler import get_warmup_cosine_scheduler, get_freeze_warmup_scheduler
from src.model.diff_trans.models.monitor import EpochCollapseMonitor
from src.model.diff_trans.models.clustering import cluster_and_visualize, visualize_with_gt, evaluate_and_plot_clustering
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
def get_param_groups(module, base_lr, weight_decay, amsgrad):
    """
    自动将模块内的参数分为两组：
    1. decay_group: 权重 (Weights) -> 使用 weight_decay
    2. no_decay_group: 偏置 (Bias) 和 Norm层参数 -> weight_decay = 0.0
    """
    decay_params = []
    no_decay_params = []
    
    # 遍历模块内所有参数
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        
        # 判断是否应该取消 decay
        # 1. 是一维向量 (通常是 Bias 或 Norm 的 scale/shift)
        # 2. 名字里包含 bias
        # 3. 名字里包含 norm (如 LayerNorm, BatchNorm)
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
            'weight_decay': 0.0,  # 关键：这里强制为 0
            'amsgrad': amsgrad
        })
    
    return groups

class Pdiff4SSG_Pretraining():
    def __init__(self, config, val_cls_mode=False):
        self.config = config
        self.model_name = 'pdiff_SGG'
        self.save_dir = self.config.PATH
        # os.makedirs(self.save_dir, exist_ok=True)
        ''' Build dataset ''' 
        if val_cls_mode:      
            self.dataset_train = build_dataset_for_clustering(self.config)
        else:
            self.dataset_train = build_dataset(
                "train_scannet",
                True,
                root_ScanNet=self.config.root_ScanNet,
                json_path=self.config.json_path,
            )
        
        self.total = self.config.total = len(self.dataset_train) // self.config.Batch_Size
        self.max_iteration = self.config.max_iteration = int(float(self.config.MAX_EPOCHES)*len(self.dataset_train) // self.config.Batch_Size)
        self.max_iteration_scheduler = self.config.max_iteration_scheduler = int(float(self.config.MAX_EPOCHES)*len(self.dataset_train) // self.config.Batch_Size)
        
        # --- [新增] 初始化 TensorBoard Writer ---
        # 建议加上时间戳，防止每次运行覆盖之前的日志
        log_root = getattr(self.config, 'analysis_save_dir', None)
        if log_root is None:
            log_root = os.path.join(os.getcwd(), "outputs")
        log_dir = os.path.join(log_root, "log_runs", "experiment_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        
        self.swav_monitor = EpochCollapseMonitor(80)
        
        ''' Build Model '''
        self.model = PointDif(self.config).cuda()
        
        # --- 构建最终的 param_groups ---
        param_groups = []

        # 1. 常规模块 (自动拆分 decay/no_decay)
        # 注意：你需要为每个模块调用一次这个函数
        param_groups.extend(get_param_groups(self.model.mask_encoder, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.rel_encoder_3d, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.ca_net, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.mlp_3d, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.point_diffusion.net, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.predictor_triplet, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.predictor_edge, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.predictor_obj, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        # 2. 特殊模块: SwAV Prototypes (完全不 decay)
        param_groups.append({
            'params': self.model.swav_reg_triplet.parameters(), 
            'lr': float(config.LR), 
            'weight_decay': 0.0, # 规范写法
            'amsgrad': self.config.AMSGRAD
        })
        
        # 3. 特殊模块: MMG (学习率减半)
        param_groups.extend(get_param_groups(self.model.mmg, float(config.LR)/2, self.config.W_DECAY, self.config.AMSGRAD))

        # 4. 特殊参数: Mask Token
        # Token 通常被视为 Weight，需要 decay
        param_groups.append({
            'params': self.model.mask_token, 
            'lr': float(config.LR), 
            'weight_decay': self.config.W_DECAY, 
            'amsgrad': self.config.AMSGRAD
        })
        param_groups.append({
            'params': self.model.edge_mask_token.parameters(), 
            'lr': float(config.LR), 
            'weight_decay': self.config.W_DECAY, 
            'amsgrad': self.config.AMSGRAD
        })

        # 初始化优化器
        self.optimizer = optim.AdamW(param_groups)
        
        self.lr_scheduler = get_freeze_warmup_scheduler(self.optimizer, self.total*10, self.config.max_iteration)
        self.optimizer.zero_grad()
    
    def load_pretrained_mask_encoder(self, checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # 1. 加载权重文件 (假设是 .pth 或 .pt)
        # map_location='cpu' 是一种安全做法，防止GPU显存不足或设备ID不匹配
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        
        # 处理 checkpoint 可能包含 'state_dict' 键的情况，也可能直接就是字典
        raw_state_dict = checkpoint['pointdif'] if 'pointdif' in checkpoint else checkpoint

        # 准备一个新的字典用于存放清洗后的 mask_encoder 权重
        mask_encoder_dict = {}

        print("Processing state_dict keys...")

        for k, v in raw_state_dict.items():
            # --- 步骤 1: 去掉 'module.' 前缀 ---
            # 多卡训练保存时会自动加上 'module.'，单卡加载需要去掉
            if k.startswith('module.'):
                name = k[7:]  # 去掉前7个字符 'module.'
            else:
                name = k

            # --- 步骤 2 & 3: 筛选 mask_encoder 并处理 Key ---
            # 只有以 'mask_encoder' 开头的才是我们需要的部分
            if name.startswith('mask_encoder.'):
                
                new_key = name.replace('mask_encoder.', '', 1)
                mask_encoder_dict[new_key] = v

        # --- 步骤 4: 加载到子模型 ---
        if len(mask_encoder_dict) > 0:
            missing, unexpected = self.model.mask_encoder.load_state_dict(mask_encoder_dict, strict=True)
            
            print(f"Success! Loaded {len(mask_encoder_dict)} keys into self.model.mask_encoder.")
            if len(missing) > 0:
                print(f"Warning: Missing keys in sub-model: {missing}")
            if len(unexpected) > 0:
                print(f"Warning: Unexpected keys in dict: {unexpected}")
        else:
            print("Error: No keys starting with 'mask_encoder' found in the checkpoint!")
    
    def load(self, best=False):
        return self.model.load(best)
    @torch.no_grad()
    def data_processing_train_pdiff(self, items):
        obj_points, descriptor, edge_indices, anchor_ids, batch_ids, obj_points_spatial, cur_obj_texts = items
        obj_points = obj_points.permute(0, 2, 1).contiguous()
        obj_points, edge_indices, descriptor, batch_ids, obj_points_spatial = \
            self.cuda(obj_points, edge_indices, descriptor, batch_ids, obj_points_spatial)

        return obj_points, descriptor, edge_indices, anchor_ids, batch_ids, obj_points_spatial, cur_obj_texts
    
    @torch.no_grad()
    def data_processing_train(self, items):
        obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids,_ = items
        obj_points = obj_points.permute(0, 2, 1).contiguous()
        obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = \
            self.cuda(obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids)
        return obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids
       
    def train(self):
        ''' create data loader '''
        drop_last = True
        train_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_train,
            batch_size=self.config.Batch_Size,
            num_workers=4, #self.config.WORKERS
            drop_last=drop_last,
            shuffle=True,
            collate_fn=collate_fn_mmg_diff,
        )
                
        keep_training = True

        # --- [START] 新增 Checkpoint 加载逻辑 ---
        
        start_epoch = 0
        
        # 尝试从 config 中获取 RESUME_PATH
        resume_path = getattr(self.config, 'RESUME_PATH', None)
        
        if resume_path and os.path.exists(resume_path):
            print(f"Resuming training from checkpoint: {resume_path}")
            
            # 确保 checkpoint 加载到正确的设备
            checkpoint = torch.load(resume_path, map_location=torch.device('cuda'))
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            try:
                self.model.load_state_dict(state_dict, strict=True)
                print("[Info] Loaded model_state_dict with strict=True.")
            except Exception as e:
                print(f"[Warning] strict=True load failed, will try strict=False. Error: {e}")
                sd_no_module = {
                    (k[len('module.'):] if k.startswith('module.') else k): v
                    for k, v in state_dict.items()
                }
                incompatible = self.model.load_state_dict(sd_no_module, strict=False)
                print(f"[Warning] Loaded model_state_dict with strict=False. Missing: {len(incompatible.missing_keys)}, Unexpected: {len(incompatible.unexpected_keys)}")

            optimizer_loaded = False
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                optimizer_loaded = True
            except Exception as e:
                print(f"[Warning] Failed to load optimizer_state_dict (param_groups may have changed). Will continue with fresh optimizer. Error: {e}")
            
            if optimizer_loaded and 'scheduler_state_dict' in checkpoint:
                try:
                    self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    print(f"[Warning] Failed to load scheduler_state_dict. Will continue with fresh scheduler state. Error: {e}")
            
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}. Last recorded loss: {checkpoint.get('loss', 'N/A')}")
        
        else:
            if resume_path:
                print(f"Warning: RESUME_PATH '{resume_path}' was specified but file not found. Starting from scratch.")
                self.load_pretrained_mask_encoder("/home/hyc/hyc_work/sceneGraph/SGG_pretrain/ckpt-epoch-300.pth")
            else:
                print("No RESUME_PATH specified. Starting training from scratch (epoch 0).")
                

        self.model.epoch = start_epoch
        # --- [END] 新增 Checkpoint 加载逻辑 ---
        
        if self.total == 1:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        ''' Resume data loader to the last read location '''
        loader = iter(train_loader)


        for k, p in self.model.named_parameters():
            if p.requires_grad:
                print(f"Para {k} need grad")

        ''' Train '''
        self.model.train()

        validate_at_start = bool(getattr(self.config, 'VALIDATE_AT_START', False))
        if validate_at_start:
            print(f"\n[Epoch {self.model.epoch}] Starting Validation (at start)...")
            self.validation_for_cls(epoch=self.model.epoch)
            print(f"[Epoch {self.model.epoch}] Validation Finished (at start).")
            self.model.train()

        while(keep_training):
            
            self.swav_monitor.reset()
            
            if self.model.epoch > self.config.MAX_EPOCHES:#
                break

            # --- [修改 1] 使用 tqdm 包装 loader ---
            # dynamic_ncols=True 可以让进度条自动适应终端宽度
            pbar = tqdm(loader, total=self.total, desc=f'Epoch {self.model.epoch}', dynamic_ncols=True)

            # --- [设置] ---
            log_interval = 50  # 每 50 个 batch 记录并打印一次
            # 初始化累加器字典
            running_metrics = {
                'total_loss': 0.0,
                'diff_loss': 0.0,
                'triplet_loss': 0.0,
                'edge_loss': 0.0,
                'contrastive_loss': 0.0,
                'total_metric': 0.0
            }
            
            for batch_idx, items in enumerate(pbar):
                ''' get data '''
                # print("------training------") # --- [修改 2] 必须注释掉，否则会打断进度条 ---
                
                obj_points, descriptor, edge_indices, anchor_ids, batch_ids, obj_points_spatial, cur_obj_texts = self.data_processing_train_pdiff(items)
                
                ''' forward '''
                total_loss, diff_loss, triplet_loss, edge_loss, contrastive_loss, total_metric, gcn_edge_feature_3d = self.model(
                    obj_points.permute(0,2,1).contiguous(), edge_indices, obj_points_spatial, 
                    descriptor=descriptor, batch_ids=batch_ids, anchor_id=anchor_ids, 
                    istrain=True, cur_obj_texts=cur_obj_texts
                )
                
                # ------------------- [新增] TensorBoard 记录逻辑 -------------------
                # 1. 计算当前的全局步数 (Global Step)
                # 这样在 TensorBoard 的 x 轴上，不同 epoch 的曲线会连起来，而不是重叠
                # 假设 self.model.epoch 是当前 epoch 序号，len(pbar) 是一个 epoch 中的总 batch 数
                global_step = self.model.epoch * len(pbar) + batch_idx

                # 2. 记录 Loss (使用 .item() 将 Tensor 转为 Python 数值)
                # 建议使用 'Train/' 前缀将它们归类在同一组
                if self.writer is not None:
                    self.writer.add_scalar('Train/Total_Loss', total_loss.item(), global_step)
                    self.writer.add_scalar('Train/Diff_Loss', diff_loss.item(), global_step)
                    self.writer.add_scalar('Train/Triplet_Loss', triplet_loss.item(), global_step)
                    self.writer.add_scalar('Train/Edge_Loss', edge_loss.item(), global_step)
                    self.writer.add_scalar('Train/Obj_Loss', contrastive_loss.item(), global_step)
                    self.writer.add_scalar('Train/total_metric', total_metric, global_step)
                
                with torch.no_grad():
                    z1 = torch.nn.functional.normalize(gcn_edge_feature_3d, dim=1, p=2)
                    mcr_stats = getattr(self.model, '_last_edge_mcr_stats', None)
        
                # 3. 更新统计 (每个 Batch)
                # 只需要把其中一个 view (z1, q1) 传进去即可代表本 Batch 情况
                self.swav_monitor.update(embeddings=z1, swav_q=None, mcr_stats=mcr_stats)
                
                current_lr = self.optimizer.param_groups[1]['lr']

                # --- [修改 3] 使用 set_postfix 在同一行刷新显示关键指标 ---
                # 为了美观，保留小数位
                pbar.set_postfix({
                    # 'Epoch': self.model.epoch,                  # 显示当前 Epoch
                    # 'Iter': f'{batch_idx + 1}/{self.total}',    # 显示 Iteration: 当前/总数
                    'dif': f'{diff_loss.item():.4f}',
                    'tri_ls': f'{triplet_loss.item():.4f}',
                    'rel_ls': f'{edge_loss.item():.4f}',
                    'obj_ls': f'{contrastive_loss.item():.4f}',
                    'met': f'{total_metric:.1f}',
                    'lr': f'{current_lr:.6f}'
                })
                
                self.backward(total_loss)
            
            if self.model.epoch % 10 == 0:
                if hasattr(self, 'save_dir'):
                    save_path = os.path.join(self.save_dir, f'model_epoch_{self.model.epoch}.pth')

                    state_dict = self.model.state_dict()
                    
                    # --- [START] 修改 Checkpoint 保存逻辑 ---
                    checkpoint = {
                        'epoch': self.model.epoch,
                        'model_state_dict': state_dict,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.lr_scheduler.state_dict(), # <-- [修改] 增加 scheduler 状态
                        'loss': total_loss.item()
                            }
                    # --- [END] 修改 Checkpoint 保存逻辑 ---

                    print(f'\n[Epoch {self.model.epoch}] 保存模型到: {save_path}')
                    torch.save(checkpoint, save_path)
                else:
                    print(f'\n[Epoch {self.model.epoch}] 警告: 未定义 self.save_dir, 跳过保存。')
            
            self.swav_monitor.report(epoch_idx=self.model.epoch)

            validate_every_n_epoch = bool(getattr(self.config, 'VALIDATE_EVERY_N_EPOCH', True))
            valid_interval = int(getattr(self.config, 'VALID_INTERVAL', 10))
            if validate_every_n_epoch and valid_interval > 0 and self.model.epoch > 0 and self.model.epoch % valid_interval == 0:
                print(f"\n[Epoch {self.model.epoch}] Starting Validation...")
                self.validation_for_cls(epoch=self.model.epoch)
                print(f"[Epoch {self.model.epoch}] Validation Finished.")
                self.model.train()

            self.model.epoch += 1
            loader = iter(train_loader)
    
    def validation(self):
        ''' create data loader '''
        train_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_train,
            batch_size=self.config.Batch_Size,
            num_workers=1, #self.config.WORKERS
            drop_last=False,
            shuffle=False,
            collate_fn=collate_fn_mmg_diff,
        )
        
        def remove_module_prefix(state_dict):
            from collections import OrderedDict
            """
            去除权重字典键名中的 'module.' 前缀
            """
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # 如果键名以 'module.' 开头，则去掉前7个字符
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            return new_state_dict
        
        keep_training = True

        self.model.epoch = 0
        
        if self.total == 1:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        ''' Resume data loader to the last read location '''
        loader = iter(train_loader)

        # model_dicts = torch.load("/home/hyc/hyc_work/sceneGraph/SGG_DIR/save_path/model_epoch_300.pth")
        
        # self.model.load_state_dict(model_dicts['model_state_dict'])
        
        all_edge_feats = []
        all_obj_feats = []
        ''' Train '''
        self.model.eval()

                    
        for batch_idx, items in enumerate(loader):
            ''' get data '''
            obj_points, descriptor, edge_indices, anchor_ids, batch_ids, obj_points_spatial, cur_obj_texts = self.data_processing_train_pdiff(items)
        
            with torch.no_grad():
                diff_loss, total_x0_metric, gcn_edge_feature_3d, gcn_obj_feature_3d = self.model.forward_ori(obj_points.permute(0,2,1).contiguous(), edge_indices, obj_points_spatial, descriptor=descriptor, batch_ids=batch_ids, anchor_id=anchor_ids, istrain=False)
                print('Epoch: %d, Iteration: %d / %d, total_x0_metric: %.4f, diff_loss: %.4f' % (self.model.epoch, batch_idx+1, self.total, total_x0_metric, diff_loss))
        
        
            all_edge_feats.append(gcn_edge_feature_3d)
            all_obj_feats.append(gcn_obj_feature_3d)
        
        all_edge_feats = torch.cat(all_edge_feats, dim=0)
        all_obj_feats = torch.cat(all_obj_feats, dim=0)
        
       
        cluster_and_visualize(all_obj_feats, 100, title_prefix="Object Features",\
                              save_path="/home/hyc/hyc_work/sceneGraph/SGG_DIR/clustering_new1/object_features_cluster.png")
        cluster_and_visualize(all_edge_feats, 30, title_prefix="Edge Features",\
                              save_path="/home/hyc/hyc_work/sceneGraph/SGG_DIR/clustering_new1/edge_features_cluster.png")
    
    
    def validation_for_cls(self, epoch=None):
        ''' create data loader '''
        
        train_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_train,
            batch_size=self.config.Batch_Size,
            num_workers=4, #self.config.WORKERS
            drop_last=False,
            shuffle=False,
            collate_fn=collate_fn_mmg,
        )

        ''' Resume data loader to the last read location '''
        loader = iter(train_loader)

        # self.load_pretrained_mask_encoder("/home/honsen/tartan/ckpt-epoch-300.pth")
        # 离线评测场景：模型权重可能已由外部（例如 main.py --eval_only --eval_ckpt）加载
        if not getattr(self, '_ckpt_loaded', False):
            if epoch is not None:
                model_dicts = torch.load(f"/home/hyc/hyc_work/sceneGraph/SGG_DIR/save_path/model_epoch_{epoch}.pth")
            else:
                model_dicts = torch.load("/home/hyc/hyc_work/sceneGraph/SGG_DIR/save_path/model_epoch_80.pth")
            self.model.load_state_dict(model_dicts['model_state_dict'])
        
        # self.load_pretrained_mask_encoder("/home/hyc/hyc_work/sceneGraph/SGG_pretrain/ckpt-epoch-300.pth")
        
        all_edge_feats = []
        all_obj_feats = []
        all_gt_rel_cls = []
        all_gt_obj_cls = []
        all_obj_points = []
        ''' Train '''
        self.model.eval()

                    
        for batch_idx, items in enumerate(loader):
            ''' get data '''
            obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = self.data_processing_train(items)
        
            with torch.no_grad():
                gcn_edge_feature_3d, gcn_obj_feature_3d = self.model.forward_cls(obj_points.permute(0,2,1).contiguous(), edge_indices,\
                    descriptor=descriptor, batch_ids=batch_ids, istrain=False)        
        
            all_edge_feats.append(gcn_edge_feature_3d)
            all_obj_feats.append(gcn_obj_feature_3d)
            all_gt_obj_cls.append(gt_class)
            all_gt_rel_cls.append(gt_rel_cls)
            all_obj_points.append(obj_points)
            print('Iteration: %d / %d' % (batch_idx+1, self.total))
        
        all_edge_feats = torch.cat(all_edge_feats, dim=0)
        all_obj_feats = torch.cat(all_obj_feats, dim=0)
        all_gt_obj_cls = torch.cat(all_gt_obj_cls, dim=0)
        all_gt_rel_cls = torch.cat(all_gt_rel_cls, dim=0)   
        all_obj_points = torch.cat(all_obj_points, dim=0)
        
        save_dict = {
                "edge_feats": all_edge_feats.detach().cpu(),    # 移除梯度并转到CPU
                "obj_feats": all_obj_feats.detach().cpu(),
                "gt_obj_cls": all_gt_obj_cls.detach().cpu(),
                "gt_rel_cls": all_gt_rel_cls.detach().cpu(),
                "obj_points": all_obj_points.detach().cpu()
            }
        # 2. 保存到文件
        # torch.save(save_dict, "/home/honsen/honsen/SceneGraph/SG_pretrain_diff/scene_graph_features.pt")
        
        # analyze_kmeans_clusters(all_obj_feats, raw_point_clouds=all_obj_points, n_clusters=100, save_dir="/home/honsen/honsen/SceneGraph/SG_pretrain_diff/clustering_show")
        
        # 4. 计算指标并画图
        # 确保目录存在
        cm_save_dir = os.path.join(self.config.analysis_save_dir, "cm_save")
        if not os.path.exists(cm_save_dir):
            os.makedirs(cm_save_dir)

        # # 评估 Object
        metrics_obj = evaluate_and_plot_clustering(
            all_obj_feats, 
            all_gt_obj_cls, 
            save_path=os.path.join(cm_save_dir, f"cls_obj_{epoch}.png"),
            metric_prefix="val_obj"
        )
        # print("Object Clustering Metrics:", metrics_obj)
        # # 评估 Edge (Relation)
        metrics_edge = evaluate_and_plot_clustering(
            all_edge_feats, 
            all_gt_rel_cls, 
            save_path=os.path.join(cm_save_dir, f"cls_edge_{epoch}.png"),
            metric_prefix="val_edge"
        )
        
        vis_save_dir = os.path.join(self.config.analysis_save_dir, "clustering_new1_ssg")
        if not os.path.exists(vis_save_dir):
            os.makedirs(vis_save_dir)
        # print("Edge Clustering Metrics:", metrics_edge)
        visualize_with_gt(all_obj_feats, all_gt_obj_cls, title_prefix="Object Features",\
                              save_path=os.path.join(vis_save_dir, f"object_features_cluster_{epoch}.png"))
        visualize_with_gt(all_edge_feats, all_gt_rel_cls, title_prefix="Edge Features",\
                              save_path=os.path.join(vis_save_dir, f"edge_features_cluster_{epoch}.png"), ignore_zero_label=True)

        # cluster_and_visualize(all_obj_feats, 100, title_prefix="Object Features",\
        #                       save_path="/home/honsen/honsen/SceneGraph/SG_pretrain_diff/clustering_dir_310/object_features_cluster.png")
        # cluster_and_visualize(all_edge_feats, 26, title_prefix="Edge Features",\
        #                       save_path="/home/honsen/honsen/SceneGraph/SG_pretrain_diff/clustering_dir_310/edge_features_cluster.png")
    
    def cuda(self, *args):
        return [item.to(self.config.DEVICE) for item in args]

    def save(self,epoch):
        self.model.save(epoch)

    def backward(self, loss):
        loss.backward()
        # 有返回值的！它会返回裁剪前的总范数 (Total Norm)
        # total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
        
        # 调试用：如果发现 total_norm 经常飙到 10 以上，说明 1.0 的裁剪起大作用了
        # if total_norm > 5.0:
        #     print(f"Gradient exploded! Norm: {total_norm:.2f} (Clipped to 1.0)")
        self.optimizer.step()
        self.optimizer.zero_grad()
        # update lr
        self.lr_scheduler.step()
