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
from src.model.diff_trans.models.PointDif_dino_que import PointDif
from src.model.optimizer.scheduler import get_warmup_cosine_scheduler, get_freeze_warmup_scheduler
from src.model.diff_trans.models.monitor import EpochCollapseMonitor
from src.model.diff_trans.models.clustering import cluster_and_visualize, visualize_with_gt, analyze_kmeans_clusters

# [DDP] 辅助函数
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
    # ... (保持原样) ...
    decay_params = []
    no_decay_params = []
    for name, param in module.named_parameters():
        if not param.requires_grad: continue
        if param.ndim <= 1 or "bias" in name or "norm" in name or "bn" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups = []
    if len(decay_params) > 0:
        groups.append({'params': decay_params, 'lr': base_lr, 'weight_decay': weight_decay, 'amsgrad': amsgrad})
    if len(no_decay_params) > 0:
        groups.append({'params': no_decay_params, 'lr': base_lr, 'weight_decay': 0.0, 'amsgrad': amsgrad})
    return groups

class Pdiff4SSG_Pretraining_ddp():
    def __init__(self, config, val_cls_mode=False):
        self.config = config
        self.model_name = 'pdiff_SGG'
        self.save_dir = self.config.PATH
        self.cnt = 0
        os.makedirs(self.save_dir, exist_ok=True)
        
        # [DDP Init]
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.local_rank = int(os.environ['LOCAL_RANK'])
            if not dist.is_initialized(): dist.init_process_group(backend='nccl')
            print(f"[Info] Running in DDP mode. Rank: {dist.get_rank()}")
        else:
            print("[Warning] No DDP environment found. Fallback to Single-GPU mode.")
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            self.local_rank = 0
        
        if not dist.is_initialized(): dist.init_process_group(backend='nccl')
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device('cuda', self.local_rank)
        
        ''' Build dataset ''' 
        if is_main_process():
            dataset_cfg = getattr(self.config, 'dataset', None)
            val_root = getattr(dataset_cfg, 'root', None) if dataset_cfg is not None else None
            val_root_3rscan = getattr(dataset_cfg, 'root_3rscan', None) if dataset_cfg is not None else None
            print(f"[Config] Validation dataset roots: dataset.root={val_root}, dataset.root_3rscan={val_root_3rscan}")

        if val_cls_mode:
            self.dataset_train = build_dataset_for_clustering(self.config)
        else:
            self.dataset_train = build_pretrain_dataset(self.config, for_train=True)
        
        self.writer = None
        if is_main_process():
            if not os.path.exists(os.path.join(self.config.analysis_save_dir, "log_runs")):
                os.makedirs(os.path.join(self.config.analysis_save_dir, "log_runs"))
            log_dir = os.path.join(self.config.analysis_save_dir, "log_runs", "experiment_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            self.writer = SummaryWriter(log_dir=log_dir)
        
        self.swav_monitor = EpochCollapseMonitor(200)
        
        # Model
        self.model = PointDif(self.config).to(self.device)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # Optimizer
        param_groups = []
        param_groups.extend(get_param_groups(self.model.mask_encoder, float(config.LR)/2, self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.rel_encoder_3d, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.ca_net, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.mlp_3d, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        if bool(getattr(self.config, "DIFFUSION_ENABLED", True)):
            param_groups.extend(get_param_groups(self.model.point_diffusion.net, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.predictor_triplet, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.predictor_edge, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.predictor_obj, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        if bool(getattr(self.config, "ATLAS_ALIGN_ENABLED", False)):
            param_groups.extend(get_param_groups(self.model.atlas_align_head, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        
        param_groups.append({'params': self.model.swav_reg_edge.parameters(), 'lr': float(config.LR), 'weight_decay': 0.0, 'amsgrad': self.config.AMSGRAD})
        param_groups.append({'params': self.model.swav_reg_triplet.parameters(), 'lr': float(config.LR), 'weight_decay': 0.0, 'amsgrad': self.config.AMSGRAD})
        param_groups.append({'params': self.model.swav_reg_obj.parameters(), 'lr': float(config.LR), 'weight_decay': 0.0, 'amsgrad': self.config.AMSGRAD})
        param_groups.extend(get_param_groups(self.model.mmg, float(config.LR)/2, self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.append({'params': self.model.mask_token, 'lr': float(config.LR), 'weight_decay': self.config.W_DECAY, 'amsgrad': self.config.AMSGRAD})
        param_groups.append({'params': self.model.edge_mask_token.parameters(), 'lr': float(config.LR), 'weight_decay': self.config.W_DECAY, 'amsgrad': self.config.AMSGRAD})

        self.optimizer = optim.AdamW(param_groups)
        self.optimizer.zero_grad()
        self.raw_model = self.model 
        self.model = DDP(self.raw_model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

    def _model_ref(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _branch_flags(self):
        model_ref = self._model_ref()
        return {
            'diffusion': bool(getattr(model_ref, 'diffusion_enabled', True)),
            'text_contrastive': bool(getattr(model_ref, 'text_contrastive_enabled', False)),
            'obj_label_contrastive': bool(getattr(model_ref, 'obj_label_contrastive_enabled', False)) and float(getattr(model_ref, 'obj_label_contrastive_weight', 0.0)) > 0,
            'atlas': bool(getattr(model_ref, 'atlas_align_enabled', False)),
        }

    def _count_effective_clusters(self, label_tensor, ignore_zero_label=True):
        if label_tensor is None:
            return 0
        if torch.is_tensor(label_tensor):
            labels_np = label_tensor.detach().cpu().numpy()
        else:
            labels_np = np.asarray(label_tensor)

        if labels_np.ndim > 1:
            labels_np = np.argmax(labels_np, axis=1)

        labels_np = labels_np.astype(int)
        if ignore_zero_label:
            labels_np = labels_np[labels_np != 0]

        if labels_np.size == 0:
            return 0
        return int(len(np.unique(labels_np)))
    
    # ... [load_pretrained_mask_encoder, data_processing_train_pdiff, data_processing_train 保持不变] ...
    def load_pretrained_mask_encoder(self, checkpoint_path):
        if is_main_process(): print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        raw_state_dict = extract_checkpoint_state_dict(checkpoint)
        raw_state_dict = normalize_state_dict_keys(raw_state_dict)
        mask_encoder_dict = {}
        for k, v in raw_state_dict.items():
            name = k
            if name.startswith('mask_encoder.'):
                mask_encoder_dict[name.replace('mask_encoder.', '', 1)] = v
        if len(mask_encoder_dict) > 0:
            self.raw_model.mask_encoder.load_state_dict(mask_encoder_dict, strict=True)
            if is_main_process(): print(f"Success! Loaded {len(mask_encoder_dict)} keys.")
        else:
            if is_main_process(): print("Error: No keys starting with 'mask_encoder' found!")

    @torch.no_grad()
    def data_processing_train_pdiff(self, items):
        atlas_embeddings = None
        atlas_valid_mask = None
        if len(items) >= 12:
            obj_points, obj_points_spatial, descriptor, edge_indices, anchor_ids, obj_points_view2, descriptor_view2, cur_obj_texts, batch_ids, obj_labels, atlas_embeddings, atlas_valid_mask = items
        elif len(items) == 10:
            obj_points, obj_points_spatial, descriptor, edge_indices, anchor_ids, obj_points_view2, descriptor_view2, cur_obj_texts, batch_ids, obj_labels = items
        else:
            obj_points, obj_points_spatial, descriptor, edge_indices, anchor_ids, obj_points_view2, descriptor_view2, cur_obj_texts, batch_ids = items
            obj_labels = None
        obj_points = obj_points.permute(0, 2, 1).contiguous()
        tensors_to_cuda = [obj_points, edge_indices, descriptor, batch_ids, obj_points_spatial, obj_points_view2, descriptor_view2]
        if obj_labels is not None:
            tensors_to_cuda.append(obj_labels)
        if atlas_embeddings is not None:
            tensors_to_cuda.append(atlas_embeddings)
        if atlas_valid_mask is not None:
            tensors_to_cuda.append(atlas_valid_mask)

        moved_tensors = self.cuda(*tensors_to_cuda)
        obj_points, edge_indices, descriptor, batch_ids, obj_points_spatial, obj_points_view2, descriptor_view2 = moved_tensors[:7]

        cursor = 7
        if obj_labels is not None:
            obj_labels = moved_tensors[cursor]
            cursor += 1
        if atlas_embeddings is not None:
            atlas_embeddings = moved_tensors[cursor]
            cursor += 1
        if atlas_valid_mask is not None:
            atlas_valid_mask = moved_tensors[cursor]

        return obj_points, descriptor, edge_indices, anchor_ids, batch_ids, obj_points_spatial, obj_points_view2, descriptor_view2, cur_obj_texts, obj_labels, atlas_embeddings, atlas_valid_mask
    
    @torch.no_grad()
    def data_processing_train(self, items):
        obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids,_ = items
        obj_points = obj_points.permute(0, 2, 1).contiguous()
        obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = \
            self.cuda(obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids)
        return obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids

    # =========================================================
    # [新增] Warm-up Queue 填充函数
    # =========================================================
    def fill_queue_warmup(self, dataloader):
        """
        在开始训练前，通过运行模型的前向传播（不计算梯度，不更新参数）
        来填满队列。
        """
        # 检查是否已经满了
        if hasattr(self.model, 'module'):
            if self.model.module.queue_is_full[0]:
                if is_main_process(): print("[Info] Queue is already full. Skipping warm-up.")
                return
        else:
            if self.model.queue_is_full[0]:
                if is_main_process(): print("[Info] Queue is already full. Skipping warm-up.")
                return

        if is_main_process():
            print("\n" + "="*40)
            print("  Starting Queue Warm-up (Pre-filling)  ")
            print("="*40)
        
        self.model.train() 
        with torch.no_grad():
            loader_iter = iter(dataloader)
            while True:
                # 1. 获取当前进程(Local)的队列状态
                # 注意：这里假设 queue_is_full 是一个 tensor 或 bool
                # 如果是 tensor 直接用，如果是 bool 转为 tensor
                if hasattr(self.model, 'module'):
                    local_is_full = self.model.module.queue_is_full[0]
                else:
                    local_is_full = self.model.queue_is_full[0]
                
                # 转换为 Tensor 以便进行 DDP 通信
                flag_tensor = torch.tensor(1 if local_is_full else 0, device=self.device)

                # 2. [关键修复] 使用 AllReduce 统一状态
                # ReduceOp.MIN 意味着：只有当所有人的 flag 都是 1 时，结果才是 1 (逻辑 AND)
                # 只要有一个人是 0 (没满)，global_all_full 就是 0
                if dist.is_initialized():
                    dist.all_reduce(flag_tensor, op=dist.ReduceOp.MIN)
                    global_all_full = (flag_tensor.item() == 1)
                else:
                    global_all_full = local_is_full

                # 3. 只有当“所有人”都满时，才一起 Break
                if global_all_full:
                    if is_main_process(): print(f"[Info] All queues filled successfully!")
                    break
                
                # ... (数据加载和 Forward 代码保持不变) ...
                try:
                    items = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(dataloader)
                    items = next(loader_iter)

                obj_points, descriptor, edge_indices, anchor_ids, batch_ids, \
                obj_points_spatial, obj_points_view2, descriptor_view2, cur_obj_texts, obj_labels, atlas_embeddings, atlas_valid_mask = self.data_processing_train_pdiff(items)
                
                # 即使当前卡满了，只要有人没满，就继续跑 Forward 以配合 SyncBatchNorm
                _ = self.model(
                    obj_points.permute(0,2,1).contiguous(), edge_indices, obj_points_spatial, pts_v2=obj_points_view2,
                    descriptor=descriptor, descriptor_v2=descriptor_view2, batch_ids=batch_ids, anchor_id=anchor_ids, 
                    istrain=True, cur_obj_texts=cur_obj_texts, obj_labels=obj_labels,
                    atlas_embeddings=atlas_embeddings, atlas_valid_mask=atlas_valid_mask
                )
                
                self.cnt += 1
                if is_main_process() and self.cnt % 10 == 0: # 减少打印频率
                    print(f"Warmup step: {self.cnt}", end='\r')

        if dist.is_initialized():
            dist.barrier()
        if is_main_process():
            print("\n" + "="*40 + "\n")

    def train(self):
        train_sampler = DistributedSampler(self.dataset_train, shuffle=True)
        
        worker_count = int(getattr(self.config, 'WORKERS', 4))

        train_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_train,
            batch_size=self.config.Batch_Size,
            num_workers=worker_count, 
            drop_last=True,
            shuffle=False,
            collate_fn=collate_fn_mmg_diff,
            sampler=train_sampler
        )
        
        self.total = len(train_loader)
        self.max_iteration = self.config.max_iteration = int(self.config.MAX_EPOCHES * self.total)
        self.lr_scheduler = get_warmup_cosine_scheduler(self.optimizer, self.total*20, self.max_iteration)
                
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

            checkpoint = torch.load(resume_path, map_location=self.device)
            state_dict = extract_checkpoint_state_dict(checkpoint)
            normalized_state_dict = normalize_state_dict_keys(state_dict)

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
                print("Warning: RESUME_PATH not found. Starting from scratch.")
                mask_init_path = getattr(self.config, 'MASK_ENCODER_INIT_PATH', None)
                if mask_init_path and os.path.exists(mask_init_path):
                    self.load_pretrained_mask_encoder(mask_init_path)
                elif mask_init_path:
                    print(f"Warning: MASK_ENCODER_INIT_PATH '{mask_init_path}' not found. Skip mask encoder init.")
            elif is_main_process():
                print("No RESUME_PATH. Starting from scratch.")

        self.raw_model.epoch = start_epoch
        if self.total == 0:
            if is_main_process():
                print('No training data provided!')
            return

        # =========================================================
        # [关键] 在正式训练循环前，执行 Warm-up
        # =========================================================
        self.fill_queue_warmup(train_loader)

        ''' Train '''
        self.model.train()

        while(keep_training):
            train_sampler.set_epoch(self.raw_model.epoch)
            self.swav_monitor.reset()
            if self.raw_model.epoch >= self.config.MAX_EPOCHES:
                break

            if is_main_process():
                loader_iter = tqdm(
                    train_loader,
                    total=self.total,
                    desc=f'Epoch {self.raw_model.epoch + 1}/{self.config.MAX_EPOCHES}',
                    dynamic_ncols=True,
                )
            else:
                loader_iter = train_loader

            for batch_idx, items in enumerate(loader_iter):
                obj_points, descriptor, edge_indices, anchor_ids, batch_ids, \
                obj_points_spatial, obj_points_view2, descriptor_view2, cur_obj_texts, obj_labels, atlas_embeddings, atlas_valid_mask = self.data_processing_train_pdiff(items)
                
                # Forward
                total_loss, diff_loss, triplet_loss, edge_loss, obj_loss, contrastive_loss, obj_label_contrastive_loss, atlas_align_loss, total_metric, gcn_edge_feature_3d = self.model(
                    obj_points.permute(0,2,1).contiguous(), edge_indices, obj_points_spatial, pts_v2=obj_points_view2,
                    descriptor=descriptor, descriptor_v2=descriptor_view2, batch_ids=batch_ids, anchor_id=anchor_ids, 
                    istrain=True, cur_obj_texts=cur_obj_texts, obj_labels=obj_labels,
                    atlas_embeddings=atlas_embeddings, atlas_valid_mask=atlas_valid_mask
                )
                
                # Log
                global_step = self.raw_model.epoch * self.total + batch_idx
                base_momentum = 0.99 
                final_momentum = 1.0
                current_momentum = self.get_current_momentum(global_step, self.max_iteration, base_momentum, final_momentum)
                branch_flags = self._branch_flags()
                
                if is_main_process() and self.writer is not None:
                    self.writer.add_scalar('Train/Total_Loss', total_loss.item(), global_step)
                    self.writer.add_scalar('Train/Triplet_Loss', triplet_loss.item(), global_step)
                    self.writer.add_scalar('Train/Edge_Loss', edge_loss.item(), global_step)
                    self.writer.add_scalar('Train/Obj_Loss', obj_loss.item(), global_step)
                    if branch_flags['diffusion']:
                        self.writer.add_scalar('Train/Diff_Loss', diff_loss.item(), global_step)
                        self.writer.add_scalar('Train/total_metric', total_metric, global_step)
                    if branch_flags['text_contrastive']:
                        self.writer.add_scalar('Train/ctrs_Loss', contrastive_loss.item(), global_step)
                    if branch_flags['obj_label_contrastive']:
                        self.writer.add_scalar('Train/obj_label_ctr_loss', obj_label_contrastive_loss.item(), global_step)
                    if branch_flags['atlas']:
                        self.writer.add_scalar('Train/atlas_align_loss', atlas_align_loss.item(), global_step)
                
                # Monitor
                with torch.no_grad():
                    z1, q1 = self.model.module.swav_reg_edge.forward_test(gcn_edge_feature_3d)
                if is_main_process(): self.swav_monitor.update(embeddings=z1, swav_q=q1)
                
                current_lr = self.optimizer.param_groups[1]['lr']
                if is_main_process():
                    postfix = {
                        'tri_ls': f'{triplet_loss.item():.4f}',
                        'rel_ls': f'{edge_loss.item():.4f}',
                        'obj_ls': f'{obj_loss.item():.4f}',
                        'lr': f'{current_lr:.6f}'
                    }
                    if branch_flags['diffusion']:
                        postfix['dif'] = f'{diff_loss.item():.4f}'
                        postfix['met'] = f'{total_metric:.1f}'
                    if branch_flags['text_contrastive']:
                        postfix['ctrs_ls'] = f'{contrastive_loss.item():.4f}'
                    if branch_flags['obj_label_contrastive']:
                        postfix['obj_ctr'] = f'{obj_label_contrastive_loss.item():.4f}'
                    if branch_flags['atlas']:
                        postfix['atlas'] = f'{atlas_align_loss.item():.4f}'
                    loader_iter.set_postfix(postfix)
                
                self.backward(total_loss, current_momentum)
            
            # Save & Validate
            current_epoch = self.raw_model.epoch
            completed_epoch = current_epoch + 1

            if completed_epoch % 10 == 0 and is_main_process():
                if hasattr(self, 'save_dir'):
                    save_path = os.path.join(self.save_dir, f'model_epoch_{completed_epoch}.pth')
                    state_dict = self.raw_model.state_dict()
                    checkpoint = {
                        'epoch': current_epoch,
                        'model_state_dict': state_dict,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.lr_scheduler.state_dict(),
                        'loss': total_loss.item()
                    }
                    print(f'\n[Epoch {completed_epoch}] Saving Checkpoint: {save_path}')
                    torch.save(checkpoint, save_path)

            validate_every_n_epoch = bool(getattr(self.config, 'VALIDATE_EVERY_N_EPOCH', True))
            valid_interval = int(getattr(self.config, 'VALID_INTERVAL', 10))
            if validate_every_n_epoch and valid_interval > 0 and completed_epoch % valid_interval == 0:
                if is_main_process():
                    print(f"\n[Epoch {completed_epoch}] Starting Validation...")
                    self.validation_for_cls(epoch=completed_epoch)
                    print(f"[Epoch {completed_epoch}] Validation Finished.")
                
                # [DDP] 关键：让其他 GPU 等待 Rank 0 跑完验证，防止不同步进入下一个 Epoch
                dist.barrier()
            
            self.model.train()
            
            if is_main_process():
                self.swav_monitor.report(epoch_idx=completed_epoch)
            
            self.raw_model.epoch += 1
    
    def cuda(self, *args):
        return [item.to(self.device, non_blocking=True) for item in args]

    def get_current_momentum(self, current_step, max_steps, base_tau=0.99, final_tau=1.0):
        return base_tau + (final_tau - base_tau) * (current_step / max_steps)
    
    def backward(self, loss, current_momentum):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        if hasattr(self.model, 'module'):
            self.model.module._update_target(momentum=current_momentum)
        else:
            self.model._update_target(momentum=current_momentum)
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
        val_dataset = build_dataset_for_clustering(self.config)
        val_loader = CustomDataLoader(
            config = self.config,
            dataset=val_dataset,
            batch_size=self.config.Batch_Size, # 可以稍微大一点，因为不存梯度
            num_workers=int(getattr(self.config, 'WORKERS', 4)),
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
        tsne_save_dir = os.path.join(self.config.analysis_save_dir, "tsne_save")
        if not os.path.exists(tsne_save_dir):
            os.makedirs(tsne_save_dir)

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

        obj_cluster_count = self._count_effective_clusters(all_gt_obj_cls, ignore_zero_label=True)
        edge_cluster_count = self._count_effective_clusters(all_gt_rel_cls, ignore_zero_label=True)

        if obj_cluster_count >= 2:
            try:
                visualize_with_gt(
                    all_obj_feats,
                    all_gt_obj_cls,
                    title_prefix=f"Object Features GT Epoch {epoch}",
                    save_path=os.path.join(tsne_save_dir, f"obj_gt_tsne_{epoch}.png"),
                    class_names=getattr(val_dataset, "classNames", None),
                )
            except Exception as exc:
                print(f"[Validation] Failed to save object GT t-SNE plot at epoch {epoch}: {exc}")
        else:
            print(f"[Validation] Skip object GT t-SNE at epoch {epoch}: effective classes = {obj_cluster_count}")

        if edge_cluster_count >= 2:
            try:
                visualize_with_gt(
                    all_edge_feats,
                    all_gt_rel_cls,
                    title_prefix=f"Edge Features GT Epoch {epoch}",
                    save_path=os.path.join(tsne_save_dir, f"edge_gt_tsne_{epoch}.png"),
                    class_names=getattr(val_dataset, "relationNames", None),
                )
            except Exception as exc:
                print(f"[Validation] Failed to save edge GT t-SNE plot at epoch {epoch}: {exc}")
        else:
            print(f"[Validation] Skip edge GT t-SNE at epoch {epoch}: effective classes = {edge_cluster_count}")

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
