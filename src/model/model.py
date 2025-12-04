from tqdm import tqdm
import copy
import os
import torch.optim as optim
import numpy as np
import torch
from src.dataset.DataLoader import (CustomDataLoader, collate_fn_mmg_diff)
from src.dataset.dataset_builder import build_dataset
from src.model.diff_trans.models.PointDif import PointDif
from src.model.optimizer.scheduler import get_warmup_cosine_scheduler, get_freeze_warmup_scheduler
from src.model.diff_trans.models.monitor import EpochCollapseMonitor
from src.model.diff_trans.models.clustering import cluster_and_visualize
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
    def __init__(self, config):
        self.config = config
        self.model_name = 'pdiff_SGG'
        self.save_dir = self.config.PATH
        # os.makedirs(self.save_dir, exist_ok=True)
        ''' Build dataset '''       
        self.dataset_train = build_dataset("train_scannet", True)
        self.total = self.config.total = len(self.dataset_train) // self.config.Batch_Size
        self.max_iteration = self.config.max_iteration = int(float(self.config.MAX_EPOCHES)*len(self.dataset_train) // self.config.Batch_Size)
        self.max_iteration_scheduler = self.config.max_iteration_scheduler = int(float(self.config.MAX_EPOCHES)*len(self.dataset_train) // self.config.Batch_Size)
        
        self.swav_monitor = EpochCollapseMonitor(60)
        
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

        # 2. 特殊模块: SwAV Prototypes (完全不 decay)
        param_groups.append({
            'params': self.model.swav_reg_rel.parameters(), 
            'lr': float(config.LR), 
            'weight_decay': 0.0, # 规范写法
            'amsgrad': self.config.AMSGRAD
        })
        
        param_groups.append({
            'params': self.model.swav_reg_obj.parameters(), 
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

        # 初始化优化器
        self.optimizer = optim.AdamW(param_groups)
        
        self.lr_scheduler = get_warmup_cosine_scheduler(self.optimizer, self.total*10, self.config.max_iteration)
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
            
            # 1. 加载模型状态
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 2. 加载优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 3. 加载 Scheduler 状态 (如果存在)
            if 'scheduler_state_dict' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Successfully loaded LR scheduler state.")
            else:
                print("Warning: 'scheduler_state_dict' not found in checkpoint. LR scheduler will restart.")
            
            # 4. 设置起始 epoch
            # 我们从保存的 epoch 的 *下一个* epoch 开始
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}. Last recorded loss: {checkpoint.get('loss', 'N/A')}")
        
        else:
            if resume_path:
                print(f"Warning: RESUME_PATH '{resume_path}' was specified but file not found. Starting from scratch.")
                # self.load_pretrained_mask_encoder("/home/honsen/tartan/ckpt-epoch-300.pth")
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

        while(keep_training):
            
            self.swav_monitor.reset()
            
            if self.model.epoch > self.config.MAX_EPOCHES:#
                break

            # --- [修改 1] 使用 tqdm 包装 loader ---
            # dynamic_ncols=True 可以让进度条自动适应终端宽度
            pbar = tqdm(loader, total=self.total, desc=f'Epoch {self.model.epoch}', dynamic_ncols=True)

            for batch_idx, items in enumerate(pbar):
                ''' get data '''
                # print("------training------") # --- [修改 2] 必须注释掉，否则会打断进度条 ---
                
                obj_points, descriptor, edge_indices, anchor_ids, batch_ids, obj_points_spatial, cur_obj_texts = self.data_processing_train_pdiff(items)
                
                ''' forward '''
                total_loss, diff_loss, cls_loss, cls_loss_obj, total_metric, gcn_edge_feature_3d = self.model(
                    obj_points.permute(0,2,1).contiguous(), edge_indices, obj_points_spatial, 
                    descriptor=descriptor, batch_ids=batch_ids, anchor_id=anchor_ids, 
                    istrain=True, cur_obj_texts=cur_obj_texts
                )
                
                with torch.no_grad():
                    z1,q1 = self.model.swav_reg_rel.forward1(gcn_edge_feature_3d)
        
                # 3. 更新统计 (每个 Batch)
                # 只需要把其中一个 view (z1, q1) 传进去即可代表本 Batch 情况
                self.swav_monitor.update(embeddings=z1, swav_q=q1)
                
                current_lr = self.optimizer.param_groups[1]['lr']

                # --- [修改 3] 使用 set_postfix 在同一行刷新显示关键指标 ---
                # 为了美观，保留小数位
                pbar.set_postfix({
                    # 'Epoch': self.model.epoch,                  # 显示当前 Epoch
                    # 'Iter': f'{batch_idx + 1}/{self.total}',    # 显示 Iteration: 当前/总数
                    'diff': f'{diff_loss.item():.4f}',
                    'cls': f'{cls_loss.item():.4f}',
                    'ctrs': f'{cls_loss_obj.item():.4f}',
                    'met': f'{total_metric:.4f}',
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

        model_dicts = torch.load("/home/honsen/honsen/SceneGraph/SG_pretrain_diff/save_path/model_epoch_250.pth")
        self.model.load_state_dict(model_dicts['model_state_dict'],strict=False)

        # model_dicts = torch.load("/home/honsen/tartan/ckpt-epoch-300.pth")
        # model_dicts = model_dicts['pointdif']
        # model_dicts = remove_module_prefix(model_dicts)
        # self.model.load_state_dict(model_dicts,)
        
        all_edge_feats = []
        all_obj_feats = []
        ''' Train '''
        self.model.eval()

                    
        for batch_idx, items in enumerate(loader):
            ''' get data '''
            print("------training------")
            obj_points, descriptor, edge_indices, anchor_ids, batch_ids, obj_points_spatial, cur_obj_texts = self.data_processing_train_pdiff(items)
        
            with torch.no_grad():
                diff_loss, total_x0_metric, gcn_edge_feature_3d, gcn_obj_feature_3d = self.model(obj_points.permute(0,2,1).contiguous(), edge_indices, obj_points_spatial, descriptor=descriptor, batch_ids=batch_ids, anchor_id=anchor_ids, istrain=False)
                print('Epoch: %d, Iteration: %d / %d, total_x0_metric: %.4f, diff_loss: %.4f' % (self.model.epoch, batch_idx+1, self.total, total_x0_metric, diff_loss))
        
        
            all_edge_feats.append(gcn_edge_feature_3d)
            all_obj_feats.append(gcn_obj_feature_3d)
        
        all_edge_feats = torch.cat(all_edge_feats, dim=0)
        all_obj_feats = torch.cat(all_obj_feats, dim=0)
        
        cluster_and_visualize(all_obj_feats, 100, title_prefix="Object Features",\
                              save_path="/home/honsen/honsen/SceneGraph/SG_pretrain_diff/clustering_dir/object_features_cluster.png")
        cluster_and_visualize(all_edge_feats, 50, title_prefix="Edge Features",\
                              save_path="/home/honsen/honsen/SceneGraph/SG_pretrain_diff/clustering_dir/edge_features_cluster.png")
        
    def cuda(self, *args):
        return [item.to(self.config.DEVICE) for item in args]

    def save(self,epoch):
        self.model.save(epoch)

    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # update lr
        self.lr_scheduler.step()
