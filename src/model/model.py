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

        ''' Build Model '''
        self.model = PointDif(self.config).cuda()
        
        # mmg_obj, mmg_rel = [], []
        # for name, para in self.model.mmg.named_parameters():
        #     if 'nn_edge' in name:
        #         mmg_rel.append(para)
        #     else:
        #         mmg_obj.append(para)
        
        self.optimizer = optim.AdamW([
            {'params':self.model.mask_encoder.parameters(), 'lr':float(config.LR)/10, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            # {'params':self.model.rel_encoder_3d.parameters() , 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            # {'params':self.model.mmg.parameters(), 'lr':float(config.LR)/2, 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.model.mask_token, 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.model.ca_net.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            # {'params':self.model.mlp_3d.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            # {'params':self.model.bboxes_head.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
            {'params':self.model.point_diffusion.net.parameters(), 'lr':float(config.LR), 'weight_decay':self.config.W_DECAY, 'amsgrad':self.config.AMSGRAD},
        ])
        
        #96000
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
        obj_points, descriptor, edge_indices, anchor_ids, batch_ids, obj_points_spatial = items
        obj_points = obj_points.permute(0, 2, 1).contiguous()
        obj_points, edge_indices, descriptor, batch_ids, obj_points_spatial = \
            self.cuda(obj_points, edge_indices, descriptor, batch_ids, obj_points_spatial)

        return obj_points, descriptor, edge_indices, anchor_ids, batch_ids, obj_points_spatial
          
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
                self.load_pretrained_mask_encoder("/home/honsen/tartan/ckpt-epoch-300.pth")
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

            if self.model.epoch > self.config.MAX_EPOCHES:#
                break

            print('\n\nTraining epoch: %d' % self.model.epoch)
                    
            for batch_idx, items in enumerate(loader):
                ''' get data '''
                print("------training------")
                obj_points, descriptor, edge_indices, anchor_ids, batch_ids, obj_points_spatial = self.data_processing_train_pdiff(items)
                
                ''' forward '''
                total_loss, total_metric = self.model(obj_points.permute(0,2,1).contiguous(), edge_indices, obj_points_spatial, descriptor=descriptor, batch_ids=batch_ids, anchor_id=anchor_ids, istrain=True)
                current_lr = self.optimizer.param_groups[1]['lr']
                # print('Epoch: %d, Iteration: %d / %d, diff_spatial_loss: %.4f, diff_loss: %.4f, total_spatial_metric: %.4f, total_metric: %.4f' % (self.model.epoch, batch_idx+1, self.total, diff_spatial_loss.item(), diff_loss.item(), total_spatial_metric, total_metric))
                print('Epoch: %d, Iteration: %d / %d, diff_loss: %.4f, total_metric: %.4f, LR: %.6f'\
                      % (self.model.epoch, batch_idx+1, self.total, total_loss.item(), total_metric, current_lr))
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
            
            self.model.epoch += 1
            loader = iter(train_loader)
    
    def validation(self):
        ''' create data loader '''
        drop_last = True
        train_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_train,
            batch_size=self.config.Batch_Size,
            num_workers=1, #self.config.WORKERS
            drop_last=drop_last,
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

        model_dicts = torch.load("/home/honsen/honsen/SceneGraph/SG_pretrain_diff/save_path/model_epoch_200.pth")
        self.model.load_state_dict(model_dicts['model_state_dict'])

        # model_dicts = torch.load("/home/honsen/tartan/ckpt-epoch-300.pth")
        # model_dicts = model_dicts['pointdif']
        # model_dicts = remove_module_prefix(model_dicts)
        # self.model.load_state_dict(model_dicts,)
        
        ''' Train '''
        self.model.eval()
        istrain = False
        while(keep_training):

            if self.model.epoch > 1:#
                break

            print('\n\nTraining epoch: %d' % self.model.epoch)
                    
            for batch_idx, items in enumerate(loader):
                ''' get data '''
                print("------training------")
                obj_points, descriptor, edge_indices, anchor_ids, batch_ids, obj_points_spatial = self.data_processing_train_pdiff(items)
                
                ''' forward '''
                if istrain:
                    total_loss, diff_loss, diff_spatial_loss, total_spatial_metric, total_metric = self.model(obj_points.permute(0,2,1).contiguous(), edge_indices, obj_points_spatial, descriptor=descriptor, batch_ids=batch_ids, anchor_id=anchor_ids, istrain=istrain)
                    print('Epoch: %d, Iteration: %d / %d, diff_spatial_loss: %.4f, diff_loss: %.4f, total_spatial_metric: %.4f, total_metric: %.4f' % (self.model.epoch, batch_idx+1, self.total, diff_spatial_loss.item(), diff_loss.item(), total_spatial_metric, total_metric))
                else:
                    with torch.no_grad():
                        diff_loss, total_x0_metric = self.model(obj_points.permute(0,2,1).contiguous(), edge_indices, obj_points_spatial, descriptor=descriptor, batch_ids=batch_ids, anchor_id=anchor_ids, istrain=False)
                        print('Epoch: %d, Iteration: %d / %d, total_x0_metric: %.4f, diff_loss: %.4f' % (self.model.epoch, batch_idx+1, self.total, total_x0_metric, diff_loss))
            
            self.model.epoch += 1
            loader = iter(train_loader)
    
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
