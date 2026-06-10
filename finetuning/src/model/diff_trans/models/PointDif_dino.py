import os
import torch
import sys
sys.path.append('/home/honsen/honsen/SceneGraph/SG_pretrain_diff')

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from src.model.diff_trans.models.build import MODELS
from src.model.diff_trans.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from src.model.diff_trans.utils.logger import *
from src.model.diff_trans.utils import misc
from src.model.diff_trans.models.mask_encoder import Mask_Encoder, Group, Encoder, TransformerEncoder
from src.model.diff_trans.models.generator import CPDM, CANet
from src.model.model_utils.model_base import BaseModel
from src.model.model_utils.network_MMRGR import MMG
from src.model.model_utils.network_PointNet import (PointNetfeat)
from src.utils import op_utils
from src.dataset.dataset_diffPoint import visualize_scenes_plt, visualize_scenes_plt_with_points, visualize_scenes_batch, visualize_and_save_sequence

from src.model.diff_trans.models.weight_focal_loss import compute_adaptive_weight, compute_local_complexity_weight
from src.model.diff_trans.models.mcr_loss import MCRLoss
from src.model.diff_trans.models.contrastive_loss import TextSupervisedContrastiveLoss, ObjectLabelContrastiveLoss
import copy

class MaskedEdgeEncoder(nn.Module):
    def __init__(self, edge_dim):
        super().__init__()
        # 1. 定义一个全局共享的 Mask Token
        # 形状是 [1, edge_dim]，表示这是一个通用的“未知边”特征
        self.mask_token = nn.Parameter(torch.randn(1, edge_dim))
        
        # 初始化参数（可选，但推荐）
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, num_edges):
        """
        Args:
            num_edges: int, 当前Batch中总的边数量
        Returns:
            masked_edge_features: [E, D] 准备送入GNN的初始边特征
        """
        
        # 2. 广播 (Broadcasting)
        # 将 [1, D] 扩展为 [E, D]
        # .expand 不会分配新内存，只是改变视图，非常高效
        masked_edge_features = self.mask_token.expand(num_edges, -1)
        
        # 注意：这里返回的特征所有行都是完全一样的数值
        # 差异化将在后续的 GNN Message Passing 中产生
        return masked_edge_features

@MODELS.register_module()
class PointDif(BaseModel):
    def __init__(self, config, dim_descriptor=11):
        super().__init__('Diff_sg', config)
        print_log(f'[Diff_sg] ', logger ='Diff_sg')
        
        self.mconfig = mconfig = config.sg_model
        
        with_bn = mconfig.WITH_BN

        dim_point = 3
        if mconfig.USE_RGB:
            dim_point +=3
        if mconfig.USE_NORMAL:
            dim_point +=3

        dim_f_spatial = dim_descriptor
        dim_point_rel = dim_f_spatial

        self.dim_point = dim_point
        self.dim_edge = dim_point_rel
        self.flow = 'target_to_source'

        self.momentum = 0.996
        self.model_pre = None

        # 把原始几何/局部描述压到一个统一的高维 embedding
        # 点云关系编码器
        self.rel_encoder_3d = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=11,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=512)
        # 图/关系建模模块 
        # 在做“点-点/组-组关系推理”，像一个带注意力的图网络/Transformer
        self.mmg = MMG(
            dim_node=512,
            dim_edge=512,
            dim_atten=256, #self.mconfig.DIM_ATTEN
            depth=2, #self.mconfig.N_LAYERS
            num_heads=8, #self.mconfig.NUM_HEADS
            aggr="max", #self.mconfig.GCN_AGGR
            flow=self.flow,
            attention="fat",#self.mconfig.ATTENTION
            use_edge=True,#self.mconfig.USE_GCN_EDGE
            DROP_OUT_ATTEN=0.5)#self.mconfig.DROP_OUT_ATTEN
        
        # PointNet 把“每个局部”变成向量；
        # MMG 再让这些向量之间“互相看一眼”，把关系信息编码进去
        # 这里是典型的“把点云分块 + 做 MAE 风格的 mask 学习”：
        # Group(num_group, group_size)：把点云划分成 G 个 group，每组 S 个点
        # Mask_Encoder：对“可见的 group tokens”做 Transformer 编码
        # mask_token：被 mask 的位置用一个可学习 token 代替（类似 BERT/MAE）
        self.config = config.maskTrans
        self.group_size = self.config.group_size
        self.num_group = self.config.num_group
        self.trans_dim = self.config.encoder_config.trans_dim
        self.mask_encoder = Mask_Encoder(self.config)
        self.encoder_dims = self.config.encoder_config.encoder_dims
        # 把 512 压成 512-11（看起来是为了和某个 11 维描述拼接/对齐，
        # 或者留出描述维度做残差/拼接）
        self.mlp_3d = torch.nn.Sequential(
            torch.nn.Linear(512, 512 - 11),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3)
        )
        # 用来把 transformer 编码出来的东西和 3D/关系特征融合到 512 维
        self.ca_net = CANet(self.encoder_dims, 512)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        # 说明除了点/组 token，还对边/关系也做 mask 编码（更像“结构化 mask”）
        self.edge_mask_token = MaskedEdgeEncoder(512)

        # --- 2. Target Network (Teacher) ---
        # 深度拷贝 Online 网络
        self.mask_encoder_target = copy.deepcopy(self.mask_encoder)
        self.rel_encoder_3d_target = copy.deepcopy(self.rel_encoder_3d)
        self.mmg_target = copy.deepcopy(self.mmg)
        self.mlp_3d_target = copy.deepcopy(self.mlp_3d)
        self.ca_net_target = copy.deepcopy(self.ca_net)
        self.mask_token_target = copy.deepcopy(self.mask_token)
        self.edge_mask_token_target = copy.deepcopy(self.edge_mask_token)
     

        # 冻结 Teacher 参数
        for p in self.get_target_params():
            p.requires_grad = False

        self.contrastive_loss_fn = TextSupervisedContrastiveLoss(temperature=0.07, num_negatives=1000)
        self.object_label_contrastive_loss_fn = ObjectLabelContrastiveLoss(
            temperature=float(getattr(config, "OBJ_LABEL_CONTRASTIVE_TEMPERATURE", 0.07))
        )
        self.obj_label_contrastive_enabled = bool(getattr(config, "OBJ_LABEL_CONTRASTIVE_ENABLED", True))
        self.obj_label_contrastive_weight = float(getattr(config, "OBJ_LABEL_CONTRASTIVE_WEIGHT", 0.1))

        self.mcr_edge_loss = MCRLoss(out_dim=512, reduce_cov=1)
        self._last_edge_mcr_stats = None

        self.mcr_obj_loss = MCRLoss(out_dim=512, reduce_cov=1)
        self._last_obj_mcr_stats = None
        
        self.mcr_trip_loss = MCRLoss(out_dim=512*3, reduce_cov=1) # Triplet dim is 512*3
        self._last_trip_mcr_stats = None

        self.point_diffusion = CPDM(self.config)
        self.count = 0
        
        # 1.5 Predictors (Student Only! Teacher 不包含 Predictor)
        # 用于防止 collapse，Student 输出需要通过 Predictor 才能去拟合 Teacher
        self.predictor_triplet = nn.Sequential(
            nn.Linear(512*3, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Linear(1024, 512*3))
        
        # [新增] Edge 和 Obj 的 Predictor
        self.predictor_edge = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 512))
        self.predictor_obj = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 512))

        print_log(f'[PointDif] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointDif')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        trunc_normal_(self.mask_token, std=.02)

   # ==================== EMA Helpers (关键修改) ====================
    def get_online_params(self):
        """
        返回所有需要 EMA 更新的 Student 参数列表。
        必须与 get_target_params 的顺序严格一一对应。
        注意：Predictor 不在这里，因为 Teacher 没有 Predictor。
        """
        return (list(self.mask_encoder.parameters()) + 
                list(self.rel_encoder_3d.parameters()) + 
                list(self.mmg.parameters()) +
                list(self.mlp_3d.parameters()) + 
                list(self.ca_net.parameters()) +
                [self.mask_token] + # 注意 Parameter 要放入 list
                list(self.edge_mask_token.parameters())
               )

    def get_target_params(self):    
        """
        返回 Teacher 对应的参数列表。
        """
        return (list(self.mask_encoder_target.parameters()) + 
                list(self.rel_encoder_3d_target.parameters()) + 
                list(self.mmg_target.parameters()) +
                list(self.mlp_3d_target.parameters()) + 
                list(self.ca_net_target.parameters()) +
                [self.mask_token_target] +
                list(self.edge_mask_token_target.parameters()) 
               )

    @torch.no_grad()
    def _update_target(self, momentum):
        self.momentum = momentum
        for p_o, p_t in zip(self.get_online_params(), self.get_target_params()):
            p_t.data = p_t.data * self.momentum + p_o.data * (1. - self.momentum)

    @torch.no_grad()
    def _gather_variable_tensor(self, tensor):
        if not dist.is_available() or not dist.is_initialized():
            return tensor

        local_count = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
        gathered_counts = [torch.zeros_like(local_count) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_counts, local_count)
        counts = [int(item.item()) for item in gathered_counts]
        max_count = max(counts) if len(counts) > 0 else 0

        padded_shape = (max_count,) + tuple(tensor.shape[1:])
        padded_tensor = tensor.new_zeros(padded_shape)
        if tensor.shape[0] > 0:
            padded_tensor[: tensor.shape[0]] = tensor

        gathered_tensors = [tensor.new_zeros(padded_shape) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_tensors, padded_tensor)

        valid_tensors = []
        for gathered_tensor, count in zip(gathered_tensors, counts):
            if count > 0:
                valid_tensors.append(gathered_tensor[:count])

        if len(valid_tensors) == 0:
            return tensor.new_zeros((0,) + tuple(tensor.shape[1:]))
        return torch.cat(valid_tensors, dim=0)

    @torch.no_grad()
    def _build_global_instance_ids(self, num_instances, device):
        local_count = torch.tensor([num_instances], device=device, dtype=torch.long)
        if not dist.is_available() or not dist.is_initialized():
            offset = 0
        else:
            gathered_counts = [torch.zeros_like(local_count) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_counts, local_count)
            rank = dist.get_rank()
            offset = int(sum(int(item.item()) for item in gathered_counts[:rank]))
        return torch.arange(num_instances, device=device, dtype=torch.long) + offset

    def obj_feat_extractor(self, pts, anchor_id, batch_ids, descriptor, mask_ratio=None, istarget=False, isview3 = False):
         ##### 
        # Point object features Extractors
        # ###
        
        if anchor_id is not None:
            anchor_set = 1
        else:
            anchor_set = 0
        
        B,_,_ = pts.shape        
        # get patch
        neighborhood, center = self.group_divider(pts)
        
        if istarget:
            # mask and encoder
            encoder_token, mask = self.mask_encoder_target(neighborhood, center, mask_ratio=mask_ratio)
            _,N,_ = (center[mask].reshape(B,-1,3)).shape
            # learnable masked token
            mask_token = self.mask_token_target.expand(B, N, -1)
            encoder_token[mask] = mask_token.reshape(-1, self.trans_dim)
            point_agg_features = self.ca_net_target(encoder_token)
        else:
            encoder_token, mask = self.mask_encoder(neighborhood, center, mask_ratio=mask_ratio)
            _,N,_ = (center[mask].reshape(B,-1,3)).shape
            # learnable masked token
            mask_token = self.mask_token.expand(B, N, -1)
            encoder_token[mask] = mask_token.reshape(-1, self.trans_dim)
            point_agg_features = self.ca_net(encoder_token)
        
        if isview3:
            return point_agg_features
        
        if anchor_set:
            device = point_agg_features.device

            # 2. 将本地锚点索引列表 (Python list) 转换为 Tensor
            #    例如: anchor_id = [1, 0, 2] (B=3, 第0个图的锚点是idx 1, ...)
            local_anchor_ids_tensor = torch.tensor(anchor_id, device=device, dtype=torch.long)

            # 3. 计算每个图的节点数
            #    batch_ids (N_total, 1) -> (N_total)
            batch_ids_squeezed = batch_ids.squeeze()
            #    bincount 会统计 [0, 0, 0, 1, 1, 2, 2, 2] -> [3, 2, 3]
            #    (即: 图0有3个节点, 图1有2个节点, 图2有3个节点)
            counts = torch.bincount(batch_ids_squeezed)

            # 4. 计算每个图的起始偏移量 (offsets)
            #    counts [3, 2, 3] -> cumsum [3, 5, 8] -> shifted [0, 3, 5]
            #    (即: 图0从idx 0开始, 图1从idx 3开始, 图2从idx 5开始)
            offsets = torch.cat([torch.tensor([0], device=device), torch.cumsum(counts, dim=0)[:-1]])

            # 5. 计算锚点的 *全局* 索引
            #    offsets [0, 3, 5] + local_ids [1, 0, 2] = global_ids [1, 3, 7]
            global_anchor_indices = offsets + local_anchor_ids_tensor
            
            # 6. 使用 *全局* 索引来提取和注入锚点特征
            #    (旧代码: anchor_obj = point_agg_features[anchor_id:anchor_id+1])
            anchor_obj_features = point_agg_features[global_anchor_indices]
            
            if istarget:
                anchor_obj_features = self.mlp_3d_target(anchor_obj_features)
            else:
                anchor_obj_features = self.mlp_3d(anchor_obj_features)
            
            if self.mconfig.USE_SPATIAL:
                tmp = descriptor.clone()
                tmp[:,6:] = tmp[:,6:].log() # only log on volume and length

                # (旧代码: tmp[anchor_id:anchor_id+1])
                anchor_spatial_info = tmp[global_anchor_indices]
                
                # (旧代码: abs_obj = torch.cat([anchor_obj, ...]))
                abs_obj_features = torch.cat([anchor_obj_features, anchor_spatial_info], dim=-1)
                
                # (旧代码: point_agg_features[anchor_id:anchor_id+1] = abs_obj.squeeze(0))
                point_agg_features[global_anchor_indices] = abs_obj_features
            
            return point_agg_features, global_anchor_indices
            # --- [!!! 锚点逻辑修改结束 !!!] ---
        
        else:
            if istarget:
                point_agg_features1 = self.mlp_3d_target(point_agg_features)
            else:
                point_agg_features1 = self.mlp_3d(point_agg_features)
                
            if self.mconfig.USE_SPATIAL:
                tmp = descriptor.clone()
                tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
                
                point_agg_features1 = torch.cat([point_agg_features1, tmp], dim=-1)
                return point_agg_features1
    
    def generate_object_pair_features(self, obj_feats, edges_feats, edge_indice):
        obj_pair_feats = []
        for (edge_feat, edge_index) in zip(edges_feats, edge_indice.t()):
            obj_pair_feats.append(torch.cat([obj_feats[edge_index[0]], obj_feats[edge_index[1]], edge_feat], dim=-1))
        obj_pair_feats = torch.vstack(obj_pair_feats)
        return obj_pair_feats
    
    def edge_masking(self, rel_feature_3d_view, edge_mask_ratio=0.4, istarget=False):
        # 2. 定义掩码比例 (建议放在 config 中，此处暂定 0.3)
        
        num_edges = rel_feature_3d_view.shape[0]
        
        # 3. 生成随机掩码 (True 表示该边被 Mask 掉)
        # device 保持一致
        mask_indices = torch.rand(num_edges, device=rel_feature_3d_view.device) < edge_mask_ratio
        
        # 4. 执行替换操作
        # 如果存在需要掩码的边
        if mask_indices.any():
            # 获取 mask token [1, 512]
            if istarget:
                mask_token = self.edge_mask_token_target.mask_token
            else:
                mask_token = self.edge_mask_token.mask_token
            
            # 计算需要掩码的数量
            num_masked = mask_indices.sum()
            
            # 将 mask_token 广播并覆盖原特征
            # 注意：这里直接修改 rel_feature_3d_view 的部分行
            rel_feature_3d_view[mask_indices] = mask_token.expand(num_masked, -1)
        return rel_feature_3d_view, mask_indices
    
    def student_view_construct(self, pts, edge_indices, descriptor, batch_ids, edge_mask_ratio=0.3,
                               obj_center=None, anchor_id=None, mask_ratio=None, istrain=False):
        edge_indices = edge_indices.long()
        batch_ids = batch_ids.long()
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor()(descriptor, edge_indices)
            
        if anchor_id is not None:
            point_agg_features, global_anchor_indices = self.obj_feat_extractor(
                pts, anchor_id, batch_ids, descriptor, mask_ratio=mask_ratio
            )
        else:
            point_agg_features = self.obj_feat_extractor(
                pts, None, batch_ids, descriptor, mask_ratio=mask_ratio
            )
        
        rel_feature_3d = self.rel_encoder_3d(edge_feature)
        rel_feature_3d, mask_indices =self.edge_masking(rel_feature_3d, edge_mask_ratio=edge_mask_ratio)
        
        if anchor_id is not None:
            gcn_obj_feature_3d, gcn_edge_feature_3d \
                    = self.mmg(point_agg_features, rel_feature_3d, edge_indices, batch_ids, global_anchor_indices, obj_center=obj_center, istrain=istrain)
            return gcn_obj_feature_3d, gcn_edge_feature_3d, mask_indices
        else:
            gcn_obj_feature_3d, gcn_edge_feature_3d \
            = self.mmg.forward_no_anchor(point_agg_features, rel_feature_3d,\
                                    edge_indices, batch_ids, obj_center=obj_center, istrain=istrain, GRU=True)
            return gcn_obj_feature_3d, gcn_edge_feature_3d, mask_indices
        
    def teacher_view_construct(self, pts, edge_indices, descriptor, batch_ids,\
        obj_center, mask_ratio = 0.2, edge_mask_ratio=0.2, istrain=False):
        edge_indices = edge_indices.long()
        batch_ids = batch_ids.long()
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor()(descriptor, edge_indices)
        with torch.no_grad():
            # self._update_target()
            point_agg_features_view2 = self.obj_feat_extractor(pts, None, batch_ids, descriptor, mask_ratio = mask_ratio, istarget=True)
            rel_feature_3d_view2 = self.rel_encoder_3d_target(edge_feature)
            
            rel_feature_3d_view2, _ =self.edge_masking(rel_feature_3d_view2, edge_mask_ratio=edge_mask_ratio, istarget=True)
            
            gcn_obj_feature_3d_view2, gcn_edge_feature_3d_view2 \
            = self.mmg_target.forward_no_anchor(point_agg_features_view2, rel_feature_3d_view2,\
                                    edge_indices, batch_ids, obj_center, istrain=istrain)
            
        return gcn_obj_feature_3d_view2, gcn_edge_feature_3d_view2
    
    def forward(self, pts, edge_indices, obj_points_spatial, descriptor=None, batch_ids=None,
                anchor_id=[], istrain=False, cur_obj_texts=None,
                obj_labels=None):
        if descriptor is None:
            raise ValueError("descriptor must be provided for PointDif pretraining.")

        # 记录中心点 (用位置编码 for weight computation)
        obj_center_v1 = descriptor[:, :3].clone()
        
        # =====================================================================
        # [Step 1] 单视图构建: v1 做 edge guide + node guide
        # =====================================================================
        
        # --- 1. Edge Guided Student: 使用锚点拓扑 ---
        obj_feat_stu_v1, edg_feat_stu_v1, _ = \
            self.student_view_construct(pts, edge_indices, descriptor, \
            batch_ids, anchor_id=anchor_id, edge_mask_ratio=0.1, mask_ratio=0.8, istrain=istrain)

        # --- 2. Node Guided Student: 同一份 v1，仅提高节点 mask ratio ---
        obj_feat_stu_v2, edg_feat_stu_v2, _ = \
            self.student_view_construct(pts, edge_indices, descriptor, \
            batch_ids, anchor_id=None, edge_mask_ratio=0.2, mask_ratio=0.8, istrain=istrain)
        
        # --- 3. Teacher: 同一份 v1，较弱 mask，提供稳定目标 ---
        with torch.no_grad():
            obj_feat_tea_v1, edg_feat_tea_v1 = \
                self.teacher_view_construct(pts, edge_indices, descriptor, \
                batch_ids, obj_center_v1, mask_ratio=0.2, edge_mask_ratio=0.2, istrain=istrain)
            obj_feat_tea_v2, edg_feat_tea_v2 = \
                self.teacher_view_construct(pts, edge_indices, descriptor, \
                batch_ids, obj_center_v1, mask_ratio=0.1, edge_mask_ratio=0.1, istrain=istrain)

        # =====================================================================
        # [Step 2] 扩散生成损失 (使用 Student v1 引导)
        # =====================================================================
        # 使用 spatial 坐标计算局部复杂度权重，重点关注复杂区域
        weights = compute_local_complexity_weight(obj_points_spatial)
        # 扩散模型恢复 obj_points_spatial，由 obj_feat_stu_v1 提供语义引导
        diff_loss, total_metric = self.point_diffusion.get_loss(obj_points_spatial, obj_feat_stu_v1, weights)

        # =====================================================================
        # [Step 3] 生成三元组特征 (Triplets: Object-Edge-Object)
        # =====================================================================
        # 辅助函数：根据边索引聚合节点和边特征
        def get_triplet(obj_feat, edge_feat):
            return self.generate_object_pair_features(obj_feat, edge_feat, edge_indices.t())

        # Students Triplets
        triplet_stu_v1 = get_triplet(obj_feat_stu_v1, edg_feat_stu_v1)
        triplet_stu_v2 = get_triplet(obj_feat_stu_v2, edg_feat_stu_v2)

        # Teachers Triplets (No Grad)
        with torch.no_grad():
            triplet_tea_v1 = get_triplet(obj_feat_tea_v1, edg_feat_tea_v1)
            triplet_tea_v2 = get_triplet(obj_feat_tea_v2, edg_feat_tea_v2)

        # =====================================================================
        # [Step 4] 自蒸馏预测头 (Predictors)
        # 只有学生需要通过 Predictor 来预测教师
        # =====================================================================
        
        # --- Object Predictors ---
        pred_obj_v1 = self.predictor_obj(obj_feat_stu_v1)
        pred_obj_v2 = self.predictor_obj(obj_feat_stu_v2)

        # --- Edge Predictors ---
        pred_edg_v1 = self.predictor_edge(edg_feat_stu_v1)
        pred_edg_v2 = self.predictor_edge(edg_feat_stu_v2)

        # --- Triplet Predictors ---
        pred_trip_v1 = self.predictor_triplet(triplet_stu_v1)
        pred_trip_v2 = self.predictor_triplet(triplet_stu_v2)

        # =====================================================================
        # [Step 5] 计算非对称自蒸馏损失 (Asymmetric Loss)
        # 核心逻辑：单视图双分支学生共同学习两个 teacher
        # =====================================================================
        
        # -------------------------------------------------------------
        # 1. Object Loss (SwAV / DINO Style)
        # -------------------------------------------------------------
        loss_obj_cross1, stats_obj_1 = self.mcr_obj_loss([pred_obj_v1], [obj_feat_tea_v1])
        loss_obj_cross2, stats_obj_2 = self.mcr_obj_loss([pred_obj_v1], [obj_feat_tea_v2])
        loss_obj_cross3, stats_obj_3 = self.mcr_obj_loss([pred_obj_v2], [obj_feat_tea_v1])
        loss_obj_cross4, stats_obj_4 = self.mcr_obj_loss([pred_obj_v2], [obj_feat_tea_v2])
        obj_loss = 0.25 * (loss_obj_cross1 + loss_obj_cross2 + loss_obj_cross3 + loss_obj_cross4)

        with torch.no_grad():
            comp_vals = [
                stats_obj_1['comp_loss'], stats_obj_2['comp_loss'],
                stats_obj_3['comp_loss'], stats_obj_4['comp_loss']
            ]
            expa_vals = [
                stats_obj_1['expa_loss'], stats_obj_2['expa_loss'],
                stats_obj_3['expa_loss'], stats_obj_4['expa_loss']
            ]
            self._last_obj_mcr_stats = {
                'obj_loss': obj_loss.detach(),
                'comp_loss': torch.stack(comp_vals).mean().detach(),
                'expa_loss': torch.stack(expa_vals).mean().detach(),
            }

        # -------------------------------------------------------------
        # 2. Edge Loss
        # -------------------------------------------------------------
        loss_edg_cross1, stats_edg_1 = self.mcr_edge_loss([pred_edg_v1], [edg_feat_tea_v1])
        loss_edg_cross2, stats_edg_2 = self.mcr_edge_loss([pred_edg_v1], [edg_feat_tea_v2])
        loss_edg_cross3, stats_edg_3 = self.mcr_edge_loss([pred_edg_v2], [edg_feat_tea_v1])
        loss_edg_cross4, stats_edg_4 = self.mcr_edge_loss([pred_edg_v2], [edg_feat_tea_v2])
        edge_loss = 0.25 * (loss_edg_cross1 + loss_edg_cross2 + loss_edg_cross3 + loss_edg_cross4)

        with torch.no_grad():
            comp_vals = [
                stats_edg_1['comp_loss'], stats_edg_2['comp_loss'],
                stats_edg_3['comp_loss'], stats_edg_4['comp_loss']
            ]
            expa_vals = [
                stats_edg_1['expa_loss'], stats_edg_2['expa_loss'],
                stats_edg_3['expa_loss'], stats_edg_4['expa_loss']
            ]
            self._last_edge_mcr_stats = {
                'edge_loss': edge_loss.detach(),
                'comp_loss': torch.stack(comp_vals).mean().detach(),
                'expa_loss': torch.stack(expa_vals).mean().detach(),
            }

        # -------------------------------------------------------------
        # 3. Triplet Loss
        # -------------------------------------------------------------
        loss_trip_cross1, stats_trip_1 = self.mcr_trip_loss([pred_trip_v1], [triplet_tea_v1])
        loss_trip_cross2, stats_trip_2 = self.mcr_trip_loss([pred_trip_v1], [triplet_tea_v2])
        loss_trip_cross3, stats_trip_3 = self.mcr_trip_loss([pred_trip_v2], [triplet_tea_v1])
        loss_trip_cross4, stats_trip_4 = self.mcr_trip_loss([pred_trip_v2], [triplet_tea_v2])
        triplet_loss = 0.25 * (loss_trip_cross1 + loss_trip_cross2 + loss_trip_cross3 + loss_trip_cross4)

        # Collect Statistics for Logging
        with torch.no_grad():
            comp_vals = [
                stats_trip_1['comp_loss'], stats_trip_2['comp_loss'],
                stats_trip_3['comp_loss'], stats_trip_4['comp_loss']
            ]
            expa_vals = [
                stats_trip_1['expa_loss'], stats_trip_2['expa_loss'],
                stats_trip_3['expa_loss'], stats_trip_4['expa_loss']
            ]
                
            self._last_trip_mcr_stats = {
                'triplet_loss': triplet_loss.detach(),
                'comp_loss': torch.stack(comp_vals).mean().detach(),
                'expa_loss': torch.stack(expa_vals).mean().detach(),
            }
        

        # =====================================================================
        # [Step 6] 文本对比损失 (Optional)
        # =====================================================================
        contrastive_loss = pred_obj_v1.new_zeros(())
        if cur_obj_texts is not None and istrain:
            # 假设这里是 CLIP 文本特征对齐
            # 注意：这里应该使用 Student 的特征来对齐文本
            contrastive_loss = self.contrastive_loss_fn(pred_obj_v1, cur_obj_texts) + \
                               self.contrastive_loss_fn(pred_obj_v2, cur_obj_texts)
            contrastive_loss = contrastive_loss / 2.0

        obj_label_contrastive_loss = pred_obj_v1.new_zeros(())
        if self.obj_label_contrastive_enabled and obj_labels is not None and istrain:
            instance_ids = self._build_global_instance_ids(obj_labels.shape[0], obj_labels.device)
            base_view_ids = torch.zeros_like(instance_ids)
            node_view_ids = torch.ones_like(instance_ids)

            anchor_features = torch.cat([pred_obj_v1, pred_obj_v2], dim=0)
            anchor_labels = obj_labels.repeat(2)
            anchor_instance_ids = instance_ids.repeat(2)
            anchor_view_ids = torch.cat([base_view_ids, node_view_ids], dim=0)

            local_teacher_features = torch.cat([obj_feat_tea_v1.detach(), obj_feat_tea_v2.detach()], dim=0)
            local_teacher_labels = obj_labels.repeat(2)
            local_teacher_instance_ids = instance_ids.repeat(2)
            local_teacher_view_ids = torch.cat([base_view_ids, node_view_ids], dim=0)

            contrast_features = self._gather_variable_tensor(local_teacher_features)
            contrast_labels = self._gather_variable_tensor(local_teacher_labels)
            contrast_instance_ids = self._gather_variable_tensor(local_teacher_instance_ids)
            contrast_view_ids = self._gather_variable_tensor(local_teacher_view_ids)

            obj_label_contrastive_loss = self.object_label_contrastive_loss_fn(
                anchor_features,
                anchor_labels,
                contrast_features,
                contrast_labels,
                anchor_instance_ids=anchor_instance_ids,
                contrast_instance_ids=contrast_instance_ids,
                anchor_view_ids=anchor_view_ids,
                contrast_view_ids=contrast_view_ids,
            )

        # =====================================================================
        # [Step 7] 总损失聚合
        # =====================================================================
        total_loss = diff_loss + \
                    0.1 * obj_loss + \
                    0.1 * edge_loss + \
                    0.1 * triplet_loss + \
                    0.1 * contrastive_loss + \
                    self.obj_label_contrastive_weight * obj_label_contrastive_loss
        
        # 返回 v1 的特征供 visualization 或 logging 使用
        return total_loss, diff_loss, triplet_loss, edge_loss, obj_loss, contrastive_loss, obj_label_contrastive_loss, total_metric

    def forward_cls(self, pts, edge_indices, descriptor=None,\
                batch_ids=None, istrain=False):
        
        point_agg_features = self.obj_feat_extractor(pts, None, batch_ids, descriptor, istarget=True)
        
        ##### 
        # Predicate features Extractors
        # ###
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor()(descriptor, edge_indices)
            
        rel_feature_3d = self.rel_encoder_3d_target(edge_feature)

        obj_center = descriptor[:, :3].clone()
        
        gcn_obj_feature_3d, gcn_edge_feature_3d \
            = self.mmg_target.forward_no_anchor(point_agg_features, rel_feature_3d, edge_indices, batch_ids, obj_center, istrain=istrain)
                       
            
        return gcn_edge_feature_3d, gcn_obj_feature_3d

    def forward_ori(self, pts, edge_indices, obj_points_spatial, descriptor=None,\
                batch_ids=None, anchor_id=[], istrain=False, cur_obj_texts=None):
        
        point_agg_features, global_anchor_indices = self.obj_feat_extractor(pts, anchor_id, batch_ids, descriptor)
    
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor()(descriptor, edge_indices)
            
        rel_feature_3d = self.rel_encoder_3d(edge_feature)
      
        obj_center = descriptor[:, :3].clone()
        
        gcn_obj_feature_3d, gcn_edge_feature_3d \
            = self.mmg(point_agg_features, rel_feature_3d, edge_indices, batch_ids, global_anchor_indices, obj_center, istrain=istrain)

       
        point_agg_features_spatial_view1 = gcn_obj_feature_3d 
      
        self.count += 1
        pred_points = self.point_diffusion.sample(1024, point_agg_features_spatial_view1, "cuda")
        # pred_points, collected_frames = self.point_diffusion.sampleN(1024, point_agg_features_spatial, "cuda", capture_range=(500,0),capture_num=20)
        diff_loss, total_x0_metric = self.point_diffusion.get_loss1(pts, point_agg_features_spatial_view1)
            
        visualize_scenes_plt(pred_points, obj_points_spatial, output_filename=f'/home/hyc/hyc_work/sceneGraph/SGG_DIR/sample_dir/sample_{self.count}.png')
        # visualize_scenes_batch(pred_points, obj_points_spatial, output_dir=f'/home/honsen/honsen/SceneGraph/SG_pretrain_diff/sample_dir/batch_sample_{self.count}')
        # visualize_and_save_sequence(collected_frames, save_path=f'/home/honsen/honsen/SceneGraph/SG_pretrain_diff/sample_dir/sequence_sample_{self.count}')
        return diff_loss, total_x0_metric, gcn_edge_feature_3d, gcn_obj_feature_3d
