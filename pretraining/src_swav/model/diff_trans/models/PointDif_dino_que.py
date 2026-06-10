import os
import torch
import sys
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from src.model.diff_trans.utils.logger import *
from src.model.diff_trans.models.build import MODELS
from src.model.diff_trans.models.mask_encoder import Mask_Encoder, Group, Encoder, TransformerEncoder
from src.model.diff_trans.models.generator import CPDM, CANet
from src.model.model_utils.model_base import BaseModel
from src.model.model_utils.network_MMRGR import MMG
from src.model.model_utils.network_PointNet import PointNetfeat
from src.utils import op_utils
from src.model.diff_trans.models.weight_focal_loss import compute_adaptive_weight, compute_local_complexity_weight
from src.model.diff_trans.models.swav_loss_que import SwAVLoss
from src.model.diff_trans.models.contrastive_loss import TextSupervisedContrastiveLoss, ObjectLabelContrastiveLoss

class MaskedEdgeEncoder(nn.Module):
    def __init__(self, edge_dim):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(1, edge_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, num_edges):
        masked_edge_features = self.mask_token.expand(num_edges, -1)
        return masked_edge_features

# @MODELS.register_module()
class PointDif(BaseModel):
    def __init__(self, config, dim_descriptor=11):
        super().__init__('Diff_sg', config)
        print_log(f'[Diff_sg] ', logger ='Diff_sg')
        
        self.mconfig = mconfig = config.sg_model
        with_bn = mconfig.WITH_BN
        dim_point = 3
        if mconfig.USE_RGB: dim_point +=3
        if mconfig.USE_NORMAL: dim_point +=3
        dim_f_spatial = dim_descriptor
        self.dim_point = dim_point
        self.dim_edge = dim_f_spatial
        self.flow = 'target_to_source'
        self.momentum = 0.996

        self.rel_encoder_3d = PointNetfeat(global_feat=True, batch_norm=with_bn, point_size=11, input_transform=False, feature_transform=mconfig.feature_transform, out_size=512)

        self.mmg = MMG(dim_node=512, dim_edge=512, dim_atten=256, depth=2, num_heads=8, aggr="max", flow=self.flow, attention="fat", use_edge=True, DROP_OUT_ATTEN=0.5)
        
        self.config = config.maskTrans
        self.group_size = self.config.group_size
        self.num_group = self.config.num_group
        self.trans_dim = self.config.encoder_config.trans_dim
        self.mask_encoder = Mask_Encoder(self.config)
        self.encoder_dims = self.config.encoder_config.encoder_dims
        
        self.mlp_3d = torch.nn.Sequential(torch.nn.Linear(512, 512 - 11), torch.nn.ReLU(), torch.nn.Dropout(0.3))
        self.ca_net = CANet(self.encoder_dims, 512)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.edge_mask_token = MaskedEdgeEncoder(512)
        
        # Prototypes
        self.prototypes_triplet = nn.Linear(512*3, 500, bias=False)
        self.prototypes_edge = nn.Linear(512, 200, bias=False)
        self.prototypes_obj = nn.Linear(512, 1000, bias=False)

        # =========================================================
        # [新增] 队列 (Queue) 初始化 - 支持 Object, Edge, Triplet
        # =========================================================
        self.queue_length = 3840 # 约 30-40 个 batch
        self.feature_dim = 512
        
        # 1. Object Queue
        self.register_buffer("queue_obj", torch.randn(self.queue_length, self.feature_dim))
        self.queue_obj = F.normalize(self.queue_obj, dim=1)
        self.register_buffer("queue_obj_ptr", torch.zeros(1, dtype=torch.long))

        # 2. Edge Queue
        self.register_buffer("queue_edge", torch.randn(self.queue_length, self.feature_dim))
        self.queue_edge = F.normalize(self.queue_edge, dim=1)
        self.register_buffer("queue_edge_ptr", torch.zeros(1, dtype=torch.long))

        # 3. Triplet Queue (注意维度是 512*3 = 1536)
        self.register_buffer("queue_trip", torch.randn(self.queue_length, self.feature_dim * 3))
        self.queue_trip = F.normalize(self.queue_trip, dim=1)
        self.register_buffer("queue_trip_ptr", torch.zeros(1, dtype=torch.long))
        
        # [新增] 队列满标志位 (用于 Warm-up 控制)
        self.register_buffer("queue_is_full", torch.zeros(1, dtype=torch.bool))

        # --- Teacher Network ---
        self.mask_encoder_target = copy.deepcopy(self.mask_encoder)
        self.rel_encoder_3d_target = copy.deepcopy(self.rel_encoder_3d)
        self.mmg_target = copy.deepcopy(self.mmg)
        self.mlp_3d_target = copy.deepcopy(self.mlp_3d)
        self.ca_net_target = copy.deepcopy(self.ca_net)
        self.mask_token_target = copy.deepcopy(self.mask_token)
        self.edge_mask_token_target = copy.deepcopy(self.edge_mask_token)
        self.prototypes_triplet_target = copy.deepcopy(self.prototypes_triplet)
        self.prototypes_edge_target = copy.deepcopy(self.prototypes_edge)
        self.prototypes_obj_target = copy.deepcopy(self.prototypes_obj)

        for p in self.get_target_params():
            p.requires_grad = False

        text_emb_path = getattr(config, "TEXT_EMB_PATH", getattr(config, "SCANNET_TEXT_EMB_PATH", None))
        self.contrastive_loss_fn = TextSupervisedContrastiveLoss(
            temperature=0.07,
            num_negatives=1000,
            text_embeddings_path=text_emb_path,
        )
        self.object_label_contrastive_loss_fn = ObjectLabelContrastiveLoss(
            temperature=float(getattr(config, "OBJ_LABEL_CONTRASTIVE_TEMPERATURE", 0.07))
        )
        self.text_contrastive_enabled = bool(getattr(config, "TEXT_CONTRASTIVE_ENABLED", False))
        self.obj_label_contrastive_enabled = bool(getattr(config, "OBJ_LABEL_CONTRASTIVE_ENABLED", False))
        self.obj_label_contrastive_weight = float(getattr(config, "OBJ_LABEL_CONTRASTIVE_WEIGHT", 0.0))
        self.diffusion_enabled = bool(getattr(config, "DIFFUSION_ENABLED", True))
        self.diffusion_loss_weight = float(getattr(config, "DIFFUSION_LOSS_WEIGHT", 10.0))
        self.atlas_align_enabled = bool(getattr(config, "ATLAS_ALIGN_ENABLED", False))
        self.atlas_latent_dim = int(getattr(config, "ATLAS_LATENT_DIM", 1024))
        self.atlas_align_weight = float(getattr(config, "ATLAS_ALIGN_WEIGHT", 0.1))
        self.atlas_align_reg_weight = float(getattr(config, "ATLAS_ALIGN_REG_WEIGHT", 1.0))
        self.atlas_align_cosine_weight = float(getattr(config, "ATLAS_ALIGN_COSINE_WEIGHT", 1.0))
        self.atlas_align_head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, self.atlas_latent_dim),
        )
        self.atlas_align_loss_fn = nn.SmoothL1Loss(reduction='mean')
        # 注意: SwAVLoss 初始化时的 sinkhorn_iterations 建议设为 10
        self.swav_reg_triplet = SwAVLoss(self.prototypes_triplet, self.prototypes_triplet_target, sinkhorn_iterations=10)
        self.swav_reg_edge = SwAVLoss(self.prototypes_edge, self.prototypes_edge_target, sinkhorn_iterations=10)   
        self.swav_reg_obj = SwAVLoss(self.prototypes_obj, self.prototypes_obj_target, sinkhorn_iterations=10)        
        
        self.point_diffusion = CPDM(self.config)
        self.count = 0
        
        # Predictors
        self.predictor_triplet = nn.Sequential(nn.Linear(512*3, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Linear(1024, 512*3))
        self.predictor_edge = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 512))
        self.predictor_obj = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 512))

        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        trunc_normal_(self.mask_token, std=.02)

    def get_online_params(self):
        return (list(self.mask_encoder.parameters()) + list(self.rel_encoder_3d.parameters()) + list(self.mmg.parameters()) + list(self.mlp_3d.parameters()) + list(self.ca_net.parameters()) + [self.mask_token] + list(self.edge_mask_token.parameters()) + list(self.prototypes_triplet.parameters()) + list(self.prototypes_edge.parameters()) + list(self.prototypes_obj.parameters()))

    def get_target_params(self):    
        return (list(self.mask_encoder_target.parameters()) + list(self.rel_encoder_3d_target.parameters()) + list(self.mmg_target.parameters()) + list(self.mlp_3d_target.parameters()) + list(self.ca_net_target.parameters()) + [self.mask_token_target] + list(self.edge_mask_token_target.parameters()) + list(self.prototypes_triplet_target.parameters()) + list(self.prototypes_edge_target.parameters()) + list(self.prototypes_obj_target.parameters()))

    @torch.no_grad()
    def _update_target(self, momentum):
        self.momentum = momentum
        for p_o, p_t in zip(self.get_online_params(), self.get_target_params()):
            p_t.data = p_t.data * self.momentum + p_o.data * (1. - self.momentum)

    # ==================== Queue Operations (核心逻辑) ====================
    @torch.no_grad()
    def dequeue_and_enqueue(self, keys, queue_name='obj'):
        """
        更新队列：将当前 keys 入队，覆盖最旧的特征
        keys: (B, D) Tensor
        queue_name: 'obj', 'edge', or 'trip'
        """
        keys = self._gather_variable_tensor(keys.detach())
        
        # 2. 选择对应的队列和指针
        if queue_name == 'obj':
            queue = self.queue_obj
            ptr = self.queue_obj_ptr
        elif queue_name == 'edge':
            queue = self.queue_edge
            ptr = self.queue_edge_ptr
        elif queue_name == 'trip':
            queue = self.queue_trip
            ptr = self.queue_trip_ptr
        else:
            return

        batch_size = keys.shape[0]
        ptr_val = int(ptr)

        # 3. 环形写入
        if ptr_val + batch_size > self.queue_length:
            len_part1 = self.queue_length - ptr_val
            len_part2 = batch_size - len_part1
            queue[ptr_val:] = keys[:len_part1]
            queue[:len_part2] = keys[len_part1:]
            ptr[0] = len_part2
            # 只要发生回绕，说明填满过至少一次
            # 注意：我们需要三个队列都填满，才算 is_full
            # 这里简化逻辑：由于三个特征是同时入队的，只要其中一个满了，理论上都满了
            if queue_name == 'obj': 
                self.queue_is_full[0] = True
        else:
            queue[ptr_val:ptr_val + batch_size] = keys
            ptr[0] = (ptr_val + batch_size) % self.queue_length
            
            # 刚好填满的情况
            if ptr[0] == 0 and batch_size > 0 and queue_name == 'obj':
                 self.queue_is_full[0] = True

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

    def compute_atlas_alignment_loss(self, obj_features, atlas_embeddings, atlas_valid_mask=None):
        loss = obj_features.new_zeros(())
        if atlas_embeddings is None or obj_features.shape[0] == 0:
            return loss

        if atlas_valid_mask is None:
            atlas_valid_mask = torch.ones(
                obj_features.shape[0], device=obj_features.device, dtype=torch.bool
            )
        else:
            atlas_valid_mask = atlas_valid_mask.bool()

        if not torch.any(atlas_valid_mask):
            return loss

        pred_embeddings = self.atlas_align_head(obj_features[atlas_valid_mask])
        target_embeddings = atlas_embeddings[atlas_valid_mask].detach()

        # AtlasNet latent vectors have a strong global bias and fairly large norms.
        # Standardizing each sample before regression avoids the old "very small almost-constant"
        # loss caused by unit-normalizing then averaging 1024 tiny per-dim errors.
        pred_embeddings = F.layer_norm(pred_embeddings, (pred_embeddings.shape[-1],))
        target_embeddings = F.layer_norm(target_embeddings, (target_embeddings.shape[-1],))

        reg_loss = self.atlas_align_loss_fn(pred_embeddings, target_embeddings)
        cosine_loss = 1.0 - F.cosine_similarity(pred_embeddings, target_embeddings, dim=-1).mean()
        return self.atlas_align_reg_weight * reg_loss + self.atlas_align_cosine_weight * cosine_loss

    # ... [obj_feat_extractor, generate_object_pair_features, edge_masking, student_view_construct, teacher_view_construct 保持原有逻辑不变] ...
    # (此处省略中间的特征提取函数，直接复制原来的即可)
    def obj_feat_extractor(self, pts, anchor_id, batch_ids, descriptor, mask_ratio=None, istarget=False, isview3 = False):
        # ... (使用原来的代码) ...
        # 注意: 确保此处包含之前讨论过的锚点索引修复逻辑
        if anchor_id is not None:
            anchor_set = 1
        else:
            anchor_set = 0
        B,_,_ = pts.shape        
        neighborhood, center = self.group_divider(pts)
        if istarget:
            encoder_token, mask = self.mask_encoder_target(neighborhood, center, mask_ratio=mask_ratio)
            _,N,_ = (center[mask].reshape(B,-1,3)).shape
            mask_token = self.mask_token_target.expand(B, N, -1)
            encoder_token[mask] = mask_token.reshape(-1, self.trans_dim)
            point_agg_features = self.ca_net_target(encoder_token)
        else:
            encoder_token, mask = self.mask_encoder(neighborhood, center, mask_ratio=mask_ratio)
            _,N,_ = (center[mask].reshape(B,-1,3)).shape
            mask_token = self.mask_token.expand(B, N, -1)
            encoder_token[mask] = mask_token.reshape(-1, self.trans_dim)
            point_agg_features = self.ca_net(encoder_token)
        if isview3: return point_agg_features
        if anchor_set:
            device = point_agg_features.device
            local_anchor_ids_tensor = torch.tensor(anchor_id, device=device, dtype=torch.long)
            batch_ids_squeezed = batch_ids.squeeze()
            counts = torch.bincount(batch_ids_squeezed)
            offsets = torch.cat([torch.tensor([0], device=device), torch.cumsum(counts, dim=0)[:-1]])
            global_anchor_indices = offsets + local_anchor_ids_tensor
            anchor_obj_features = point_agg_features[global_anchor_indices]
            if istarget: anchor_obj_features = self.mlp_3d_target(anchor_obj_features)
            else: anchor_obj_features = self.mlp_3d(anchor_obj_features)
            if self.mconfig.USE_SPATIAL:
                tmp = descriptor.clone()
                tmp[:,6:] = tmp[:,6:].log()
                anchor_spatial_info = tmp[global_anchor_indices]
                abs_obj_features = torch.cat([anchor_obj_features, anchor_spatial_info], dim=-1)
                point_agg_features[global_anchor_indices] = abs_obj_features
            return point_agg_features, global_anchor_indices
        else:
            if istarget: point_agg_features1 = self.mlp_3d_target(point_agg_features)
            else: point_agg_features1 = self.mlp_3d(point_agg_features)   
            if self.mconfig.USE_SPATIAL:
                tmp = descriptor.clone()
                tmp[:,6:] = tmp[:,6:].log()
                point_agg_features1 = torch.cat([point_agg_features1, tmp], dim=-1)
                return point_agg_features1

    def generate_object_pair_features(self, obj_feats, edges_feats, edge_indice):
        obj_pair_feats = []
        for (edge_feat, edge_index) in zip(edges_feats, edge_indice.t()):
            obj_pair_feats.append(torch.cat([obj_feats[edge_index[0]], obj_feats[edge_index[1]], edge_feat], dim=-1))
        obj_pair_feats = torch.vstack(obj_pair_feats)
        return obj_pair_feats
    
    def edge_masking(self, rel_feature_3d_view, edge_mask_ratio=0.4, istarget=False):
        num_edges = rel_feature_3d_view.shape[0]
        mask_indices = torch.rand(num_edges, device=rel_feature_3d_view.device) < edge_mask_ratio
        if mask_indices.any():
            if istarget: mask_token = self.edge_mask_token_target.mask_token
            else: mask_token = self.edge_mask_token.mask_token
            num_masked = mask_indices.sum()
            rel_feature_3d_view[mask_indices] = mask_token.expand(num_masked, -1)
        return rel_feature_3d_view, mask_indices

    def student_view_construct(self, pts, edge_indices, descriptor, batch_ids, edge_mask_ratio=0.3, obj_center=None, anchor_id=None, istrain=False):
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor()(descriptor, edge_indices)
        if anchor_id is not None:
            point_agg_features, global_anchor_indices = self.obj_feat_extractor(pts, anchor_id, batch_ids, descriptor)
        else:
            point_agg_features = self.obj_feat_extractor(pts, None, batch_ids, descriptor)
        rel_feature_3d = self.rel_encoder_3d(edge_feature)
        rel_feature_3d, mask_indices =self.edge_masking(rel_feature_3d, edge_mask_ratio=edge_mask_ratio)
        if anchor_id is not None:
            gcn_obj_feature_3d, gcn_edge_feature_3d = self.mmg(point_agg_features, rel_feature_3d, edge_indices, batch_ids, global_anchor_indices, obj_center=obj_center, istrain=istrain)
            return gcn_obj_feature_3d, gcn_edge_feature_3d, mask_indices
        else:
            gcn_obj_feature_3d, gcn_edge_feature_3d = self.mmg.forward_no_anchor(point_agg_features, rel_feature_3d, edge_indices, batch_ids, obj_center=obj_center, istrain=istrain, GRU=True)
            return gcn_obj_feature_3d, gcn_edge_feature_3d, mask_indices

    def teacher_view_construct(self, pts, edge_indices, descriptor, batch_ids, obj_center, mask_ratio = 0.2, edge_mask_ratio=0.2, istrain=False):
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor()(descriptor, edge_indices)
        with torch.no_grad():
            point_agg_features_view2 = self.obj_feat_extractor(pts, None, batch_ids, descriptor, mask_ratio = mask_ratio, istarget=True)
            rel_feature_3d_view2 = self.rel_encoder_3d_target(edge_feature)
            rel_feature_3d_view2, _ =self.edge_masking(rel_feature_3d_view2, edge_mask_ratio=edge_mask_ratio, istarget=True)
            gcn_obj_feature_3d_view2, gcn_edge_feature_3d_view2 = self.mmg_target.forward_no_anchor(point_agg_features_view2, rel_feature_3d_view2, edge_indices, batch_ids, obj_center, istrain=istrain)
        return gcn_obj_feature_3d_view2, gcn_edge_feature_3d_view2

    def forward(self, pts, edge_indices, obj_points_spatial, descriptor=None, pts_v2=None, \
                descriptor_v2=None, batch_ids=None, anchor_id=[], istrain=False, cur_obj_texts=None,
                obj_labels=None, atlas_embeddings=None, atlas_valid_mask=None):
        
        obj_center_v1 = descriptor[:, :3].clone()
        obj_center_v2 = descriptor_v2[:, :3].clone()
        
        # [Step 1] 构建视图
        obj_feat_stu_v1, edg_feat_stu_v1, _ = \
            self.student_view_construct(pts, edge_indices, descriptor, \
            batch_ids, anchor_id = anchor_id, obj_center=obj_center_v1, edge_mask_ratio=0.1, istrain=istrain)
        
        obj_feat_stu_v2, edg_feat_stu_v2, mask_indices_v2 = \
            self.student_view_construct(pts_v2, edge_indices, descriptor_v2, \
            batch_ids, obj_center=obj_center_v2, edge_mask_ratio=0.4, istrain=istrain)
        
        # Student v3 is temporarily disabled.
        # obj_feat_stu_v3, edg_feat_stu_v3, mask_indices_v3 = \
        #     self.student_view_construct(pts_v2, edge_indices, descriptor, \
        #     batch_ids, obj_center=obj_center_v1, edge_mask_ratio=0.4, istrain=istrain)
        
        # Teachers
        with torch.no_grad():
            obj_feat_tea_v6, edg_feat_tea_v6 = \
                self.teacher_view_construct(pts_v2, edge_indices, descriptor, \
                batch_ids, obj_center_v1, mask_ratio=0.2, edge_mask_ratio=0.1, istrain=istrain)
            
            obj_feat_tea_v5, edg_feat_tea_v5 = \
                self.teacher_view_construct(pts_v2, edge_indices, descriptor_v2, \
                batch_ids, obj_center_v2, mask_ratio=0.1, edge_mask_ratio=0.1, istrain=istrain)

        # [Step 2] 扩散生成损失
        diff_loss = obj_feat_stu_v1.new_zeros(())
        total_metric = 0.0
        if self.diffusion_enabled:
            weights = compute_local_complexity_weight(obj_points_spatial)
            diff_loss, total_metric = self.point_diffusion.get_loss(obj_points_spatial, obj_feat_stu_v1, weights)

        # [Step 3] 生成三元组特征
        def get_triplet(obj_feat, edge_feat):
            return self.generate_object_pair_features(obj_feat, edge_feat, edge_indices.t())

        triplet_stu_v1 = get_triplet(obj_feat_stu_v1, edg_feat_stu_v1)
        triplet_stu_v2 = get_triplet(obj_feat_stu_v2, edg_feat_stu_v2)
        # triplet_stu_v3 = get_triplet(obj_feat_stu_v3, edg_feat_stu_v3)
        
        
        with torch.no_grad():
            triplet_tea_v6 = get_triplet(obj_feat_tea_v6, edg_feat_tea_v6)
            triplet_tea_v5 = get_triplet(obj_feat_tea_v5, edg_feat_tea_v5)

        # [Step 4] Predictors
        pred_obj_v1 = self.predictor_obj(obj_feat_stu_v1)
        pred_obj_v2 = self.predictor_obj(obj_feat_stu_v2)
        # pred_obj_v3 = self.predictor_obj(obj_feat_stu_v3)
        
        pred_edg_v1 = self.predictor_edge(edg_feat_stu_v1)
        pred_edg_v2 = self.predictor_edge(edg_feat_stu_v2)
        # pred_edg_v3 = self.predictor_edge(edg_feat_stu_v3)

        pred_trip_v1 = self.predictor_triplet(triplet_stu_v1)
        pred_trip_v2 = self.predictor_triplet(triplet_stu_v2)
        # pred_trip_v3 = self.predictor_triplet(triplet_stu_v3)

        # [Step 5] 计算非对称自蒸馏损失
        
        # =========================================================
        # [关键] 准备队列
        # 如果队列满了(queue_is_full=True)，则传入 SwAVLoss
        # 如果没满(Warm-up阶段或刚开始)，传入 None，SwAVLoss 退化为纯 Batch 模式
        # =========================================================
        q_obj = self.queue_obj.clone().detach() if self.queue_is_full[0] else None
        q_edge = self.queue_edge.clone().detach() if self.queue_is_full[0] else None
        q_trip = self.queue_trip.clone().detach() if self.queue_is_full[0] else None

        # 1. Object Loss
        loss_obj_cross1 = self.swav_reg_obj.forward_asymmetric(obj_feat_tea_v6, pred_obj_v2, queue=q_obj)
        loss_obj_cross3 = self.swav_reg_obj.forward_asymmetric(obj_feat_tea_v5, pred_obj_v1, queue=q_obj)
        obj_loss = (loss_obj_cross1 + loss_obj_cross3) / 2.0

        # 2. Edge Loss
        loss_edg_cross1 = self.swav_reg_edge.forward_asymmetric(edg_feat_tea_v6, pred_edg_v2, queue=q_edge)
        loss_edg_cross2 = self.swav_reg_edge.forward_asymmetric(edg_feat_tea_v5, pred_edg_v2, queue=q_edge)
        loss_edg_cross3 = self.swav_reg_edge.forward_asymmetric(edg_feat_tea_v5, pred_edg_v1, queue=q_edge)
        edge_loss = (loss_edg_cross1 + loss_edg_cross2 + loss_edg_cross3) / 3.0
        # 3. Triplet Loss
        loss_trip_cross1 = self.swav_reg_triplet.forward_asymmetric(triplet_tea_v6, pred_trip_v2, queue=q_trip)
        loss_trip_cross2 = self.swav_reg_triplet.forward_asymmetric(triplet_tea_v5, pred_trip_v2, queue=q_trip)
        loss_trip_cross3 = self.swav_reg_triplet.forward_asymmetric(triplet_tea_v5, pred_trip_v1, queue=q_trip)
        triplet_loss = (loss_trip_cross1 + loss_trip_cross2 + loss_trip_cross3) / 3.0

        # [Step 6] 文本损失
        contrastive_loss = pred_obj_v1.new_zeros(())
        if self.text_contrastive_enabled and cur_obj_texts is not None and istrain:
            contrastive_loss = self.contrastive_loss_fn(pred_obj_v1, cur_obj_texts) + \
                                self.contrastive_loss_fn(pred_obj_v2, cur_obj_texts)
            contrastive_loss = contrastive_loss / 2.0

        obj_label_contrastive_loss = pred_obj_v1.new_zeros(())
        if self.obj_label_contrastive_enabled and obj_labels is not None and istrain:
            instance_ids = self._build_global_instance_ids(obj_labels.shape[0], obj_labels.device)
            base_view_ids = torch.zeros_like(instance_ids)
            aug_view_ids = torch.ones_like(instance_ids)

            anchor_features = torch.cat([pred_obj_v1, pred_obj_v2], dim=0)
            anchor_labels = obj_labels.repeat(2)
            anchor_instance_ids = instance_ids.repeat(2)
            anchor_view_ids = torch.cat([base_view_ids, aug_view_ids], dim=0)

            local_teacher_features = torch.cat([obj_feat_tea_v5.detach(), obj_feat_tea_v6.detach()], dim=0)
            local_teacher_labels = obj_labels.repeat(2)
            local_teacher_instance_ids = instance_ids.repeat(2)
            local_teacher_view_ids = torch.cat([aug_view_ids, base_view_ids], dim=0)

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

        atlas_align_loss = pred_obj_v1.new_zeros(())
        if self.atlas_align_enabled and atlas_embeddings is not None and istrain:
            atlas_align_loss = self.compute_atlas_alignment_loss(
                obj_feat_stu_v1,
                atlas_embeddings,
                atlas_valid_mask=atlas_valid_mask,
            )
        
        # [Step 7] 总损失
        total_loss = self.diffusion_loss_weight * diff_loss + \
                     0.1 * obj_loss + \
                     0.1 * edge_loss + \
                     0.1 * triplet_loss + \
                     0.1 * contrastive_loss + \
                     self.obj_label_contrastive_weight * obj_label_contrastive_loss + \
                     self.atlas_align_weight * atlas_align_loss
        
        # =========================================================
        # [新增] 更新队列 (Queue Update)
        # 只有在训练模式下才更新
        # =========================================================
        if istrain:
            self.dequeue_and_enqueue(obj_feat_tea_v6, queue_name='obj')
            self.dequeue_and_enqueue(edg_feat_tea_v6, queue_name='edge')
            self.dequeue_and_enqueue(triplet_tea_v6, queue_name='trip')

        return total_loss, diff_loss, triplet_loss, edge_loss, obj_loss, contrastive_loss, obj_label_contrastive_loss, atlas_align_loss, total_metric, edg_feat_tea_v6

    # ... [forward_cls, forward_ori 保持不变] ...
    def forward_cls(self, pts, edge_indices, descriptor=None, batch_ids=None, istrain=False):
        point_agg_features = self.obj_feat_extractor(pts, None, batch_ids, descriptor, istarget=True)
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor()(descriptor, edge_indices)
        rel_feature_3d = self.rel_encoder_3d_target(edge_feature)
        obj_center = descriptor[:, :3].clone()
        gcn_obj_feature_3d, gcn_edge_feature_3d = self.mmg_target.forward_no_anchor(point_agg_features, rel_feature_3d, edge_indices, batch_ids, obj_center, istrain=istrain)
        return gcn_edge_feature_3d, gcn_obj_feature_3d

    def forward_ori(self, pts, edge_indices, obj_points_spatial, descriptor=None, batch_ids=None, anchor_id=[], istrain=False, cur_obj_texts=None):
        point_agg_features, global_anchor_indices = self.obj_feat_extractor(pts, anchor_id, batch_ids, descriptor)
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor()(descriptor, edge_indices)
        rel_feature_3d = self.rel_encoder_3d(edge_feature)
        obj_center = descriptor[:, :3].clone()
        gcn_obj_feature_3d, gcn_edge_feature_3d = self.mmg(point_agg_features, rel_feature_3d, edge_indices, batch_ids, global_anchor_indices, obj_center, istrain=istrain)
        return gcn_edge_feature_3d, gcn_obj_feature_3d
