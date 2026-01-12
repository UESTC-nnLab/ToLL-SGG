import torch
import sys
sys.path.append('/home/hyc/hyc_work/sceneGraph/SGG_DIR')

import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from src.model.diff_trans.models.build import MODELS
from src.model.diff_trans.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from src.model.diff_trans.utils.logger import *
from src.model.diff_trans.utils import misc
from src.model.diff_trans.models.mask_encoder import Mask_Encoder, Group, Encoder, TransformerEncoder
from src.model.diff_trans.models.generator import CPDM, CANet
from src.model.model_utils.model_base import BaseModel
import math
from src.model.model_utils.network_MMRGR import MMG
from src.model.model_utils.network_PointNet import (PointNetfeat,
                                                    PointNetRelCls,
                                                    PointNetRelClsMulti)
from torch.optim.lr_scheduler import LambdaLR
from src.utils import op_utils
from src.model.diff_trans.models.edge_obj_fusion import VertexEdgeCrossAttention
from src.dataset.dataset_diffPoint import visualize_scenes_plt, visualize_scenes_plt_with_points, visualize_scenes_batch, visualize_and_save_sequence

from src.model.diff_trans.models.shapeNet import BBoxPredictionHead, compute_aabb_ground_truth
from src.model.diff_trans.models.weight_focal_loss import compute_local_complexity_weight,visualize_batch_with_weights
from src.model.diff_trans.models.swav_loss import SwAVLoss
from src.model.diff_trans.models.contrastive_loss import TextSupervisedContrastiveLoss


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

        self.momentum = 0.1
        self.model_pre = None

        self.rel_encoder_3d = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=11,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=512)

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

        self.mlp_3d = torch.nn.Sequential(
            torch.nn.Linear(512, 512 - 11),
            torch.nn.LayerNorm(512 - 11),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3)
        )
        
        self.contrastive_loss_fn = TextSupervisedContrastiveLoss(temperature=0.07, num_negatives=1000)
        
        self.config = config.maskTrans
        self.group_size = self.config.group_size
        self.num_group = self.config.num_group
        self.trans_dim = self.config.encoder_config.trans_dim
        self.mask_encoder = Mask_Encoder(self.config)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.edge_mask_token = MaskedEdgeEncoder(512)
        
        self.drop_path_rate = self.config.encoder_config.drop_path_rate

        self.encoder_dims = self.config.encoder_config.encoder_dims
        self.cond_dims =  self.config.generator_config.cond_dims
        self.ca_net = CANet(self.encoder_dims, 512)
        
        # self.ob_fusion_net = VertexEdgeCrossAttention(embed_dim=512, num_heads=8)
        # self.bboxes_head = BBoxPredictionHead(512)
        # self.bbox_loss_fn = nn.SmoothL1Loss(reduction='mean')
        
        self.swav_reg_rel = SwAVLoss(num_prototypes=60)
        # self.swav_reg_obj = SwAVLoss(num_prototypes=800)
        
        self.point_diffusion = CPDM(self.config)
        self.count = 0
        
        print_log(f'[PointDif] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointDif')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        trunc_normal_(self.mask_token, std=.02)

    def obj_feat_extractor(self, pts, anchor_id, batch_ids, descriptor):
         ##### 
        # Point object features Extractors
        # ###
        
        B,_,_ = pts.shape        
        # get patch
        neighborhood, center = self.group_divider(pts)
        # mask and encoder
        encoder_token, mask = self.mask_encoder(neighborhood, center)
        _,N,_ = (center[mask].reshape(B,-1,3)).shape
        # learnable masked token
        mask_token = self.mask_token.expand(B, N, -1)
        encoder_token[mask] = mask_token.reshape(-1, self.trans_dim)
        
        point_agg_features = self.ca_net(encoder_token)
        
        if anchor_id is not None:
            anchor_set = 1
        else:
            anchor_set = 0
        
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
            point_agg_features = self.mlp_3d(point_agg_features)
            if self.mconfig.USE_SPATIAL:
                tmp = descriptor.clone()
                tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
                
                point_agg_features = torch.cat([point_agg_features, tmp], dim=-1)
        
            return point_agg_features
    
    def generate_object_pair_features(self, obj_feats, edges_feats, edge_indice):
        obj_pair_feats = []
        for (edge_feat, edge_index) in zip(edges_feats, edge_indice.t()):
            obj_pair_feats.append(torch.cat([obj_feats[edge_index[0]], obj_feats[edge_index[1]], edge_feat], dim=-1))
        obj_pair_feats = torch.vstack(obj_pair_feats)
        return obj_pair_feats
    
    def forward(self, pts, edge_indices, obj_points_spatial, descriptor=None,\
                batch_ids=None, anchor_id=[], istrain=False, cur_obj_texts=None):
        
        point_agg_features_view1, global_anchor_indices = self.obj_feat_extractor(pts, anchor_id, batch_ids, descriptor)
        point_agg_features_view2 = self.obj_feat_extractor(pts, None, batch_ids, descriptor)
        
        ##### 
        # Predicate features Extractors
        # ###
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor()(descriptor, edge_indices)
            
        rel_feature_3d = self.rel_encoder_3d(edge_feature)
        rel_feature_3d_view2 = self.rel_encoder_3d(edge_feature)#self.edge_mask_token(edge_feature.shape[0])

        if True:
            # 2. 定义掩码比例 (建议放在 config 中，此处暂定 0.3)
            edge_mask_ratio = 0.5 
            
            num_edges = rel_feature_3d_view2.shape[0]
            
            # 3. 生成随机掩码 (True 表示该边被 Mask 掉)
            # device 保持一致
            mask_indices = torch.rand(num_edges, device=rel_feature_3d_view2.device) < edge_mask_ratio
            
            # 4. 执行替换操作
            # 如果存在需要掩码的边
            if mask_indices.any():
                # 获取 mask token [1, 512]
                mask_token = self.edge_mask_token.mask_token
                
                # 计算需要掩码的数量
                num_masked = mask_indices.sum()
                
                # 将 mask_token 广播并覆盖原特征
                # 注意：这里直接修改 rel_feature_3d_view2 的部分行
                rel_feature_3d_view2[mask_indices] = mask_token.expand(num_masked, -1)
        # --- [修改结束] ---
        
        obj_center = descriptor[:, :3].clone()
        
        gcn_obj_feature_3d, gcn_edge_feature_3d \
            = self.mmg(point_agg_features_view1, rel_feature_3d, edge_indices, batch_ids, global_anchor_indices, obj_center, istrain=istrain)

        gcn_obj_feature_3d_view2, gcn_edge_feature_3d_view2 \
            = self.mmg.forward_no_anchor(point_agg_features_view2, rel_feature_3d_view2, edge_indices, batch_ids, obj_center, istrain=istrain)
        
        point_agg_features_spatial_view1 = gcn_obj_feature_3d
        point_agg_features_spatial_view2 = gcn_obj_feature_3d_view2
        
        if istrain:
            # diff_spatial_loss, total_spatial_metric = self.point_diffusion.get_loss(obj_points_spatial, point_agg_features_spatial)
            weights = compute_local_complexity_weight(obj_points_spatial, 16)
            
            # visualize_batch_with_weights(pts, weights, save_path=f'/home/honsen/honsen/SceneGraph/SG_pretrain_diff/src/dataset/batch_weights_{self.count}.png')
            
            # if mask_indices is not None and mask_indices.any():
            #     target_edge_feat = rel_feature_3d.detach() # 同样 detach
            #     loss_edge_recon = F.mse_loss(gcn_edge_feature_3d_view2[mask_indices], target_edge_feat[mask_indices])
            # else:
            #     loss_edge_recon = 0.0
            
            diff_loss, total_metric = self.point_diffusion.get_loss(obj_points_spatial, point_agg_features_spatial_view1, weights=weights)

            contrastive_loss = 0.0
            if cur_obj_texts is not None:
                text_embeddings_dict = torch.load('/home/hyc/hyc_work/sceneGraph/SGG_DIR/scannet_text_embeddings.pt')
                contrastive_loss = self.contrastive_loss_fn(point_agg_features_spatial_view1, cur_obj_texts, text_embeddings_dict)#+self.contrastive_loss_fn(point_agg_features_spatial_view2, cur_obj_texts, text_embeddings_dict)

            triplet_view1 = self.generate_object_pair_features(point_agg_features_spatial_view1, gcn_edge_feature_3d, edge_indices.t())
            
            triplet_view2 = self.generate_object_pair_features(point_agg_features_spatial_view2, gcn_edge_feature_3d_view2, edge_indices.t())
            
            triplet_loss_edge = self.swav_reg_rel(triplet_view1, triplet_view2)
            
            total_loss = diff_loss + 0.01*contrastive_loss
            
            return total_loss, diff_loss, contrastive_loss, total_metric, gcn_edge_feature_3d
        else:
            
            # pred_points = self.point_diffusion.sample(1024, point_agg_features_spatial, "cuda")
            # pred_points, collected_frames = self.point_diffusion.sampleN(1024, point_agg_features_spatial, "cuda", capture_range=(500,0),capture_num=20)
            diff_loss, total_x0_metric = self.point_diffusion.get_loss1(pts, point_agg_features_spatial_view1)
                
            # visualize_scenes_plt(pred_points, obj_points_spatial, output_filename=f'/home/honsen/honsen/SceneGraph/SG_pretrain_diff/sample_dir/sample_{self.count}.png')
            # visualize_scenes_batch(pred_points, obj_points_spatial, output_dir=f'/home/honsen/honsen/SceneGraph/SG_pretrain_diff/sample_dir/batch_sample_{self.count}')
            # visualize_and_save_sequence(collected_frames, save_path=f'/home/honsen/honsen/SceneGraph/SG_pretrain_diff/sample_dir/sequence_sample_{self.count}')
           
            self.count += 1
            return diff_loss, total_x0_metric, gcn_edge_feature_3d, gcn_obj_feature_3d
    
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
        if istrain:
            weights = compute_local_complexity_weight(obj_points_spatial, 8)
            
            diff_loss, total_metric = self.point_diffusion.get_loss(obj_points_spatial, point_agg_features, weights=None)

            contrastive_loss = 0.0
            if cur_obj_texts is not None:
                text_embeddings_dict = torch.load('/home/hyc/hyc_work/sceneGraph/SGG_DIR/scannet_text_embeddings.pt')
                contrastive_loss = self.contrastive_loss_fn(point_agg_features_spatial_view1, cur_obj_texts, text_embeddings_dict)
            
            total_loss = diff_loss #+ 0.02*contrastive_loss
            
            return total_loss, diff_loss, contrastive_loss, total_metric, gcn_edge_feature_3d
        else:
            self.count += 1
            pred_points = self.point_diffusion.sample(1024, point_agg_features, "cuda")
            # pred_points, collected_frames = self.point_diffusion.sampleN(1024, point_agg_features_spatial, "cuda", capture_range=(500,0),capture_num=20)
            diff_loss, total_x0_metric = self.point_diffusion.get_loss1(pts, point_agg_features_spatial_view1)
                
            visualize_scenes_plt(pred_points, obj_points_spatial, output_filename=f'/home/hyc/hyc_work/sceneGraph/SGG_DIR/sample_dir/sample_{self.count}.png')
            # visualize_scenes_batch(pred_points, obj_points_spatial, output_dir=f'/home/honsen/honsen/SceneGraph/SG_pretrain_diff/sample_dir/batch_sample_{self.count}')
            # visualize_and_save_sequence(collected_frames, save_path=f'/home/honsen/honsen/SceneGraph/SG_pretrain_diff/sample_dir/sequence_sample_{self.count}')
            return diff_loss, total_x0_metric, gcn_edge_feature_3d, gcn_obj_feature_3d
        
        
    def forward_cls(self, pts, edge_indices, descriptor=None,\
                batch_ids=None, istrain=False):
        
        point_agg_features = self.obj_feat_extractor(pts, None, batch_ids, descriptor)
        
        ##### 
        # Predicate features Extractors
        # ###
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor()(descriptor, edge_indices)
            
        rel_feature_3d = self.rel_encoder_3d(edge_feature)

        obj_center = descriptor[:, :3].clone()
        
        gcn_obj_feature_3d, gcn_edge_feature_3d \
            = self.mmg.forward_no_anchor(point_agg_features, rel_feature_3d, edge_indices, batch_ids, obj_center, istrain=istrain)
                       
            
        return gcn_edge_feature_3d, point_agg_features
    
    # def forward_cls(self, pts, edge_indices, descriptor=None,\
    #             batch_ids=None, istrain=False):
        
    #     B,_,_ = pts.shape        
    #     # get patch
    #     neighborhood, center = self.group_divider(pts)
    #     # mask and encoder
    #     encoder_token, mask = self.mask_encoder(neighborhood, center)
    #     _,N,_ = (center[mask].reshape(B,-1,3)).shape
    #     # learnable masked token
    #     mask_token = self.mask_token.expand(B, N, -1)
    #     encoder_token[mask] = mask_token.reshape(-1, self.trans_dim)
        
    #     point_agg_features = self.ca_net(encoder_token)
        
        
    #     #     ##### 
    #     # Predicate features Extractors
    #     # ###
    #     with torch.no_grad():
    #         edge_feature = op_utils.Gen_edge_descriptor()(descriptor, edge_indices)
            
    #     rel_feature_3d = self.rel_encoder_3d(edge_feature)
        
    #     return rel_feature_3d , point_agg_features