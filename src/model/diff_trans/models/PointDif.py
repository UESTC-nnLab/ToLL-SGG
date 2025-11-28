import torch
import sys
sys.path.append('/home/honsen/honsen/SceneGraph/SG_pretrain_diff')

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
from src.dataset.dataset_diffPoint import visualize_scenes_plt, visualize_scenes_plt_with_points, visualize_scenes_batch
from src.model.diff_trans.models.clustering import cluster_and_visualize
from src.model.diff_trans.models.shapeNet import BBoxPredictionHead, compute_aabb_ground_truth
from src.model.diff_trans.models.weight_focal_loss import compute_local_complexity_weight,visualize_batch_with_weights

class Config:
    """
    一个通用的配置类，可以将嵌套的字典递归地转换为类的属性。
    """
    def __init__(self, data_dict: dict):
        for key, value in data_dict.items():
            # 如果值是字典，则递归地创建一个新的Config实例
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            # 否则，直接将值设置为属性
            else:
                setattr(self, key, value)
    
    def __repr__(self, indent=0):
        """
        为类提供一个可读性好的打印输出格式。
        """
        lines = []
        indent_str = "  " * indent
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                lines.append(f"{indent_str}{key}:")
                lines.append(value.__repr__(indent + 1))
            else:
                lines.append(f"{indent_str}{key}: {repr(value)}")
        return "\n".join(lines)

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

        # self.rel_encoder_3d = PointNetfeat(
        #     global_feat=True,
        #     batch_norm=with_bn,
        #     point_size=11,
        #     input_transform=False,
        #     feature_transform=mconfig.feature_transform,
        #     out_size=512)

        # self.mmg = MMG(
        #     dim_node=512,
        #     dim_edge=512,
        #     dim_atten=256, #self.mconfig.DIM_ATTEN
        #     depth=2, #self.mconfig.N_LAYERS
        #     num_heads=8, #self.mconfig.NUM_HEADS
        #     aggr="max", #self.mconfig.GCN_AGGR
        #     flow=self.flow,
        #     attention="fat",#self.mconfig.ATTENTION
        #     use_edge=True,#self.mconfig.USE_GCN_EDGE
        #     DROP_OUT_ATTEN=0.5)#self.mconfig.DROP_OUT_ATTEN

        # self.mlp_3d = torch.nn.Sequential(
        #     torch.nn.Linear(512, 512 - 11),
        #     torch.nn.LayerNorm(512 - 11),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.1)
        # )
        
        self.config = config.maskTrans
        self.group_size = self.config.group_size
        self.num_group = self.config.num_group
        self.trans_dim = self.config.encoder_config.trans_dim
        self.mask_encoder = Mask_Encoder(self.config)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.drop_path_rate = self.config.encoder_config.drop_path_rate

        self.encoder_dims = self.config.encoder_config.encoder_dims
        self.cond_dims =  self.config.generator_config.cond_dims
        self.ca_net = CANet(self.encoder_dims, 512)
        
        # self.ob_fusion_net = VertexEdgeCrossAttention(embed_dim=512, num_heads=8)
        # self.bboxes_head = BBoxPredictionHead(512)
        # self.bbox_loss_fn = nn.SmoothL1Loss(reduction='mean')
        
        self.point_diffusion = CPDM(self.config)
        self.count = 0
        
        print_log(f'[PointDif] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointDif')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        trunc_normal_(self.mask_token, std=.02)

    def forward(self, pts, edge_indices, obj_points_spatial, descriptor=None, batch_ids=None, anchor_id=[], istrain=False, **kwargs):
        
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
        
        # anchor_set = 1
        
        # if anchor_set:
        #     device = point_agg_features.device

        #     # 2. 将本地锚点索引列表 (Python list) 转换为 Tensor
        #     #    例如: anchor_id = [1, 0, 2] (B=3, 第0个图的锚点是idx 1, ...)
        #     local_anchor_ids_tensor = torch.tensor(anchor_id, device=device, dtype=torch.long)

        #     # 3. 计算每个图的节点数
        #     #    batch_ids (N_total, 1) -> (N_total)
        #     batch_ids_squeezed = batch_ids.squeeze()
        #     #    bincount 会统计 [0, 0, 0, 1, 1, 2, 2, 2] -> [3, 2, 3]
        #     #    (即: 图0有3个节点, 图1有2个节点, 图2有3个节点)
        #     counts = torch.bincount(batch_ids_squeezed)

        #     # 4. 计算每个图的起始偏移量 (offsets)
        #     #    counts [3, 2, 3] -> cumsum [3, 5, 8] -> shifted [0, 3, 5]
        #     #    (即: 图0从idx 0开始, 图1从idx 3开始, 图2从idx 5开始)
        #     offsets = torch.cat([torch.tensor([0], device=device), torch.cumsum(counts, dim=0)[:-1]])

        #     # 5. 计算锚点的 *全局* 索引
        #     #    offsets [0, 3, 5] + local_ids [1, 0, 2] = global_ids [1, 3, 7]
        #     global_anchor_indices = offsets + local_anchor_ids_tensor
            
        #     # 6. 使用 *全局* 索引来提取和注入锚点特征
        #     #    (旧代码: anchor_obj = point_agg_features[anchor_id:anchor_id+1])
        #     anchor_obj_features = point_agg_features[global_anchor_indices]
        #     anchor_obj_features = self.mlp_3d(anchor_obj_features)
            
        #     if self.mconfig.USE_SPATIAL:
        #         tmp = descriptor.clone()
        #         tmp[:,6:] = tmp[:,6:].log() # only log on volume and length

        #         # (旧代码: tmp[anchor_id:anchor_id+1])
        #         anchor_spatial_info = tmp[global_anchor_indices]
                
        #         # (旧代码: abs_obj = torch.cat([anchor_obj, ...]))
        #         abs_obj_features = torch.cat([anchor_obj_features, anchor_spatial_info], dim=-1)
                
        #         # (旧代码: point_agg_features[anchor_id:anchor_id+1] = abs_obj.squeeze(0))
        #         point_agg_features[global_anchor_indices] = abs_obj_features
            
        #     # --- [!!! 锚点逻辑修改结束 !!!] ---
        
        # else:
        #     point_agg_features = self.mlp_3d(point_agg_features)
        #     if self.mconfig.USE_SPATIAL:
        #         tmp = descriptor.clone()
        #         tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
                
        #         point_agg_features = torch.cat([point_agg_features, tmp], dim=-1)
        
        # ##### 
        # # Predicate features Extractors
        # # ###
        # with torch.no_grad():
        #     edge_feature = op_utils.Gen_edge_descriptor()(descriptor, edge_indices)
        # rel_feature_3d = self.rel_encoder_3d(edge_feature)

        # obj_center = descriptor[:, :3].clone()
        
        # gcn_obj_feature_3d, gcn_edge_feature_3d \
        #     = self.mmg(point_agg_features, rel_feature_3d, edge_indices, batch_ids, global_anchor_indices, obj_center, istrain=istrain)

        # point_agg_features_spatial = gcn_obj_feature_3d
        
        # cluster_and_visualize(gcn_obj_feature_3d, 20, title_prefix="Object Features",\
        #                       save_path="/home/honsen/honsen/SceneGraph/SG_pretrain_diff/clustering_dir/object_features_cluster.png")
        # cluster_and_visualize(gcn_edge_feature_3d, 20, title_prefix="Edge Features",\
        #                       save_path="/home/honsen/honsen/SceneGraph/SG_pretrain_diff/clustering_dir/edge_features_cluster.png")
        if istrain:
            # diff_spatial_loss, total_spatial_metric = self.point_diffusion.get_loss(obj_points_spatial, point_agg_features_spatial)
            weights = compute_local_complexity_weight(pts)
            
            # visualize_batch_with_weights(pts, weights, save_path=f'/home/honsen/honsen/SceneGraph/SG_pretrain_diff/src/dataset/batch_weights_{self.count}.png')
            
            diff_loss, total_metric = self.point_diffusion.get_loss(pts, point_agg_features,weights=weights)
            # gt_bboxes = compute_aabb_ground_truth(obj_points_spatial)
            # pred_bboxes = self.bboxes_head(gcn_obj_feature_3d)
            # bboxes_loss = self.bbox_loss_fn(pred_bboxes, gt_bboxes)
            
            total_loss = diff_loss#+0.5*bboxes_loss
            
            return total_loss, total_metric
            # return total_loss, diff_loss, diff_spatial_loss, total_spatial_metric, total_metric
        else:
            
            pred_points = self.point_diffusion.sample(512, point_agg_features, "cuda")
            diff_loss, total_x0_metric = self.point_diffusion.get_loss1(pts, point_agg_features)
                
                # gt_bboxes = compute_aabb_ground_truth(obj_points_spatial)
                # pred_bboxes = self.bboxes_head(gcn_obj_feature_3d)
                # bboxes_loss = self.bbox_loss_fn(pred_bboxes, gt_bboxes)
            
            
            visualize_scenes_batch(pred_points, pts, output_dir=f'/home/honsen/honsen/SceneGraph/SG_pretrain_diff/sample_dir/batch_sample_{self.count}')
            self.count += 1
            return diff_loss, total_x0_metric

# finetune model
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]

        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.mask_encoder.", ""): v for k, v in ckpt['pointdif'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('pointdif'):
                    base_ckpt[k[len('pointdif.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)

        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        # A.max(1)
        
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        # ret = self.cls_head_finetune(concat_f)
        return concat_f#ret

def load_config_from_yaml(file_path):
    import yaml
    from types import SimpleNamespace
    """
    从嵌套的 YAML 文件加载模型配置。
    """
    try:
        with open(file_path, 'r') as f:
            full_config_dict = yaml.safe_load(f)
        
        # --- 关键改动在这里 ---
        # 提取 'model' 键下的字典
        if 'model' in full_config_dict:
            model_config_dict = full_config_dict['model']
        else:
            raise KeyError("在配置文件中找不到 'model' 这个主键。")
        # ----------------------

        # 将提取出的模型配置字典转换为 SimpleNamespace 对象
        config = SimpleNamespace(**model_config_dict)
        print(f"成功从 {file_path} 加载 'model' 配置。")
        return config
    except FileNotFoundError:
        print(f"错误：找不到配置文件 {file_path}")
        return None
    except Exception as e:
        print(f"加载或解析配置文件时出错: {e}")
        return None

def gen_descriptor(pts:torch.tensor):
    '''
    centroid_pts,std_pts,segment_dims,segment_volume,segment_lengths
    [3, 3, 3, 1, 1]
    '''
    assert pts.ndim==2
    assert pts.shape[-1]==3
    # centroid [n, 3]
    centroid_pts = pts.mean(0) 
    # # std [n, 3]
    std_pts = pts.std(0)
    # dimensions [n, 3]
    segment_dims = pts.max(dim=0)[0] - pts.min(dim=0)[0]
    # volume [n, 1]
    segment_volume = (segment_dims[0]*segment_dims[1]*segment_dims[2]).unsqueeze(0)
    # length [n, 1]
    segment_lengths = segment_dims.max().unsqueeze(0)
    return torch.cat([centroid_pts,std_pts,segment_dims,segment_volume,segment_lengths],dim=0)

def generate_and_sample_digraph_edges(num_vertices, num_edges_to_sample):
    import random
    """
    生成一个完全有向图的所有边，并从中随机抽取指定数量的边。

    参数:
    num_vertices (int): 图中的顶点数量。
    num_edges_to_sample (int): 需要随机抽取的边数。

    返回:
    list: 一个包含随机抽取的边的列表，每条边是一个元组 (u, v)。
    """
    all_edges = []
    
    # --- 步骤 1: 生成完全有向图的所有边 ---
    # 对于每个顶点u，创建一条指向所有其他顶点v的边
    for u in range(num_vertices):
        for v in range(num_vertices):
            if u != v:
                all_edges.append((u, v))

    # 检查理论最大边数是否足够进行抽样
    max_edges = num_vertices * (num_vertices - 1)
    if num_edges_to_sample > max_edges:
        print(f"错误：请求抽样的边数 ({num_edges_to_sample}) 超过了图中可能的最大边数 ({max_edges})。")
        return None

    # --- 步骤 2: 从所有边中随机抽取50条 ---
    # 使用 random.sample 进行无放回抽样，确保抽出的边不重复
    sampled_edges = random.sample(all_edges, num_edges_to_sample)
    
    return sampled_edges

if __name__ == '__main__':
    import json
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config_file_path = '/home/honsen/honsen/SceneGraph/SG_pretrain_diff/config/Diff_pretrain.json'
    
    # b. 从JSON文件读取内容并加载到Python字典
    try:
        with open(config_file_path, 'r') as f:
            # 使用 json.load() 从文件对象中读取
            config_dict = json.load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件 '{config_file_path}' 未找到。")
        exit()
    except json.JSONDecodeError:
        print(f"错误: 配置文件 '{config_file_path}' 不是有效的JSON格式。")
        exit()
    config = Config(config_dict)
    # config = load_config_from_yaml("/home/hyc/hyc_work/sceneGraph/SG_pretrain_diff/config/Diff_pretrain.json")
    
    model = PointDif(config).cuda()
    # model.load_model_from_ckpt("/home/hyc/hyc_work/sceneGraph/PointDif/save_dir/ckpt-epoch-300.pth")
    # #input: B N 3
    
    sample_edges = generate_and_sample_digraph_edges(512, 4600)
    
    sample_edges = torch.tensor(sample_edges).cuda().permute(1,0)  # 2 E
    
    inps = torch.randn((512, 256, 3)).cuda()
    
    for i in range(inps.shape[0]):
        desc = gen_descriptor(inps[i])
        if i==0:
            descriptor = desc.unsqueeze(0)
        else:
            descriptor = torch.cat([descriptor, desc.unsqueeze(0)], dim=0)
    
    batch_ids = torch.zeros((512, 1), dtype=torch.long).cuda()
    
    qwe = model(inps, sample_edges, descriptor=descriptor, batch_ids=batch_ids, istrain=True)
    
    
    print(qwe)
    
    # print()
    