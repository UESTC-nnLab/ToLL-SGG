import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.model_utils.network_util import (MLP, Aggre_Index, Gen_Index,
                                                build_mlp)
from src.model.transformer.attention import MultiHeadAttention

class GraphEdgeAttenNetwork(torch.nn.Module):
    def __init__(self, num_heads, dim_node, dim_edge, dim_atten, aggr= 'max', use_bn=False,
                 flow='target_to_source',attention = 'fat',use_edge:bool=True, **kwargs):
        super().__init__() 
        self.name = 'edgeatten'
        self.dim_node=dim_node
        self.dim_edge=dim_edge
        self.index_get = Gen_Index()
        if attention == 'fat':       
            self.index_aggr = Aggre_Index(aggr=aggr)
        elif attention == 'distance':
            aggr = 'add'
            self.index_aggr = Aggre_Index(aggr=aggr)
        else:
            raise NotImplementedError()

        self.edgeatten = MultiHeadedEdgeAttention(
            dim_node=dim_node,dim_edge=dim_edge,dim_atten=dim_atten,
            num_heads=num_heads,use_bn=use_bn,attention=attention,use_edge=use_edge, **kwargs)
        self.prop = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node],
                               do_bn= use_bn, on_last=False)

    def forward(self, x, edge_feature, edge_index, weight=None, istrain=False):
        assert x.ndim == 2
        assert edge_feature.ndim == 2
        x_i, x_j = self.index_get(x, edge_index)
        xx, gcn_edge_feature, prob = self.edgeatten(x_i, edge_feature, x_j, weight, istrain=istrain)
        xx = self.index_aggr(xx, edge_index, dim_size = x.shape[0])
        xx = self.prop(torch.cat([x,xx],dim=1))
        return xx, gcn_edge_feature
 
class MultiHeadedEdgeAttention(torch.nn.Module):
    def __init__(self, num_heads: int, dim_node: int, dim_edge: int, dim_atten: int, use_bn=False,
                 attention = 'fat', use_edge:bool = True, **kwargs):
        super().__init__()
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.name = 'MultiHeadedEdgeAttention'
        self.dim_node=dim_node
        self.dim_edge=dim_edge
        self.d_n = d_n = dim_node // num_heads
        self.d_e = d_e = dim_edge // num_heads
        self.d_o = d_o = dim_atten // num_heads
        self.num_heads = num_heads
        self.use_edge = use_edge
        self.nn_edge = build_mlp([dim_node*2+dim_edge,(dim_node+dim_edge),dim_edge],
                           do_bn= use_bn, on_last=False)
        self.mask_obj = 0.5
        
        DROP_OUT_ATTEN = None
        if 'DROP_OUT_ATTEN' in kwargs:
            DROP_OUT_ATTEN = kwargs['DROP_OUT_ATTEN']
        
        self.attention = attention
        assert self.attention in ['fat']
        
        if self.attention == 'fat':
            if use_edge:
                self.nn = MLP([d_n+d_e, d_n+d_e, d_o],do_bn=use_bn,drop_out = DROP_OUT_ATTEN)
            else:
                self.nn = MLP([d_n, d_n*2, d_o],do_bn=use_bn,drop_out = DROP_OUT_ATTEN)
                 
            self.proj_edge  = build_mlp([dim_edge,dim_edge])
            self.proj_query = build_mlp([dim_node,dim_node])
            self.proj_value = build_mlp([dim_node,dim_atten])
        elif self.attention == 'distance':
            self.proj_value = build_mlp([dim_node,dim_atten])

        
    def forward(self, query, edge, value, weight=None, istrain=False):
        batch_dim = query.size(0)
        
        edge_feature = torch.cat([query, edge, value],dim=1)
        
        edge_feature = self.nn_edge( edge_feature )

        if self.attention == 'fat':
            value = self.proj_value(value)
            query = self.proj_query(query).view(batch_dim, self.d_n, self.num_heads)
            edge = self.proj_edge(edge).view(batch_dim, self.d_e, self.num_heads)
            if self.use_edge:
                prob = self.nn(torch.cat([query,edge],dim=1)) 
            else:
                prob = self.nn(query) 
            prob = prob.softmax(1)
            x = torch.einsum('bm,bm->bm', prob.reshape_as(value), value)
        
        elif self.attention == 'distance':
            raise NotImplementedError()
        
        else:
            raise NotImplementedError('')
        
        return x, edge_feature, prob
    
    
class MMG(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, dim_atten, num_heads=1, aggr= 'max', 
                 use_bn=False,flow='target_to_source', attention = 'fat', 
                 hidden_size=512, depth=1, use_edge:bool=True, **kwargs,
                 ):
        
        super().__init__()

        self.num_heads = num_heads
        self.depth = depth # 这是 L=2 (GNN单元的深度)

        self.self_attn = nn.ModuleList(
            MultiHeadAttention(d_model=dim_node, d_k=dim_node // num_heads, d_v=dim_node // num_heads, h=num_heads) for i in range(depth))
        
        self.gcn_3ds = torch.nn.ModuleList()
        
        for _ in range(self.depth):
            
            self.gcn_3ds.append(GraphEdgeAttenNetwork(
                                num_heads,
                                dim_node,
                                dim_edge,
                                dim_atten,
                                aggr,
                                use_bn=use_bn,
                                flow=flow,
                                attention=attention,
                                use_edge=use_edge, 
                                **kwargs))
           
        self.self_attn_fc = nn.Sequential( 
            nn.Linear(4, 32),   
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, num_heads)
        )
        
        # --- [R-GNN + GRU 修改 START] ---
        
        # 1. 定义 T=5 的循环步数
        self.recurrent_steps = 5 
        
        # 2. 添加 GRUCell
        #    它的作用是融合 h_t (旧状态) 和 m_t (新消息)
        self.gru_cell = nn.GRUCell(dim_node, dim_node)
        
        self.gru_norm = nn.LayerNorm(dim_node)
        # --- [R-GNN + GRU 修改 END] ---
        
        self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
    
    
    def _run_gnn_l_loop(self, h_t, e_true, edge_index, batch_ids, obj_center, istrain, obj_mask, obj_distance_weight, attention_matrix_way):
        """
        辅助函数：运行内部的 L=2 GNN循环。
        这部分代码就是你原来 for 循环的主体。
        它计算 GNN 的 "消息" (message) m_t。
        """
        l_loop_input_obj = h_t                # L-loop 的节点输入是 T-loop 的当前状态 h_t
        l_loop_input_edge = e_true            # L-loop 的边输入始终是 "真实" 的边特征
        
        l_loop_obj_out = l_loop_input_obj     # 初始化
        l_loop_edge_out = l_loop_input_edge   # 初始化

        for i in range(self.depth):

            # a. Self-Attention
            attn_input = l_loop_input_obj.unsqueeze(0)
            l_loop_obj_out = self.self_attn[i](attn_input, attn_input, attn_input, 
                                               attention_weights=obj_distance_weight, way=attention_matrix_way, 
                                               attention_mask=obj_mask, use_knn=False)
            l_loop_obj_out = l_loop_obj_out.squeeze(0)

            # b. GCN 消息传递
            l_loop_obj_out, l_loop_edge_out = self.gcn_3ds[i](l_loop_obj_out,    # l_loop 的节点输入
                                                               l_loop_input_edge, # l_loop 的边输入
                                                               edge_index, istrain=istrain)
            
            # c. 激活
            if i < (self.depth-1) or self.depth==1:
                l_loop_obj_out = F.relu(l_loop_obj_out)
                l_loop_obj_out = self.drop_out(l_loop_obj_out)

                l_loop_edge_out = F.relu(l_loop_edge_out)
                l_loop_edge_out = self.drop_out(l_loop_edge_out)
            
            # d. 链式更新 L-loop 的输入
            l_loop_input_obj = l_loop_obj_out  # h_t_(i+1)
            l_loop_input_edge = l_loop_edge_out # e_t_(i+1)
            
        # L-loop 结束，返回最终的节点和边特征
        return l_loop_obj_out, l_loop_edge_out


    def forward(self, obj_feature_3d, edge_feature_3d, edge_index, batch_ids, global_anchor_indices, obj_center=None, istrain=False):
        
        """
        MMG 的前向传播 (已集成 GRU 和全局锚点索引)
        """
        
        true_anchor_features = obj_feature_3d[global_anchor_indices].clone()
        # 3. 存储 "真实" 的边特征 (不变)
        static_edge_feat = edge_feature_3d
        
        # 4. 初始化 T 循环的递归隐藏状态 h_0 (不变)
        h_t = obj_feature_3d 
        
        # --- [R-GNN + GRU 修改 END] ---


        N_K = obj_feature_3d.shape[0]

        # --- (计算静态的注意力权重 - 未修改) ---
        if obj_center is not None:
            batch_size = batch_ids.max().item() + 1
            N_K = obj_feature_3d.shape[0] 
            obj_mask = torch.zeros(1, 1, N_K, N_K).cuda()
            obj_distance_weight = torch.zeros(1, self.num_heads, N_K, N_K).cuda()
            count = 0

            for i in range(batch_size):
                idx_i = torch.where(batch_ids == i)[0]
                obj_mask[:, :, count:count + len(idx_i), count:count + len(idx_i)] = 1
                center_A = obj_center[None, idx_i, :].clone().detach().repeat(len(idx_i), 1, 1)
                center_B = obj_center[idx_i, None, :].clone().detach().repeat(1, len(idx_i), 1)
                center_dist = (center_A - center_B)
                dist = center_dist.pow(2)
                dist = torch.sqrt(torch.sum(dist, dim=-1))[:, :, None]
                weights = torch.cat([center_dist, dist], dim=-1).unsqueeze(0)   
                dist_weights = self.self_attn_fc(weights).permute(0,3,1,2) 
                attention_matrix_way = 'add'
                obj_distance_weight[:, :, count:count + len(idx_i), count:count + len(idx_i)] = dist_weights
                count += len(idx_i)
        else:
            # View 3: 没有 obj_center，我们要“无痕”操作
            obj_mask = None
            
            # 1. 创建全 0 张量
            # 建议用 .to(obj_feature_3d.device) 替代 .cuda()，兼容性更好
            obj_distance_weight = torch.zeros(1, self.num_heads, N_K, N_K).to(obj_feature_3d.device)
            
            # 2. 关键！必须改成 'add'
            # 因为 Score + 0 = Score，这样才不会破坏原本的注意力
            attention_matrix_way = 'add'
        # --- (静态权重计算结束) ---


        # --- [R-GNN + GRU 循环] ---
        
        gcn_edge_feature_output = static_edge_feat.clone() 
        
        for t in range(self.recurrent_steps):
            
            # 1. 运行 L=2 GNN 循环 (不变)
            m_t, gcn_edge_feature_output = self._run_gnn_l_loop(
                h_t, static_edge_feat, edge_index, batch_ids, 
                obj_center, istrain, obj_mask, obj_distance_weight, 
                attention_matrix_way
            )
            
            # 2. GRU 门控更新 (不变)
            h_t_plus_1 = self.gru_cell(m_t, h_t)
            
            # h_t_plus_1 = h_t_plus_1 + h_prev
            
            # h_t_plus_1 = self.gru_norm(h_t_plus_1)
            
            # 3. 锚点硬重置 (Hard Reset)
            #    (使用传入的 global_anchor_indices)
            h_t_plus_1[global_anchor_indices] = true_anchor_features
            
            # 4. 更新隐藏状态 (不变)
            h_t = h_t_plus_1

        # --- (T=5 循环结束) ---

        return h_t, gcn_edge_feature_output

    def forward_no_anchor(self, obj_feature_3d, edge_feature_3d, edge_index, batch_ids, obj_center=None, istrain=False, GRU=False):
        
        """
        MMG 的前向传播 (已集成 GRU 和全局锚点索引)
        """
        N_K = obj_feature_3d.shape[0]

        # --- (计算静态的注意力权重 - 未修改) ---
        if obj_center is not None:
            batch_size = batch_ids.max().item() + 1
            N_K = obj_feature_3d.shape[0] 
            obj_mask = torch.zeros(1, 1, N_K, N_K).cuda()
            obj_distance_weight = torch.zeros(1, self.num_heads, N_K, N_K).cuda()
            count = 0

            for i in range(batch_size):
                idx_i = torch.where(batch_ids == i)[0]
                obj_mask[:, :, count:count + len(idx_i), count:count + len(idx_i)] = 1
                center_A = obj_center[None, idx_i, :].clone().detach().repeat(len(idx_i), 1, 1)
                center_B = obj_center[idx_i, None, :].clone().detach().repeat(1, len(idx_i), 1)
                center_dist = (center_A - center_B)
                dist = center_dist.pow(2)
                dist = torch.sqrt(torch.sum(dist, dim=-1))[:, :, None]
                weights = torch.cat([center_dist, dist], dim=-1).unsqueeze(0)   
                dist_weights = self.self_attn_fc(weights).permute(0,3,1,2) 
                attention_matrix_way = 'add'
                obj_distance_weight[:, :, count:count + len(idx_i), count:count + len(idx_i)] = dist_weights
                count += len(idx_i)
        else:
            # View 3: 没有 obj_center，我们要“无痕”操作
            obj_mask = None
            
            # 1. 创建全 0 张量
            # 建议用 .to(obj_feature_3d.device) 替代 .cuda()，兼容性更好
            obj_distance_weight = torch.zeros(1, self.num_heads, N_K, N_K).to(obj_feature_3d.device)
            
            # 2. 关键！必须改成 'add'
            # 因为 Score + 0 = Score，这样才不会破坏原本的注意力
            attention_matrix_way = 'add'
        # --- (静态权重计算结束) ---


        if not GRU:
            # 1. 运行 L=2 GNN 循环 (不变)
            m_t, gcn_edge_feature_output = self._run_gnn_l_loop(
                obj_feature_3d, edge_feature_3d, edge_index, batch_ids, 
                obj_center, istrain, obj_mask, obj_distance_weight, 
                attention_matrix_way
            )
            return m_t, gcn_edge_feature_output
        else:
            # 1. 初始化
            static_edge_feat = edge_feature_3d
            h_t = obj_feature_3d # 初始状态
            
            # 2. 开启循环 (和主 forward 保持一致的步数)
            # 这样 Student 也有 T=5 的思考时间，足够通过多跳邻居恢复信息
            for t in range(self.recurrent_steps):
                                
                # A. 运行 GNN 提取消息
                m_t, gcn_edge_feature_output = self._run_gnn_l_loop(
                    h_t, static_edge_feat, edge_index, batch_ids, 
                    obj_center, istrain, obj_mask, obj_distance_weight, 
                    attention_matrix_way
                )
                
                # B. GRU 门控更新
                h_t = self.gru_cell(m_t, h_t)
             
            return h_t, gcn_edge_feature_output

    