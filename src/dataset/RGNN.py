import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import GINEConv

# 1. 定义一个简单的 MLP，GINEConv 需要它
#    它将作为 GINEConv 中的 f_theta (见公式)
def create_mlp(in_feats, out_feats):
    return nn.Sequential(
        nn.Linear(in_feats, out_feats),
        nn.ReLU(),
        nn.Linear(out_feats, out_feats)
    )

# 2. 这就是你的自定义 Recurrent GNN 模型
class RecurrentGNN_with_EdgeFeatures(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, hidden_feats, num_steps):
        super().__init__()
        
        self.num_steps = num_steps # 循环的步数 T
        self.hidden_feats = hidden_feats
        
        # 初始的节点特征投影层
        self.node_embed = nn.Linear(node_in_feats, hidden_feats)
        # 初始的边特征投影层
        self.edge_embed = nn.Linear(edge_in_feats, hidden_feats)

        # -----------------------------------------------------
        # 关键：我们只创建 *一个* GINEConv 层实例。
        # 它的权重将在所有 T 步中共享。
        # -----------------------------------------------------
        mlp = create_mlp(hidden_feats, hidden_feats)
        self.conv_layer = GINEConv(apply_func=mlp, aggregator_type='sum')
        
        # 你也可以在这里加入一个 GRU 单元来做更平滑的状态更新
        # self.gru = nn.GRUCell(hidden_feats, hidden_feats)

    def forward(self, g, node_features, edge_features):
        # g: 你的 DGL 图
        # node_features: 你的原始节点特征 (例如 [N, node_in_feats])
        # edge_features: 你的原始边特征 (例如 [E, edge_in_feats])
        
        # 1. 初始投影
        h = self.node_embed(node_features)
        e = self.edge_embed(edge_features)

        # -----------------------------------------------------
        # 关键：手动循环，解决 K-hop 问题
        # -----------------------------------------------------
        for _ in range(self.num_steps):
            # 调用同一个 GINEConv 层实例
            # 它在内部会执行 h_i' = MLP(h_i + sum(ReLU(h_j + e_ij)))
            h_new = self.conv_layer(g, h, e)
            
            # --- 可选项：使用 GRU 更新 ---
            # h = self.gru(h_new, h) 
            # --------------------------
            
            # --- 简单选项：直接替换 ---
            h = h_new
            # --------------------------
            
        return h

# --- 如何使用它 ---
if __name__ == '__main__':
    N = 10  # 10 个节点
    E = 30  # 30 条边
    
    # 随机创建一个图
    u, v = torch.randint(0, N, (2, E))
    g = dgl.graph((u, v))
    
    # 锚点信息（你的空间位置）可以放在这里
    node_feat = torch.randn(N, 16) # 16维的节点原始特征
    
    # 关系信息 ("on top of" 等) 的嵌入向量可以放在这里
    edge_feat = torch.randn(E, 8)  # 8维的边（关系）特征

    # 假设隐藏层64维，我们循环 5 步
    model = RecurrentGNN_with_EdgeFeatures(
        node_in_feats=16, 
        edge_in_feats=8, 
        hidden_feats=64,
        num_steps=5
    )
    
    # 运行模型
    final_node_embeddings = model(g, node_feat, edge_feat)
    
    print(f"输入节点特征维度: {node_feat.shape}")
    print(f"输入边特征维度:   {edge_feat.shape}")
    print(f"输出节点特征维度: {final_node_embeddings.shape}")