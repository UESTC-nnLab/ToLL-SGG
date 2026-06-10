import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VertexEdgeCrossAttention(nn.Module):
    """
    一个用于融合顶点和边特征的交叉注意力模块。
    顶点特征作为 Query，边特征作为 Key 和 Value。
    """
    def __init__(self, embed_dim, num_heads):
        """
        初始化交叉注意力模块。

        参数:
        - embed_dim (int): 输入特征的维度 (这里是512)。
        - num_heads (int): 注意力头的数量。embed_dim 必须能被 num_heads 整除。
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"嵌入维度 ({embed_dim}) 必须能被注意力头的数量 ({num_heads}) 整除。"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 线性投影层
        # W_q, W_k, W_v
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 最终的输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, vertex_features, edge_features):
        """
        前向传播。

        参数:
        - vertex_features (torch.Tensor): 顶点特征，形状为 (N, embed_dim)。作为 Query。
        - edge_features (torch.Tensor): 边特征，形状为 (M, embed_dim)。作为 Key 和 Value。

        返回:
        - torch.Tensor: 更新后的顶点特征，形状为 (N, embed_dim)。
        """
        N, _ = vertex_features.shape  # N = 顶点数量
        M, _ = edge_features.shape    # M = 边的数量

        # 1. 线性投影 Q, K, V
        # Q: (N, embed_dim) -> (N, embed_dim)
        # K: (M, embed_dim) -> (M, embed_dim)
        # V: (M, embed_dim) -> (M, embed_dim)
        q = self.q_proj(vertex_features)
        k = self.k_proj(edge_features)
        v = self.v_proj(edge_features)

        # 2. 准备多头注意力
        # 将 Q, K, V 的形状变为 (Batch_Size, Num_Heads, Seq_Len, Head_Dim)
        # 在我们的例子中，Batch_Size=1, Seq_Len 分别是 N 或 M
        # q: (N, embed_dim) -> (N, num_heads, head_dim) -> (num_heads, N, head_dim)
        # k: (M, embed_dim) -> (M, num_heads, head_dim) -> (num_heads, M, head_dim)
        # v: (M, embed_dim) -> (M, num_heads, head_dim) -> (num_heads, M, head_dim)
        q = q.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(M, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(M, self.num_heads, self.head_dim).transpose(0, 1)

        # 3. 计算注意力分数并应用到 Value 上
        # scaled_dot_product_attention:
        # attn_weights: (num_heads, N, M)
        # output: (num_heads, N, head_dim)
        # PyTorch 2.0+ 内置了高效的实现，我们直接调用
        output = F.scaled_dot_product_attention(q, k, v)

        # 4. 合并多头并进行最终投影
        # output: (num_heads, N, head_dim) -> (N, num_heads, head_dim) -> (N, embed_dim)
        output = output.transpose(0, 1).contiguous().view(N, self.embed_dim)
        
        # out_proj: (N, embed_dim) -> (N, embed_dim)
        updated_vertex_features = self.out_proj(output)

        return updated_vertex_features

### 如何使用

if __name__ == '__main__':
    # --- 参数定义 ---
    N = 10  # 顶点数量
    M = 20  # 边的数量
    embed_dim = 512 # 特征维度
    num_heads = 8   # 注意力头的数量

    # --- 实例化模块 ---
    cross_attention_layer = VertexEdgeCrossAttention(embed_dim=embed_dim, num_heads=num_heads)
    print("交叉注意力模块结构:")
    print(cross_attention_layer)

    # --- 创建模拟输入数据 ---
    # 顶点特征 (Query)
    vertex_feat = torch.randn(N, embed_dim)
    # 边特征 (Key, Value)
    edge_feat = torch.randn(M, embed_dim)

    print(f"\n输入顶点特征形状: {vertex_feat.shape}")
    print(f"输入边特征形状:   {edge_feat.shape}")

    # --- 前向传播，得到融合了边信息的顶点特征 ---
    updated_vertices = cross_attention_layer(vertex_feat, edge_feat)

    print(f"输出特征形状:     {updated_vertices.shape}")

    # --- 验证 ---
    # 输出的形状应该和输入顶点的形状完全一致，可以用于后续处理
    # 例如，进行残差连接
    final_vertices = vertex_feat + updated_vertices # 残差连接
    print(f"残差连接后形状:   {final_vertices.shape}")