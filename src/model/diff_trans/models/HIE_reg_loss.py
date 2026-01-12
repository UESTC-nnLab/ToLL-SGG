import torch
import torch.nn as nn
import torch.nn.functional as F

class PoincareBall:
    """
    轻量级庞加莱球流形实现，仅包含 HIE 所需的核心映射操作。
    """
    def __init__(self, c=1.0, clip_r=0.999):
        super().__init__()
        self.c = c
        self.sqrt_c = c ** 0.5
        self.clip_r = clip_r  # 数值稳定性截断半径

    def _check_point_on_manifold(self, x, atol=1e-5):
        """检查点是否在流形内 (模长 < 1/sqrt(c))"""
        norm = x.norm(dim=-1, keepdim=True)
        max_norm = 1.0 / self.sqrt_c
        return (norm < max_norm - atol).all()

    def proj(self, x):
        """
        将点投影回球内，防止数值溢出。
        通常在 expmap 后调用，确保点不会因为精度问题跑出边界。
        """
        norm = x.norm(dim=-1, keepdim=True)
        max_norm = (1.0 - 1e-5) / self.sqrt_c
        cond = norm > max_norm
        projected = x / norm * max_norm
        return torch.where(cond, projected, x)

    def expmap0(self, v):
        """
        指数映射: 切空间 (原点) -> 双曲空间 (庞加莱球)
        输入: v (欧氏空间特征 / 切向量)
        输出: x (双曲空间坐标)
        公式: y = tanh(sqrt(c)/2 * ||v||) * (v / (sqrt(c) * ||v||))
        """
        v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        scale = torch.tanh(self.sqrt_c * v_norm / 2) / (self.sqrt_c * v_norm)
        return self.proj(v * scale)

    def logmap0(self, y):
        """
        对数映射: 双曲空间 (庞加莱球) -> 切空间 (原点)
        输入: y (双曲空间坐标)
        输出: v (切向量 / 近似欧氏特征)
        公式: v = 2/sqrt(c) * atanh(sqrt(c) * ||y||) * (y / ||y||)
        """
        y_norm = y.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        # 数值稳定性保护：防止 atanh 输入 >= 1
        y_norm = torch.clamp(y_norm, max=(1.0 - 1e-6) / self.sqrt_c)
        scale = 2 * torch.atanh(self.sqrt_c * y_norm) / (self.sqrt_c * y_norm)
        return y * scale

class HIE_Edge_Regularizer(nn.Module):
    def __init__(self, c=1.0, lambda_reg=0.1, mode='hir_tangent'):
        """
        Args:
            c (float): 双曲空间的曲率，默认为 1.0
            lambda_reg (float): Loss 的权重
            mode (str): HIE 的模式，对应论文中的不同消融实验设置
        """
        super().__init__()
        self.manifold = PoincareBall(c=c)
        self.c = c
        self.lambda_reg = lambda_reg
        self.mode = mode
        # 激活函数，用于将负距离转换为 Loss (论文中使用 monotonically increasing function)
        # 这里使用 Softplus 保证平滑，或者直接用 exp
        self.activation = nn.Softplus() 

    def get_c(self):
        return self.c

    def forward(self, euclidean_edge_features):
        """
        输入: (N, 512) 的欧氏空间 Edge 特征
        返回: HIE 正则化 Loss
        """
        # -----------------------------------------------------------
        # Step 1: 投影到双曲空间 (Scenario B 关键步骤)
        # -----------------------------------------------------------
        # 将 GNN 输出的欧氏特征视为原点处的切向量，通过指数映射投影到庞加莱球
        # 这步操作赋予了特征“双曲几何约束”，使得模长受限 (< 1/sqrt(c))
        embeddings_hyp = self.manifold.expmap0(euclidean_edge_features)

        # -----------------------------------------------------------
        # Step 2: HIE 核心逻辑 (基于你提供的代码片段)
        # -----------------------------------------------------------
        loss = self._hir_loss_logic(embeddings_hyp)
        
        return self.lambda_reg * loss

    def _hir_loss_logic(self, embeddings):
        """
        完全复用你提供的 HIE 代码逻辑，基于论文 Tangent HIE 实现
        """
        c = self.get_c()
        
        # 1. 完整的 HIE (Tangent Version): 根对齐 + 拉伸
        if self.mode == 'hir_tangent':
            # 将双曲特征映射回切空间
            embeddings_tan = self.manifold.logmap0(embeddings)
            
            # [Root Alignment] 去中心化: 找到 batch 的重心并将其对齐到原点
            # 对应论文 Eq (7)
            embeddings_tan = embeddings_tan - embeddings_tan.mean(dim=0)
            
            # [Hierarchical Stretching] 层级拉伸: 鼓励模长变大
            # 对应论文 Eq (8) & (9)
            # 计算切空间模长的平方和的均值
            tangent_mean_norm = (1e-6 + embeddings_tan.pow(2).sum(dim=1).mean())
            
            # 最小化 Loss = 最小化 activation(-norm) = 最大化 norm
            # 这会将非核心节点推向双曲空间的边界（高层级/具体语义）
            loss = self.activation(-tangent_mean_norm)
            return loss

        # 2. 仅拉伸 (Without Alignment) - 对应论文消融实验
        elif self.mode == 'hir_tangent_stretching_only':
            embeddings_tan = self.manifold.logmap0(embeddings)
            # 不做减均值操作
            tangent_mean_norm = (1e-6 + embeddings_tan.pow(2).sum(dim=1).mean())
            loss = self.activation(-tangent_mean_norm)
            return loss
            
        else:
            return torch.tensor(0.0, device=embeddings.device)

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 假设：
    # Batch size = 64
    # Feature dim = 512
    edge_features_from_gnn = torch.randn(64, 512)  # 模拟 GNN 输出的欧氏特征

    # 初始化 HIE 正则化器
    # 建议 c=1.0 起步，lambda_reg 可以设小一点 (如 0.1) 作为辅助 Loss
    hie_loss_fn = HIE_Edge_Regularizer(c=1.0, lambda_reg=0.1, mode='hir_tangent')

    # 计算 Loss
    loss = hie_loss_fn(edge_features_from_gnn)

    print(f"Input Shape: {edge_features_from_gnn.shape}")
    print(f"HIE Regularization Loss: {loss.item()}")

    # 反向传播示例
    loss.backward()
    print("Backward pass successful.")

class TotalCodingRateRegularizer(nn.Module):
    def __init__(self, eps=0.01):
        """
        Total Coding Rate (TCR) Regularizer (Expansion term).
        
        Args:
            eps (float): 失真率 (Distortion rate)，通常设为 0.01 或 0.5 / sqrt(dim).
                         较小的值会施加更强的膨胀约束。
        """
        super(TotalCodingRateRegularizer, self).__init__()
        self.eps = eps

    def forward(self, z):
        """
        Args:
            z (torch.Tensor): 输入张量，形状 (N, D)，例如 (Batch_Size, 512)
        
        Returns:
            loss (torch.Scalar): 需要最小化的 Loss 值 (即 -R(z))
        """
        # 1. 获取维度
        N, D = z.shape
        
        # 2. 归一化 (Normalization)
        # MCR² 理论要求特征分布在单位球面上，这一步至关重要
        z = F.normalize(z, p=2, dim=1)
        
        # 3. 计算缩放系数
        scalar = D / (N * self.eps**2)
        
        # 4. 计算 Log-Det
        # 利用恒等式 det(I + AB) = det(I + BA) 来优化计算量
        # 我们选择计算维度较小的那个矩阵的行列式
        
        if N < D:
            # 如果 Batch Size (N) 小于 特征维度 (512)
            # 计算 Gram Matrix (N, N)
            mat = torch.eye(N, device=z.device) + scalar * torch.matmul(z, z.T)
        else:
            # 如果 Batch Size 很大 (例如 4096)，特征维度较小 (512)
            # 计算 Covariance-like Matrix (D, D)
            mat = torch.eye(D, device=z.device) + scalar * torch.matmul(z.T, z)
            
        # 5. 计算 Coding Rate R(z)
        # 这里的 0.5 是公式中的系数
        coding_rate = 0.5 * torch.logdet(mat)
        
        # 6. 作为 Loss，我们需要最大化 Coding Rate，所以返回负值
        return -coding_rate