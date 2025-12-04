import torch
import torch.nn as nn
import torch.nn.functional as F

class SwAVLoss(nn.Module):
    def __init__(self, 
                 feature_dim=512, 
                 num_prototypes=50, 
                 temperature=0.1, 
                 sinkhorn_iterations=3, 
                 epsilon=0.05):
        """
        Args:
            feature_dim (int): 输入 embedding 的维度，你提到是 512。
            num_prototypes (int): 原型（聚类中心）的数量。对于 3DSSG，建议设置比实际关系类别数大一些，例如 50 或 100。
            temperature (float): 计算相似度时的温度系数。
            sinkhorn_iterations (int): Sinkhorn 算法迭代次数，通常 3 次足够。
            epsilon (float): Sinkhorn 算法中的正则化参数。
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon

        # === 1. 原型初始化 ===
        # 原型就是一个 Linear 层，不带 bias
        self.prototypes = nn.Linear(feature_dim, num_prototypes, bias=False)
        
        # 随机均匀初始化原型权重
        # 实际上 PyTorch 的 Linear 默认就是 Uniform 初始化，这里显式写出以便理解
        nn.init.xavier_uniform_(self.prototypes.weight)
        
        # 关键：初始化后立即进行归一化，保证原型在单位球面上
        self.prototypes.weight.data = F.normalize(self.prototypes.weight.data, dim=1, p=2)

    @torch.no_grad()
    def normalize_prototypes(self):
        """
        在每次迭代 forward 之前或之后调用，强制原型保持归一化。
        """
        self.prototypes.weight.data = F.normalize(self.prototypes.weight.data, dim=1, p=2)

    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        """
        Sinkhorn-Knopp 算法：将相似度分数转换为均匀分布的“软聚类分配” Q。
        输入 out: (B, K) - 样本与原型的相似度 logits
        输出 q:   (B, K) - 软分配概率
        """
        Q = torch.exp(out / self.epsilon).t() # (K, B)
        B = Q.shape[1]
        K = Q.shape[0]

        # 初始化向量
        sum_Q = torch.sum(Q)
        Q /= sum_Q # 归一化整个矩阵

        for _ in range(self.sinkhorn_iterations):
            # 行归一化 (让每个原型被分配的总量趋向于 1/K)
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # 列归一化 (让每个样本分配出去的总概率为 1/B)
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            Q /= sum_of_cols
            Q /= B

        Q *= B # 恢复量级以便作为概率使用
        return Q.t() # (B, K)

    def forward1(self, z1):
      
        # 0. 确保原型是归一化的
        self.normalize_prototypes()

        # 1. 归一化输入特征
        z1 = F.normalize(z1, dim=1, p=2)

        # 2. 计算与原型的相似度 (B, K)
        # scores = z @ prototypes.T
        scores1 = self.prototypes(z1)

        # 3. 使用 Sinkhorn 算法计算“伪标签” (Target Codes)
        # 注意：伪标签的计算不需要梯度 (detach)
        with torch.no_grad():
            q1 = self.distributed_sinkhorn(scores1)
            
        return z1, q1
    
    def forward(self, z1, z2):
        """
        Args:
            z1: (B, 512) 视图1 的 edge embeddings
            z2: (B, 512) 视图2 的 edge embeddings (来自同一批 edge，但经过了不同的增强/Dropout)
        Returns:
            loss: 标量 SwAV Loss
        """
        # 0. 确保原型是归一化的
        self.normalize_prototypes()

        # 1. 归一化输入特征
        z1 = F.normalize(z1, dim=1, p=2)
        z2 = F.normalize(z2, dim=1, p=2)

        # 2. 计算与原型的相似度 (B, K)
        # scores = z @ prototypes.T
        scores1 = self.prototypes(z1)
        scores2 = self.prototypes(z2)

        # 3. 使用 Sinkhorn 算法计算“伪标签” (Target Codes)
        # 注意：伪标签的计算不需要梯度 (detach)
        with torch.no_grad():
            q1 = self.distributed_sinkhorn(scores1)
            q2 = self.distributed_sinkhorn(scores2)

        # 4. 计算 Swapped Prediction Loss
        # 用 z1 预测 q2，用 z2 预测 q1
        loss1 = -torch.mean(torch.sum(q2 * F.log_softmax(scores1 / self.temperature, dim=1), dim=1))
        loss2 = -torch.mean(torch.sum(q1 * F.log_softmax(scores2 / self.temperature, dim=1), dim=1))

        return loss1 + loss2

# ================= 使用示例 =================

# 假设 batch_size = 32, dim = 512
B, D = 32, 512
swav = SwAVLoss(feature_dim=D, num_prototypes=50).cuda()

# 模拟输入：
# 假设你在预训练阶段，对同一个 Batch 的边，通过两次 Forward (开启 Dropout) 
# 或者对点云特征加噪，得到了两个稍微不同的 embedding
edge_emb_view1 = torch.randn(B, D).cuda()
edge_emb_view2 = edge_emb_view1 + 0.1 * torch.randn(B, D).cuda() # 模拟增强

# 计算 Loss
loss = swav(edge_emb_view1, edge_emb_view2)

print(f"SwAV Clustering Loss: {loss.item()}")
loss.backward()