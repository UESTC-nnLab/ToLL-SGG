import torch
import torch.nn as nn
import torch.nn.functional as F

class SwAVLoss(nn.Module):
    def __init__(self,
                 stu_learnable_proto,
                 teach_learnable_proto, 
                 temperature=0.1, 
                 sinkhorn_iterations=6, 
                 epsilon=0.08):
        """
        Args:
            stu_learnable_proto (nn.Linear): 学生网络的原型层 (必须包含参数)
            teach_learnable_proto (nn.Linear): 教师网络的原型层 (必须包含参数，且参数通过EMA更新)
            temperature (float): Softmax 温度系数
            sinkhorn_iterations (int): Sinkhorn 迭代次数
            epsilon (float): Sinkhorn 正则化参数
        """
        super().__init__()
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon

        # === 1. 分别存储学生和教师原型 ===
        # 注意：这里传入的是 nn.Linear 对象
        self.stu_prototypes = stu_learnable_proto
        self.teach_prototypes = teach_learnable_proto
        
        # === 2. 初始化权重 ===
        # 通常只初始化学生的，因为教师稍后会被学生的权重覆盖(在EMA第一步)
        # 但为了安全起见，这里对学生原型做标准的 Xavier 初始化
        self._init_weights(self.stu_prototypes)
        
        # 教师原型不需要梯度（它的更新在外部通过 EMA 进行）
        for p in self.teach_prototypes.parameters():
            p.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            # 初始化后立即归一化，确保在球面上
            if module.bias is None:
                module.weight.data = F.normalize(module.weight.data, dim=1, p=2)

    @torch.no_grad()
    def normalize_prototypes(self):
        """
        每次迭代前调用。
        必须同时归一化 学生 和 教师 的原型向量，
        因为 SwAV/DINO 的度量基于余弦相似度。
        """
        # 归一化学生原型
        self.stu_prototypes.weight.data = F.normalize(self.stu_prototypes.weight.data, dim=1, p=2)
        # 归一化教师原型
        self.teach_prototypes.weight.data = F.normalize(self.teach_prototypes.weight.data, dim=1, p=2)

    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        """
        Sinkhorn-Knopp 算法：生成软聚类分配 Q
        """
        Q = torch.exp(out / self.epsilon).t() # (K, B)
        B = Q.shape[1]
        K = Q.shape[0]

        # 1. 归一化整个矩阵
        sum_Q = torch.sum(Q)
        Q /= sum_Q 

        for _ in range(self.sinkhorn_iterations):
            # 2. 行归一化 (约束每个原型分配到的样本量均匀 1/K)
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # 3. 列归一化 (约束每个样本分配出去的概率和为 1/B)
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            Q /= sum_of_cols
            Q /= B

        Q *= B # 恢复量级
        return Q.t() # (B, K)
    
    @torch.no_grad()
    def forward_test(self, z1):
      
        # 0. 确保原型是归一化的
        self.normalize_prototypes()

        # 1. 归一化输入特征
        z1 = F.normalize(z1, dim=1, p=2)

        # 2. 计算与原型的相似度 (B, K)
        # scores = z @ prototypes.T
        scores1 = self.teach_prototypes(z1)

        # 3. 使用 Sinkhorn 算法计算“伪标签” (Target Codes)
        # 注意：伪标签的计算不需要梯度 (detach)
        with torch.no_grad():
            q1 = self.distributed_sinkhorn(scores1)
            
        return z1, q1
    
    def forward_asymmetric(self, z_teacher, z_student):
        """
        [核心方法] 非对称损失计算
        逻辑：Student 预测 Teacher 的聚类分配
        
        Args:
            z_teacher: (B, D) 来自 Teacher 网络的特征 (Target)
            z_student: (B, D) 来自 Student 网络的特征 (Prediction)
        """
        # 0. 确保所有原型都在单位球面上
        self.normalize_prototypes()

        # 1. 特征归一化
        z_t = F.normalize(z_teacher.detach(), dim=1, p=2) # 教师特征切断梯度
        z_s = F.normalize(z_student, dim=1, p=2)

        # ================= Teacher 分支 (生成 Target) =================
        with torch.no_grad():
            # [关键] 使用 "教师原型" 计算 logits
            scores_t = self.teach_prototypes(z_t)
            # 使用 Sinkhorn 计算目标分布 Q (伪标签)
            q_t = self.distributed_sinkhorn(scores_t)

        # ================= Student 分支 (生成 Predict) =================
        # [关键] 使用 "学生原型" 计算 logits
        scores_s = self.stu_prototypes(z_s)

        # ================= 计算 Loss =================
        # Cross Entropy: H(Target, Predict)
        # q_t 是软标签，scores_s 是预测的 logits
        loss = -torch.mean(
            torch.sum(q_t * F.log_softmax(scores_s / self.temperature, dim=1), dim=1)
        )

        return loss

    def forward(self, z1, z2):
        """
        保留旧接口：对称损失 (仅使用学生原型)
        如果你的架构是完全对称的 (SimCLR/Original SwAV style)，没有 Teacher 网络，
        那么两个视图都应该使用 stu_prototypes。
        """
        self.normalize_prototypes()
        z1 = F.normalize(z1, dim=1, p=2)
        z2 = F.normalize(z2, dim=1, p=2)

        # 此时两个都用 stu_prototypes
        scores1 = self.stu_prototypes(z1)
        scores2 = self.stu_prototypes(z2)

        with torch.no_grad():
            q1 = self.distributed_sinkhorn(scores1)
            q2 = self.distributed_sinkhorn(scores2)

        loss1 = -torch.mean(torch.sum(q2 * F.log_softmax(scores1 / self.temperature, dim=1), dim=1))
        loss2 = -torch.mean(torch.sum(q1 * F.log_softmax(scores2 / self.temperature, dim=1), dim=1))

        return loss1 + loss2