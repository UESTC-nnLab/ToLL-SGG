import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class SwAVLoss(nn.Module):
    def __init__(self,
                 stu_learnable_proto,
                 teach_learnable_proto, 
                 temperature=0.1, 
                 sinkhorn_iterations=10, # [建议] 提高迭代次数以保证分配质量
                 epsilon=0.05):
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
        self.stu_prototypes = stu_learnable_proto
        self.teach_prototypes = teach_learnable_proto
        
        # === 2. 初始化权重 ===
        self._init_weights(self.stu_prototypes)
        
        # 教师原型不需要梯度
        for p in self.teach_prototypes.parameters():
            p.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is None:
                module.weight.data = F.normalize(module.weight.data, dim=1, p=2)

    @torch.no_grad()
    def normalize_prototypes(self):
        """
        每次迭代前调用。归一化原型向量到单位球面上。
        """
        self.stu_prototypes.weight.data = F.normalize(self.stu_prototypes.weight.data, dim=1, p=2)
        self.teach_prototypes.weight.data = F.normalize(self.teach_prototypes.weight.data, dim=1, p=2)

    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        """
        Sinkhorn-Knopp 算法：生成软聚类分配 Q
        """
        Q = torch.exp(out / self.epsilon).t() # (K, B)
        B = Q.shape[1]
        K = Q.shape[0]
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        B_total = B * world_size

        # 1. 归一化整个矩阵
        sum_Q = torch.sum(Q)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q 

        for _ in range(self.sinkhorn_iterations):
            # 2. 行归一化 (约束每个原型分配到的样本量均匀 1/K)
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # 3. 列归一化 (约束每个样本分配出去的概率和为 1/B)
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            Q /= sum_of_cols
            Q /= B_total

        Q *= B_total # 恢复量级
        return Q.t() # (B, K)
    
    @torch.no_grad()
    def forward_test(self, z1):
        self.normalize_prototypes()
        z1 = F.normalize(z1, dim=1, p=2)
        scores1 = self.teach_prototypes(z1)
        with torch.no_grad():
            q1 = self.distributed_sinkhorn(scores1)
        return z1, q1
    
    def forward_asymmetric(self, z_teacher, z_student, queue=None):
        """
        [核心方法] 非对称损失计算
        Args:
            z_teacher: (B, D) 来自 Teacher 网络的特征 (Target)
            z_student: (B, D) 来自 Student 网络的特征 (Prediction)
            queue: (Q, D) 队列中的历史特征 (Optional) [新增参数]
        """
        # 0. 确保所有原型都在单位球面上
        self.normalize_prototypes()

        # 1. 特征归一化
        z_t = F.normalize(z_teacher.detach(), dim=1, p=2) # 教师特征切断梯度
        z_s = F.normalize(z_student, dim=1, p=2)
        
        # =========================================================
        # [关键修改] 使用 Queue 增强 Teacher 分配
        # =========================================================
        bs = z_t.size(0)
        
        if queue is not None:
            # 再次归一化 Queue 以防万一
            queue = F.normalize(queue.detach(), dim=1, p=2)
            # 拼接: [Batch_Teacher; Queue] -> (B + Q, D)
            z_t_combined = torch.cat([z_t, queue], dim=0)
        else:
            z_t_combined = z_t

        # ================= Teacher 分支 (生成 Target) =================
        with torch.no_grad():
            # 使用 "教师原型" 计算 logits
            # scores_t shape: (B + Q, K)
            scores_t = self.teach_prototypes(z_t_combined)
            
            # 使用 Sinkhorn 计算目标分布 Q (伪标签)
            q_t = self.distributed_sinkhorn(scores_t)
            
            # [关键] 我们只需要当前 Batch 的 assignment 来计算 Loss
            # 队列里的样本只是为了帮当前 Batch 抢位置，不参与 Loss 计算
            q_t = q_t[:bs]

        # ================= Student 分支 (生成 Predict) =================
        # [关键] 使用 "学生原型" 计算 logits
        # Student 不需要看 Queue，只预测当前 Batch
        scores_s = self.stu_prototypes(z_s)

        # ================= 计算 Loss =================
        # Cross Entropy: H(Target, Predict)
        loss = -torch.mean(
            torch.sum(q_t * F.log_softmax(scores_s / self.temperature, dim=1), dim=1)
        )

        return loss
