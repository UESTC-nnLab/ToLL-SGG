import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class SwAVLoss(nn.Module):
    def __init__(
        self,
        stu_learnable_proto,
        teach_learnable_proto,
        temperature=0.1,
        sinkhorn_iterations=10,
        epsilon=0.05,
    ):
        super().__init__()
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon
        self.stu_prototypes = stu_learnable_proto
        self.teach_prototypes = teach_learnable_proto
        self._init_weights(self.stu_prototypes)

        for p in self.teach_prototypes.parameters():
            p.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is None:
                module.weight.data = F.normalize(module.weight.data, dim=1, p=2)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.stu_prototypes.weight.data = F.normalize(self.stu_prototypes.weight.data, dim=1, p=2)
        self.teach_prototypes.weight.data = F.normalize(self.teach_prototypes.weight.data, dim=1, p=2)

    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        Q = torch.exp(out / self.epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        B_total = B * world_size

        sum_Q = torch.sum(Q)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for _ in range(self.sinkhorn_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            Q /= sum_of_cols
            Q /= B_total

        Q *= B_total
        return Q.t()

    @torch.no_grad()
    def forward_test(self, z1):
        self.normalize_prototypes()
        z1 = F.normalize(z1, dim=1, p=2)
        scores1 = self.teach_prototypes(z1)
        q1 = self.distributed_sinkhorn(scores1)
        return z1, q1

    def forward_asymmetric(self, z_teacher, z_student, queue=None):
        self.normalize_prototypes()
        z_t = F.normalize(z_teacher.detach(), dim=1, p=2)
        z_s = F.normalize(z_student, dim=1, p=2)

        batch_size = z_t.size(0)
        if queue is not None:
            queue = F.normalize(queue.detach(), dim=1, p=2)
            z_t_combined = torch.cat([z_t, queue], dim=0)
        else:
            z_t_combined = z_t

        with torch.no_grad():
            scores_t = self.teach_prototypes(z_t_combined)
            q_t = self.distributed_sinkhorn(scores_t)
            q_t = q_t[:batch_size]

        scores_s = self.stu_prototypes(z_s)
        loss = -torch.mean(torch.sum(q_t * F.log_softmax(scores_s / self.temperature, dim=1), dim=1))
        return loss
