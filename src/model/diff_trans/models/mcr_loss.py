import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

try:
    from torch.distributed.nn.functional import all_reduce as _dist_all_reduce
except Exception:
    _dist_all_reduce = None


def _half_logdet(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.cholesky_ex(x)[0].diagonal().log().sum()


class MCRLoss(nn.Module):
    def __init__(self, out_dim: int, expa_type: int = 0, reduce_cov: int = 0, eps: float = 0.05, coeff: float = 1.0):
        super().__init__()
        self.out_dim = out_dim
        self.eps = eps
        self.coeff = coeff
        self.expa_type = expa_type
        self.reduce_cov = reduce_cov

    def forward(self, student_feat_list, teacher_feat_list, no_diag: bool = False, normalized: bool = False):
        student_feat = torch.stack(student_feat_list)
        teacher_feat = torch.stack(teacher_feat_list)

        if not normalized:
            student_feat = F.normalize(student_feat, p=2, dim=-1)
            teacher_feat = F.normalize(teacher_feat, p=2, dim=-1)

        comp_loss, global_comp_loss = self.calc_compression(student_feat, teacher_feat, no_diag=no_diag)

        if self.expa_type == 0:
            expa_feat = student_feat[: teacher_feat.shape[0]]
        elif self.expa_type == 1:
            expa_feat = (student_feat[: teacher_feat.shape[0]] + teacher_feat) / 2
        else:
            raise ValueError(f"Unsupported expa_type: {self.expa_type}")

        expa_loss = self.calc_expansion(expa_feat)
        loss = -self.coeff * comp_loss - expa_loss
        return loss, {
            'loss': loss.detach(),
            'comp_loss': comp_loss.detach(),
            'global_comp_loss': global_comp_loss.detach(),
            'expa_loss': expa_loss.detach(),
        }

    def calc_compression(self, student_feat, teacher_feat, no_diag: bool = False):
        sim = (teacher_feat.unsqueeze(1) * student_feat.unsqueeze(0)).sum(-1).mean(-1)
        if no_diag:
            sim.view(-1)[:: (student_feat.shape[0] + 1)].fill_(0)
        if no_diag:
            n_loss_terms = teacher_feat.shape[0] * student_feat.shape[0] - min(teacher_feat.shape[0], student_feat.shape[0])
        else:
            n_loss_terms = teacher_feat.shape[0] * student_feat.shape[0]
        comp_loss = sim.sum() / n_loss_terms
        global_comp_loss = sim[:, : teacher_feat.shape[0]].detach().sum().div_(teacher_feat.shape[0])
        return comp_loss, global_comp_loss

    def calc_expansion(self, feat_list) -> torch.Tensor:
        num_views = feat_list.shape[0]
        m, p = feat_list[0].shape
        cov = torch.einsum('nbc,nbd->ncd', feat_list, feat_list)

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if dist.is_initialized() and self.reduce_cov == 1:
            if _dist_all_reduce is not None:
                cov = _dist_all_reduce(cov, op=dist.ReduceOp.SUM)
            else:
                dist.all_reduce(cov)

        scalar = p / (m * world_size * self.eps)
        I = torch.eye(p, device=cov.device, dtype=cov.dtype)
        loss = sum([_half_logdet(I + scalar * cov[i]) for i in range(num_views)])
        loss = loss / num_views
        loss = loss * (p + world_size * m) / (p * world_size * m)
        return loss
