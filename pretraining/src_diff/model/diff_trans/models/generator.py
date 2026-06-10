import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from src.model.diff_trans.utils import misc
from src.model.diff_trans.utils.logger import *
from SoftPool import soft_pool2d, SoftPool2d

def get_chamfer_distance(s1, s2, chunk_size=16):
    """
    计算两个点云之间的 Chamfer Distance (Squared L2)。
    
    Args:
        s1: (B, N, 3) 预测点云 (或真实点云)
        s2: (B, M, 3) 真实点云 (或预测点云)
    Returns:
        loss: (B,) 每个样本的 CD 损失
    """
    batch_size, num_points_1, _ = s1.shape
    _, num_points_2, _ = s2.shape

    min_dist_s1_chunks = []
    for start in range(0, num_points_1, chunk_size):
        end = min(start + chunk_size, num_points_1)
        dist_chunk = torch.cdist(s1[:, start:end, :], s2).pow(2)
        min_dist_chunk = dist_chunk.min(dim=2).values
        min_dist_s1_chunks.append(min_dist_chunk)
    min_dist_s1 = torch.cat(min_dist_s1_chunks, dim=1)
    term1 = min_dist_s1.mean(dim=1)

    min_dist_s2_chunks = []
    for start in range(0, num_points_2, chunk_size):
        end = min(start + chunk_size, num_points_2)
        dist_chunk = torch.cdist(s2[:, start:end, :], s1).pow(2)
        min_dist_chunk = dist_chunk.min(dim=2).values
        min_dist_s2_chunks.append(min_dist_chunk)
    min_dist_s2 = torch.cat(min_dist_s2_chunks, dim=1)
    term2 = min_dist_s2.mean(dim=1)

    return term1 + term2

class VarianceSchedule(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.num_steps = self.config.generator_config.time_schedule.num_steps
        self.beta_start = self.config.generator_config.time_schedule.beta_start
        self.beta_end = self.config.generator_config.time_schedule.beta_end
        self.mode = self.config.generator_config.time_schedule.mode
        
        if self.mode == 'linear':
            betas = torch.linspace(self.beta_start, self.beta_end, steps=self.num_steps)
            
        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding
        alphas = 1 - betas
        
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    # original sampling strategy
    # def uniform_sampling(self, batch_size):
    #     ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
    #     return ts.tolist()

    # Recurrent Uniform Sampling Strategy
    def recurrent_uniform_sampling(self, batch_size, interval_nums):
        interval_size = self.num_steps / interval_nums
        sampled_intervals = []
        for i in range(interval_nums):
            start = int(i * interval_size) + 1
            end = int((i + 1) * interval_size)
            sampled_interval = np.random.choice(np.arange(start, end + 1), batch_size)
            sampled_intervals.append(sampled_interval)
        ts = np.vstack(sampled_intervals)
        ts = torch.tensor(ts)
        ts = torch.stack([ts[:, i][torch.randperm(interval_nums)] for i in range(batch_size)], dim=1)
        return ts


# Condition Aggregation Network
class CANet(nn.Module): 
    def __init__(self, encoder_dims, cond_dims):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.cond_dims = cond_dims

        self.mlp1 = nn.Sequential(
            nn.Conv2d(self.encoder_dims, 512, kernel_size=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=1, bias=True),
            nn.ReLU(True),
        )

        self.mlp2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, self.cond_dims, kernel_size=1, bias=True),
            nn.ReLU(True),
        )

    def forward(self, patch_fea):
        '''
            patch_feature : B G 384
            -----------------
            point_condition : B 384
        '''
        
        patch_fea = patch_fea.transpose(1, 2)     # B 384 G
        patch_fea = patch_fea.unsqueeze(-1)       # B 384 G 1
        patch_fea = self.mlp1(patch_fea)          # B 512 G 1
        # soft_pool2d
        global_fea = soft_pool2d(patch_fea, kernel_size=[patch_fea.size(2), 1])  # B 512 1 1
        global_fea = global_fea.expand(-1, -1, patch_fea.size(2), -1)            # B 512 G 1
        combined_fea = torch.cat([patch_fea, global_fea], dim=1)                 # B 1024 G 1
        combined_fea = self.mlp2(combined_fea)                                       # B F G 1
        condition_fea = soft_pool2d(combined_fea, kernel_size=[combined_fea.size(2), 1])  # B F 1 1
        condition_fea = condition_fea.squeeze(-1).squeeze(-1)                          #  B F
        return condition_fea

# Point Condition Network 
class PCNet(nn.Module):
    def __init__(self, dim_in, dim_out, dim_cond):
        super(PCNet, self).__init__()
        self.fea_layer = nn.Linear(dim_in, dim_out)
        self.cond_bias = nn.Linear(dim_cond, dim_out, bias=False)
        self.cond_gate = nn.Linear(dim_cond, dim_out)

    def forward(self, fea, cond):
        gate = torch.sigmoid(self.cond_gate(cond))
        bias = self.cond_bias(cond)
        out = self.fea_layer(fea) * gate + bias
        return out


class TokenToPointCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads=4, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(query_dim)
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.context_proj = nn.Linear(context_dim, query_dim)
        self.context_pos_proj = nn.Sequential(
            nn.Linear(3, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
        )
        self.kv_proj = nn.Linear(query_dim, query_dim * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.out_drop = nn.Dropout(proj_drop)

        self.ffn_norm = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 2),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(query_dim * 2, query_dim),
        )

    def forward(self, point_features, visible_tokens, token_positions=None):
        if visible_tokens is None or visible_tokens.numel() == 0:
            return point_features

        batch_size, num_points, query_dim = point_features.shape
        _, num_tokens, _ = visible_tokens.shape

        query = self.query_proj(self.query_norm(point_features))
        query = query.reshape(batch_size, num_points, self.num_heads, self.head_dim).transpose(1, 2)

        context = self.context_proj(visible_tokens)
        if token_positions is not None:
            context = context + self.context_pos_proj(token_positions)
        context = self.context_norm(context)

        kv = self.kv_proj(context)
        kv = kv.reshape(batch_size, num_tokens, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        key, value = kv[0], kv[1]

        attn = (query @ key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attended = (attn @ value).transpose(1, 2).reshape(batch_size, num_points, query_dim)
        point_features = point_features + self.out_drop(self.out_proj(attended))
        point_features = point_features + self.ffn(self.ffn_norm(point_features))
        return point_features


# Point Denoising Network
class DenoisingNet(nn.Module):

    def __init__(self, point_dim, cond_dims, residual, token_cond_dim=None, cross_attn_heads=4):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = nn.ModuleList([
            PCNet(3, 128, cond_dims+3),
            PCNet(128, 256, cond_dims+3),
            PCNet(256, 512, cond_dims+3),
            PCNet(512, 256, cond_dims+3),
            PCNet(256, 128, cond_dims+3),
            PCNet(128, 3, cond_dims+3)
        ])
        self.cross_attn_blocks = nn.ModuleDict()
        if token_cond_dim is not None and token_cond_dim > 0:
            self.cross_attn_blocks["1"] = TokenToPointCrossAttention(
                query_dim=256,
                context_dim=token_cond_dim,
                num_heads=cross_attn_heads,
            )
            self.cross_attn_blocks["2"] = TokenToPointCrossAttention(
                query_dim=512,
                context_dim=token_cond_dim,
                num_heads=cross_attn_heads,
            )

    def forward(self, coords, beta, cond, visible_tokens=None, token_positions=None):
        """
        Args:
            coords:   Noise point clouds at timestep t, (B, N, 3).
            beta:     Time. (B, ).
            cond:     Condition. (B, F).
        """

        batch_size = coords.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        cond = cond.view(batch_size, 1, -1)         # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        cond_emb = torch.cat([time_emb, cond], dim=-1)    # (B, 1, F+3)

        out = coords
        for i, layer in enumerate(self.layers):
            out = layer(fea=out, cond=cond_emb)
            if i < len(self.layers) - 1:
                out = self.act(out)
                if str(i) in self.cross_attn_blocks:
                    out = self.cross_attn_blocks[str(i)](
                        out,
                        visible_tokens,
                        token_positions=token_positions,
                    )

        if self.residual:
            return coords + out
        else:
            return out


# Conditional Point Diffusion Model
class CPDM(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.cond_dims = self.config.generator_config.cond_dims
        self.token_cond_dim = self.config.encoder_config.trans_dim
        self.condition_dropout_prob = float(getattr(self.config.generator_config, "condition_dropout_prob", 0.0))
        self.cfg_guidance_scale = float(
            getattr(
                self.config.generator_config,
                "cfg_guidance_scale",
                getattr(self.config.generator_config, "condition_force_weight", 0.0),
            )
        )
        self.cross_attn_heads = int(getattr(self.config.generator_config, "cross_attn_heads", 4))
        self.sample_verbose = bool(getattr(self.config.generator_config, "sample_verbose", False))
        self.sample_log_interval = int(getattr(self.config.generator_config, "sample_log_interval", 50))
        self.net = DenoisingNet(
            point_dim=3,
            cond_dims=self.cond_dims,
            residual=True,
            token_cond_dim=self.token_cond_dim,
            cross_attn_heads=self.cross_attn_heads,
        )
        self.var_sched = VarianceSchedule(config)
        self.interval_nums = self.config.generator_config.interval_nums
        self.null_global_condition = nn.Parameter(torch.zeros(1, self.cond_dims))
        self.null_visible_token = nn.Parameter(torch.zeros(1, 1, self.token_cond_dim))
        nn.init.normal_(self.null_global_condition, std=0.02)
        nn.init.normal_(self.null_visible_token, std=0.02)

    def _apply_condition_dropout(self, cond, token_condition=None, token_positions=None):
        if (not self.training) or self.condition_dropout_prob <= 0:
            return cond, token_condition, token_positions

        drop_mask = torch.rand(cond.size(0), device=cond.device) < self.condition_dropout_prob
        if not drop_mask.any():
            return cond, token_condition, token_positions

        cond = cond.clone()
        drop_count = int(drop_mask.sum().item())
        cond[drop_mask] = self.null_global_condition.to(device=cond.device, dtype=cond.dtype).expand(drop_count, -1)

        if token_condition is not None:
            token_condition = token_condition.clone()
            token_condition[drop_mask] = self.null_visible_token.to(
                device=token_condition.device,
                dtype=token_condition.dtype,
            ).expand(
                drop_count,
                token_condition.size(1),
                -1,
            )

        if token_positions is not None:
            token_positions = token_positions.clone()
            token_positions[drop_mask] = 0.0

        return cond, token_condition, token_positions

    def _build_unconditional_condition(self, cond, token_condition=None, token_positions=None):
        batch_size = cond.size(0)
        null_global = self.null_global_condition.to(device=cond.device, dtype=cond.dtype).expand(batch_size, -1)

        null_tokens = None
        if token_condition is not None:
            null_tokens = self.null_visible_token.to(
                device=token_condition.device,
                dtype=token_condition.dtype,
            ).expand(
                batch_size,
                token_condition.size(1),
                -1,
            )

        null_positions = None
        if token_positions is not None:
            null_positions = token_positions.new_zeros(token_positions.shape)

        return null_global, null_tokens, null_positions

    def _resolve_condition_inputs(
        self,
        cond,
        token_condition=None,
        token_positions=None,
        force_unconditional=False,
    ):
        if force_unconditional:
            return self._build_unconditional_condition(
                cond,
                token_condition=token_condition,
                token_positions=token_positions,
            )
        return cond, token_condition, token_positions

    @torch.no_grad()
    def _compute_step_reconstruction_metrics(
        self,
        coords,
        cond,
        ts,
        noises,
        token_condition=None,
        token_positions=None,
        force_unconditional=False,
    ):
        device = coords.device
        cond_eval, token_condition_eval, token_positions_eval = self._resolve_condition_inputs(
            cond,
            token_condition=token_condition,
            token_positions=token_positions,
            force_unconditional=force_unconditional,
        )

        total_x0_metric = 0.0
        total_cd_metric = 0.0
        recovered_coords = coords

        for i in range(self.interval_nums):
            t = ts[i].tolist()
            alphas_cumprod = self.var_sched.alphas_cumprod[t].to(device)
            beta = self.var_sched.betas[t].to(device)

            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod).view(-1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod).view(-1, 1, 1)

            noise = noises[i]
            noised_coords = sqrt_alphas_cumprod_t * coords + sqrt_one_minus_alphas_cumprod_t * noise
            pred_noise = self.net(
                noised_coords,
                beta=beta,
                cond=cond_eval,
                visible_tokens=token_condition_eval,
                token_positions=token_positions_eval,
            )
            recovered_coords = (
                noised_coords - sqrt_one_minus_alphas_cumprod_t * pred_noise
            ) / sqrt_alphas_cumprod_t

            metric_x0_mse = F.mse_loss(coords, recovered_coords, reduction='mean')
            total_x0_metric += (metric_x0_mse.item() * (1.0 / self.interval_nums))
            metric_cd = get_chamfer_distance(coords, recovered_coords).mean()
            total_cd_metric += (metric_cd.item() * (1.0 / self.interval_nums))

        return {
            "x0_mse": total_x0_metric,
            "cd": total_cd_metric,
            "recovered_points": recovered_coords.detach(),
        }

    @torch.no_grad()
    def diagnose_reconstruction(
        self,
        coords,
        cond,
        num_points=None,
        token_condition=None,
        token_positions=None,
        compare_unconditional=True,
        guidance_scale=None,
    ):
        batch_size = coords.size(0)
        if num_points is None:
            num_points = coords.size(1)

        ts = self.var_sched.recurrent_uniform_sampling(batch_size, self.interval_nums)
        noises = [torch.randn_like(coords) for _ in range(self.interval_nums)]

        conditioned_step = self._compute_step_reconstruction_metrics(
            coords,
            cond,
            ts,
            noises,
            token_condition=token_condition,
            token_positions=token_positions,
            force_unconditional=False,
        )
        conditioned_sample = self.sample(
            num_points,
            cond,
            coords.device,
            token_condition=token_condition,
            token_positions=token_positions,
            guidance_scale=guidance_scale,
            force_unconditional=False,
        )

        diagnostics = {
            "conditioned_one_step_x0_mse": conditioned_step["x0_mse"],
            "conditioned_one_step_cd": conditioned_step["cd"],
            "conditioned_sample_cd": float(
                get_chamfer_distance(coords, conditioned_sample).mean().item()
            ),
            "conditioned_one_step_points": conditioned_step["recovered_points"],
            "conditioned_sample_points": conditioned_sample.detach(),
        }

        if compare_unconditional:
            unconditional_step = self._compute_step_reconstruction_metrics(
                coords,
                cond,
                ts,
                noises,
                token_condition=token_condition,
                token_positions=token_positions,
                force_unconditional=True,
            )
            unconditional_sample = self.sample(
                num_points,
                cond,
                coords.device,
                token_condition=token_condition,
                token_positions=token_positions,
                guidance_scale=guidance_scale,
                force_unconditional=True,
            )
            diagnostics.update({
                "unconditioned_one_step_x0_mse": unconditional_step["x0_mse"],
                "unconditioned_one_step_cd": unconditional_step["cd"],
                "unconditioned_sample_cd": float(
                    get_chamfer_distance(coords, unconditional_sample).mean().item()
                ),
                "unconditioned_one_step_points": unconditional_step["recovered_points"],
                "unconditioned_sample_points": unconditional_sample.detach(),
            })
            diagnostics["condition_gain_one_step_cd"] = (
                diagnostics["unconditioned_one_step_cd"] - diagnostics["conditioned_one_step_cd"]
            )
            diagnostics["condition_gain_sample_cd"] = (
                diagnostics["unconditioned_sample_cd"] - diagnostics["conditioned_sample_cd"]
            )

        return diagnostics
    
    def get_loss1(self, coords, cond, ts=None, token_condition=None, token_positions=None):
        """
        Args:
            coords:   point cloud, (B, N, 3).
            cond:     condition (B, F).
        """

        batch_size, _, point_dim = coords.size()

        cond, token_condition, token_positions = self._apply_condition_dropout(
            cond,
            token_condition=token_condition,
            token_positions=token_positions,
        )

        if ts == None:
            ts = self.var_sched.recurrent_uniform_sampling(batch_size, self.interval_nums)

        total_loss = 0

        for i in range(self.interval_nums):
            t = ts[i].tolist()
            
            alphas_cumprod = self.var_sched.alphas_cumprod[t]
            beta = self.var_sched.betas[t]
            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod).view(-1, 1, 1)       # (B, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod).view(-1, 1, 1)   # (B, 1, 1)
            
            noise = torch.randn_like(coords)  # (B, N, d)
            # 加噪
            noised_coords = sqrt_alphas_cumprod_t * coords + sqrt_one_minus_alphas_cumprod_t * noise
            
            # 预测
            pred_noise = self.net(
                noised_coords,
                beta=beta,
                cond=cond,
                visible_tokens=token_condition,
                token_positions=token_positions,
            ) # (B, N, 3)
            loss = F.mse_loss(noise.view(-1, point_dim), pred_noise.view(-1, point_dim), reduction='mean')
            total_loss += (loss * (1.0 / self.interval_nums))

        total_x0_metric = 0.0
        with torch.no_grad():
                # 1. 恢复 x_0
                recovered_coords = (noised_coords - sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t
                
                # 2. 计算 x_0 恢复指标 (Metric 通常保持未加权，反映真实的物理误差)
                # 如果你也希望 metric 加权，可以用同样的逻辑处理
                metric_x0_mse = F.mse_loss(coords, recovered_coords, reduction='mean')
                
                # 3. 累加
                total_x0_metric += (metric_x0_mse.item() * (1.0 / self.interval_nums))
        
        return total_loss, total_x0_metric
    
    def get_loss(self, coords, cond, weights=None, ts=None, token_condition=None, token_positions=None):
        """
        Args:
            coords:   point cloud, (B, N, 3).
            cond:     condition (B, F).
            weights:  (B,)  Sample-wise weights.
        """

        batch_size, _, point_dim = coords.size()
        device = coords.device
        cond, token_condition, token_positions = self._apply_condition_dropout(
            cond,
            token_condition=token_condition,
            token_positions=token_positions,
        )

        if ts == None:
            ts = self.var_sched.recurrent_uniform_sampling(batch_size, self.interval_nums)

        total_loss = 0
        total_x0_metric = 0.0
        total_cd_metric = 0.0
        # --- [处理权重] ---
        # 如果传入了 weights，先将其变为 (B, 1, 1) 以便广播乘法
        if weights is not None:
            # 确保 weights 在正确的设备上
            # view(batch_size, 1, 1) 使得 (B) -> (B, 1, 1)
            # 这样乘以 (B, N, 3) 时，每个 batch 中的 N 个点都会乘以同一个 weight
            batch_weights = weights.view(batch_size, 1, 1).to(device)
        else:
            batch_weights = None

        for i in range(self.interval_nums):
            t = ts[i].tolist()
            
            # ... (调度器参数获取，保持不变) ...
            alphas_cumprod = self.var_sched.alphas_cumprod[t].to(device)
            beta = self.var_sched.betas[t].to(device)
            
            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod).view(-1, 1, 1)       
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod).view(-1, 1, 1) 
            
            noise = torch.randn_like(coords) # (B, N, 3)
            
            # 加噪
            noised_coords = sqrt_alphas_cumprod_t * coords + sqrt_one_minus_alphas_cumprod_t * noise
            
            # 预测
            pred_noise = self.net(
                noised_coords,
                beta=beta,
                cond=cond,
                visible_tokens=token_condition,
                token_positions=token_positions,
            ) # (B, N, 3)
            
            # --- [修改 1: 加权 Loss 计算] ---
            # 1. 计算 element-wise MSE，不进行 reduction (B, N, 3)
            loss_elementwise = F.mse_loss(pred_noise, noise, reduction='none')
            
            # 2. 应用权重
            if batch_weights is not None:
                # (B, N, 3) * (B, 1, 1) -> (B, N, 3)
                loss_weighted = loss_elementwise * batch_weights
                # 3. 求均值
                loss = loss_weighted.mean()
            else:
                # 如果没有权重，退化为普通 MSE
                loss = loss_elementwise.mean()

            total_loss += (loss * (1.0 / self.interval_nums))
            
            recovered_coords = (noised_coords - sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t
                  
            with torch.no_grad():
                metric_x0_mse = F.mse_loss(coords, recovered_coords, reduction='mean')
                total_x0_metric += (metric_x0_mse.item() * (1.0 / self.interval_nums))
                metric_cd = get_chamfer_distance(coords, recovered_coords).mean()
                total_cd_metric += (metric_cd.item() * (1.0 / self.interval_nums))
        
        return total_loss, total_x0_metric, total_cd_metric
    
    def get_loss_withCD(self, coords, cond, weights=None, ts=None, cd_weight=0.1, token_condition=None, token_positions=None):
        """
        Args:
            coords:   point cloud, (B, N, 3).
            cond:     condition (B, F).
            weights:  (B,)  Sample-wise weights.
            ts:       Time steps.
            cd_weight: Weight for Chamfer Distance regularization.
        """

        batch_size, _, point_dim = coords.size()
        device = coords.device
        cond, token_condition, token_positions = self._apply_condition_dropout(
            cond,
            token_condition=token_condition,
            token_positions=token_positions,
        )

        if ts == None:
            ts = self.var_sched.recurrent_uniform_sampling(batch_size, self.interval_nums)

        total_loss = 0
        total_diff_loss = 0
        total_x0_metric = 0.0
        total_cd_metric = 0.0

        # --- [处理权重] ---
        # 如果传入了 weights，先将其变为 (B, 1) 以便广播乘法 (适配 Chamfer Distance 输出的 (B,) 维度)
        if weights is not None:
            # 用于 MSE 的权重 (B, 1, 1)
            batch_weights_mse = weights.view(batch_size, 1, 1).to(device)
            # 用于 CD 的权重 (B,)
            batch_weights_cd = weights.to(device)
        else:
            batch_weights_mse = None
            batch_weights_cd = None

        for i in range(self.interval_nums):
            t = ts[i].tolist()
            
            # ... (调度器参数获取) ...
            alphas_cumprod = self.var_sched.alphas_cumprod[t].to(device)
            beta = self.var_sched.betas[t].to(device)
            
            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod).view(-1, 1, 1)       
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod).view(-1, 1, 1) 
            
            noise = torch.randn_like(coords) # (B, N, 3)
            
            # 加噪
            noised_coords = sqrt_alphas_cumprod_t * coords + sqrt_one_minus_alphas_cumprod_t * noise
            
            # 预测噪声
            pred_noise = self.net(
                noised_coords,
                beta=beta,
                cond=cond,
                visible_tokens=token_condition,
                token_positions=token_positions,
            ) # (B, N, 3)
            
            # --- [Part 1: 基础 MSE Loss] ---
            loss_elementwise = F.mse_loss(pred_noise, noise, reduction='none')
            
            if batch_weights_mse is not None:
                loss_mse = (loss_elementwise * batch_weights_mse).mean()
            else:
                loss_mse = loss_elementwise.mean()

            # --- [Part 2: 预测 x_0 并计算 Chamfer Distance] ---
            # 关键：这里必须在 no_grad 之外计算，才能传梯度
            # 根据 DDPM 公式反推 x_0
            pred_x0 = (noised_coords - sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t
            
            # 计算 Chamfer Distance (B,)
            loss_cd_batch = get_chamfer_distance(coords, pred_x0)
            
            # 对 CD Loss 应用样本权重
            if batch_weights_cd is not None:
                loss_cd = (loss_cd_batch * batch_weights_cd).mean()
            else:
                loss_cd = loss_cd_batch.mean()

            # --- [Part 3: 总 Loss 组合] ---
            # 现在的 loss 既包含噪声预测准确度，也包含几何形状还原度
            current_iter_loss = loss_mse + cd_weight * loss_cd
            
            current_diff_loss = loss_mse
            total_diff_loss += current_diff_loss * (1.0 / self.interval_nums)
            
            total_loss += (current_iter_loss * (1.0 / self.interval_nums))
                        
            # --- [Metric 计算] ---
            with torch.no_grad():
                metric_x0_mse = F.mse_loss(coords, pred_x0, reduction='mean')
                total_x0_metric += (metric_x0_mse.item() * (1.0 / self.interval_nums))
                metric_cd = get_chamfer_distance(coords, pred_x0).mean()
                total_cd_metric += (metric_cd.item() * (1.0 / self.interval_nums))

        return total_loss, total_diff_loss, total_x0_metric, total_cd_metric
    
    def sample(
        self,
        num_points,
        cond,
        device,
        token_condition=None,
        token_positions=None,
        guidance_scale=None,
        force_unconditional=False,
    ):
        """
        Generates a point cloud by reversing the diffusion process.
        Args:
            num_points: The number of points to generate (e.g., 1024).
            cond: The condition vector from the encoder, shape (B, F).
            device: The torch device to run on.
        Returns:
            A list of point clouds, one for each timestep of the reverse process.
        """
        batch_size = cond.size(0)
        cond_eval, token_condition_eval, token_positions_eval = self._resolve_condition_inputs(
            cond,
            token_condition=token_condition,
            token_positions=token_positions,
            force_unconditional=force_unconditional,
        )
        if guidance_scale is None:
            guidance_scale = self.cfg_guidance_scale
        if force_unconditional:
            guidance_scale = 0.0
        # Start with pure gaussian noise
        x_t = torch.randn(batch_size, num_points, 3).to(device)

        if self.sample_verbose:
            print("Starting reverse diffusion process...")
        # Reverse process loop
        for t in range(self.var_sched.num_steps, 0, -1):
            # Prepare timestep tensor
            ts = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Get diffusion schedule variables for timestep t
            alphas_t = self.var_sched.alphas[ts].view(-1, 1, 1)
            alphas_cumprod_t = self.var_sched.alphas_cumprod[ts].view(-1, 1, 1)
            betas_t = self.var_sched.betas[ts].view(-1, 1, 1)
            
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod_t)
            sqrt_recip_alphas_t = torch.sqrt(1.0 / alphas_t)
            
            # Predict noise using the denoising network
            # Note: The network expects beta values, not the timestep t
            if guidance_scale > 0:
                null_cond, null_tokens, null_positions = self._build_unconditional_condition(
                    cond_eval,
                    token_condition=token_condition_eval,
                    token_positions=token_positions_eval,
                )
                pred_noise_uncond = self.net(
                    x_t,
                    beta=betas_t,
                    cond=null_cond,
                    visible_tokens=null_tokens,
                    token_positions=null_positions,
                )
                pred_noise_cond = self.net(
                    x_t,
                    beta=betas_t,
                    cond=cond_eval,
                    visible_tokens=token_condition_eval,
                    token_positions=token_positions_eval,
                )
                pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
            else:
                pred_noise = self.net(
                    x_t,
                    beta=betas_t,
                    cond=cond_eval,
                    visible_tokens=token_condition_eval,
                    token_positions=token_positions_eval,
                )
            
            # DDPM sampling formula:
            # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * pred_noise) + sigma_t * z
            mean = sqrt_recip_alphas_t * (x_t - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)
            
            if t > 1:
                # Add noise for all steps except the last one
                alphas_cumprod_t_prev = self.var_sched.alphas_cumprod[ts-1].view(-1, 1, 1)
                posterior_variance = (1. - alphas_cumprod_t_prev) / (1. - alphas_cumprod_t) * betas_t
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(posterior_variance) * noise
            else:
                # No noise at the final step
                x_t = mean
            
            if self.sample_verbose:
                if (self.sample_log_interval > 0 and t % self.sample_log_interval == 0) or t < 10:
                    print(f"  Timestep {t}/{self.var_sched.num_steps} processed.")

        if self.sample_verbose:
            print("Reverse diffusion finished.")
        recon_points = x_t
        return recon_points

    def sampleN(self, num_points, cond, device, capture_range=None, capture_num=10, token_condition=None, token_positions=None, guidance_scale=None):
        """
        Generates a point cloud by reversing the diffusion process.
        
        Args:
            num_points: The number of points to generate.
            cond: The condition vector.
            device: The torch device.
            capture_range: tuple (start_t, end_t), e.g., (1900, 2000). 
                        Specify the range of timesteps to collect.
            capture_num: int, number of frames to collect evenly within the range.
        
        Returns:
            recon_points: The final generated point cloud.
            collected_frames: A list of dicts [{'t': t, 'data': point_cloud}, ...].
        """
        batch_size = cond.size(0)
        if guidance_scale is None:
            guidance_scale = self.cfg_guidance_scale
        # Start with pure gaussian noise
        x_t = torch.randn(batch_size, num_points, 3).to(device)
        
        # Calculate timesteps to save
        steps_to_save = set()
        if capture_range is not None:
            start_t, end_t = capture_range
            # 使用 linspace 在区间内均匀取点，并转为整数
            # 注意：扩散模型通常是从大到小遍历，所以这里生成的数字是用来在循环中匹配的
            save_indices = np.linspace(start_t, end_t, capture_num).astype(int)
            steps_to_save = set(save_indices)
            if self.sample_verbose:
                print(f"Will capture frames at steps: {sorted(list(steps_to_save), reverse=True)}")

        collected_frames = []

        if self.sample_verbose:
            print("Starting reverse diffusion process...")
        # Reverse process loop (from T down to 1)
        for t in range(self.var_sched.num_steps, 0, -1):
            
            # --- [Modification] Capture logic ---
            # 我们在去噪发生之前捕获 x_t，这样捕获的就是时刻 t 的状态
            if t in steps_to_save:
                # Clone and detach to CPU, convert to numpy
                # 保存 batch 中的第一个样本，或者保存整个 batch，这里默认保存整个 batch
                current_data = x_t.detach().cpu().numpy() 
                collected_frames.append({
                    't': t,
                    'data': current_data 
                })
            # ------------------------------------

            # Prepare timestep tensor
            ts = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Get diffusion schedule variables
            alphas_t = self.var_sched.alphas[ts].view(-1, 1, 1)
            alphas_cumprod_t = self.var_sched.alphas_cumprod[ts].view(-1, 1, 1)
            betas_t = self.var_sched.betas[ts].view(-1, 1, 1)
            
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod_t)
            sqrt_recip_alphas_t = torch.sqrt(1.0 / alphas_t)
            
            # Predict noise
            if guidance_scale > 0:
                null_cond, null_tokens, null_positions = self._build_unconditional_condition(
                    cond,
                    token_condition=token_condition,
                    token_positions=token_positions,
                )
                pred_noise_uncond = self.net(
                    x_t,
                    beta=betas_t,
                    cond=null_cond,
                    visible_tokens=null_tokens,
                    token_positions=null_positions,
                )
                pred_noise_cond = self.net(
                    x_t,
                    beta=betas_t,
                    cond=cond,
                    visible_tokens=token_condition,
                    token_positions=token_positions,
                )
                pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
            else:
                pred_noise = self.net(
                    x_t,
                    beta=betas_t,
                    cond=cond,
                    visible_tokens=token_condition,
                    token_positions=token_positions,
                )
            
            # DDPM sampling formula
            mean = sqrt_recip_alphas_t * (x_t - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)
            
            if t > 1:
                alphas_cumprod_t_prev = self.var_sched.alphas_cumprod[ts-1].view(-1, 1, 1)
                posterior_variance = (1. - alphas_cumprod_t_prev) / (1. - alphas_cumprod_t) * betas_t
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(posterior_variance) * noise
            else:
                x_t = mean
            
            if self.sample_verbose:
                if (self.sample_log_interval > 0 and t % self.sample_log_interval == 0) or t < 10:
                    print(f"  Timestep {t}/{self.var_sched.num_steps} processed.")

        if self.sample_verbose:
            print("Reverse diffusion finished.")
        recon_points = x_t
        
        return recon_points, collected_frames
