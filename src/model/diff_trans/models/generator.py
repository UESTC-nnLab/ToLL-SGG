import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from src.model.diff_trans.utils import misc
from src.model.diff_trans.utils.logger import *
from SoftPool import soft_pool2d, SoftPool2d

def get_chamfer_distance(s1, s2):
    """
    计算两个点云之间的 Chamfer Distance (Squared L2)。
    
    Args:
        s1: (B, N, 3) 预测点云 (或真实点云)
        s2: (B, M, 3) 真实点云 (或预测点云)
    Returns:
        loss: (B,) 每个样本的 CD 损失
    """
    # 1. 计算距离矩阵 dist_mat: (B, N, M)
    # torch.cdist 计算的是欧氏距离 (L2 norm)
    dist_mat = torch.cdist(s1, s2)
    
    # 2. 取平方，因为通常 Chamfer Loss 定义为距离的平方和，这样梯度更强
    dist_mat_sq = dist_mat.pow(2)
    
    # 3. 对于 s1 中的每个点，找到 s2 中最近的点，并求平均
    min_dist_s1, _ = torch.min(dist_mat_sq, dim=2) # (B, N)
    term1 = torch.mean(min_dist_s1, dim=1)         # (B,)
    
    # 4. 对于 s2 中的每个点，找到 s1 中最近的点，并求平均
    min_dist_s2, _ = torch.min(dist_mat_sq, dim=1) # (B, M)
    term2 = torch.mean(min_dist_s2, dim=1)         # (B,)
    
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

# Point Denoising Network
class DenoisingNet(nn.Module):

    def __init__(self, point_dim, cond_dims, residual):
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

    def forward(self, coords, beta, cond):
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
        self.net = DenoisingNet(point_dim=3, cond_dims=512, residual=True)
        self.var_sched = VarianceSchedule(config)
        self.interval_nums = self.config.generator_config.interval_nums
    
    def get_loss1(self, coords, cond, ts=None):
        """
        Args:
            coords:   point cloud, (B, N, 3).
            cond:     condition (B, F).
        """

        batch_size, _, point_dim = coords.size()

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
            pred_noise = self.net(noised_coords, beta=beta, cond=cond) # (B, N, 3)
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
    
    def get_loss(self, coords, cond, weights=None, ts=None):
        """
        Args:
            coords:   point cloud, (B, N, 3).
            cond:     condition (B, F).
            weights:  (B,)  Sample-wise weights.
        """

        batch_size, _, point_dim = coords.size()
        device = coords.device
        if ts == None:
            ts = self.var_sched.recurrent_uniform_sampling(batch_size, self.interval_nums)

        total_loss = 0
        total_x0_metric = 0.0
        total_loss_cd_batch = 0.0
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
            pred_noise = self.net(noised_coords, beta=beta, cond=cond) # (B, N, 3)
            
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
            # loss_cd_batch = F.mse_loss(torch.clamp(recovered_coords, min=-1.5, max=1.5), coords, reduction='mean')
            # total_loss_cd_batch += (loss_cd_batch * (1.0 / self.interval_nums))
                  
            with torch.no_grad():
                # 1. 恢复 x_0
                # recovered_coords = (noised_coords - sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t
                
                # 2. 计算 x_0 恢复指标 (Metric 通常保持未加权，反映真实的物理误差)
                # 如果你也希望 metric 加权，可以用同样的逻辑处理
                metric_x0_mse = F.mse_loss(coords, recovered_coords, reduction='mean')
                # 3. 累加
                total_x0_metric += (metric_x0_mse.item() * (1.0 / self.interval_nums))
        
        return total_loss, total_x0_metric
    
    def get_loss_withCD(self, coords, cond, weights=None, ts=None, cd_weight=0.1):
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
        if ts == None:
            ts = self.var_sched.recurrent_uniform_sampling(batch_size, self.interval_nums)

        total_loss = 0
        total_x0_metric = 0.0

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
            pred_noise = self.net(noised_coords, beta=beta, cond=cond) # (B, N, 3)
            
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
            
            total_diff_loss = current_diff_loss * (1.0 / self.interval_nums)
            
            total_loss += (current_iter_loss * (1.0 / self.interval_nums))
                        
            # --- [Metric 计算] ---
            with torch.no_grad():
                # 为了指标统计，我们直接复用上面算好的 pred_x0
                # 这样省去一次计算
                metric_x0_mse = F.mse_loss(coords, pred_x0, reduction='mean')
                total_x0_metric += (metric_x0_mse.item() * (1.0 / self.interval_nums))

        return total_loss, total_diff_loss, total_x0_metric
    
    def sample(self, num_points, cond, device):
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
        # Start with pure gaussian noise
        x_t = torch.randn(batch_size, num_points, 3).to(device)
        
        # A list to store intermediate point clouds
        intermediate_points = [x_t.clone().cpu().numpy()]

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
            pred_noise = self.net(x_t, beta=betas_t, cond=cond)
            
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
            
            if t % 50 == 0 or t < 10: # Log progress and save more frames at the end
                 print(f"  Timestep {t}/{self.var_sched.num_steps} processed.")
            
            # intermediate_points.append(x_t.clone().detach().cpu().numpy())
            
        print("Reverse diffusion finished.")
        # The final denoised point cloud is the last one in the list
        recon_points = x_t
        return recon_points#, intermediate_points

    def sampleN(self, num_points, cond, device, capture_range=None, capture_num=10):
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
            print(f"Will capture frames at steps: {sorted(list(steps_to_save), reverse=True)}")

        collected_frames = []

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
            pred_noise = self.net(x_t, beta=betas_t, cond=cond)
            
            # DDPM sampling formula
            mean = sqrt_recip_alphas_t * (x_t - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)
            
            if t > 1:
                alphas_cumprod_t_prev = self.var_sched.alphas_cumprod[ts-1].view(-1, 1, 1)
                posterior_variance = (1. - alphas_cumprod_t_prev) / (1. - alphas_cumprod_t) * betas_t
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(posterior_variance) * noise
            else:
                x_t = mean
            
            if t % 50 == 0 or t < 10: 
                print(f"  Timestep {t}/{self.var_sched.num_steps} processed.")
        
        print("Reverse diffusion finished.")
        recon_points = x_t
        
        return recon_points, collected_frames