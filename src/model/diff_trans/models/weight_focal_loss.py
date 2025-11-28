import torch
def compute_batch_complexity_weight(points, base_weight=1.0, boost_factor=2.0):
    """
    Batch 计算点云几何复杂性权重 (Vectorized Version)。
    
    Args:
        points: (B, N, 3) 点云数据
        base_weight: 基础权重
        boost_factor: 复杂物体权重的放大系数
        
    Returns:
        weights: (B,) 每个点云对应的权重
    """
    B, N, C = points.shape
    
    # ------------------------------------------------------------
    # 1. 计算协方差矩阵 (Batch Covariance Matrix)
    # ------------------------------------------------------------
    # 去中心化: (B, N, 3) - (B, 1, 3)
    mean = points.mean(dim=1, keepdim=True)
    centered = points - mean
    
    # 计算 Covariance: (X^T * X) / (N-1)
    # transpose(1, 2) 变为 (B, 3, N)
    # bmm((B, 3, N), (B, N, 3)) -> (B, 3, 3)
    cov = torch.bmm(centered.transpose(1, 2), centered) / (N - 1)
    
    # ------------------------------------------------------------
    # 2. 特征值分解 (Batch Eigen Decomposition)
    # ------------------------------------------------------------
    # torch.linalg.eigh 专门用于对称矩阵 (Symmetric/Hermitian)，比 eig 快且稳
    # 返回的 eigenvalues 默认是升序排列的: lambda_1 <= lambda_2 <= lambda_3
    # eigs shape: (B, 3)
    eigs, _ = torch.linalg.eigh(cov)
    
    # 加上微小值防止数值问题 (特别是全0输入的padding情况)
    eigs = torch.clamp(eigs, min=1e-8)
    
    # ------------------------------------------------------------
    # 3. 计算权重
    # ------------------------------------------------------------
    lambda_min = eigs[:, 0] # (B,) 最小特征值 (厚度方向)
    lambda_max = eigs[:, 2] # (B,) 最大特征值 (主方向)
    
    # 复杂性得分: 0 (平面/线) -> 1 (球体/噪声)
    # 对于墙壁/地板，lambda_min 极小，score -> 0
    complexity_score = lambda_min / (lambda_max + 1e-8)
    
    # 映射到权重
    weights = base_weight + boost_factor * complexity_score
    
    return weights


def compute_local_complexity_weight(points, k=16, base_weight=1.0, boost_factor=2.0):
    """
    基于 [局部表面变化率] 计算复杂度权重。
    相比全局PCA，这种方法能识别出球体是"简单"的，而保留对边缘/杂乱结构的敏感性。
    
    Args:
        points: (B, N, 3) 点云
        k: 局部邻域大小 (建议 16 或 32)
        base_weight: 基础权重
        boost_factor: 放大系数
        
    Returns:
        weights: (B,)
    """
    B, N, C = points.shape
    device = points.device
    
    # ----------------------------------------------------------------------
    # 1. KNN 构建 (Find k-Nearest Neighbors)
    # ----------------------------------------------------------------------
    # 计算所有点对距离矩阵 (B, N, N)
    # 对于 N=1024/2048，显存占用可控。如果 N 很大(>10k)，需要用 pytorch3d.ops.knn_points
    dist_mat = torch.cdist(points, points) 
    
    # 选取最近的 k 个点 (包含自己)
    # knn_indices: (B, N, k)
    _, knn_indices = torch.topk(dist_mat, k=k, dim=-1, largest=False)
    
    # ----------------------------------------------------------------------
    # 2. Gather 邻域点 (Gather Neighbor Points)
    # ----------------------------------------------------------------------
    # 我们需要构建 (B, N, k, 3) 的张量
    
    # 构造 batch 索引: (B, N, k)
    batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, k)
    
    # Gather 操作
    # points: (B, N, 3) -> 扩展索引以匹配 gathered 形状
    # 最终 neighborhood: (B, N, k, 3)
    knn_indices_flat = knn_indices.view(B, N * k) # 展平方便 gather
    points_gather = points.gather(1, knn_indices_flat.unsqueeze(-1).expand(-1, -1, 3))
    neighborhood = points_gather.view(B, N, k, 3)
    
    # ----------------------------------------------------------------------
    # 3. 局部 PCA (Local PCA)
    # ----------------------------------------------------------------------
    # 去中心化: 每个邻域减去该邻域的均值
    # mean: (B, N, 1, 3)
    local_mean = neighborhood.mean(dim=2, keepdim=True)
    local_centered = neighborhood - local_mean
    
    # 协方差矩阵: (B, N, 3, 3)
    # 实际上我们将 (B*N, k, 3) 视为 batch 做 bmm
    local_centered_flat = local_centered.view(B * N, k, 3)
    # (B*N, 3, k) @ (B*N, k, 3) -> (B*N, 3, 3)
    cov = torch.bmm(local_centered_flat.transpose(1, 2), local_centered_flat) / (k - 1)
    
    # 特征值分解
    # eigs: (B*N, 3), 升序 lambda1 <= lambda2 <= lambda3
    eigs, _ = torch.linalg.eigh(cov)
    eigs = torch.clamp(eigs, min=1e-8)
    
    # ----------------------------------------------------------------------
    # 4. 计算局部表面变化率 (Surface Variation)
    # ----------------------------------------------------------------------
    # Surface Variation公式: sigma = lambda_1 / (lambda_1 + lambda_2 + lambda_3)
    # lambda_1 是最小特征值 (法线方向的方差)
    # 对于平面区域，lambda_1 ≈ 0
    
    sum_eigs = eigs.sum(dim=1)
    # curvature: (B*N,)
    local_curvature = eigs[:, 0] / sum_eigs
    
    # ----------------------------------------------------------------------
    # 5. 聚合得到物体级权重
    # ----------------------------------------------------------------------
    # 恢复形状 (B, N)
    local_curvature = local_curvature.view(B, N)
    
    # 计算每个物体的平均曲率
    # 这里也可以用 max，但 mean 更鲁棒
    object_complexity = local_curvature.mean(dim=1) 
    
    # 映射到权重
    # 平面/球体: object_complexity 很小 (< 0.01)
    # 复杂物体: object_complexity 较大 (> 0.05)
    
    # 使用 sqrt 或 log 来拉伸分布，让微小的差异更明显（可选）
    # object_complexity = torch.sqrt(object_complexity)
    
    weights = base_weight + boost_factor * (object_complexity / 0.1) # 0.1 是一个经验归一化值
    # 截断一下防止权重过大爆炸
    weights = torch.clamp(weights, max=base_weight + boost_factor * 3.0)
    
    return weights

def visualize_batch_with_weights(points_tensor, weights_tensor, save_path="batch_vis.png"):
    import math
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    """
    将 Batch 点云可视化在一张大图上，并显示对应的权重。
    
    Args:
        points_tensor: (B, N, 3) Torch Tensor or Numpy Array
        weights_tensor: (B,) Torch Tensor or Numpy Array
        save_path: 图片保存路径
    """
    # 1. 数据转换与准备
    if isinstance(points_tensor, torch.Tensor):
        points = points_tensor.detach().cpu().numpy()
    else:
        points = points_tensor
        
    if isinstance(weights_tensor, torch.Tensor):
        weights = weights_tensor.detach().cpu().numpy()
    else:
        weights = weights_tensor

    B, N, _ = points.shape
    
    # 2. 计算网格布局 (Rows x Cols)
    cols = int(math.ceil(math.sqrt(B)))
    rows = int(math.ceil(B / cols))
    
    # 3. 创建画布 (根据子图数量自动调整大小)
    fig = plt.figure(figsize=(4 * cols, 3.5 * rows))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    print(f"正在绘制 {B} 个物体...")
    
    for i in range(B):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        pts = points[i]
        w = weights[i]
        
        # 提取坐标
        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        
        # 绘制散点 (颜色根据 Z 轴映射，增加立体感)
        ax.scatter(xs, ys, zs, c=zs, cmap='viridis', s=2, alpha=0.8)
        
        # 设置标题 (显示 Weight)
        # 根据权重大小改变标题颜色：红色代表高权重(复杂)，蓝色代表低权重(简单)
        title_color = 'red' if w > np.mean(weights) else 'blue'
        ax.set_title(f"Obj {i}\nWeight: {w:.4f}", color=title_color, fontsize=12, fontweight='bold')
        
        # 移除坐标轴刻度，只看形状
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # 设置视角 (可选)
        ax.view_init(elev=30, azim=45)
        
        # 统一坐标轴比例 (Visual fidelity)
        # 找到当前物体的最大范围，强制 Box 为立方体，防止扁平物体看起来像立方体
        max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
        mid_x = (xs.max()+xs.min()) * 0.5
        mid_y = (ys.max()+ys.min()) * 0.5
        mid_z = (zs.max()+zs.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 4. 保存
    save_dir = os.path.dirname(os.path.abspath(save_path))
    if not os.path.exists(save_dir) and save_dir != '':
        os.makedirs(save_dir)
        
    plt.suptitle(f"Batch Complexity Analysis (Red=High Weight, Blue=Low Weight)", fontsize=16)
    plt.savefig(save_path, bbox_inches='tight', dpi=150) # DPI 150 保证清晰度
    plt.close(fig)
    print(f"可视化结果已保存至: {save_path}")