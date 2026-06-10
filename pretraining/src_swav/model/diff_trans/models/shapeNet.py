import torch
import torch.nn as nn
def compute_aabb_ground_truth(object_point_clouds: torch.Tensor) -> torch.Tensor:
    """
    从一批 object 点云计算 AABB 真值。

    参数:
    object_point_clouds (torch.Tensor): 
        一批 object 的点云。
        形状为 (B, N_points, 3)，其中 B 是 object 的数量, N_points 是每个 object 的点数。

    返回:
    gt_bboxes (torch.Tensor): 
        AABB 真值，形状为 (B, 6)。
        每行是 (cx, cy, cz, sx, sy, sz)。
    """
    # 1. 找到最小和最大的坐标
    # min_coords 和 max_coords 的形状都将是 (B, 3)
    min_coords = torch.min(object_point_clouds, dim=1).values
    max_coords = torch.max(object_point_clouds, dim=1).values
    
    # 2. 计算中心 (center)
    gt_centers = (min_coords + max_coords) / 2.0
    
    # 3. 计算尺寸 (size)
    # 加上一个很小的 epsilon (1e-6) 来防止尺寸为0，避免数值不稳定
    gt_sizes = (max_coords - min_coords) + 1e-6
    
    # 4. 拼接 (center, size)
    gt_bboxes = torch.cat([gt_centers, gt_sizes], dim=1)
    
    return gt_bboxes

class BBoxPredictionHead(nn.Module):
    """
    一个简单的MLP头，用于从 object embedding 
    回归 3D Bounding Box (AABB) 参数。
    
    AABB 被定义为 (center_x, center_y, center_z, size_x, size_y, size_z)
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        """
        参数:
        embedding_dim (int): 输入的 object embedding 维度 (例如 512)
        hidden_dim (int): MLP 隐藏层维度
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim), # 使用 LayerNorm 提高稳定性
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            # 输出层：6个参数 (cx, cy, cz, sx, sy, sz)
            nn.Linear(hidden_dim // 2, 6)
        )

    def forward(self, obj_embeddings: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
        obj_embeddings (torch.Tensor): 形状为 (B, D) 或 (B, N_obj, D)
                                       B=batch_size, N_obj=objects_per_scene, D=embedding_dim

        返回:
        pred_bboxes (torch.Tensor): 形状为 (B, 6) 或 (B, N_obj, 6)
        """
        return self.mlp(obj_embeddings)