import json
import os
import numpy as np
import torch
import torch.utils.data as data
from src.dataset.atlasnet_cache import fetch_object_embeddings, load_embedding_cache
from src.utils import op_utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入 3D 绘图工具包
from utils import util_ply
import trimesh
import os


def normalize_text_label(text):
    if text is None:
        return ""
    return str(text).strip().lower()


def load_text_embedding_cache(cache_path):
    if not cache_path:
        return None

    cache_path = os.path.expanduser(str(cache_path))
    if not os.path.exists(cache_path):
        return None

    cache = torch.load(cache_path, map_location="cpu")
    if not isinstance(cache, dict):
        raise ValueError(f"Invalid text embedding cache at {cache_path}: expected a dict.")

    normalized_cache = {}
    embedding_dim = None
    for key, value in cache.items():
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.float().contiguous().view(-1)
        if embedding_dim is None:
            embedding_dim = int(value.shape[0])
        elif value.shape[0] != embedding_dim:
            raise ValueError(
                f"Inconsistent text embedding dimension at {cache_path}: "
                f"expected {embedding_dim}, got {value.shape[0]} for key '{key}'."
            )
        normalized_cache[normalize_text_label(key)] = value

    return {
        "path": cache_path,
        "embedding_dim": int(embedding_dim or 0),
        "embeddings": normalized_cache,
    }


def fetch_text_embeddings(cache_bundle, object_texts):
    if cache_bundle is None:
        return None, None

    emb_dim = int(cache_bundle["embedding_dim"])
    embeddings = torch.zeros((len(object_texts), emb_dim), dtype=torch.float32)
    valid_mask = torch.zeros((len(object_texts),), dtype=torch.bool)
    cache = cache_bundle["embeddings"]

    for row_idx, text in enumerate(object_texts):
        normalized_text = normalize_text_label(text)
        embedding = cache.get(normalized_text)
        if embedding is None:
            continue
        embeddings[row_idx] = embedding
        valid_mask[row_idx] = True

    return embeddings, valid_mask

def uniform_upsample_gpu(points, target_num, k=3):
            import torch.nn.functional as F
            """
            基于局部稀疏度加权的均匀点云上采样 (PyTorch Vectorized)
            
            Args:
                points: (B, N, 3) 原始点云 Tensor
                target_num: 目标点数 (必须 > N)
                k: 用于构建局部三角形的邻居数量 (推荐 3)
                
            Returns:
                upsampled_points: (B, target_num, 3) 上采样后的点云
            """
            B, N, C = points.shape
            assert target_num > N, "目标点数必须大于原始点数"
            num_new_points = target_num - N
            
            # 1. 计算点与点之间的距离矩阵 (B, N, N)
            # 注意：如果 N 非常大 (>10000)，cdist 可能会显存溢出，需分批处理
            dists = torch.cdist(points, points)
            
            # 2. 找到每个点的 k 个最近邻 (排除自己，所以取 k+1)
            # values: (B, N, k+1), indices: (B, N, k+1)
            knn_dists, knn_indices = torch.topk(dists, k=k+1, dim=-1, largest=False)
            
            # 去掉第一个（也就是自己），保留 k 个邻居
            # neighbor_indices: (B, N, k)
            neighbor_indices = knn_indices[:, :, 1:]
            
            # 3. 计算稀疏度权重 (Sparsity Weights)
            # 计算每个点到其 k 个邻居的平均距离。距离越大，说明周围越空，越需要插值。
            # mean_dists: (B, N)
            avg_local_dist = knn_dists[:, :, 1:].mean(dim=-1)
            
            # 归一化为概率分布 (Probability Distribution)
            # 使用 Softmax 放大差异，让稀疏区域更容易被选中
            # 也可以直接用 avg_local_dist / sum，Softmax 会更激进地填充大洞
            weights = F.softmax(avg_local_dist, dim=-1) 
            
            # 4. 根据稀疏度选择要进行插值的中心点 (B, num_new_points)
            # torch.multinomial 可以根据权重进行采样
            # 这里的 indices 是我们要在这个点周围生成新点的索引
            sample_indices = torch.multinomial(weights, num_new_points, replacement=True)
            
            # 5. 收集生成所需的几何信息
            # 获取中心点坐标: (B, num_new, 3)
            batch_indices = torch.arange(B, device=points.device).unsqueeze(-1).expand(-1, num_new_points)
            center_points = points[batch_indices, sample_indices] # P_A
            
            # 获取中心点的邻居索引
            # neighbor_indices 的形状是 (B, N, k)，我们需要取 sample_indices 对应的行
            # select_neighbor_indices: (B, num_new, k)
            select_neighbor_indices = neighbor_indices[batch_indices, sample_indices]
            
            # 随机从 k 个邻居中选 2 个，加上中心点构成一个三角形
            # 这里为了简单，我们直接取第1个和第2个邻居 (也可以随机选)
            n1_indices = select_neighbor_indices[:, :, 0]
            n2_indices = select_neighbor_indices[:, :, 1]
            
            p1 = points[batch_indices, n1_indices] # P_B
            p2 = points[batch_indices, n2_indices] # P_C
            
            # 6. 在三角形 (center, p1, p2) 内部进行均匀采样
            # 使用重心坐标 (Barycentric Coordinates) 确保三角形内均匀
            # 公式: P = (1 - sqrt(r1)) * A + sqrt(r1) * (1 - r2) * B + sqrt(r1) * r2 * C
            r1 = torch.rand((B, num_new_points, 1), device=points.device)
            r2 = torch.rand((B, num_new_points, 1), device=points.device)
            
            sqrt_r1 = torch.sqrt(r1)
            
            # 计算新点坐标
            new_points = (1 - sqrt_r1) * center_points + \
                        sqrt_r1 * (1 - r2) * p1 + \
                        sqrt_r1 * r2 * p2
                        
            # 7. 拼接原始点和新点
            final_points = torch.cat([points, new_points], dim=1)
            
            return final_points

def visualize_and_save_sequence(collected_frames, save_path, batch_idx=0):
    """
    Visualizes a sequence of point clouds and saves the figure.
    
    Args:
        collected_frames: List of dicts [{'t': int, 'data': (B, N, 3) array}, ...]
        save_path: File path to save the image (e.g., 'outputs/vis.png').
        batch_idx: Index of the batch item to visualize (default 0).
    """
    if not collected_frames:
        print("No frames to visualize.")
        return

    num_frames = len(collected_frames)
    
    # 创建画布：根据帧数动态设置宽度，高度固定
    fig = plt.figure(figsize=(num_frames * 3, 3.5))
    
    # 按照时间步从大到小（Diffusion过程）或者从小到大排序展示
    # 这里我们直接按 collected_frames 的顺序展示 (通常是从 T 到 0)
    for i, frame_info in enumerate(collected_frames):
        t = frame_info['t']
        points = frame_info['data'] # 取出指定 batch 的点云 (N, 3)
        
        points = points.reshape(-1, 3)
        
        # 添加 3D 子图
        ax = fig.add_subplot(1, num_frames, i + 1, projection='3d')
        
        # 绘制散点
        # s=1 表示点的大小，alpha 控制透明度
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='b', alpha=0.6)
        
        # 设置标题
        ax.set_title(f"t = {t}")
        
        #为了美观，通常隐藏坐标轴刻度，只看形状
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # 可选：设置固定的视角或坐标轴范围，避免画面抖动
        # ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)

    plt.tight_layout()
    
    # 确保目录存在
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Visualization saved to {save_path}")


def visualize_scenes_batch(scene1_tensor, scene2_tensor, output_dir="batch_visualizations"):
    """
    将两个 (B, N, 3) 的点云 Tensor 中的每一个样本分别绘制，
    并将 B 张对比图保存到同一个文件夹下。
    """
    
    # --- 1. 检查与准备 ---
    batch_size = scene1_tensor.shape[0]
    num_points = scene1_tensor.shape[1]
    
    if batch_size == 0 or scene2_tensor.shape[0] == 0:
        print("错误：输入的 Tensor 为空 (Batch size 为 0)。")
        return

    # 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出文件夹: {output_dir}")
    else:
        print(f"输出文件夹已存在: {output_dir}")

    print(f"开始处理 Batch，共包含 {batch_size} 组点云...")

    # --- 2. 循环遍历 Batch 中的每一个点云 ---
    for i in range(batch_size):
        
        # 提取第 i 个样本: (N, 3) -> 转为 NumPy
        # 注意：这里直接用 [i] 索引，而不是 view(-1, 3)，确保只取当前样本
        scene1_np = scene1_tensor[i].detach().cpu().numpy()
        scene2_np = scene2_tensor[i].detach().cpu().numpy()
        
        # --- 3. 创建画布 (每次循环都新建一张图) ---
        fig = plt.figure(figsize=(12, 6))
        
        # --- 4. 左子图 (Scene 1) ---
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        x1, y1, z1 = scene1_np[:, 0], scene1_np[:, 1], scene1_np[:, 2]
        ax1.scatter(x1, y1, z1, s=1, c='r', label='Scene 1')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'Sample {i} - Scene 1 (Red)')
        # 为了保持视角一致，可以手动设置坐标轴范围（可选）
        # ax1.set_xlim([-1, 1]); ax1.set_ylim([-1, 1]); ax1.set_zlim([-1, 1])
        
        # --- 5. 右子图 (Scene 2) ---
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        x2, y2, z2 = scene2_np[:, 0], scene2_np[:, 1], scene2_np[:, 2]
        ax2.scatter(x2, y2, z2, s=1, c='b', label='Scene 2')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title(f'Sample {i} - Scene 2 (Blue)')
        
        # --- 6. 布局调整与保存 ---
        fig.suptitle(f'Point Cloud Comparison - Batch Index {i}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 生成独立的文件名，例如: batch_visualizations/sample_00.png
        save_path = os.path.join(output_dir, f"sample_{i:02d}.png")
        plt.savefig(save_path)
        
        # --- 7. 关键：关闭画布释放内存 ---
        # 如果不关闭，循环多次后内存会溢出
        plt.close(fig)
        
        if (i + 1) % 5 == 0:
            print(f"已保存 {i + 1}/{batch_size} 张图片...")

    print(f"所有图片已保存至: {os.path.abspath(output_dir)}")

def visualize_scenes_plt_with_points(scene1_tensor, scene2_tensor, output_filename="scene_comparison_plt.png"):
    """
    使用 Matplotlib 将两个 (B, N, 3) 的点云 Tensor 
    在同一张图的两个子图中展示，并保存图片到本地。
    同时：在图片所在目录创建一个 points 文件夹，保存对应的点云数据 (.npy)。
    """
    
    print(f"开始处理点云，目标图片路径: {output_filename}")

    # --- 1. 从 Batch 中提取点云并转为 NumPy ---
    if scene1_tensor.shape[0] == 0 or scene2_tensor.shape[0] == 0:
        print("错误：输入的 Tensor 为空 (Batch size 为 0)。")
        return

    # 将 (B, N, 3) -> (N, 3) 并转为 numpy
    # 注意：这里假设只取 batch 中的第一个样本
    scene1_np = scene1_tensor.view(-1,3).detach().cpu().numpy()
    scene2_np = scene2_tensor.view(-1,3).detach().cpu().numpy()
    
    print(f"  - pcd1 包含 {len(scene1_np)} 个点。")
    print(f"  - pcd2 包含 {len(scene2_np)} 个点。")

    # ============================================================
    # <--- 新增功能：保存点云数据 (.npy) --->
    # ============================================================
    try:
        # 1. 获取绝对路径，确保路径解析正确
        abs_path = os.path.abspath(output_filename)
        
        # 2. 解析目录和文件名
        # sample_dir: /.../sample_dir
        dir_name = os.path.dirname(abs_path)
        # file_stem: batch_sample_0 (去掉了 .png)
        base_name = os.path.basename(abs_path)
        file_stem = os.path.splitext(base_name)[0]
        
        # 3. 创建 points 子目录
        points_dir = os.path.join(dir_name, "points")
        os.makedirs(points_dir, exist_ok=True)
        
        # 4. 构建点云保存路径
        # 例如: .../sample_dir/points/batch_sample_0_scene1.npy
        pcd1_path = os.path.join(points_dir, f"{file_stem}_scene1.npy")
        pcd2_path = os.path.join(points_dir, f"{file_stem}_scene2.npy")
        
        # 5. 保存为 .npy 格式 (读取方便，体积小)
        np.save(pcd1_path, scene1_np)
        np.save(pcd2_path, scene2_np)
        
        print(f"  - 点云数据已保存至: {points_dir}")
        
    except Exception as e:
        print(f"警告: 保存点云文件失败，错误信息: {e}")
    # ============================================================


    # --- 2. 创建 Matplotlib 画布和子图 ---
    fig = plt.figure(figsize=(12, 6))
    
    # --- 3. 绘制第一个子图 (Scene 1) ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    x1, y1, z1 = scene1_np[:, 0], scene1_np[:, 1], scene1_np[:, 2]
    ax1.scatter(x1, y1, z1, s=1, c='r', label='Scene 1')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Scene 1 (Red)')
    ax1.legend()

    # --- 4. 绘制第二个子图 (Scene 2) ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    x2, y2, z2 = scene2_np[:, 0], scene2_np[:, 1], scene2_np[:, 2]
    ax2.scatter(x2, y2, z2, s=1, c='b', label='Scene 2')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Scene 2 (Blue)')
    ax2.legend()

    # --- 5. 添加总标题并调整布局 ---
    fig.suptitle(f'Point Cloud Comparison: {file_stem}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

    # --- 6. 保存图像 ---
    # 确保图片文件夹存在 (防止万一 output_filename 的文件夹还没建)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    plt.savefig(output_filename)
    
    # --- 7. 清理 ---
    plt.close(fig) 
    
    print(f"成功！图片已保存: {output_filename}")

def visualize_scenes_plt(scene1_tensor, scene2_tensor, output_filename="scene_comparison_plt.png"):
    """
    使用 Matplotlib 将两个 (B, N, 3) 的点云 Tensor 
    在同一张图的两个子图中展示，并保存到本地。
    """
    
    print(f"开始使用 Matplotlib 处理点云，准备保存到 {output_filename}...")

    # --- 1. 从 Batch 中提取点云并转为 NumPy ---
    # 我们假设您想可视化 batch 中的第一个元素 (B=0)
    if scene1_tensor.shape[0] == 0 or scene2_tensor.shape[0] == 0:
        print("错误：输入的 Tensor 为空 (Batch size 为 0)。")
        return

    # 将 (B, N, 3) -> (N, 3) 并转为 numpy
    scene1_np = scene1_tensor.view(-1,3).detach().cpu().numpy()
    scene2_np = scene2_tensor.view(-1,3).detach().cpu().numpy()
    
    
    
    print(f"  - pcd1 (左图) 包含 {len(scene1_np)} 个点。")
    print(f"  - pcd2 (右图) 包含 {len(scene2_np)} 个点。")

    # --- 2. 创建 Matplotlib 画布和子图 ---
    # 创建一个大小为 (12, 6) 英寸的画布
    fig = plt.figure(figsize=(12, 6))
    
    # --- 3. 绘制第一个子图 (Scene 1) ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # 提取 x, y, z 坐标
    x1, y1, z1 = scene1_np[:, 0], scene1_np[:, 1], scene1_np[:, 2]
    
    # 绘制散点图
    # s=1 表示点的大小
    ax1.scatter(x1, y1, z1, s=1, c='r', label='Scene 1')
    
    # 设置标签和标题
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Scene 1 (Red)')
    ax1.legend()

    # --- 4. 绘制第二个子图 (Scene 2) ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # 提取 x, y, z 坐标
    x2, y2, z2 = scene2_np[:, 0], scene2_np[:, 1], scene2_np[:, 2]
    
    # 绘制散点图
    ax2.scatter(x2, y2, z2, s=1, c='b', label='Scene 2')
    
    # 设置标签和标题
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Scene 2 (Blue)')
    ax2.legend()

    # --- 5. 添加总标题并调整布局 ---
    fig.suptitle('Point Cloud Comparison (Batch[0])', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 为总标题留出空间

    # --- 6. 保存图像 ---
    plt.savefig(output_filename)
    
    # --- 7. 清理 ---
    plt.close(fig)  # 关闭画布，释放内存
    
    print(f"成功！Matplotlib 可视化结果已保存到: {os.path.abspath(output_filename)}")

def load_mesh(path,label_file,use_rgb,use_normal):
    result=dict()
    if label_file == 'labels.instances.align.annotated.v2.ply' or label_file == 'labels.instances.annotated.v2.ply':
        
        plydata = trimesh.load(os.path.join(path,label_file), process=False)
        points = np.array(plydata.vertices)
        instances = util_ply.read_labels(plydata).flatten()
        
        if use_rgb:
            rgbs = np.array(plydata.visual.vertex_colors.tolist())[:,:3]
            points = np.concatenate((points, rgbs / 255.0), axis=1)
            
        if use_normal:
            normal = plydata.vertex_normals[:,:3]
            points = np.concatenate((points, normal), axis=1)
        
        result['points']=points
        result['instances']=instances
    else:
        raise NotImplementedError('')
    return result

class PdiffDatasetGraph(data.Dataset):
    def __init__(self,
                 split,
                 for_train,
                 point_sample_num=192,
                 point_union_num=192*2,
                 root_ScanNet="/home/honsen/tartan/ScanNet/scans/data",
                 json_path="/home/honsen/tartan/ScanNet/scans/training_samples.json",
                 max_edges=-1,
                 atlas_embedding_path=None,
                 text_embedding_path=None,
                 use_aug_scenes=True):
        assert split in ['train_scannet', 'validation_scannet']
        self.for_train = for_train
        self.root_ScanNet = root_ScanNet
        
        self.max_edges = max_edges
        self.use_descriptor = True

        self.num_points = point_sample_num
        self.num_points_union = point_union_num

        self.scene_lists = os.listdir(self.root_ScanNet)

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON 文件未找到: {json_path}")
            
        self.json_path = json_path
        self.atlas_embedding_path = atlas_embedding_path
        self.use_aug_scenes = use_aug_scenes
        self.atlas_embedding_cache = load_embedding_cache(atlas_embedding_path)
        self.text_embedding_path = text_embedding_path
        self.text_embedding_cache = load_text_embedding_cache(text_embedding_path)
        
        self.samples_list = []
        
        self._load_data()

    def _load_data(self):
        """
        私有辅助函数，用于加载和扁平化JSON数据。
        """
        print(f"正在从 {self.json_path} 加载和扁平化数据...")
        
        with open(self.json_path, 'r') as f:
            all_scenes_data = json.load(f)

        num_scenes = 0
        num_subgraphs = 0
        skipped_aug_scenes = 0
        
        for scene_id, scene_data in all_scenes_data.items():
            if (not self.use_aug_scenes) and "__aug" in scene_id:
                skipped_aug_scenes += 1
                continue

            num_scenes += 1
            subgraphs = scene_data.get("subgraphs", [])
            
            for group_id, subgraph_data in enumerate(subgraphs):
                
                sample_id = f"{scene_id}_{group_id}"
                
                sample = {
                    "sample_id": sample_id,          
                    "scene_id": scene_id,            
                    "nodes": subgraph_data.get("nodes", []),
                    "edges": subgraph_data.get("edges", []),
                    "anchor": subgraph_data.get("anchor")
                }
                
                self.samples_list.append(sample)
                num_subgraphs += 1
        
        print(f"加载完成！")
        print(f"  > 总场景数: {num_scenes}")
        if not self.use_aug_scenes:
            print(f"  > 已过滤增强场景数: {skipped_aug_scenes}")
        print(f"  > 总子图数 (训练样本数): {len(self.samples_list)}")
    def __getitem__(self, index):

        sub_graph_sample = self.samples_list[index]

        scene_id = sub_graph_sample["scene_id"]
        sample_id = sub_graph_sample["sample_id"]

        curScenePath = os.path.join(self.root_ScanNet, scene_id)

        # results = load_mesh(curScenePath,'labels.instances.align.annotated.v2.ply',False,False)
        # points = results['points'] # Nx3
        # instances = results['instances']  # N
        
        points = np.load(os.path.join(curScenePath,"sensorsData/points.npy"))
        instances = np.load(os.path.join(curScenePath, "sensorsData/instance.npy"))

        selected_rels = sub_graph_sample["edges"]
        all_nodes_cur = sub_graph_sample["nodes"]
        obj_labels = os.path.join(curScenePath, "sensorsData/object_labels.json")
        
        if not os.path.exists(obj_labels):
            cur_obj_texts = ["unknown" for _ in all_nodes_cur]
        else:
            with open(obj_labels, 'r') as f:
                obj_labels_data = json.load(f)    
            # --- [!!! 关键修改 START !!!] ---
            
            cur_obj_texts = [obj_labels_data[str(i)] for i in all_nodes_cur]

        contrastive_embeddings, contrastive_valid_mask = fetch_text_embeddings(
            self.text_embedding_cache,
            cur_obj_texts,
        )
        if contrastive_embeddings is None:
            contrastive_embeddings, contrastive_valid_mask = fetch_object_embeddings(
                self.atlas_embedding_cache,
                scene_id,
                all_nodes_cur,
                sample_id=sample_id,
            )
        
        # 1. 获取 "实例ID" (e.g., 22)
        #    这是 ScanNet 的原始掩码 ID，不是 obj_points 的索引
        anchor_instance_id = sub_graph_sample["anchor"]
        
        # 2. 在节点列表中查找该 "实例ID" 的 "本地索引" (e.g., 1)
        #    all_nodes_cur 列表的顺序与 data_preparation 中
        #    构建 obj_points 的顺序是一致的。
        #    因此 .index() 查找 22 在 [17, 22, 5] 中的位置，得到 1
        try:
            # 这就是 obj_points 中对应的真实行索引
            anchor_index = all_nodes_cur.index(anchor_instance_id) 
        except ValueError:
            # 健壮性检查：以防万一锚点不在节点列表中
            print(f"警告: 锚点 {anchor_instance_id} 不在场景 {scene_id} 的节点列表 {all_nodes_cur} 中。")
            # 采取一个回退措施，比如使用第一个节点作为锚点
            anchor_index = 0
            
        # --- [!!! 关键修改 END !!!] ---

        obj_points, obj_points_spatial, edge_indices, descriptor = \
            self.data_preparation(points, instances, self.num_points, self.num_points_union, all_nodes_cur, selected_rels,
                                  padding=0.2)
        
        while (len(edge_indices) == 0) and self.for_train:
            index = np.random.randint(self.__len__())
            obj_points, obj_points_spatial, edge_indices, descriptor, anchor_index, \
                 cur_obj_texts, contrastive_embeddings, contrastive_valid_mask = self.__getitem__(index)

        return obj_points, obj_points_spatial, edge_indices, descriptor, anchor_index, \
                 cur_obj_texts, contrastive_embeddings, contrastive_valid_mask



    def limit_dict_size(self, input_dict, max_keys=70):
        import random
        current_keys = list(input_dict.keys())
        if len(current_keys) > max_keys:
            excess = len(current_keys) - max_keys
            # 随机选择要删除的键
            keys_to_remove = random.sample(current_keys, excess)
            for key in keys_to_remove:
                del input_dict[key]
        return input_dict


    def norm_tensor(self, points):
        assert points.ndim == 2
        assert points.shape[1] == 3
        centroid = torch.mean(points, dim=0)  # N, 3
        points -= centroid  # n, 3, npts
        return points

    def zero_mean(self, point):
        mean = torch.mean(point, dim=0)
        point -= mean.unsqueeze(0)
        return point


    def __len__(self):
        return len(self.samples_list)

    def relsToInstance(self, selected_rels):

        vertices = set()  # 用集合存储顶点（自动去重）

        for u, v in selected_rels:
            vertices.add(u)
            vertices.add(v)

        # vertices = [int(v) for v in vertices]

        # 转换为排序后的列表（可选）
        vertices = sorted(vertices)

        return vertices

    def subs_rel_idx(self, arrangeidx, selected_rels):

        selected_rels = np.array(selected_rels)

        for key in arrangeidx:
            selected_rels[np.where(selected_rels==key)]=arrangeidx[key]

        return selected_rels

    def data_preparation(self, points, instances, num_points, num_points_union, all_nodes_cur, selected_rels, scene_id="",
                         padding=0.2,
                        ): 

        num_objects = len(all_nodes_cur)
        dim_point = points.shape[-1]  # xyz

        instances_box, label_node = dict(), []
        obj_points = torch.zeros([num_objects, num_points, dim_point])
        
        obj_points_spatial = torch.zeros([num_objects, num_points, dim_point])
        # obj_points1 = np.zeros([num_objects, num_points, dim_point])
        descriptor = torch.zeros([num_objects, 11])

        arrangeidx = {}

        for i, instance_id in enumerate(all_nodes_cur):
            arrangeidx[instance_id] = i
            # get node point
            obj_pointset = points[np.where(instances == instance_id)[0]]
            min_box = np.min(obj_pointset[:, :3], 0) - padding  # padding object boxes to contain all object points
            max_box = np.max(obj_pointset[:, :3], 0) + padding
            instances_box[instance_id] = (min_box, max_box)  # this two points can decide a 3D boundingbox
            choice = np.random.choice(len(obj_pointset), num_points, replace=True)
            obj_pointset = obj_pointset[choice, :]
            # obj_points1[i] = obj_pointset
            descriptor[i] = op_utils.gen_descriptor(torch.from_numpy(obj_pointset)[:, :3])
            obj_pointset1 = torch.from_numpy(obj_pointset.astype(np.float32))
            obj_points_spatial[i] = obj_pointset1.clone()
            obj_pointset1[:, :3] = self.zero_mean(obj_pointset1[:, :3])
            obj_points[i] = obj_pointset1

        # visualize_scenes_plt(obj_points, obj_points_spatial,"/home/hyc/hyc_work/sceneGraph/SGG_DIR/qwe.png")
        
        # rel_points = list()
        # for e in range(len(selected_rels)):
        #     edge = selected_rels[e]
        #     instance1 = int(edge[0])
        #     instance2 = int(edge[1])

        #     mask1 = (instances == instance1).astype(np.int32) * 1
        #     mask2 = (instances == instance2).astype(np.int32) * 2
        #     mask_ = np.expand_dims(mask1 + mask2, 1)
        #     bbox1 = instances_box[instance1]
        #     bbox2 = instances_box[instance2]
        #     min_box = np.minimum(bbox1[0], bbox2[0])
        #     max_box = np.maximum(bbox1[1], bbox2[1])
        #     filter_mask = (points[:, 0] > min_box[0]) * (points[:, 0] < max_box[0]) \
        #                   * (points[:, 1] > min_box[1]) * (points[:, 1] < max_box[1]) \
        #                   * (points[:, 2] > min_box[2]) * (points[:, 2] < max_box[2])

        #     # add with context, to distingush the different object's points
        #     points4d = np.concatenate([points, mask_], 1)

        #     pointset = points4d[np.where(filter_mask > 0)[0], :]
        #     choice = np.random.choice(len(pointset), num_points_union, replace=True)
        #     pointset = pointset[choice, :]
        #     pointset = torch.from_numpy(pointset.astype(np.float32))
        #     pointset[:, :3] = self.zero_mean(pointset[:, :3])
        #     rel_points.append(pointset)

        # if len(rel_points) > 0:
        #     rel_points = torch.stack(rel_points, 0)
        # else:
        #     rel_points = torch.tensor([])

        edge_indices = self.subs_rel_idx(arrangeidx, selected_rels)
        edge_indices = torch.tensor(edge_indices, dtype=torch.long)

        return obj_points, obj_points_spatial, edge_indices, descriptor

    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.count = n # 连通分量数量

        def find(self, i):
            if self.parent[i] != i:
                self.parent[i] = self.find(self.parent[i])
            return self.parent[i]

        def union(self, i, j):
            root_i = self.find(i)
            root_j = self.find(j)
            if root_i != root_j:
                self.parent[root_i] = root_j
                self.count -= 1
                return True
            return False

    def apply_elastic_distortion(self, points, granularity=0.2, magnitude=0.05):
        """
        基于 RBF (Radial Basis Function) 的弹性形变
        """
        # 1. 计算场景尺寸 (用于确定 sigma)
        min_coord = np.min(points, axis=0)
        max_coord = np.max(points, axis=0)
        dimensions = max_coord - min_coord
        max_dim = np.max(dimensions)

        # 2. 设定控制中心数量
        # 原代码硬编码了 num_centers=10，这里我们可以根据 granularity 动态设定
        # granularity 越小，控制点应该越多，变形越局部
        # 假设 granularity 代表“每个控制点的平均影响范围”
        num_centers = int(max(5, np.prod(dimensions) / (granularity**3)))
        # 为了防止过大导致计算慢，可以设个上限，比如 20-50
        num_centers = min(num_centers, 50) 
        
        # 3. 随机选取控制中心 (Anchors)
        if len(points) > num_centers:
            centers_indices = np.random.choice(len(points), num_centers, replace=False)
            centers = points[centers_indices]
        else:
            centers = points # 点太少就全员作为控制点

        # 4. 生成控制中心的随机位移
        # (num_centers, 3)
        vectors = np.random.normal(0, magnitude, (len(centers), 3))

        # 5. 计算距离矩阵 (N_total, num_centers)
        # P_i - C_j
        dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)

        # 6. 计算高斯权重
        # sigma 决定了形变的平滑程度。granularity 可以用来控制 sigma
        sigma = max_dim * granularity 
        weights = np.exp(- (dists ** 2) / (2 * sigma ** 2)) # (N, num_centers)

        # 7. 加权求和得到每个点的位移
        deformation = np.dot(weights, vectors)

        # 8. 归一化权重 (关键修复)
        # 原代码这里加了 1e-8 防止除零，这很好
        # 但 RBF 插值通常不需要除以权重和，除非做的是“加权平均插值”。
        # 如果只是叠加位移场，直接叠加即可。但为了防止多个中心重叠处位移过大，除以权重和是更稳健的做法。
        weight_sum = np.sum(weights, axis=1, keepdims=True) + 1e-8
        deformation = deformation / weight_sum

        return points + deformation

    # ==========================================
    # [辅助函数] 边索引增广 (保持连通)
    # ==========================================
    def sample_connected_subgraph(self, arrangeidx, selected_rels, keep_ratio=0.5):
        """
        从原图中采样一个子图，保证：
        1. 连通性 (通过生成树保证)
        2. 边数量约为原图的 keep_ratio
        
        Returns:
            view2_indices: View 2 保留的边在原列表 selected_rels 中的索引
            overlap_indices: View 1 和 View 2 重合的边的索引 (相对于 selected_rels)
        """
        num_nodes = len(arrangeidx)
        num_edges = len(selected_rels)
        
        if num_edges == 0:
            return torch.tensor([]), torch.tensor([])

        # 1. 映射 node id 到 0..N-1
        # edges_internal: list of (u_idx, v_idx, original_edge_index)
        edges_internal = []
        for idx, (u, v) in enumerate(selected_rels):
            if u in arrangeidx and v in arrangeidx:
                edges_internal.append((arrangeidx[u], arrangeidx[v], idx))
        
        # 2. 随机打乱边，以便生成随机的生成树
        np.random.shuffle(edges_internal)
        
        uf = self.UnionFind(num_nodes)
        spanning_tree_indices = []
        remaining_edges = []
        
        # 3. Kruskal 算法构建生成树 (保证连通性)
        for u, v, original_idx in edges_internal:
            if uf.union(u, v):
                spanning_tree_indices.append(original_idx)
            else:
                remaining_edges.append(original_idx)
        
        # 4. 补充边以达到目标比例 (keep_ratio)
        current_count = len(spanning_tree_indices)
        target_count = int(num_edges * keep_ratio)
        
        # 如果生成树的边已经超过目标 (对于稀疏图常见)，则至少保留生成树
        num_to_add = max(0, target_count - current_count)
        
        # 从剩余边中随机选择
        if num_to_add > 0 and len(remaining_edges) > 0:
            # remaining_edges 已经被 shuffle 过了，直接切片
            added_indices = remaining_edges[:num_to_add]
            final_indices_list = spanning_tree_indices + added_indices
        else:
            final_indices_list = spanning_tree_indices

        # 排序索引以便后续处理
        final_indices_list = sorted(final_indices_list)
        
        # 5. 构建结果
        # View 2 的边索引
        edge_indices_view2 = torch.tensor(final_indices_list, dtype=torch.long)
        
        # View 1 假设使用的是全集 (0, 1, ..., num_edges-1)
        # 因此，Overlap 就是 View 2 选中的那些边
        overlap_indices = edge_indices_view2.clone()
        
        return edge_indices_view2, overlap_indices

    def data_preparation_multi_view(self, points, instances, num_points, num_points_union, all_nodes_cur, selected_rels, scene_id="",
                                  padding=0.2):

        num_objects = len(all_nodes_cur)
        dim_point = points.shape[-1]  # xyz

        instances_box = dict()
                
        # --- View 1 的数据容器 ---
        obj_points = torch.zeros([num_objects, num_points, dim_point])
        obj_points_spatial = torch.zeros([num_objects, num_points, dim_point])
        descriptor = torch.zeros([num_objects, 11])

        # --- View 2 的数据容器 ---
        obj_points_view2 = torch.zeros([num_objects, num_points, dim_point])
        descriptor_view2 = torch.zeros([num_objects, 11])

        arrangeidx = {}

        # ====================================================================
        # [View 2 全局增广参数]
        # ====================================================================
        # 1. 全局非等比缩放系数
        aug_scale = np.random.uniform(0.5, 2, (1, 3)) 
        # 2. 噪声强度
        aug_noise_sigma = 0.005
        # 3. 弹性形变强度
        aug_elastic_mag = 0.05

        view2_raw_points_buffer = [] 

        for i, instance_id in enumerate(all_nodes_cur):
            arrangeidx[instance_id] = i
            
            raw_indices = np.where(instances == instance_id)[0]
            obj_pointset_raw = points[raw_indices]
            
            # --- View 1 生成 ---
            min_box = np.min(obj_pointset_raw[:, :3], 0) - padding
            max_box = np.max(obj_pointset_raw[:, :3], 0) + padding
            instances_box[instance_id] = (min_box, max_box)
            
            choice1 = np.random.choice(len(obj_pointset_raw), num_points, replace=True)
            obj_pointset_v1 = obj_pointset_raw[choice1, :]
            
            descriptor[i] = op_utils.gen_descriptor(torch.from_numpy(obj_pointset_v1)[:, :3])
            
            obj_pts_tensor_v1 = torch.from_numpy(obj_pointset_v1.astype(np.float32))
            obj_points_spatial[i] = obj_pts_tensor_v1.clone()
            
            # [关键修改] View 1: 零均值化 + 归一化
            # 1. 零均值化
            obj_pts_tensor_v1[:, :3] = self.zero_mean(obj_pts_tensor_v1[:, :3])
            
            # 2. 归一化到单位球 (最大半径=1)
            # 计算最大半径 m = max(sqrt(x^2+y^2+z^2))
            m = torch.max(torch.sqrt(torch.sum(obj_pts_tensor_v1[:, :3] ** 2, dim=1)))
            if m < 1e-6: m = 1.0 # 防止除零
            obj_pts_tensor_v1[:, :3] = obj_pts_tensor_v1[:, :3] / m
            
            obj_points[i] = obj_pts_tensor_v1

            # --- View 2 数据收集 ---
            choice2 = np.random.choice(len(obj_pointset_raw), num_points, replace=True)
            obj_pointset_v2_raw = obj_pointset_raw[choice2, :].copy()
            view2_raw_points_buffer.append(obj_pointset_v2_raw)

        # ====================================================================
        # [View 2 全局增广应用]
        # ====================================================================
        
        if len(view2_raw_points_buffer) > 0:
            all_points_v2 = np.concatenate(view2_raw_points_buffer, axis=0)
            
            # 增广操作
            all_points_v2[:, :3] = all_points_v2[:, :3] * aug_scale
            if np.random.rand() < 1: 
                all_points_v2[:, :3] = self.apply_elastic_distortion(
                    all_points_v2[:, :3], magnitude=aug_elastic_mag
                )
            noise = np.random.normal(0, aug_noise_sigma, all_points_v2[:, :3].shape)
            all_points_v2[:, :3] += noise
            
            # 拆分并处理 View 2
            curr_idx = 0
            for i in range(num_objects):
                obj_pts_aug = all_points_v2[curr_idx : curr_idx + num_points, :]
                curr_idx += num_points
                
                descriptor_view2[i] = op_utils.gen_descriptor(torch.from_numpy(obj_pts_aug)[:, :3])
                
                obj_pts_tensor_v2 = torch.from_numpy(obj_pts_aug.astype(np.float32))
                
                # [关键修改] View 2: 零均值化 + 归一化
                # 1. 零均值化
                obj_pts_tensor_v2[:, :3] = self.zero_mean(obj_pts_tensor_v2[:, :3])
                
                # 2. 归一化到单位球
                m = torch.max(torch.sqrt(torch.sum(obj_pts_tensor_v2[:, :3] ** 2, dim=1)))
                if m < 1e-6: m = 1.0
                obj_pts_tensor_v2[:, :3] = obj_pts_tensor_v2[:, :3] / m
                
                obj_points_view2[i] = obj_pts_tensor_v2
        
        # -----------------------------------------------------------
        # 处理 spatial (保持原样: 需要保留相对位置信息，所以通常不做个体归一化，只做零均值)
        # 如果 spatial 也需要归一化(例如全局归一化)，请告诉我
        # -----------------------------------------------------------
        B_dim, N_dim, _ = obj_points_spatial.shape
        obj_points_spatial = obj_points_spatial.view(B_dim * N_dim, -1)
        obj_points_spatial[:, :3] = self.zero_mean(obj_points_spatial[:, :3]) 
        obj_points_spatial = obj_points_spatial.view(B_dim, N_dim, -1)
       

        # ... (后续边索引处理代码保持不变) ...

        # visualize_scenes_plt(obj_points, obj_points_spatial,"/home/hyc/hyc_work/sceneGraph/SGG_DIR/qwe1.png")
        # visualize_scenes_plt(obj_points, obj_points_view2,"/home/hyc/hyc_work/sceneGraph/SGG_DIR/qwe2.png")
     
        edge_indices_view1 = self.subs_rel_idx(arrangeidx, selected_rels)
        edge_indices_view1 = torch.tensor(edge_indices_view1, dtype=torch.long)
        
        return (obj_points, obj_points_spatial, edge_indices_view1, descriptor, 
                obj_points_view2, descriptor_view2)
