import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple
import random
import itertools
import open3d as o3d
import matplotlib.pyplot as plt

# ==================================================================
# 您的原始代码 (未修改)
# ==================================================================

def get_object_centroids(point_cloud: np.ndarray, object_labels: np.ndarray) -> dict:
    """
    计算每个物体的质心。
    """
    centroids = {}
    unique_labels = np.unique(object_labels)
    
    for label in unique_labels:
        if label == 0:  # 忽略背景
            continue
        
        object_mask = (object_labels == label)
        if np.any(object_mask):
            centroids[label] = np.mean(point_cloud[object_mask], axis=0)
            
    return centroids

def get_cluster_centroids(clusters_map: dict, node_centroids: dict) -> dict:
    """计算当前所有簇的质心"""
    cluster_centroids = {}
    for cluster_id, node_ids in clusters_map.items():
        # 从原始节点质心字典中提取质心
        centroids_in_cluster = np.array([node_centroids[node_id] for node_id in node_ids])
        cluster_centroids[cluster_id] = np.mean(centroids_in_cluster, axis=0)
    return cluster_centroids

def get_nearest_neighbor_cluster(target_cluster_id: int, 
                                 cluster_centroids: dict) -> int:
    """找到距离目标簇最近的邻居簇ID"""
    target_centroid = cluster_centroids[target_cluster_id]
    
    min_dist = float('inf')
    neighbor_id = -1
    
    for cluster_id, centroid in cluster_centroids.items():
        if cluster_id == target_cluster_id:
            continue
            
        dist = np.linalg.norm(target_centroid - centroid)
        if dist < min_dist:
            min_dist = dist
            neighbor_id = cluster_id
            
    return neighbor_id


def build_compact_subgraphs(point_cloud: np.ndarray, 
                            object_labels: np.ndarray, 
                            min_subgraph_size: int = 5,
                            node_centroids: Dict[int, np.ndarray] = None) -> List[List[int]]:
    """
    (推荐) 使用层次聚类 + 迭代合并来创建紧凑且满足最小尺寸的子图。
    
    (我稍微修改了签名，允许传入已计算的质心)
    """
    
    # 1. 计算所有物体的质心
    if node_centroids is None:
        # node_centroids: {object_id: centroid_array}
        node_centroids = get_object_centroids(point_cloud, object_labels)
    
    if len(node_centroids) < min_subgraph_size:
        if len(node_centroids) > 0:
            print(f"警告: 场景只有 {len(node_centroids)} 个物体，小于最小尺寸 {min_subgraph_size}。"
                  "将所有物体作为一个子图返回。")
            return [list(node_centroids.keys())]
        else:
            print("警告: 场景中没有检测到物体。")
            return []

    node_ids = list(node_centroids.keys())
    # (N, 3) 质心数组，顺序与 node_ids 对应
    centroid_matrix = np.array(list(node_centroids.values()))
    N = len(node_ids)

    # --- 阶段一: 层次聚类 ---
    
    # 计算初始“理想”簇数
    # max(1, ...) 确保 N=9, K_min=5 时，M=1 而不是 0
    M_initial_clusters = max(1, N // min_subgraph_size)
    
    # 使用 'ward' 方法进行聚类，'ward' 方法天生追求最小化方差，使簇内紧凑
    Z = linkage(centroid_matrix, method='ward')
    
    # 'fcluster' 根据 'maxclust' 标准将Z切割成 M_initial_clusters 个簇
    # cluster_labels 是一个 (N,) 数组，值为 1, 2, ..., M
    cluster_labels = fcluster(Z, t=M_initial_clusters, criterion='maxclust')

    # 将聚类结果转换为更易于操作的字典
    # clusters_map: {cluster_id: [node_id, ...]}
    clusters_map = {}
    for i, node_id in enumerate(node_ids):
        label = cluster_labels[i]
        if label not in clusters_map:
            clusters_map[label] = []
        clusters_map[label].append(node_id)
    
    while True:
        # 1. 找到最小的簇
        smallest_cluster_id = -1
        smallest_size = float('inf')
        
        for cluster_id, nodes in clusters_map.items():
            if len(nodes) < smallest_size:
                smallest_size = len(nodes)
                smallest_cluster_id = cluster_id
        
        # 2. 检查是否满足约束
        if smallest_size >= min_subgraph_size:
            break
            
        # 3. 检查是否只剩一个簇（但仍小于K_min，即 N < K_min 的情况）
        if len(clusters_map) <= 1:
            break
        
        # A. 计算所有当前簇的质心
        cluster_centroids = get_cluster_centroids(clusters_map, node_centroids)
        
        # B. 找到最小簇的最近邻居
        neighbor_cluster_id = get_nearest_neighbor_cluster(smallest_cluster_id, 
                                                           cluster_centroids)
        
        # C. 执行合并
        # 将 smallest_cluster_id 的所有节点合并到 neighbor_cluster_id
        nodes_to_merge = clusters_map.pop(smallest_cluster_id)
        clusters_map[neighbor_cluster_id].extend(nodes_to_merge)

    # 5. 格式化输出
    final_subgraphs = list(clusters_map.values())
    return final_subgraphs

# ==================================================================
# 新增功能 1: 构建子图内部的边
# ==================================================================

def build_subgraph_edges(
    subgraphs: List[List[int]], 
    min_removal_ratio: float = 0.4,  # 最小移除比例 (例如: 0.1 表示 10%)
    max_removal_ratio: float = 0.8   # 最大移除比例 (例如: 0.4 表示 40%)
) -> List[Tuple[int, int]]:
    """
    为每个子图构建一个“几乎”全连接的有向图。
    
    策略:
    1. 为子图中的 N 个节点创建一个 N*(N-1) 的完全有向图。
    2. 从这个图中随机移除一定比例的边。
    3. 移除的比例在 [min_removal_ratio, max_removal_ratio] 之间随机确定。
    
    Args:
        subgraphs: 子图划分列表，如 [[1, 2, 5], [3, 4]]
        min_removal_ratio: 要移除的边的最小比例 (0.0 到 1.0 之间)。
        max_removal_ratio: 要移除的边的最大比例 (0.0 到 1.0 之间)。
        
    Returns:
        一个包含所有 (源, 目标) 边元组的列表。
    """
    all_edges = []
    
    # 确保比例值在 [0.0, 1.0] 范围内
    min_ratio = max(0.0, min(1.0, min_removal_ratio))
    max_ratio = max(0.0, min(1.0, max_removal_ratio))
    
    # 确保 min_ratio <= max_ratio
    if min_ratio > max_ratio:
        min_ratio, max_ratio = max_ratio, min_ratio # 如果输反了，自动纠正
        print(f"Warning: min_removal_ratio > max_removal_ratio. Swapping values.")

    for subgraph_nodes in subgraphs:
        num_nodes = len(subgraph_nodes)
        
        # 如果节点数少于2，无法创建边
        if num_nodes < 2:
            continue
            
        # 1. 创建一个完全有向图
        possible_edges = list(itertools.permutations(subgraph_nodes, 2))
        num_possible_edges = len(possible_edges) # N * (N-1)
        
        if num_possible_edges == 0:
            continue
            
        # --- 核心修改 ---
        # 2. 确定一个在此范围内的随机移除比例
        current_removal_ratio = random.uniform(min_ratio, max_ratio)
        
        # 3. 计算要移除和保留的边数
        num_to_remove = int(num_possible_edges * current_removal_ratio)
        num_to_keep = num_possible_edges - num_to_remove
        # ------------------
        
        # 4. 随机抽样以“移除”边
        #    (通过“随机选择要保留的边”来实现“随机移除”)
        if num_to_keep > 0:
            edges_to_keep = random.sample(possible_edges, num_to_keep)
        else:
            edges_to_keep = []
        
        all_edges.extend(edges_to_keep)
        
    return all_edges

# ==================================================================
# 新增功能 2: 3D 可视化
# ==================================================================

def get_rotation_matrix_between_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """ 
    计算从 vec1 旋转到 vec2 的旋转矩阵 (兼容 O3D 的 axis-angle vector 输入)。
    """
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    dot_prod = np.dot(vec1, vec2)
    
    # 1. 向量几乎相同
    if np.allclose(dot_prod, 1.0):
        return np.identity(3)
        
    # 2. 向量几乎相反 (180度)
    if np.allclose(dot_prod, -1.0):
        # 找到一个与 vec1 正交的轴
        axis = np.array([0.0, 1.0, 0.0])
        if np.allclose(np.abs(vec1), [0.0, 1.0, 0.0]):
            axis = np.array([1.0, 0.0, 0.0])
            
        axis = np.cross(vec1, axis)
        axis = axis / np.linalg.norm(axis)
        
        # --- (修改点 1) ---
        # 合并 轴 和 角度(pi)
        rotation_vector = axis * np.pi
        return o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)

    # 3. 正常情况
    axis = np.cross(vec1, vec2)
    axis_norm = axis / np.linalg.norm(axis)
    angle = np.arccos(dot_prod) # 角度
    
    # --- (修改点 2) ---
    # 合并 轴(axis_norm) 和 角度(angle)
    rotation_vector = axis_norm * angle
    return o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)

def visualize_subgraphs_and_edges(
    point_cloud: np.ndarray,
    object_labels: np.ndarray,
    node_centroids: Dict[int, np.ndarray],
    subgraphs: List[List[int]],
    edges: List[Tuple[int, int]],
    node_sphere_radius: float = 0.1,
    arrow_cylinder_radius: float = 0.03,
    arrow_head_radius: float = 0.06,
    arrow_head_length: float = 0.15
):
    """
    使用 Open3D 可视化点云、子图节点和它们之间的边。
    """
    
    # --- 1. 准备点云 (PCD) ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    # 为点云着色 (按物体实例)
    unique_labels = np.unique(object_labels)
    max_label = np.max(unique_labels)
    if max_label == 0: max_label = 1 # 避免除以零
        
    # 使用 'jet' colormap 为每个实例上色
    # cmap(0) -> 蓝色, cmap(1) -> 红色
    cmap = plt.cm.get_cmap("jet") 
    
    # 创建一个 label -> color 的映射
    label_to_color = {}
    for label in unique_labels:
        if label == 0:
            label_to_color[0] = [0.8, 0.8, 0.8] # 背景为灰色
        else:
            # 归一化到 [0, 1]
            norm_label = (label % max_label) / float(max_label) 
            label_to_color[label] = cmap(norm_label)[:3] # (R, G, B)
            
    pcd_colors = np.array([label_to_color.get(l, [0,0,0]) for l in object_labels])
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    
    # --- 2. 准备子图颜色 ---
    num_subgraphs = len(subgraphs)
    # 使用 'tab10' colormap 为每个子图上色
    subgraph_cmap = plt.cm.get_cmap("tab10") 
    
    # 创建一个 node_id -> subgraph_color 的映射
    node_to_subgraph_color = {}
    for i, subgraph_nodes in enumerate(subgraphs):
        color = subgraph_cmap(i / float(num_subgraphs))[:3]
        for node_id in subgraph_nodes:
            node_to_subgraph_color[node_id] = color

    geometries = [pcd]

    # --- 3. 创建节点球体 (Nodes) ---
    for node_id, centroid in node_centroids.items():
        # 默认为黑色，如果它属于一个子图，则使用子图颜色
        color = node_to_subgraph_color.get(node_id, [0.1, 0.1, 0.1])
        
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=node_sphere_radius)
        sphere.paint_uniform_color(color)
        sphere.translate(centroid)
        geometries.append(sphere)

    # --- 4. 创建边 (Edges) ---
    # Open3D 绘制有向箭头比较繁琐，我们用 LineSet 并用颜色区分
    # 或者为每条边创建一个箭头几何体
    
    for u, v in edges:
        # 确保两个节点都存在
        if u not in node_centroids or v not in node_centroids:
            continue
            
        u_pos = node_centroids[u]
        v_pos = node_centroids[v]
        
        # 使用源节点 u 的子图颜色
        color = node_to_subgraph_color.get(u, [0.1, 0.1, 0.1])
        
        # 创建一个箭头
        vec = v_pos - u_pos
        length = np.linalg.norm(vec)
        if length < 1e-6: continue # 避免零长度
            
        vec_norm = vec / length
        
        # ==================== (修改开始) ====================
        
        # 1. 计算旋转矩阵
        
        # Open3D 默认沿 Z 轴 [0,0,1] 创建几何体
        # 我们在代码中先将 Z 轴转到了 X 轴 [1,0,0] (绕 Y 轴转 90 度)
        R_Z_to_X = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
        
        # 现在我们使用新辅助函数计算从 X 轴 [1,0,0] 到目标向量 vec_norm 的旋转
        R_X_to_Vec = get_rotation_matrix_between_vectors(
            np.array([1.0, 0.0, 0.0]), 
            vec_norm
        )
        
        # 箭头圆柱体
        cyl_len = length - arrow_head_length
        if cyl_len > 0:
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=arrow_cylinder_radius, 
                height=cyl_len
            )
            # 旋转圆柱体
            cylinder.rotate(R_Z_to_X, center=(0,0,0))     # 先从 Z 转到 X
            cylinder.rotate(R_X_to_Vec, center=(0,0,0))   # 再从 X 转到 vec_norm
            
            # (平移逻辑不变)
            cylinder.translate(u_pos + vec_norm * (cyl_len / 2.0))
            cylinder.paint_uniform_color(color)
            geometries.append(cylinder)

        # 箭头头部
        cone = o3d.geometry.TriangleMesh.create_cone(
            radius=arrow_head_radius, 
            height=arrow_head_length
        )
        # 旋转圆锥体
        cone.rotate(R_Z_to_X, center=(0,0,0))   # 先从 Z 转到 X
        cone.rotate(R_X_to_Vec, center=(0,0,0)) # 再从 X 转到 vec_norm
        
        # (平移逻辑不变)
        cone.translate(v_pos - vec_norm * (arrow_head_length / 2.0))
        cone.paint_uniform_color(color)
        geometries.append(cone)
        
        # ==================== (修改结束) ====================

    # --- 5. 启动可视化窗口 ---
    print("\n启动 Open3D 可视化窗口...")
    print("  - 点云按 [实例ID] 着色。")
    print("  - 球体和箭头按 [子图ID] 着色。")
    o3d.visualization.draw_geometries(geometries)


# ==================================================================
# 主执行程序
# ==================================================================



if __name__ == "__main__":
    
    # --- 1. 加载数据 ---
    # !! 警告: 请确保以下路径在您的系统上有效
    # SCENE_PATH = "/home/honsen/tartan/ScanNet/scans/data/scene0000_00/sensorsData"
    # 您提供的路径。如果失败，将使用虚拟数据。
    
    try:
        # object_labels = np.load(f"{SCENE_PATH}/instance.npy")
        # point_cloud = np.load(f"{SCENE_PATH}/points.npy")
        object_labels = np.load("/home/honsen/tartan/ScanNet/scans/data/scene0000_00/sensorsData/instance.npy")
        point_cloud = np.load("/home/honsen/tartan/ScanNet/scans/data/scene0000_00/sensorsData/points.npy")
        print(f"成功加载 Sence0000_00 数据: {len(point_cloud)} 个点。")

    except FileNotFoundError:
        print("错误: 无法在指定路径加载ScanNet数据。")
        


    K_MIN = 4  # 每个子图的最小尺寸
    EDGES_TO_REMOVE = 6 # 每个子图移除的边数

    print("\n--- 开始计算质心 ---")
    # 0. (新增) 单独计算质心，因为可视化和子图构建都需要它
    node_centroids = get_object_centroids(point_cloud, object_labels)
    
    if not node_centroids:
         print("错误: 未找到物体质心，无法继续。")
    else:
        print(f"--- 找到 {len(node_centroids)} 个物体 ---")

        print("\n--- 开始构建子图 ---")
        # 1. (来自用户) 构建子图
        #    我们传入 node_centroids 以避免重复计算
        subgraphs = build_compact_subgraphs(point_cloud, 
                                            object_labels, 
                                            min_subgraph_size=K_MIN,
                                            node_centroids=node_centroids)
        
        print(f"\n--- 最终划分为 {len(subgraphs)} 个子图 ---")
        for i, sg in enumerate(subgraphs):
            print(f"  子图 {i+1} (大小 {len(sg)}): {sg[:10]}...") # 最多显示前10个
        print("\n--- 开始为子图构建边 ---")

        edges = build_subgraph_edges(subgraphs, 
                                     point_cloud, 
                                     object_labels, 
                                     edges_to_remove=EDGES_TO_REMOVE)
        
        print(f"--- 总共生成 {len(edges)} 条边 ---")
        if edges:
             print(f"  示例边: {edges[:5]}...")

        print("\n--- 准备 3D 可视化 ---")
        # 3. (新功能) 可视化
        visualize_subgraphs_and_edges(
            point_cloud,
            object_labels,
            node_centroids,
            subgraphs,
            edges
        )