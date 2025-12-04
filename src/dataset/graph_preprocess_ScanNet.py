import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple
import random
import itertools
import open3d as o3d
import matplotlib.pyplot as plt
from seg_graph import get_object_centroids, build_compact_subgraphs, build_subgraph_edges
import os
import networkx as nx
from typing import Set
from find_mini_v_cover import find_minimal_anchor_set_undirected
import json
from tqdm import tqdm

class NumpyEncoder(json.JSONEncoder):
    """
    一个自定义的 JSON 编码器，用于处理 NumPy 的数据类型。
    当 json.dump 遇到它不认识的类型时，会调用 default() 方法。
    """
    def default(self, obj):
        # 如果对象是 NumPy 整数类型...
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            # ...将其转换为 Python 原生 int
            return int(obj)
        
        # 如果对象是 NumPy 浮点数类型...
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
                              np.float64)):
            # ...将其转换为 Python 原生 float
            return float(obj)
            
        # 如果对象是 NumPy 数组...
        elif isinstance(obj, np.ndarray):
            # ...将其转换为 Python 列表
            return obj.tolist()
            
        # 其他类型，交给基类处理
        return json.JSONEncoder.default(self, obj)

def filter_small_instances(points: np.ndarray, labels: np.ndarray, min_points: int = 1024):
    """
    过滤掉点数少于 min_points 的实例。
    
    Args:
        points: (N, 3) 点云数据
        labels: (N,) 实例标签
        min_points: 最小点数阈值
        
    Returns:
        points, labels: 过滤后的数据
    """
    # 统计每个实例标签出现的次数
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # 找到点数 >= min_points 的标签
    valid_labels = unique_labels[counts >= min_points]
    
    # 打印一下过滤情况（可选，用于调试，嫌烦可以注释掉）
    # print(f"原始实例数: {len(unique_labels)}, 过滤后实例数: {len(valid_labels)}")
    
    if len(valid_labels) != len(unique_labels):
        print("-----pointNum < 1024--------")
    
    # 创建掩码：只保留标签在 valid_labels 中的点
    # np.isin 用于检查 labels 中的元素是否在 valid_labels 中
    mask = np.isin(labels, valid_labels)
    
    return points[mask], labels[mask]

def filter_invalid_labels(points: np.ndarray, labels: np.ndarray):
   
    mask = (labels != -1)

    filtered_points = points[mask]
    filtered_labels = labels[mask]
    
    return filtered_points, filtered_labels

def build_training_sample(scene_id: str, 
                          all_edges: List[Tuple[int, int]], 
                          connected_components: List[Set[int]]):
    """
    根据您的JSON结构需求，为单个场景生成训练样本字典。
    
    Args:
        scene_id: 场景ID (例如: "scene0000_00")
        all_edges: 该场景中 *所有* (源, 目标) 边的列表
        connected_components: 连通分量列表，每个分量是一个包含节点ID的集合
                          
    Returns:
        一个字典，包含该场景的格式化数据。
    """
    
    # 1. 创建一个从 node_id -> component_index 的快速查找字典
    #    这使我们能够 O(1) 找到任何节点属于哪个子图
    node_to_component_id: Dict[int, int] = {}
    for i, component_set in enumerate(connected_components):
        for node in component_set:
            node_to_component_id[node] = i
            
    # 2. 为每个连通子图初始化一个边列表
    component_edge_lists: List[List[Tuple[int, int]]] = [[] for _ in range(len(connected_components))]
    
    # 3. 遍历 all_edges 一次，将每条边分配到对应的子图中
    for u, v in all_edges:
        component_id_u = node_to_component_id.get(u)
        component_id_v = node_to_component_id.get(v)
        
        # 确保 u 和 v 都在同一个连通分量中
        if component_id_u is not None and component_id_u == component_id_v:
            component_edge_lists[component_id_u].append((u, v))
            
    # 4. 构建最终的JSON结构
    scene_subgraphs_data = []
    for i, component_set in enumerate(connected_components):
        if not component_set:  # 跳过空的连通分量
            continue
            
        # 将节点从集合转为排序列表，以便在JSON中保持一致
        component_nodes_list = sorted(list(component_set))
        
        # 按照您的要求，为该子图随机选择一个锚点
        anchor_node = random.choice(component_nodes_list)
        
        # 获取我们预先计算好的该子图的边
        subgraph_edges = component_edge_lists[i]
        
        # 构建子图条目
        subgraph_entry = {
            "nodes": component_nodes_list,  # 连通子图的节点列表
            "edges": subgraph_edges,        # 连通子图的边列表
            "anchor": anchor_node           # 连通子图的锚点
        }
        scene_subgraphs_data.append(subgraph_entry)
        
    # 5. 构建该场景的根字典
    #    这符合您需求的第 2, 3, 4 点
    scene_data = {
        "scene_id": scene_id,
        "subgraphs": scene_subgraphs_data
    }
    
    return scene_data

def preprocess_scan_scene(scan_id: str, data_path: str):
    """
    预处理ScanNet数据，生成子图和边信息。
    
    参数:
        scan_id (str): ScanNet扫描的ID。
        data_path (str): 数据存储路径。
    """
    try:
        K_MIN = 3  # 每个子图的最小尺寸
        # 加载点云数据
        point_cloud = np.load(f"{data_path}/sensorsData/points.npy")  # Nx3
        object_labels = np.load(f"{data_path}/sensorsData/instance.npy")  # N
        
        point_cloud, object_labels = filter_invalid_labels(point_cloud, object_labels)
        
        point_cloud, object_labels = filter_small_instances(point_cloud, object_labels, min_points=512)
        
        node_centroids = get_object_centroids(point_cloud, object_labels)
        
        subgraphs = build_compact_subgraphs(point_cloud, 
                                            object_labels, 
                                            min_subgraph_size=K_MIN,
                                            node_centroids=node_centroids)
        
        edges = build_subgraph_edges(subgraphs)

        Graph4Filter = nx.DiGraph()
        Graph4Filter.add_edges_from(edges)
        _, connected_components = find_minimal_anchor_set_undirected(Graph4Filter)
        if not connected_components:
            print(f"场景 {scan_id} 未找到连通分量，跳过。")
            return None
            
        # --- 新增步骤: 生成训练样本 ---
        scene_data = build_training_sample(scan_id, edges, connected_components)
        
        return scene_data
        
    except Exception as e:
        print(f"处理场景 {scan_id} 时发生错误: {e}")
        return None

def preprocess_scanNet_main(ScanNet_dirs):
    
    scan_lists = os.listdir(ScanNet_dirs)

    all_scenes_data = {}
    
    for scan in tqdm(scan_lists):
        data_path = os.path.join(ScanNet_dirs, scan)
        scene_data = preprocess_scan_scene(scan, data_path)

        # 如果场景处理成功，将其添加到大字典中
        if scene_data:
            scan_id = scene_data["scene_id"]
            all_scenes_data[scan_id] = scene_data
            print(f"--- 成功处理并添加场景: {scan_id} ---")
    
    # 将JSON文件保存在 ScanNet_dirs 的上一级目录中
    output_directory = os.path.dirname(ScanNet_dirs)
    output_filepath = os.path.join(output_directory, "training_samples2.json")
    
    if not all_scenes_data:
        print("未处理任何有效场景，不生成JSON文件。")
        return

    print(f"\n... 正在将 {len(all_scenes_data)} 个场景的数据写入到 {output_filepath} ...")
    
    try:
        with open(output_filepath, 'w') as f:
            # indent=2 使JSON文件可读性更强
            json.dump(all_scenes_data, f, indent=2, cls=NumpyEncoder)
        print("JSON 文件保存成功！")
        
    except Exception as e:
        print(f"保存JSON文件时出错: {e}")
    
if __name__ == '__main__':
    ScanNet_dirs = '/home/honsen/tartan/ScanNet/scans'  # 替换为实际路径
    preprocess_scanNet_main(ScanNet_dirs)
    # scene0345_01