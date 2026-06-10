import networkx as nx
from typing import Set

def find_minimal_anchor_set_undirected(graph: nx.DiGraph) -> Set:
    """
    在 *无向图* 的假设下，找到一个最小的“锚点”顶点集。

    这个假设是：
    1. 图中的关系(边)是双向的（无向的）。
    2. 在任何一个“连通分量”中，只需要知道一个锚点，
       就可以通过边推断该分量中所有其他节点的信息。

    因此，这个函数找到所有的连通分量，并从每个分量中
    选择一个代表性节点作为锚点。

    Args:
        graph: 一个 networkx.DiGraph 对象。
               该函数会将其视为无向图进行处理。

    Returns:
        一个包含最小锚点集顶点名称的Python set。
    """
    
    # 1. 找到所有的连通分量。
    #    首先，我们需要一个图的无向视图。
    try:
        undirected_graph = graph.to_undirected()
    except Exception as e:
        print(f"错误：无法将图转换为无向图: {e}")
        return set()

    # 2. 找到无向图的连通分量
    #    connected_components 返回一个迭代器，
    #    每个元素是一个包含该分量所有节点名称的 set。
    #    例如: [ {'lamp', 'book', 'table', 'floor', 'room', 'chair'}, {'window'} ]
    try:
        connected_components = list(nx.connected_components(undirected_graph))
    except Exception as e:
        print(f"错误：在查找连通分量时出错: {e}")
        return set()

    anchor_nodes = set()
    
    # 3. 遍历每一个连通分量
    for component in connected_components:
        
        # 4. 从每个非空的连通分量中，任选一个顶点作为锚点
        if component:
            # 使用 next(iter(set)) 从集合中安全地获取一个元素
            anchor_node = next(iter(component))
            anchor_nodes.add(anchor_node)

    return anchor_nodes, connected_components

# --- 示例 ---
if __name__ == '__main__':
    # 1. 创建一个有向图 (场景图)
    G = nx.DiGraph()
    
    # 示例1：一个复杂的场景 (与之前相同)
    G.add_edges_from([
        ("lamp", "table"),   
        ("book", "table"),   
        ("table", "floor"),  
        ("floor", "room"),   
        ("room", "floor"),   # 这条边在无向图中是多余的，但不影响
        ("chair", "floor")   
    ])
    
    # 也添加一个孤立节点
    G.add_node("window") 

    # 查找最小锚点集 (在无向图假设下)
    anchors = find_minimal_anchor_set_undirected(G)
    
    print(f"--- 示例图 1 (无向图假设) ---")
    print(f"图的节点: {list(G.nodes)}")
    print(f"找到的最小锚点集 (大小 {len(anchors)}): {anchors}")
    print("\n分析：在这个假设下，'lamp', 'book', 'table', 'floor', 'room', 'chair'")
    print("      都通过'table'和'floor'连接在一起，属于 *同一个* 连通分量。")
    print("      因此我们只需要从这个大分量中任选一个节点。")
    print("      'window' 是一个独立的连通分量，必须选它自己。")
    print("      所以最终的锚点集大小为 2。")

    
    # 示例2：一个只有环的图
    G_ring = nx.DiGraph()
    G_ring.add_edges_from([
        ("A", "B"),
        ("B", "C"),
        ("C", "A")
    ])

    anchors_ring = find_minimal_anchor_set_undirected(G_ring)
    
    print(f"\n--- 示例图 2 (无向图假设) ---")
    print(f"图的节点: {list(G_ring.nodes)}")
    print(f"找到的最小锚点集 (大小 {len(anchors_ring)}): {anchors_ring}")
    print("分析：A, B, C 构成一个连通分量。")
    print("      因此，我们只需要从 {'A', 'B', 'C'} 中任选一个作为锚点。")

    # 示例3：两个独立的组件
    G_disjoint = nx.DiGraph()
    G_disjoint.add_edges_from([
        ("A", "B"),
        ("C", "D")
    ])

    anchors_disjoint = find_minimal_anchor_set_undirected(G_disjoint)
    print(f"\n--- 示例图 3 (无向图假设) ---")
    print(f"图的节点: {list(G_disjoint.nodes)}")
    print(f"找到的最小锚点集 (大小 {len(anchors_disjoint)}): {anchors_disjoint}")
    print("分析：{'A', 'B'} 是一个连通分量，{'C', 'D'} 是另一个。")
    print("      因此，我们从 {'A', 'B'} 中选一个，从 {'C', 'D'} 中选一个。")
    print("      所以最终的锚点集大小为 2。")