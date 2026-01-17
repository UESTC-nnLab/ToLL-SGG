import os
import json
import argparse
from collections import Counter
import matplotlib.pyplot as plt

def count_category_distribution(scan_root, output_subdir="sensorsData", save_result=True):
    """
    统计 ScanNet 数据集中所有 object_labels.json 的类别分布
    """
    # 检查根目录是否存在
    if not os.path.exists(scan_root):
        print(f"Error: 路径 {scan_root} 不存在。")
        return

    # 获取所有场景文件夹
    scenes = sorted([d for d in os.listdir(scan_root) if os.path.isdir(os.path.join(scan_root, d))])
    print(f"正在扫描 {scan_root} 下的 {len(scenes)} 个场景...")

    # 初始化计数器
    total_counter = Counter()
    processed_files = 0
    missing_files = 0

    # 遍历每个场景
    for scene_id in scenes:
        # 构建 object_labels.json 的完整路径
        # 路径格式: /root/scene_id/sensorsData/object_labels.json
        json_path = os.path.join(scan_root, scene_id, output_subdir, "object_labels.json")

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                    # data 是一个字典: {"0": "wall", "1": "chair", ...}
                    # 我们只需要统计 values (类别名称)
                    labels = list(data.values())
                    
                    # 更新计数器
                    total_counter.update(labels)
                    processed_files += 1
            except Exception as e:
                print(f"读取错误 {scene_id}: {e}")
        else:
            missing_files += 1

    # --- 输出统计结果 ---
    print("\n" + "="*50)
    print(f"统计完成 Summary")
    print("="*50)
    print(f"成功处理文件数: {processed_files}")
    print(f"缺失文件数:   {missing_files}")
    print(f"发现不同类别数: {len(total_counter)}")
    print("-" * 50)
    print(f"{'Category (类别)':<30} | {'Count (数量)':<10}")
    print("-" * 50)

    # 按数量从多到少排序输出
    sorted_counts = total_counter.most_common()
    for category, count in sorted_counts:
        print(f"{category:<30} | {count:<10}")

    # --- 保存结果到文件 (可选) ---
    if save_result:
        result_file = "/home/honsen/honsen/SceneGraph/generateScannet_vlm/ScanNet/construct3DSSG/category_distribution.json"
        with open(result_file, 'w') as f:
            # 将 Counter 对象转换为字典保存
            json.dump(dict(sorted_counts), f, indent=4)
        print("-" * 50)
        print(f"详细分布数据已保存至当前目录下的: {result_file}")

    return sorted_counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计 ScanNet Object Labels 分布")
    
    # 默认路径设置为你提供的路径
    default_path = "/home/honsen/tartan/ScanNet/scans"
    
    parser.add_argument("--scan_root", type=str, default=default_path, 
                        help=f"ScanNet scans 根目录 (默认: {default_path})")
    parser.add_argument("--subdir", type=str, default="sensorsData", 
                        help="存放 json 文件的子目录名称 (默认: sensorsData)")

    args = parser.parse_args()

    # 执行统计
    count_category_distribution(args.scan_root, args.subdir)