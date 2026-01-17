import os
import json
import argparse
import torch
try:
    import clip
except ModuleNotFoundError:
    clip = None
from tqdm import tqdm

def generate_scannet_embeddings(root_dir: str, output_path: str, model_name: str, prompt_template: str):
    # ================= 配置区域 =================
    # 你的 ScanNet 数据根目录
    root_dir = root_dir
    
    # 结果保存路径
    output_path = output_path
    
    # CLIP 模型版本
    model_name = model_name
    
    # 提示词模板
    prompt_template = prompt_template
    # ===========================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在使用设备: {device}")

    if clip is None:
        raise ModuleNotFoundError(
            "Python package 'clip' not found. Install OpenAI CLIP first, e.g.: "
            "python3 -m pip install git+https://github.com/openai/CLIP.git"
        )

    # 1. 收集所有场景中出现过的唯一类别名称
    print("正在扫描所有场景目录以收集类别标签...")
    unique_labels = set()
    
    # 获取所有场景文件夹
    if not os.path.exists(root_dir):
        print(f"错误: 目录 {root_dir} 不存在")
        return

    scene_dirs = sorted(os.listdir(root_dir))
    
    for scene_id in tqdm(scene_dirs, desc="Scanning Metadata"):
        json_path = os.path.join(root_dir, scene_id, "sensorsData", "object_labels.json")
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # data 的格式是 {"instance_id": "class_name", ...}
                    # 我们只需要 value (class_name)
                    for class_name in data.values():
                        if class_name: # 确保不是空字符串
                            unique_labels.add(class_name)
            except Exception as e:
                print(f"读取 {json_path} 失败: {e}")

    unique_labels_list = sorted(list(unique_labels))
    print(f"扫描完成! 共发现 {len(unique_labels_list)} 个唯一语义类别。")
    print(f"示例类别: {unique_labels_list[:5]}")

    # 2. 加载 CLIP 模型
    print(f"正在加载 CLIP 模型: {model_name}...")
    model, _ = clip.load(model_name, device=device)
    model.eval()

    # 3. 生成 Embeddings
    print("开始生成文本 Embeddings...")
    
    # 构造 Prompt
    # 注意：CLIP 处理长列表可能会 OOM，如果类别非常多(>10000)，建议分 Batch 处理
    # ScanNet 的类别通常在几百个左右，直接处理即可
    prompts = [prompt_template.format(label) for label in unique_labels_list]
    
    text_tokens = clip.tokenize(prompts).to(device)

    result_dict = {}

    with torch.no_grad():
        # 编码
        text_features = model.encode_text(text_tokens)
        
        # 归一化 (非常重要，用于余弦相似度计算)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # 转回 CPU 方便保存
        text_features = text_features.cpu()

        # 4. 构建字典 { 'class_name': tensor }
        for i, label in enumerate(unique_labels_list):
            result_dict[label] = text_features[i]

    # 5. 保存结果
    print(f"正在保存结果到 {output_path}...")
    torch.save(result_dict, output_path)
    
    print("完成！")
    print(f"字典包含 {len(result_dict)} 个键值对。")
    print(f"Embedding 形状: {list(result_dict.values())[0].shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default=os.environ.get(
            "SCANNET_SCANS_ROOT",
            "/data0/jiangxiangwei/Diff-SGG/data/ScanNet_merged_v2_20k_uniform_copy/scans",
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.environ.get(
            "SCANNET_TEXT_EMB_PATH",
            "/data0/jiangxiangwei/Diff-SGG/outputs/scannet_text_embeddings.pt",
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=os.environ.get("CLIP_MODEL_NAME", "ViT-B/32"),
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=os.environ.get("CLIP_PROMPT_TEMPLATE", "a point cloud of {}."),
    )
    args = parser.parse_args()

    generate_scannet_embeddings(
        root_dir=args.root_dir,
        output_path=args.output_path,
        model_name=args.model_name,
        prompt_template=args.prompt_template,
    )