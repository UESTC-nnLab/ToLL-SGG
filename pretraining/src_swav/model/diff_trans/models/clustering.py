import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import os
def cluster_and_visualize(tensor_data: torch.Tensor, 
                          n_clusters: int, 
                          title_prefix: str = "Data", 
                          save_path: str = "cluster_plot.png",
                          use_pca: bool = True,      # 新增：是否使用 PCA 降噪
                          pca_dim: int = 64):        # 新增：PCA 降维后的维度
    """
    针对对比学习/SwAV 特征优化的聚类可视化函数。
    包含 L2 归一化和 PCA 降维步骤。
    """
    
    print(f"\n--- Processing {title_prefix} ---")
    
    # 1. Data Preparation
    if tensor_data.is_cuda:
        data_np = tensor_data.detach().cpu().numpy()
    else:
        data_np = tensor_data.numpy()

    if data_np.dtype != np.float64:
        data_np = data_np.astype(np.float64)

    # --- [关键修正 1] L2 归一化 ---
    # 这将 K-Means (欧式距离) 转化为 Spherical K-Means (余弦相似度)
    # 对于 SwAV/SimCLR 这种基于超球面特征的模型，这是必须的！
    print("Applying L2 Normalization (Crucial for Contrastive Embeddings)...")
    data_np = normalize(data_np, norm='l2', axis=1)

    # --- [关键修正 2] PCA 降维 (可选但推荐) ---
    # 去除高频噪声，保留主成分，解决维度灾难，并加速后续步骤
    if use_pca and data_np.shape[1] > pca_dim:
        print(f"Applying PCA to reduce dimensions from {data_np.shape[1]} to {pca_dim}...")
        pca = PCA(n_components=pca_dim, random_state=42)
        data_for_clustering = pca.fit_transform(data_np)
    else:
        data_for_clustering = data_np

    # 2. Clustering (K-Means)
    print(f"Starting K-Means clustering for {data_np.shape[0]} samples into {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, 
                    random_state=42, 
                    n_init=10) 
    
    # 使用处理过(归一化+PCA)的数据进行聚类
    labels = kmeans.fit_predict(data_for_clustering)
    
    print(f"Clustering complete. Assigned data to {n_clusters} clusters.")

    # 3. Dimensionality Reduction (t-SNE)
    print("Starting t-SNE dimensionality reduction...")
    
    # 注意：t-SNE 通常也建议先用 PCA 初始化，或者输入 PCA 后的数据
    tsne = TSNE(n_components=2, 
                perplexity=35.0, 
                max_iter=1000, 
                random_state=42, 
                init='pca',       # [优化] 使用 PCA 初始化通常效果更好
                learning_rate='auto',
                n_jobs=-1)
    
    # 可视化输入：可以使用 PCA 后的数据，也可以用归一化后的原数据
    # 这里使用 data_for_clustering (PCA后) 既快又能保留主要结构
    data_2d = tsne.fit_transform(data_for_clustering)
    print("t-SNE reduction complete.")

    # 4. Visualization (Matplotlib)
    plt.figure(figsize=(12, 10))
    
    cmap = plt.get_cmap('Spectral', n_clusters) # 或者 'tab10', 'jet' 等
    
    scatter = plt.scatter(data_2d[:, 0], 
                          data_2d[:, 1], 
                          c=labels, 
                          cmap=cmap,
                          s=15,          # 稍微调大一点点
                          alpha=0.6)     # 透明度

    plt.title(f"{title_prefix} - Spherical K-Means Results\n(Pre-normalized + PCA | K={n_clusters})", fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    
    # Colorbar logic (unchanged)
    cbar = plt.colorbar(scatter, ticks=np.arange(n_clusters))
    cbar.set_label('Cluster ID')
    if n_clusters > 20:
        tick_skip = n_clusters // 10
        ticks = np.arange(0, n_clusters, tick_skip)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks)

    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 5. Save
    plt.savefig(save_path, dpi=150) # 增加 dpi 让图片更清晰
    print(f"Visualization saved to {save_path}")
    plt.close()

    # 6. Save Labels
    base, _ = os.path.splitext(save_path)
    labels_save_path = f"{base}_labels.npy"
    np.save(labels_save_path, labels)
    
    return labels

def evaluate_and_plot_clustering(feature_tensor: torch.Tensor, 
                                 label_tensor: torch.Tensor, 
                                 save_path: str = "clustering_analysis.png",
                                 ignore_zero_label: bool = True,
                                 max_samples_for_sil: int = 10000,
                                 class_names: list = None,
                                 show_axis_labels: bool = True,
                                 metric_prefix: str = "val"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import KMeans
    from sklearn.metrics import confusion_matrix, silhouette_score, normalized_mutual_info_score, adjusted_rand_score
    from sklearn.preprocessing import normalize
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    import torch
    import os

    """
    【多合一功能】计算聚类指标 (NMI, ARI, ACC, Silhouette) 并绘制对齐后的混淆矩阵。
    
    修改点：
    - 增加了 label_to_idx 映射，强制将不连续的 GT 标签映射为 0~K-1 的连续索引。
    - 解决了混淆矩阵因标签索引不匹配而出现全空行/列（幽灵行）的问题。
    """
    
    print(f"\n--- Starting Comprehensive Clustering Evaluation ({metric_prefix}) ---")

    # ==========================
    # 1. 数据预处理 (Data Preprocessing)
    # ==========================
    # 处理 One-hot / Logits
    if label_tensor.dim() > 1:
        if label_tensor.shape[1] == 1: 
            label_tensor = label_tensor.squeeze(1)
        else: 
            label_tensor = torch.argmax(label_tensor, dim=1)

    # 转 Numpy
    if feature_tensor.is_cuda:
        feats_np = feature_tensor.detach().cpu().numpy()
        labels_np = label_tensor.detach().cpu().numpy()
    else:
        feats_np = feature_tensor.numpy()
        labels_np = label_tensor.numpy()

    labels_np = labels_np.astype(int)

    # 过滤背景类 (Label 0)
    if ignore_zero_label:
        mask = labels_np != 0
        feats_np = feats_np[mask]
        labels_np = labels_np[mask]
        if len(labels_np) == 0:
            print("Error: No data left after filtering label 0.")
            return {}

    n_samples = len(labels_np)
    # 获取真实类别数 (K-Means 的 K)
    unique_labels = np.unique(labels_np)
    num_classes = len(unique_labels)
    print(f"Data ready: {n_samples} samples, {num_classes} unique classes.")

    # ============================================================
    # 【关键修改】标签重映射 (Label Remapping)
    # 目的：将任意物理 ID (如 1, 5, 10...) 映射为连续索引 (0, 1, 2...)
    # 这样能确保 GT 和 K-Means (输出 0~K-1) 的标签空间完全一致
    # ============================================================
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    mapped_labels_np = np.array([label_to_idx[x] for x in labels_np])
    
    # 特征归一化 (对 Cosine Similarity 聚类至关重要)
    feats_norm = normalize(feats_np, norm='l2', axis=1)

    metrics = {}

    # ==========================
    # 2. 计算几何指标 (Silhouette)
    # ==========================
    # 轮廓系数复杂度高，需要降采样
    if n_samples > max_samples_for_sil:
        indices = np.random.choice(n_samples, max_samples_for_sil, replace=False)
        feats_sil = feats_norm[indices]
        # 注意：Silhouette 只需要聚类分组，用 mapped_labels 效果一样
        labels_sil = mapped_labels_np[indices] 
    else:
        feats_sil = feats_norm
        labels_sil = mapped_labels_np
    
    try:
        # 这里计算的是 GT label 在特征空间的分离度
        sil_score = silhouette_score(feats_sil, labels_sil, metric='euclidean')
        metrics[f"{metric_prefix}/silhouette"] = sil_score
    except Exception as e:
        print(f"Skipping Silhouette due to error: {e}")
        sil_score = -1

    # ==========================
    # 3. 核心步骤：K-Means 聚类
    # ==========================
    print(f"Running K-Means (K={num_classes})...")
    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(feats_norm)

    # ==========================
    # 4. 计算语义对齐指标 (NMI, ARI)
    # ==========================
    # 注意：这里使用 mapped_labels_np
    nmi_score = normalized_mutual_info_score(mapped_labels_np, pred_labels)
    ari_score = adjusted_rand_score(mapped_labels_np, pred_labels)
    
    metrics[f"{metric_prefix}/nmi"] = nmi_score
    metrics[f"{metric_prefix}/ari"] = ari_score

    # ==========================
    # 5. 混淆矩阵与最佳对齐 (Hungarian Matching)
    # ==========================
    # 计算原始混淆矩阵
    # 【关键】因为 mapped_labels_np 和 pred_labels 都是 0~K-1，
    # 所以生成的 cm 必然是 K x K 的方阵，不会有多余的空行。
    cm = confusion_matrix(mapped_labels_np, pred_labels)
    
    # 匈牙利算法对齐 (Maximize Trace)
    row_ind, col_ind = linear_sum_assignment(-cm)
    cm_aligned = cm[:, col_ind] # 重排列

    # 计算 ACC (Purity)
    acc = cm_aligned.trace() / np.sum(cm_aligned)
    metrics[f"{metric_prefix}/acc"] = acc

    print(f"--- Metrics Summary ---")
    print(f"ACC (Aligned): {acc:.2%}")
    print(f"NMI:           {nmi_score:.4f}")
    print(f"ARI:           {ari_score:.4f}")
    print(f"Silhouette:    {sil_score:.4f}")

    # ==========================
    # 6. 可视化绘图 (Visualization)
    # ==========================
    # 归一化矩阵 (按行，显示 Recall)
    cm_norm = cm_aligned.astype('float') / cm_aligned.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(16, 14)) # 针对160类，画布要大
    
    # 设置坐标轴标签
    # 注意：虽然矩阵内部用了 mapped index，但画图时我们依然想看原始的物理名字/ID
    # unique_labels 是排好序的原始 ID，正好对应 mapped index 的 0, 1, 2...
    if class_names is None:
        tick_labels = [str(l) for l in unique_labels]
    else:
        # 尝试匹配 class names
        try:
            # unique_labels 里的值是原始 ID，用它去 class_names 里取名字
            tick_labels = [class_names[i] for i in unique_labels]
        except:
            tick_labels = [str(l) for l in unique_labels]

    # --- 针对 160 类的特殊设置 ---
    # 如果不想显示坐标轴文字（太密），设为空列表
    if not show_axis_labels or len(unique_labels) > 100:
        # 策略：如果超过100类，且用户没强制要求显示，我们就只显示每隔5个的刻度，或者干脆不显示
        # 这里选择不显示具体文字，只显示刻度线，保持清爽
        xticklabels = False 
        yticklabels = False
        print("Notice: Hiding axis labels due to high class count (>100) for clarity.")
    else:
        xticklabels = tick_labels
        yticklabels = tick_labels

    sns.heatmap(cm_norm, 
                annot=False,    # <--- 【关键】不显示数值，只显示颜色
                fmt='.2f', 
                cmap='Blues',   # 或者是 'viridis', 'magma' 等
                xticklabels=xticklabels, 
                yticklabels=yticklabels,
                square=True,
                cbar_kws={"shrink": .8, "label": "Recall (Row Normalized)"})

    plt.xlabel('Predicted Cluster (Aligned)')
    plt.ylabel('Ground Truth Label')
    plt.title(f'Cluster Confusion Matrix (N={num_classes})\nACC:{acc:.2%} | NMI:{nmi_score:.3f} | ARI:{ari_score:.3f}', fontsize=16)
    
    # 保存目录检查
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200) # dpi 调高一点，防止像素模糊
    print(f"Visualization saved to {save_path}")
    plt.close()

    return metrics

def visualize_aesthetic_gt(feature_tensor: torch.Tensor, 
                           label_tensor: torch.Tensor, 
                           title_prefix: str = "Data", 
                           save_path: str = "gt_tsne_aesthetic.png",
                           use_pca: bool = True,
                           pca_dim: int = 64,
                           ignore_zero_label: bool = False):
    """
    [美化版] 使用真值标签 (Ground Truth) 对特征进行 t-SNE 可视化。
    特点：无坐标轴、无网格、Colorbar无数值（纯净模式）。
    """
    
    print(f"\n--- Processing {title_prefix} (Aesthetic Mode) ---")
    
    # 1. Data Preparation (Tensor -> Numpy)
    if feature_tensor.is_cuda:
        feats_np = feature_tensor.detach().cpu().numpy()
        labels_np = label_tensor.detach().cpu().numpy()
    else:
        feats_np = feature_tensor.numpy()
        labels_np = label_tensor.numpy()

    # 处理 One-hot / Logits
    if labels_np.ndim > 1:
        labels_np = np.argmax(labels_np, axis=1)
    
    labels_np = labels_np.astype(int)

    # 过滤 Label 0
    if ignore_zero_label:
        mask = labels_np != 0
        feats_np = feats_np[mask]
        labels_np = labels_np[mask]
        if len(labels_np) == 0:
            print("Error: No data left after filtering label 0!")
            return

    # 2. Preprocessing
    feats_np = normalize(feats_np, norm='l2', axis=1)

    if use_pca and feats_np.shape[1] > pca_dim:
        real_pca_dim = min(pca_dim, feats_np.shape[0])
        pca = PCA(n_components=real_pca_dim, random_state=42)
        feats_reduced = pca.fit_transform(feats_np)
    else:
        feats_reduced = feats_np

    # 3. t-SNE
    print("Running t-SNE...")
    perp = min(30.0, max(5.0, feats_reduced.shape[0] / 10.0))
    tsne = TSNE(n_components=2, perplexity=perp, max_iter=1000, 
                random_state=42, init='pca', learning_rate='auto', n_jobs=-1)
    data_2d = tsne.fit_transform(feats_reduced)

    # 4. Visualization (Clean Style)
    # 使用正方形画布，视觉更平衡
    plt.figure(figsize=(10, 9))
    
    unique_labels = np.unique(labels_np)
    cmap_name = 'tab20' if len(unique_labels) <= 20 else 'nipy_spectral' # jet或者nipy_spectral适合类别特别多的情况
    cmap = plt.get_cmap(cmap_name)
    
    scatter = plt.scatter(data_2d[:, 0], 
                          data_2d[:, 1], 
                          c=labels_np, 
                          cmap=cmap, 
                          s=5, # 稍微调大点大小
                          alpha=0.7,
                          edgecolors='none') # 去掉点的边缘线，看起来更柔和

    # --- [关键修改] 移除坐标轴、刻度和边框 ---
    plt.axis('off') 
    
    # 标题可选，如果追求极致纯净可以注释掉下面这行
    plt.title(f"{title_prefix} Distribution", fontsize=18, pad=20)

    # --- [关键修改] Colorbar 只展示颜色条，不展示数值 ---
    cbar = plt.colorbar(scatter, fraction=0.046, pad=0.04)
    cbar.set_ticks([])       # 移除刻度线和数值
    cbar.set_ticklabels([])  # 确保没有文字标签
    cbar.outline.set_visible(False) # 可选：移除colorbar的边框线，看起来更极简

    # 5. Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight') # dpi调高到300更清晰
    print(f"Saved aesthetic visualization to {save_path}")
    plt.close()

def visualize_with_gt(feature_tensor: torch.Tensor, 
                      label_tensor: torch.Tensor, 
                      title_prefix: str = "Data", 
                      save_path: str = "gt_tsne.png",
                      use_pca: bool = True,
                      pca_dim: int = 64,
                      class_names: list = None,
                      target_classes: list = None): # [修改] 替换了原来的 ignore_zero_label
    """
    使用真值标签 (Ground Truth) 对特征进行 t-SNE 可视化。
    
    Args:
        feature_tensor: (N, D) 特征
        label_tensor: (N,) 标签
        target_classes: (list, optional) 指定需要显示的类别 ID 列表，例如 [2, 3, 4, 5]。
                        如果为 None，默认显示所有类别，但会自动过滤掉 label 0 (背景/无语义类)。
    """
    
    print(f"\n--- Processing {title_prefix} with Ground Truth Labels ---")
    
    # 1. Data Preparation (Tensor -> Numpy)
    if feature_tensor.is_cuda:
        feats_np = feature_tensor.detach().cpu().numpy()
        labels_np = label_tensor.detach().cpu().numpy()
    else:
        feats_np = feature_tensor.numpy()
        labels_np = label_tensor.numpy()

    # 处理 One-hot / Logits 形状 (N, C) -> (N,)
    if labels_np.ndim > 1:
        print(f"Detected Multi-dim labels {labels_np.shape}, converting to indices via argmax...")
        labels_np = np.argmax(labels_np, axis=1)
    
    # 确保标签是整数
    labels_np = labels_np.astype(int)

    # --- [核心修改] 类别过滤逻辑 ---
    original_count = len(labels_np)
    
    if target_classes is not None:
        # 情况 A: 用户指定了具体的类别列表
        print(f"Filtering data for specific classes: {target_classes}")
        # np.isin 用于判断 labels_np 中的元素是否在 target_classes 中
        mask = np.isin(labels_np, target_classes)
        filter_msg = f"Target Classes: {target_classes}"
    else:
        # 情况 B: 默认情况，只过滤掉 0
        print("No target classes specified. Defaulting to filter out label 0 (Background).")
        mask = labels_np != 0
        filter_msg = "Filtered Label 0"

    # 应用过滤
    feats_np = feats_np[mask]
    labels_np = labels_np[mask]
    filtered_count = len(labels_np)
    
    print(f"Removed {original_count - filtered_count} samples. Remaining: {filtered_count}")
    
    if filtered_count < 2:
        print("Error: Not enough data left after filtering! Skipping visualization.")
        return

    num_classes = len(np.unique(labels_np))
    print(f"Found {num_classes} unique classes in the current batch (after filtering).")

    # 2. Feature Preprocessing (L2 Norm + PCA)
    print("Applying L2 Normalization...")
    feats_np = normalize(feats_np, norm='l2', axis=1)

    if use_pca and feats_np.shape[1] > pca_dim:
        real_pca_dim = min(pca_dim, feats_np.shape[0])
        print(f"Applying PCA ({feats_np.shape[1]} -> {real_pca_dim})...")
        pca = PCA(n_components=real_pca_dim, random_state=42)
        feats_reduced = pca.fit_transform(feats_np)
    else:
        feats_reduced = feats_np

    # 3. t-SNE
    print("Starting t-SNE...")
    # 动态调整 perplexity，防止样本过少报错
    n_samples = feats_reduced.shape[0]
    perp = min(30.0, max(5.0, n_samples / 10.0))
    if n_samples < 5: # 极端情况
        perp = 1.0
        
    tsne = TSNE(n_components=2, 
                perplexity=perp, 
                max_iter=1000, 
                random_state=42, 
                init='pca', 
                learning_rate='auto',
                n_jobs=-1)
    data_2d = tsne.fit_transform(feats_reduced)
    print("t-SNE complete.")

    # 4. Visualization
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels_np)
    # 根据类别数量自动选择配色方案
    cmap_name = 'tab20' if len(unique_labels) <= 20 else 'nipy_spectral'
    cmap = plt.get_cmap(cmap_name)
    
    # 绘制散点图
    scatter = plt.scatter(data_2d[:, 0], 
                          data_2d[:, 1], 
                          c=labels_np, 
                          cmap=cmap, 
                          s=30, # 稍微加大一点点点
                          alpha=0.7)

     # --- [关键修改] 移除坐标轴、刻度和边框 ---
    plt.axis('off') 
    
    # 标题可选，如果追求极致纯净可以注释掉下面这行
    plt.title(f"{title_prefix} Distribution", fontsize=18, pad=20)

    # --- [关键修改] Colorbar 只展示颜色条，不展示数值 ---
    cbar = plt.colorbar(scatter, fraction=0.046, pad=0.04)
    cbar.set_ticks([])       # 移除刻度线和数值
    cbar.set_ticklabels([])  # 确保没有文字标签
    cbar.outline.set_visible(False) # 可选：移除colorbar的边框线，看起来更极简

    # Colorbar 设置
    # 注意：如果过滤后剩下的类别很少（比如只有 [2, 5]），Colorbar 应该只显示这两个刻度
    # cbar = plt.colorbar(scatter, ticks=unique_labels)
    # cbar.set_label('Ground Truth Class ID')
    
    # 如果提供了 class_names，尝试映射名字
    if class_names is not None:
        try:
            # 这里的 i 是真实的 label id
            mapped_names = []
            for i in unique_labels:
                if i < len(class_names):
                    mapped_names.append(f"{i}: {class_names[i]}")
                else:
                    mapped_names.append(f"Class {i}")
            cbar.set_ticklabels(mapped_names)
        except Exception as e:
            print(f"Warning: Failed to map class names: {e}")

    # 5. Save
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()


import matplotlib
# [关键] 必须在导入 pyplot 之前设置 Agg 后端，否则服务器会报 "no display name" 错误
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import os
import torch
import numpy as np
import open3d as o3d
def visualize_single_pc_matplotlib(pc, save_path):
    """
    使用 Matplotlib 将点云保存为 PNG 图片 (服务器友好)
    Args:
        pc: (N, 3) numpy array
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制散点
    # c=pc[:, 2]: 根据 Z 轴高度上色，方便看清立体结构
    # s=1: 点的大小，点云密集时调小一点
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc[:, 2], cmap='viridis', s=2, alpha=0.8)
    
    # 移除坐标轴刻度，让图片更干净
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # [可选] 设置一个固定的视角 (Elevation, Azimuth)
    # 如果发现物体是躺着的，可以调整这些参数
    ax.view_init(elev=30, azim=45) 
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close() # [关键] 必须关闭，释放内存

def analyze_kmeans_clusters(
    feature_tensor,      # (N, 512)
    raw_point_clouds,    # (N, 3, 1024) or (N, 1024, 3)
    n_clusters=20,       
    samples_per_cluster=5, 
    save_dir="cluster_analysis"
):
    """
    执行 K-Means 聚类，并同时保存 .ply (3D文件) 和 .png (预览图)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 聚类
    print(f"Running KMeans on {feature_tensor.shape[0]} samples...")
    # SwAV 特征通常需要 L2 归一化后再聚类 (Spherical K-Means)
    feats_np = torch.nn.functional.normalize(feature_tensor, dim=1).cpu().numpy()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(feats_np)
    labels = kmeans.labels_
    
    # 2. 对每个簇进行采样和保存
    print(f"Analyzing {n_clusters} clusters -> saving results to '{save_dir}/' ...")
    
    for c_id in range(n_clusters):
        # 创建子文件夹，分类存放 (可选，防止文件太多太乱)
        cluster_dir = os.path.join(save_dir, f"cluster_{c_id:02d}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        # 找到属于该簇的所有样本索引
        indices = np.where(labels == c_id)[0]
        
        if len(indices) == 0:
            continue
            
        # 随机采样
        n_samples = min(len(indices), samples_per_cluster)
        sample_indices = np.random.choice(indices, n_samples, replace=False)
        
        # print(f"Cluster {c_id}: Found {len(indices)} samples. Saving {n_samples} examples.")
        
        for i, idx in enumerate(sample_indices):
            # --- 数据准备 ---
            pc = raw_point_clouds[idx]
            if isinstance(pc, torch.Tensor):
                pc = pc.detach().cpu().numpy()
            
            # 确保形状为 (N_points, 3)
            # 假设原始是 (3, 1024) -> 转置为 (1024, 3)
            if pc.shape[0] == 3 and pc.shape[1] > 3: 
                pc = pc.T
                
            # --- 保存方式 1: Open3D .PLY 文件 (用于下载后用 MeshLab 查看) ---
            ply_path = os.path.join(cluster_dir, f"sample_{i}_idx_{idx}.ply")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)
            o3d.io.write_point_cloud(ply_path, pcd)
            
            # --- 保存方式 2: Matplotlib .PNG 图片 (用于快速预览) ---
            png_path = os.path.join(cluster_dir, f"sample_{i}_idx_{idx}.png")
            visualize_single_pc_matplotlib(pc, png_path)

    print(f"\n[Done] Analysis saved to {save_dir}")
    print(f" - View .png images for quick check")
    print(f" - Download .ply files for interactive 3D inspection")