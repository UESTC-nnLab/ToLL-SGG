import torch
import numpy as np
from sklearn.cluster import KMeans  # <-- Changed import
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import os 
import matplotlib

# Set matplotlib backend to Agg to avoid display issues in non-GUI environments
matplotlib.use('Agg') 

def cluster_and_visualize(tensor_data: torch.Tensor, 
                          n_clusters: int,  # <-- Added parameter for K
                          title_prefix: str = "Data", 
                          save_path: str = "cluster_plot.png"):
    """
    Performs K-Means clustering on the input high-dimensional tensor
    and visualizes the result using t-SNE.

    Parameters:
    tensor_data (torch.Tensor): Input data, shape (N, 512).
    n_clusters (int): The number of clusters (K) for K-Means.
    title_prefix (str): Prefix for the plot title (e.g., "Object" or "Edge").
    save_path (str): The file path to save the visualization (e.g., "plot.png").
                     Labels will be saved to a corresponding .npy file.
    """
    
    print(f"\n--- Processing {title_prefix} ---")
    
    # 1. Data Preparation: Convert Tensor to Numpy
    if tensor_data.is_cuda:
        data_np = tensor_data.detach().cpu().numpy()
    else:
        data_np = tensor_data.numpy()

    if data_np.dtype != np.float64:
        data_np = data_np.astype(np.float64)
        
    # Note: K-Means is sensitive to feature scaling.
    # Our mock data generator already applies StandardScaler.
    # If your real data is not scaled, you should do it here.

    # 2. Clustering (K-Means)
    print(f"Starting K-Means clustering for {data_np.shape[0]} samples into {n_clusters} clusters...")
    
    # --- Replaced HDBSCAN with K-Means ---
    kmeans = KMeans(n_clusters=n_clusters, 
                    random_state=42, 
                    n_init=10)  # n_init=10 runs the algorithm 10 times
    
    labels = kmeans.fit_predict(data_np)
    
    print(f"Clustering complete. Assigned data to {n_clusters} clusters.")

    # 3. Dimensionality Reduction (t-SNE)
    print("Starting t-SNE dimensionality reduction (this may take a moment)...")
    
    tsne = TSNE(n_components=2, 
                perplexity=30.0, 
                max_iter=1000, 
                random_state=42, 
                n_jobs=-1)
    
    data_2d = tsne.fit_transform(data_np)
    print("t-SNE reduction complete.")

    # 4. Visualization (Matplotlib)
    plt.figure(figsize=(12, 10))
    
    # Get a discrete colormap
    cmap = plt.get_cmap('Spectral', n_clusters)
    
    scatter = plt.scatter(data_2d[:, 0], 
                          data_2d[:, 1], 
                          c=labels, 
                          cmap=cmap,
                          vmin=-0.5, # Center colors on integer labels
                          vmax=n_clusters - 0.5,
                          s=10,
                          alpha=0.7)

    # Updated title for K-Means
    plt.title(f"{title_prefix} - K-Means Clustering Results (t-SNE Visualization)\n (K = {n_clusters} clusters)", fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    
    # Updated colorbar for discrete cluster IDs (no noise)
    cbar = plt.colorbar(scatter, ticks=np.arange(n_clusters), label='Cluster ID')
    
    # If there are too many ticks (e.s. 30), show a subset
    if n_clusters > 20:
        tick_skip = n_clusters // 10  # Show about 10 ticks
        ticks = np.arange(0, n_clusters, tick_skip)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks)

    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 5. Save the plot to the specified path
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    
    plt.close()

    # 6. Save the cluster labels
    base, ext = os.path.splitext(save_path)
    labels_save_path = f"{base}_labels.npy"
    
    np.save(labels_save_path, labels)
    print(f"Cluster labels saved to {labels_save_path}")

    return labels