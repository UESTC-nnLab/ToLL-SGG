import torch
import torch.nn.functional as F

class EpochCollapseMonitor:
    def __init__(self, num_prototypes, device='cuda'):
        """
        Args:
            num_prototypes (int): 原型(聚类中心)的总数，例如 50
            device (str): 张量所在的设备
        """
        self.num_prototypes = num_prototypes
        self.device = device
        self.reset() # 初始化计数器

    def reset(self):
        """在每个 Epoch 开始前调用，清空累积数据"""
        self.sum_std = 0.0          # 累积特征标准差
        self.sum_comp = 0.0
        self.sum_expa = 0.0
        self.mcr_count = 0
        self.batch_count = 0        # 累积 Batch 数量
        # 全局直方图：记录整个 Epoch 中每个原型被选中的总次数
        self.global_hist = torch.zeros(self.num_prototypes, device=self.device)

    @torch.no_grad()
    def update(self, embeddings, swav_q=None, mcr_stats=None):
        """
        在每个 Batch 训练结束后调用，累积数据。
        
        Args:
            embeddings: (B, D) 当前 batch 的边特征
            swav_q: (B, K) Sinkhorn 生成的伪标签分布
        """
        # 1. 累积特征标准差 (Feature STD)
        # 计算当前 batch 的平均 STD
        current_std = F.normalize(embeddings, dim=1).std(dim=0).mean().item()
        self.sum_std += current_std

        # 2. 累积原型使用情况 (Prototype Usage)
        # 统计当前 batch 中，每个样本被分到了哪个类 (硬分配用于统计)
        if swav_q is not None:
            pred_labels = swav_q.argmax(dim=1) 
            # bincount 统计频次
            hist = torch.bincount(pred_labels, minlength=self.num_prototypes).float()
            self.global_hist += hist

        if mcr_stats is not None:
            comp = mcr_stats.get('comp_loss', None)
            expa = mcr_stats.get('expa_loss', None)
            if comp is not None and expa is not None:
                self.sum_comp += float(comp)
                self.sum_expa += float(expa)
                self.mcr_count += 1
        
        self.batch_count += 1

    def report(self, epoch_idx):
        """在 Epoch 结束时调用，打印报告"""
        if self.batch_count == 0:
            return

        # === 计算全局指标 ===
        # 1. 平均特征标准差
        avg_std = self.sum_std / self.batch_count

        # 2. 全局活跃原型数 (Global Active Prototypes)
        # 整个 Epoch 跑完，有多少个原型至少被用过一次？
        active_prototypes = (self.global_hist > 0).sum().item()

        # 3. 全局分布熵 (Global Entropy)
        # 衡量全量数据在聚类中心上的分布是否均匀
        total_samples = self.global_hist.sum()
        if total_samples > 0:
            probs = self.global_hist / total_samples
            entropy = -(probs * torch.log(probs + 1e-6)).sum().item()
            max_possible_entropy = torch.log(torch.tensor(float(self.num_prototypes))).item()
        else:
            entropy = 0.0
            max_possible_entropy = torch.log(torch.tensor(float(self.num_prototypes))).item()

        # 4. 最大主导占比 (Most Dominant Cluster)
        # 最火的那个聚类中心占了多少样本
        max_assign_ratio = (self.global_hist.max() / total_samples).item() if total_samples > 0 else 0.0

        # === 打印报表 ===
        print(f"\n{'='*20} [Epoch {epoch_idx} Collapse Report] {'='*20}")
        print(f"  > Avg Feature STD:   {avg_std:.6f}  (参考: 若 < 0.001 则极度危险)")
        if total_samples > 0:
            print(f"  > Active Prototypes: {active_prototypes}/{self.num_prototypes}  (越多越好)")
            print(f"  > Global Entropy:    {entropy:.4f} / {max_possible_entropy:.4f} (接近最大值说明分布均匀)")
            print(f"  > Dominant Cluster:  {max_assign_ratio*100:.2f}% samples  (若 > 90% 说明几乎全分到一类了)")
        if self.mcr_count > 0:
            print(f"  > Avg MCR Comp:      {self.sum_comp / self.mcr_count:.6f}")
            print(f"  > Avg MCR Expa:      {self.sum_expa / self.mcr_count:.6f}")
        
        # 自动报警
        if avg_std < 1e-3:
            print("  🚨 [严重警告] 特征坍塌 (Feature Collapse): 所有边特征都一样了！")
        elif total_samples > 0 and active_prototypes < self.num_prototypes * 0.2:
            print("  ⚠️ [警告] 聚类坍塌 (Prototype Collapse): 只有极少数聚类中心在工作。")
        else:
            print("  ✅ 模型状态健康 (Model looks healthy).")
        print(f"{'='*65}\n")