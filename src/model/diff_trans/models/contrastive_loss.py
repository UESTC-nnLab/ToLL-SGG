import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class TextSupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, num_negatives=50):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives # 指定随机采样的负样本数量
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, point_features, batch_texts, text_embeddings_dict):
        """
        参数:
            point_features: (B, Dim)
                点云特征。
            batch_texts: List[str]
                当前Batch每个点云对应的真实文本标签。
            text_embeddings_dict: Dict { str : Tensor }
                包含所有类别的大字典。
        """
        device = point_features.device
        
        # ==========================================
        # 1. 确定本次计算所需的“活跃类别集合” (Active Keys)
        # ==========================================
        
        # A. 找出当前Batch中必须存在的正样本类别 (去重)
        positive_keys = list(set(batch_texts))
        
        # B. 找出所有可能的负样本 (字典总Key - 正样本Key)
        # 注意：这里转换成 set 进行差集运算
        all_keys_set = set(text_embeddings_dict.keys())
        positive_keys_set = set(positive_keys)
        candidate_negative_keys = list(all_keys_set - positive_keys_set)
        
        # C. 随机采样负样本
        # 如果剩余的key不足 num_negatives，就取全部；否则随机取 num_negatives 个
        curr_num_neg = min(self.num_negatives, len(candidate_negative_keys))
        sampled_negative_keys = random.sample(candidate_negative_keys, curr_num_neg)
        
        # D. 合并得到本次用于计算矩阵的所有 Key
        # 顺序：[所有正样本类别, ... 随机负样本类别 ...]
        active_keys = positive_keys + sampled_negative_keys
        
        # ==========================================
        # 2. 构建小型的文本特征矩阵 (Sampled Matrix)
        # ==========================================
        # 只提取 active_keys 对应的 embedding
        # 形状: (Num_Active_Keys, Dim)  <-- 远小于总类别数
        text_embeddings_list = [text_embeddings_dict[k].to(device).view(-1) for k in active_keys]
        active_text_embeddings = torch.stack(text_embeddings_list)
        
        # ==========================================
        # 3. 构建 Ground Truth (Remapping)
        # ==========================================
        # 因为矩阵变了，索引也变了。我们需要找到 batch_texts 在 active_keys 中的新索引。
        
        # 构建快速查找表: { 类别名 : 在小矩阵中的行号 }
        key_to_idx_map = {key: idx for idx, key in enumerate(active_keys)}
        
        target_indices = []
        for text_label in batch_texts:
            # 必定能找到，因为 active_keys 包含了 positive_keys
            target_indices.append(key_to_idx_map[text_label])
            
        ground_truth_labels = torch.tensor(target_indices, device=device, dtype=torch.long)

        # ==========================================
        # 4. 对比损失计算
        # ==========================================
        
        # 归一化
        point_features = F.normalize(point_features, p=2, dim=1)
        active_text_embeddings = F.normalize(active_text_embeddings, p=2, dim=1).float()
        
        # 计算相似度 Logits: (B, Num_Active_Keys)
        # 显存占用大幅降低，因为 active_text_embeddings 很小
        logits = torch.matmul(point_features, active_text_embeddings.T) / self.temperature
        
        loss = self.cross_entropy(logits, ground_truth_labels)
        
        return loss

# ------------------------------------------------------------------
# 测试代码
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 模拟环境
    B = 4
    Dim = 128
    
    # 1. 模拟一个很大的字典 (假设有 1000 个类)
    # 我们只打印 Loss，不实际占用那么多内存以免运行缓慢
    full_dict = {f"class_{i}": torch.randn(Dim) for i in range(1000)}
    
    # 2. 当前 Batch 的真实标签 (只涉及其中几个类)
    batch_texts = ["class_1", "class_5", "class_1", "class_99"]
    
    # 3. 当前 Batch 的点云特征
    p_feats = torch.randn(B, Dim)
    
    # 4. 初始化 Loss (只采样 50 个负样本)
    criterion = TextSupervisedContrastiveLoss(temperature=0.07, num_negatives=50)
    
    loss = criterion(p_feats, batch_texts, full_dict)
    
    # 验证一下 active keys 的数量
    # 应该是: 3个独立的正样本(class_1, 5, 99) + 50个负样本 = 53 (如果字典够大的话)
    print(f"Calculated Loss: {loss.item()}")