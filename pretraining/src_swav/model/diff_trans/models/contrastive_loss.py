import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextSupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, num_negatives=50, text_embeddings_path=None):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.cross_entropy = nn.CrossEntropyLoss()
        self.text_embeddings_dict = None
        self.text_embeddings_path = text_embeddings_path

        if text_embeddings_path and os.path.exists(text_embeddings_path):
            self.text_embeddings_dict = torch.load(text_embeddings_path, map_location="cpu")

    def forward(self, point_features, batch_texts):
        device = point_features.device

        if self.text_embeddings_dict is None or batch_texts is None:
            return point_features.new_zeros(())

        valid_indices = [idx for idx, text in enumerate(batch_texts) if text in self.text_embeddings_dict]
        if len(valid_indices) == 0:
            return point_features.new_zeros(())

        point_features = point_features[valid_indices]
        batch_texts = [batch_texts[idx] for idx in valid_indices]

        positive_keys = list(set(batch_texts))
        all_keys_set = set(self.text_embeddings_dict.keys())
        positive_keys_set = set(positive_keys)
        candidate_negative_keys = list(all_keys_set - positive_keys_set)

        curr_num_neg = min(self.num_negatives, len(candidate_negative_keys))
        sampled_negative_keys = random.sample(candidate_negative_keys, curr_num_neg) if curr_num_neg > 0 else []
        active_keys = positive_keys + sampled_negative_keys

        text_embeddings_list = [self.text_embeddings_dict[k].to(device).view(-1) for k in active_keys]
        active_text_embeddings = torch.stack(text_embeddings_list)

        key_to_idx_map = {key: idx for idx, key in enumerate(active_keys)}
        target_indices = [key_to_idx_map[text_label] for text_label in batch_texts]
        ground_truth_labels = torch.tensor(target_indices, device=device, dtype=torch.long)

        point_features = F.normalize(point_features, p=2, dim=1)
        active_text_embeddings = F.normalize(active_text_embeddings, p=2, dim=1).float()

        logits = torch.matmul(point_features, active_text_embeddings.T) / self.temperature
        return self.cross_entropy(logits, ground_truth_labels)


class ObjectLabelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, eps=1e-12):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        anchor_features,
        anchor_labels,
        contrast_features,
        contrast_labels,
        anchor_instance_ids=None,
        contrast_instance_ids=None,
        anchor_view_ids=None,
        contrast_view_ids=None,
    ):
        if anchor_features.numel() == 0 or contrast_features.numel() == 0:
            return anchor_features.new_zeros(())

        anchor_labels = anchor_labels.reshape(-1)
        contrast_labels = contrast_labels.reshape(-1)

        if anchor_features.shape[0] != anchor_labels.shape[0]:
            raise ValueError("anchor_features and anchor_labels size mismatch")
        if contrast_features.shape[0] != contrast_labels.shape[0]:
            raise ValueError("contrast_features and contrast_labels size mismatch")

        anchor_features = F.normalize(anchor_features, p=2, dim=1)
        contrast_features = F.normalize(contrast_features, p=2, dim=1)

        logits = torch.matmul(anchor_features, contrast_features.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        if anchor_instance_ids is not None and contrast_instance_ids is not None:
            anchor_instance_ids = anchor_instance_ids.reshape(-1)
            contrast_instance_ids = contrast_instance_ids.reshape(-1)
            if anchor_features.shape[0] != anchor_instance_ids.shape[0]:
                raise ValueError("anchor_features and anchor_instance_ids size mismatch")
            if contrast_features.shape[0] != contrast_instance_ids.shape[0]:
                raise ValueError("contrast_features and contrast_instance_ids size mismatch")

            positive_mask = anchor_instance_ids[:, None].eq(contrast_instance_ids[None, :])
            if anchor_view_ids is not None and contrast_view_ids is not None:
                anchor_view_ids = anchor_view_ids.reshape(-1)
                contrast_view_ids = contrast_view_ids.reshape(-1)
                if anchor_features.shape[0] != anchor_view_ids.shape[0]:
                    raise ValueError("anchor_features and anchor_view_ids size mismatch")
                if contrast_features.shape[0] != contrast_view_ids.shape[0]:
                    raise ValueError("contrast_features and contrast_view_ids size mismatch")
                positive_mask = positive_mask & anchor_view_ids[:, None].ne(contrast_view_ids[None, :])
            negative_mask = anchor_labels[:, None].ne(contrast_labels[None, :])
            active_mask = positive_mask | negative_mask
        else:
            positive_mask = anchor_labels[:, None].eq(contrast_labels[None, :])
            negative_mask = ~positive_mask
            active_mask = torch.ones_like(positive_mask, dtype=torch.bool)

        positive_count = positive_mask.sum(dim=1)
        negative_count = negative_mask.sum(dim=1)
        valid_mask = (positive_count > 0) & (negative_count > 0)
        if not torch.any(valid_mask):
            return anchor_features.new_zeros(())

        exp_logits = torch.exp(logits) * active_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(self.eps))

        positive_mask = positive_mask.float()
        loss = -(positive_mask[valid_mask] * log_prob[valid_mask]).sum(dim=1) / positive_count[valid_mask].float()
        return loss.mean()
