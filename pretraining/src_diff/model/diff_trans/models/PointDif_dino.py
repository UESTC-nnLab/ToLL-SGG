import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from timm.models.layers import trunc_normal_

from src.model.diff_trans.models.build import MODELS
from src.model.diff_trans.utils.logger import print_log
from src.model.diff_trans.models.mask_encoder import Mask_Encoder, Group
from src.model.diff_trans.models.generator import CPDM, CANet, get_chamfer_distance
from src.model.model_utils.model_base import BaseModel
from src.model.model_utils.network_MMRGR import MMG
from src.model.model_utils.network_PointNet import PointNetfeat
from src.model.diff_trans.models.weight_focal_loss import compute_local_complexity_weight
from src.utils import op_utils
from src.dataset.dataset_diffPoint import visualize_scenes_plt_with_points


class MaskedEdgeEncoder(nn.Module):
    def __init__(self, edge_dim):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(1, edge_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, num_edges):
        return self.mask_token.expand(num_edges, -1)


@MODELS.register_module()
class PointDif(BaseModel):
    def __init__(self, config, dim_descriptor=11):
        super().__init__('Diff_sg', config)
        print_log('[Diff_sg]', logger='Diff_sg')

        self.mconfig = mconfig = config.sg_model
        with_bn = mconfig.WITH_BN

        dim_point = 3
        if mconfig.USE_RGB:
            dim_point += 3
        if mconfig.USE_NORMAL:
            dim_point += 3

        self.dim_point = dim_point
        self.dim_edge = dim_descriptor
        self.flow = 'target_to_source'

        self.rel_encoder_3d = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=11,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=512,
        )

        self.mmg = MMG(
            dim_node=512,
            dim_edge=512,
            dim_atten=256,
            depth=2,
            num_heads=8,
            aggr="max",
            flow=self.flow,
            attention="fat",
            use_edge=True,
            DROP_OUT_ATTEN=0.5,
        )

        self.config = config.maskTrans
        self.group_size = self.config.group_size
        self.num_group = self.config.num_group
        self.trans_dim = self.config.encoder_config.trans_dim
        self.encoder_dims = self.config.encoder_config.encoder_dims
        self.object_mask_ratio = float(self.config.encoder_config.mask_ratio)
        self.mask_encoder = Mask_Encoder(self.config)
        self.ca_net = CANet(self.encoder_dims, 512)
        self.mlp_3d = nn.Sequential(
            nn.Linear(512, 512 - 11),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.edge_mask_token = MaskedEdgeEncoder(512)
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.point_diffusion = CPDM(self.config)

        # Keep the backbone intact and only add lightweight generative heads.
        self.shape_condition_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
        )
        self.object_text_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
        )
        self.layout_position_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 3),
        )
        self.layout_scale_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 3),
        )

        self.layout_loss_fn = nn.SmoothL1Loss(reduction='mean')
        self.layout_recon_loss_fn = nn.SmoothL1Loss(reduction='mean')

        self.diffusion_cd_weight = float(getattr(config, "DIFFUSION_CD_WEIGHT", 0.0))
        self.direct_spatial_diffusion_only = bool(getattr(config, "DIRECT_SPATIAL_DIFFUSION_ONLY", False))
        self.text_contrastive_enabled = bool(getattr(config, "TEXT_CONTRASTIVE_ENABLED", False))
        self.text_contrastive_weight = float(getattr(config, "TEXT_CONTRASTIVE_WEIGHT", 0.1))
        self.text_contrastive_temperature = float(getattr(config, "TEXT_CONTRASTIVE_TEMPERATURE", 0.07))
        self.layout_position_weight = float(getattr(config, "LAYOUT_POSITION_WEIGHT", 1.0))
        self.layout_scale_weight = float(getattr(config, "LAYOUT_SCALE_WEIGHT", 1.0))
        self.layout_recon_weight = float(getattr(config, "LAYOUT_RECON_WEIGHT", 1.0))
        self.layout_debug_interval_steps = int(getattr(config, "LAYOUT_DEBUG_INTERVAL_STEPS", 0))
        self.max_layout_debug_visualizations = int(getattr(config, "LAYOUT_DEBUG_VIS_MAX", 100))
        self.enable_diagnostics = bool(getattr(config, "DIAGNOSTICS_ENABLE", False))
        self.diagnostics_compare_unconditional = bool(
            getattr(config, "DIAGNOSTICS_COMPARE_UNCONDITIONAL", True)
        )
        self.analysis_save_dir = getattr(
            config,
            "analysis_save_dir",
            os.path.join(os.getcwd(), "analysis_results", "src_diff"),
        )
        self._layout_debug_step = 0
        self._layout_debug_vis_count = 0

        self.count = 0
        trunc_normal_(self.mask_token, std=.02)
        print_log(
            f'[PointDif] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
            logger='PointDif'
        )

    def _is_main_process(self):
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

    def normalize_to_canonical_space(self, points, descriptor=None):
        centered = points - points.mean(dim=1, keepdim=True)
        if descriptor is not None:
            dims = descriptor[:, 6:9].clamp_min(1e-4).unsqueeze(1)
        else:
            dims = (centered.max(dim=1).values - centered.min(dim=1).values).clamp_min(1e-4).unsqueeze(1)
        return centered / dims

    def normalize_auxiliary_positions(self, positions, reference_points, descriptor=None):
        reference_center = reference_points.mean(dim=1, keepdim=True)
        centered = positions - reference_center
        if descriptor is not None:
            dims = descriptor[:, 6:9].clamp_min(1e-4).unsqueeze(1)
        else:
            dims = (reference_points.max(dim=1).values - reference_points.min(dim=1).values).clamp_min(1e-4).unsqueeze(1)
        return centered / dims

    def compose_layout_points(self, canonical_points, centers, sizes):
        if canonical_points is None:
            return None
        canonical_points = canonical_points - canonical_points.mean(dim=1, keepdim=True)
        extents = canonical_points.max(dim=1).values - canonical_points.min(dim=1).values
        extents = extents.clamp_min(1e-4)
        scaled_points = canonical_points / extents.unsqueeze(1) * sizes.unsqueeze(1)
        return scaled_points + centers.unsqueeze(1)

    def compute_diffusion_reconstruction_loss(self, canonical_points, shape_features, visible_tokens=None, visible_token_positions=None):
        if canonical_points is None or canonical_points.shape[0] == 0:
            zero = shape_features.new_zeros(())
            return zero, zero.detach(), 0.0

        weights = compute_local_complexity_weight(canonical_points)
        if self.diffusion_cd_weight > 0:
            diffusion_train_loss, diffusion_log_loss, _x0_metric, cd_metric = self.point_diffusion.get_loss_withCD(
                canonical_points.contiguous(),
                shape_features,
                weights=weights,
                cd_weight=self.diffusion_cd_weight,
                token_condition=visible_tokens,
                token_positions=visible_token_positions,
            )
        else:
            diffusion_log_loss, _x0_metric, cd_metric = self.point_diffusion.get_loss(
                canonical_points.contiguous(),
                shape_features,
                weights=weights,
                token_condition=visible_tokens,
                token_positions=visible_token_positions,
            )
            diffusion_train_loss = diffusion_log_loss
        return diffusion_train_loss, diffusion_log_loss, cd_metric

    def compute_cd_metric(self, pred_points, target_points):
        if pred_points is None or target_points is None or pred_points.shape[0] == 0:
            return 0.0
        with torch.no_grad():
            return float(get_chamfer_distance(pred_points, target_points).mean().item())

    def build_spatial_diffusion_target(self, pts, obj_points_spatial=None, descriptor=None):
        if obj_points_spatial is not None:
            return obj_points_spatial[:, :, :3].contiguous()

        canonical_target = self.normalize_to_canonical_space(pts.contiguous(), descriptor)
        if descriptor is None:
            return canonical_target

        gt_center = descriptor[:, :3].clone()
        gt_scale = descriptor[:, 6:9].clone().clamp_min(1e-4)
        return self.compose_layout_points(canonical_target, gt_center, gt_scale)

    def prepare_diffusion_inputs(self, pts, descriptor, diffusion_context, obj_points_spatial=None):
        if self.direct_spatial_diffusion_only:
            return (
                self.build_spatial_diffusion_target(
                    pts,
                    obj_points_spatial=obj_points_spatial,
                    descriptor=descriptor,
                ),
                diffusion_context["visible_tokens"],
                None,
            )

        canonical_target = self.normalize_to_canonical_space(pts.contiguous(), descriptor)
        return (
            canonical_target,
            diffusion_context["visible_tokens"],
            diffusion_context["visible_positions"],
        )

    @torch.no_grad()
    def sample_canonical_points(self, shape_features, num_points, device, visible_tokens=None, visible_token_positions=None):
        if shape_features is None or shape_features.shape[0] == 0:
            return None
        sampled = self.point_diffusion.sample(
            num_points,
            shape_features,
            device,
            token_condition=visible_tokens,
            token_positions=visible_token_positions,
        )
        return sampled.to(shape_features.device)

    def predict_layout_parameters(self, shape_features):
        pred_center = self.layout_position_head(shape_features)
        pred_scale = F.softplus(self.layout_scale_head(shape_features)) + 1e-4
        return pred_center, pred_scale

    def compute_object_text_contrastive_loss(self, shape_features, text_embeddings=None, text_valid_mask=None):
        zero = shape_features.new_zeros(())
        if (not self.text_contrastive_enabled) or text_embeddings is None or text_valid_mask is None:
            return zero, 0, 0.0

        valid_mask = text_valid_mask.bool()
        if valid_mask.sum().item() < 2:
            return zero, int(valid_mask.sum().item()), 0.0

        obj_feats = self.object_text_projector(shape_features[valid_mask])
        text_feats = text_embeddings[valid_mask].to(shape_features.device, dtype=shape_features.dtype)
        obj_feats = F.normalize(obj_feats, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)

        temperature = max(self.text_contrastive_temperature, 1e-4)
        logits = obj_feats @ text_feats.t() / temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.t(), targets)
        loss = 0.5 * (loss_i2t + loss_t2i)

        with torch.no_grad():
            top1 = (logits.argmax(dim=1) == targets).float().mean().item()
        return loss, int(valid_mask.sum().item()), top1

    def compute_descriptor_recovery_metric(self, canonical_points, descriptor, spatial_points=None):
        gt_center = descriptor[:, :3].clone()
        gt_scale = descriptor[:, 6:9].clone().clamp_min(1e-4)
        recovered_points = self.compose_layout_points(canonical_points, gt_center, gt_scale)
        target_points = spatial_points[:, :, :3].contiguous() if spatial_points is not None else recovered_points
        recovery_l1 = F.smooth_l1_loss(recovered_points, target_points)
        recovery_cd = self.compute_cd_metric(recovered_points, target_points)
        return recovered_points, target_points, recovery_l1, recovery_cd

    def build_reconstruction_outputs(self, canonical_target, shape_features, descriptor, spatial_points=None):
        gt_center = descriptor[:, :3].clone()
        gt_scale = descriptor[:, 6:9].clone().clamp_min(1e-4)

        pred_center, pred_scale = self.predict_layout_parameters(shape_features)
        position_loss = self.layout_loss_fn(pred_center, gt_center)
        scale_loss = self.layout_loss_fn(pred_scale, gt_scale)

        pred_layout_points = self.compose_layout_points(canonical_target, pred_center, pred_scale)
        target_layout_points, target_spatial_points, descriptor_recovery_loss, descriptor_recovery_cd = self.compute_descriptor_recovery_metric(
            canonical_target,
            descriptor,
            spatial_points=spatial_points,
        )
        layout_recon_loss = self.layout_recon_loss_fn(pred_layout_points, target_layout_points)
        total_metric = F.smooth_l1_loss(pred_layout_points, target_layout_points).item()

        return {
            "pred_center": pred_center,
            "pred_scale": pred_scale,
            "gt_center": gt_center,
            "gt_scale": gt_scale,
            "position_loss": position_loss,
            "scale_loss": scale_loss,
            "layout_recon_loss": layout_recon_loss,
            "pred_layout_points": pred_layout_points,
            "target_layout_points": target_layout_points,
            "target_spatial_points": target_spatial_points,
            "descriptor_recovery_loss": descriptor_recovery_loss,
            "descriptor_recovery_cd": descriptor_recovery_cd,
            "total_metric": total_metric,
        }

    def maybe_visualize_layout_reconstruction(self, pred_points, target_points, metric_value, tag="train"):
        if pred_points is None or target_points is None:
            return
        if not self._is_main_process():
            return
        if self.layout_debug_interval_steps <= 0 or self.max_layout_debug_visualizations <= 0:
            return
        if tag == "train":
            self._layout_debug_step += 1
            if self._layout_debug_step % self.layout_debug_interval_steps != 0:
                return
        if self._layout_debug_vis_count >= self.max_layout_debug_visualizations:
            return

        output_dir = os.path.join(self.analysis_save_dir, "layout_debug")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(
            output_dir,
            f"{tag}_layout_{self._layout_debug_vis_count:03d}_met_{metric_value:.4f}.png",
        )
        try:
            visualize_scenes_plt_with_points(
                pred_points.detach(),
                target_points.detach(),
                output_filename=filename,
            )
            self._layout_debug_vis_count += 1
            print_log(f"[PointDif] Saved layout debug visualization to {filename}", logger='PointDif')
        except Exception as exc:  # noqa: BLE001
            print_log(f"[PointDif] Failed to save layout debug visualization: {exc}", logger='PointDif')

    def obj_feat_extractor(self, pts, anchor_id, batch_ids, descriptor, mask_ratio=None):
        anchor_set = anchor_id is not None
        batch_size, _, _ = pts.shape
        neighborhood, center = self.group_divider(pts)
        encoder_token, mask, visible_tokens, visible_centers = self.mask_encoder(
            neighborhood,
            center,
            mask_ratio=mask_ratio,
        )
        _, masked_count, _ = center[mask].reshape(batch_size, -1, 3).shape
        if masked_count > 0:
            mask_token = self.mask_token.expand(batch_size, masked_count, -1)
            encoder_token = encoder_token.clone()
            encoder_token[mask] = mask_token.reshape(-1, self.trans_dim)
        point_agg_features = self.ca_net(encoder_token)
        diffusion_context = {
            "visible_tokens": visible_tokens,
            "visible_positions": self.normalize_auxiliary_positions(
                visible_centers,
                pts,
                descriptor,
            ),
        }

        if anchor_set:
            device = point_agg_features.device
            local_anchor_ids_tensor = torch.tensor(anchor_id, device=device, dtype=torch.long)
            batch_ids_squeezed = batch_ids.squeeze()
            counts = torch.bincount(batch_ids_squeezed)
            offsets = torch.cat([torch.tensor([0], device=device), torch.cumsum(counts, dim=0)[:-1]])
            global_anchor_indices = offsets + local_anchor_ids_tensor

            anchor_obj_features = self.mlp_3d(point_agg_features[global_anchor_indices])
            if self.mconfig.USE_SPATIAL:
                tmp = descriptor.clone()
                tmp[:, 6:] = tmp[:, 6:].clamp_min(1e-4).log()
                anchor_spatial_info = tmp[global_anchor_indices]
                point_agg_features[global_anchor_indices] = torch.cat(
                    [anchor_obj_features, anchor_spatial_info], dim=-1
                )
            return point_agg_features, global_anchor_indices, diffusion_context

        point_agg_features = self.mlp_3d(point_agg_features)
        if self.mconfig.USE_SPATIAL:
            tmp = descriptor.clone()
            tmp[:, 6:] = tmp[:, 6:].clamp_min(1e-4).log()
            point_agg_features = torch.cat([point_agg_features, tmp], dim=-1)
        return point_agg_features, diffusion_context

    def edge_masking(self, rel_feature_3d_view, edge_mask_ratio=0.0):
        if edge_mask_ratio <= 0:
            return rel_feature_3d_view, None
        num_edges = rel_feature_3d_view.shape[0]
        mask_indices = torch.rand(num_edges, device=rel_feature_3d_view.device) < edge_mask_ratio
        if mask_indices.any():
            num_masked = int(mask_indices.sum().item())
            rel_feature_3d_view[mask_indices] = self.edge_mask_token.mask_token.expand(num_masked, -1)
        return rel_feature_3d_view, mask_indices

    def build_object_graph_features(
        self,
        pts,
        edge_indices,
        descriptor,
        batch_ids,
        anchor_id=None,
        mask_ratio=None,
        edge_mask_ratio=0.0,
        istrain=False,
    ):
        edge_indices = edge_indices.long()
        batch_ids = batch_ids.long()
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor()(descriptor, edge_indices)

        if anchor_id is not None:
            point_agg_features, global_anchor_indices, diffusion_context = self.obj_feat_extractor(
                pts, anchor_id, batch_ids, descriptor, mask_ratio=mask_ratio
            )
        else:
            point_agg_features, diffusion_context = self.obj_feat_extractor(
                pts, None, batch_ids, descriptor, mask_ratio=mask_ratio
            )
            global_anchor_indices = None

        rel_feature_3d = self.rel_encoder_3d(edge_feature)
        rel_feature_3d, _ = self.edge_masking(rel_feature_3d, edge_mask_ratio=edge_mask_ratio)

        obj_center = descriptor[:, :3].clone()
        if anchor_id is not None:
            gcn_obj_feature_3d, gcn_edge_feature_3d = self.mmg(
                point_agg_features,
                rel_feature_3d,
                edge_indices,
                batch_ids,
                global_anchor_indices,
                obj_center=obj_center,
                istrain=istrain,
            )
        else:
            gcn_obj_feature_3d, gcn_edge_feature_3d = self.mmg.forward_no_anchor(
                point_agg_features,
                rel_feature_3d,
                edge_indices,
                batch_ids,
                obj_center=obj_center,
                istrain=istrain,
                GRU=True,
            )
        return gcn_obj_feature_3d, gcn_edge_feature_3d, diffusion_context

    def forward(
        self,
        pts,
        edge_indices,
        obj_points_spatial,
        descriptor=None,
        batch_ids=None,
        anchor_id=None,
        istrain=False,
        cur_obj_texts=None,
        obj_labels=None,
        edge_labels=None,
        atlas_embeddings=None,
        atlas_valid_mask=None,
    ):
        if descriptor is None:
            raise ValueError("descriptor must be provided for PointDif pretraining.")

        obj_feat, edge_feat, diffusion_context = self.build_object_graph_features(
            pts,
            edge_indices,
            descriptor,
            batch_ids,
            anchor_id=anchor_id,
            mask_ratio=self.object_mask_ratio,
            edge_mask_ratio=0.1,
            istrain=istrain,
        )

        shape_features = self.shape_condition_head(obj_feat)
        diffusion_target, visible_tokens, visible_token_positions = self.prepare_diffusion_inputs(
            pts,
            descriptor,
            diffusion_context,
            obj_points_spatial=obj_points_spatial,
        )
        diffusion_train_loss, diff_loss, diff_metric = self.compute_diffusion_reconstruction_loss(
            diffusion_target,
            shape_features,
            visible_tokens=visible_tokens,
            visible_token_positions=visible_token_positions,
        )

        if self.direct_spatial_diffusion_only:
            zero = diffusion_train_loss.new_zeros(())
            return (
                diffusion_train_loss,
                diff_loss,
                diffusion_train_loss.detach(),
                zero,
                zero,
                zero,
                zero,
                diff_metric,
                edge_feat,
                shape_features,
            )

        canonical_target = diffusion_target
        recon_outputs = self.build_reconstruction_outputs(
            canonical_target,
            shape_features,
            descriptor,
            spatial_points=obj_points_spatial,
        )
        text_contrastive_loss, _valid_text_count, _text_top1 = self.compute_object_text_contrastive_loss(
            shape_features,
            text_embeddings=atlas_embeddings,
            text_valid_mask=atlas_valid_mask,
        )

        total_loss = (
            diffusion_train_loss
            + self.layout_position_weight * recon_outputs["position_loss"]
            + self.layout_scale_weight * recon_outputs["scale_loss"]
            + self.layout_recon_weight * recon_outputs["layout_recon_loss"]
            + self.text_contrastive_weight * text_contrastive_loss
        )

        if istrain:
            self.maybe_visualize_layout_reconstruction(
                recon_outputs["pred_layout_points"],
                recon_outputs["target_layout_points"],
                recon_outputs["total_metric"],
                tag="train",
            )

        return (
            total_loss,
            diff_loss,
            diffusion_train_loss.detach(),
            text_contrastive_loss,
            recon_outputs["position_loss"],
            recon_outputs["scale_loss"],
            recon_outputs["layout_recon_loss"],
            recon_outputs["total_metric"],
            edge_feat,
            shape_features,
        )

    def forward_cls(self, pts, edge_indices, descriptor=None, batch_ids=None, istrain=False):
        obj_feat, edge_feat, _ = self.build_object_graph_features(
            pts,
            edge_indices,
            descriptor,
            batch_ids,
            anchor_id=None,
            mask_ratio=self.object_mask_ratio,
            edge_mask_ratio=0.0,
            istrain=istrain,
        )
        shape_features = self.shape_condition_head(obj_feat)
        return edge_feat, shape_features

    def forward_ori(
        self,
        pts,
        edge_indices,
        obj_points_spatial,
        descriptor=None,
        batch_ids=None,
        anchor_id=None,
        istrain=False,
        cur_obj_texts=None,
        atlas_embeddings=None,
        atlas_valid_mask=None,
    ):
        obj_feat, edge_feat, diffusion_context = self.build_object_graph_features(
            pts,
            edge_indices,
            descriptor,
            batch_ids,
            anchor_id=anchor_id,
            mask_ratio=self.object_mask_ratio,
            edge_mask_ratio=0.1,
            istrain=istrain,
        )
        shape_features = self.shape_condition_head(obj_feat)
        diffusion_target, visible_tokens, visible_token_positions = self.prepare_diffusion_inputs(
            pts,
            descriptor,
            diffusion_context,
            obj_points_spatial=obj_points_spatial,
        )
        pred_points = self.sample_canonical_points(
            shape_features,
            pts.shape[1],
            pts.device,
            visible_tokens=visible_tokens,
            visible_token_positions=visible_token_positions,
        )
        if pred_points is None:
            pred_points = diffusion_target

        if self.direct_spatial_diffusion_only:
            total_metric = self.compute_cd_metric(pred_points, diffusion_target)
            self.maybe_visualize_layout_reconstruction(
                pred_points,
                diffusion_target,
                total_metric,
                tag="eval",
            )
            _, diff_loss, _ = self.compute_diffusion_reconstruction_loss(
                diffusion_target,
                shape_features,
                visible_tokens=visible_tokens,
                visible_token_positions=visible_token_positions,
            )
            return diff_loss, total_metric, edge_feat, shape_features

        canonical_target = diffusion_target

        pred_center, pred_scale = self.predict_layout_parameters(shape_features)
        pred_layout_points = self.compose_layout_points(pred_points, pred_center, pred_scale)
        target_layout_points, _target_spatial_points, _descriptor_recovery_loss, _descriptor_recovery_cd = self.compute_descriptor_recovery_metric(
            canonical_target,
            descriptor,
            spatial_points=obj_points_spatial,
        )
        total_metric = F.smooth_l1_loss(pred_layout_points, target_layout_points).item()
        self.maybe_visualize_layout_reconstruction(
            pred_layout_points,
            target_layout_points,
            total_metric,
            tag="eval",
        )

        _, diff_loss, _ = self.compute_diffusion_reconstruction_loss(
            canonical_target,
            shape_features,
            visible_tokens=visible_tokens,
            visible_token_positions=visible_token_positions,
        )
        return diff_loss, total_metric, edge_feat, shape_features

    @torch.no_grad()
    def reconstruct_scene_points(
        self,
        pts,
        edge_indices,
        descriptor=None,
        batch_ids=None,
        anchor_id=None,
        obj_points_spatial=None,
    ):
        if descriptor is None:
            raise ValueError("descriptor must be provided for scene reconstruction.")

        obj_feat, _, diffusion_context = self.build_object_graph_features(
            pts,
            edge_indices,
            descriptor,
            batch_ids,
            anchor_id=anchor_id,
            mask_ratio=self.object_mask_ratio,
            edge_mask_ratio=0.1,
            istrain=False,
        )
        shape_features = self.shape_condition_head(obj_feat)
        diffusion_target, visible_tokens, visible_token_positions = self.prepare_diffusion_inputs(
            pts,
            descriptor,
            diffusion_context,
            obj_points_spatial=obj_points_spatial,
        )
        pred_points = self.sample_canonical_points(
            shape_features,
            pts.shape[1],
            pts.device,
            visible_tokens=visible_tokens,
            visible_token_positions=visible_token_positions,
        )
        if pred_points is None:
            pred_points = diffusion_target

        if self.direct_spatial_diffusion_only:
            return {
                "shape_features": shape_features,
                "pred_canonical_points": pred_points,
                "target_canonical_points": diffusion_target,
                "pred_center": None,
                "pred_scale": None,
                "gt_center": None,
                "gt_scale": None,
                "pred_layout_points": pred_points,
                "target_layout_points": diffusion_target,
                "pred_spatial_points": pred_points,
                "target_spatial_points": diffusion_target,
            }

        canonical_target = diffusion_target

        pred_center, pred_scale = self.predict_layout_parameters(shape_features)
        pred_layout_points = self.compose_layout_points(pred_points, pred_center, pred_scale)
        gt_center = descriptor[:, :3].clone()
        gt_scale = descriptor[:, 6:9].clone().clamp_min(1e-4)
        target_layout_points, target_spatial_points, descriptor_recovery_loss, descriptor_recovery_cd = self.compute_descriptor_recovery_metric(
            canonical_target,
            descriptor,
            spatial_points=obj_points_spatial,
        )

        return {
            "shape_features": shape_features,
            "pred_canonical_points": pred_points,
            "target_canonical_points": canonical_target,
            "pred_center": pred_center,
            "pred_scale": pred_scale,
            "gt_center": gt_center,
            "gt_scale": gt_scale,
            "pred_layout_points": pred_layout_points,
            "target_layout_points": target_layout_points,
            "target_spatial_points": target_spatial_points,
            "descriptor_recovery_loss": descriptor_recovery_loss,
            "descriptor_recovery_cd": descriptor_recovery_cd,
        }

    @torch.no_grad()
    def collect_reconstruction_diagnostics(
        self,
        pts,
        edge_indices,
        descriptor=None,
        batch_ids=None,
        anchor_id=None,
        obj_points_spatial=None,
        compare_unconditional=None,
    ):
        if descriptor is None:
            raise ValueError("descriptor must be provided for reconstruction diagnostics.")

        if compare_unconditional is None:
            compare_unconditional = self.diagnostics_compare_unconditional

        obj_feat, _, diffusion_context = self.build_object_graph_features(
            pts,
            edge_indices,
            descriptor,
            batch_ids,
            anchor_id=anchor_id,
            mask_ratio=self.object_mask_ratio,
            edge_mask_ratio=0.1,
            istrain=False,
        )
        shape_features = self.shape_condition_head(obj_feat)
        diffusion_target, visible_tokens, visible_token_positions = self.prepare_diffusion_inputs(
            pts,
            descriptor,
            diffusion_context,
            obj_points_spatial=obj_points_spatial,
        )
        diagnostics = self.point_diffusion.diagnose_reconstruction(
            diffusion_target,
            shape_features,
            num_points=pts.shape[1],
            token_condition=visible_tokens,
            token_positions=visible_token_positions,
            compare_unconditional=compare_unconditional,
        )

        outputs = {
            "shape_features": shape_features.detach(),
            "target_points": diffusion_target.detach(),
            "conditioned_one_step_points": diagnostics["conditioned_one_step_points"].detach(),
            "conditioned_sample_points": diagnostics["conditioned_sample_points"].detach(),
            "conditioned_one_step_x0_mse": float(diagnostics["conditioned_one_step_x0_mse"]),
            "conditioned_one_step_cd": float(diagnostics["conditioned_one_step_cd"]),
            "conditioned_sample_cd": float(diagnostics["conditioned_sample_cd"]),
            "direct_spatial_diffusion_only": self.direct_spatial_diffusion_only,
        }

        if not self.direct_spatial_diffusion_only:
            _recovered_points, _target_spatial_points, descriptor_recovery_loss, descriptor_recovery_cd = self.compute_descriptor_recovery_metric(
                diffusion_target,
                descriptor,
                spatial_points=obj_points_spatial,
            )
            outputs.update({
                "descriptor_recovery_l1": float(descriptor_recovery_loss.item()),
                "descriptor_recovery_cd": float(descriptor_recovery_cd),
            })

        if compare_unconditional and "unconditioned_sample_points" in diagnostics:
            outputs.update({
                "unconditioned_one_step_points": diagnostics["unconditioned_one_step_points"].detach(),
                "unconditioned_sample_points": diagnostics["unconditioned_sample_points"].detach(),
                "unconditioned_one_step_x0_mse": float(diagnostics["unconditioned_one_step_x0_mse"]),
                "unconditioned_one_step_cd": float(diagnostics["unconditioned_one_step_cd"]),
                "unconditioned_sample_cd": float(diagnostics["unconditioned_sample_cd"]),
                "condition_gain_one_step_cd": float(diagnostics["condition_gain_one_step_cd"]),
                "condition_gain_sample_cd": float(diagnostics["condition_gain_sample_cd"]),
            })

        return outputs
