import datetime
import json
import os
import random

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset.DataLoader import CustomDataLoader, collate_fn_mmg, collate_fn_mmg_diff
from src.dataset.dataset_builder import build_dataset_for_clustering, build_pretrain_dataset
from src.dataset.dataset_diffPoint import visualize_scenes_plt_with_points
from src.model.diff_trans.models.PointDif_dino import PointDif
from src.model.diff_trans.models.clustering import evaluate_and_plot_clustering, visualize_with_gt
from src.model.optimizer.scheduler import get_freeze_warmup_scheduler


def normalize_state_dict_keys(state_dict):
    return {
        (k[7:] if k.startswith('module.') else k): v
        for k, v in state_dict.items()
    }


def extract_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        if 'pointdif' in checkpoint:
            return checkpoint['pointdif']
    return checkpoint


def get_param_groups(module, base_lr, weight_decay, amsgrad):
    decay_params = []
    no_decay_params = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "norm" in name or "bn" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = []
    if decay_params:
        groups.append({
            'params': decay_params,
            'lr': base_lr,
            'weight_decay': weight_decay,
            'amsgrad': amsgrad,
        })
    if no_decay_params:
        groups.append({
            'params': no_decay_params,
            'lr': base_lr,
            'weight_decay': 0.0,
            'amsgrad': amsgrad,
        })
    return groups


class Pdiff4SSG_Pretraining:
    def __init__(self, config, val_cls_mode=False):
        self.config = config
        self.model_name = 'pdiff_SGG'
        self.save_dir = self.config.PATH
        os.makedirs(self.save_dir, exist_ok=True)

        if val_cls_mode:
            self.dataset_train = build_dataset_for_clustering(self.config)
        else:
            self.dataset_train = build_pretrain_dataset(self.config, for_train=True)

        self.total = self.config.total = max(1, len(self.dataset_train) // self.config.Batch_Size)
        self.max_iteration = self.config.max_iteration = int(
            float(self.config.MAX_EPOCHES) * len(self.dataset_train) // self.config.Batch_Size
        )

        log_root = getattr(self.config, 'analysis_save_dir', None) or os.path.join(os.getcwd(), "outputs")
        log_dir = os.path.join(log_root, "log_runs", "experiment_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.model = PointDif(self.config).cuda()

        param_groups = []
        param_groups.extend(get_param_groups(self.model.mask_encoder, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.rel_encoder_3d, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.ca_net, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.mlp_3d, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.point_diffusion.net, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.shape_condition_head, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.object_text_projector, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.layout_position_head, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.layout_scale_head, float(config.LR), self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.extend(get_param_groups(self.model.mmg, float(config.LR) / 2, self.config.W_DECAY, self.config.AMSGRAD))
        param_groups.append({
            'params': self.model.mask_token,
            'lr': float(config.LR),
            'weight_decay': self.config.W_DECAY,
            'amsgrad': self.config.AMSGRAD,
        })
        param_groups.append({
            'params': self.model.edge_mask_token.parameters(),
            'lr': float(config.LR),
            'weight_decay': self.config.W_DECAY,
            'amsgrad': self.config.AMSGRAD,
        })

        self.optimizer = optim.AdamW(param_groups)
        self.lr_scheduler = get_freeze_warmup_scheduler(self.optimizer, self.total * 10, self.config.max_iteration)
        self.optimizer.zero_grad()
        self.diagnostics_enable = bool(getattr(self.config, 'DIAGNOSTICS_ENABLE', False))
        self.diagnostics_interval = int(getattr(self.config, 'DIAGNOSTICS_INTERVAL', 10))
        self.diagnostics_sample_count = int(getattr(self.config, 'DIAGNOSTICS_SAMPLE_COUNT', 2))
        self.diagnostics_save_visuals = bool(getattr(self.config, 'DIAGNOSTICS_SAVE_VISUALS', True))
        self.diagnostics_compare_unconditional = bool(
            getattr(self.config, 'DIAGNOSTICS_COMPARE_UNCONDITIONAL', True)
        )
        self._diagnostic_sample_indices = None

    def load_pretrained_mask_encoder(self, checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        raw_state_dict = extract_checkpoint_state_dict(checkpoint)
        raw_state_dict = normalize_state_dict_keys(raw_state_dict)
        mask_encoder_dict = {}
        for key, value in raw_state_dict.items():
            if key.startswith('mask_encoder.'):
                mask_encoder_dict[key.replace('mask_encoder.', '', 1)] = value
        if mask_encoder_dict:
            self.model.mask_encoder.load_state_dict(mask_encoder_dict, strict=True)
            print(f"Success! Loaded {len(mask_encoder_dict)} keys into self.model.mask_encoder.")
        else:
            print("Error: No keys starting with 'mask_encoder' found in the checkpoint!")

    def maybe_load_mask_encoder_init(self):
        mask_init_path = getattr(self.config, 'MASK_ENCODER_INIT_PATH', None)
        if not mask_init_path:
            return
        if os.path.exists(mask_init_path):
            self.load_pretrained_mask_encoder(mask_init_path)
        else:
            print(f"Warning: MASK_ENCODER_INIT_PATH '{mask_init_path}' not found. Skip mask encoder init.")

    @torch.no_grad()
    def data_processing_train_pdiff(self, items):
        atlas_embeddings = None
        atlas_valid_mask = None
        edge_labels = None
        if len(items) == 11:
            obj_points, obj_points_spatial, descriptor, edge_indices, anchor_ids, cur_obj_texts, batch_ids, obj_labels, edge_labels, atlas_embeddings, atlas_valid_mask = items
        elif len(items) == 10:
            obj_points, obj_points_spatial, descriptor, edge_indices, anchor_ids, cur_obj_texts, batch_ids, obj_labels, atlas_embeddings, atlas_valid_mask = items
        elif len(items) == 9:
            obj_points, obj_points_spatial, descriptor, edge_indices, anchor_ids, cur_obj_texts, batch_ids, atlas_embeddings, atlas_valid_mask = items
            obj_labels = None
        elif len(items) == 8:
            obj_points, obj_points_spatial, descriptor, edge_indices, anchor_ids, cur_obj_texts, batch_ids, obj_labels = items
        else:
            obj_points, obj_points_spatial, descriptor, edge_indices, anchor_ids, cur_obj_texts, batch_ids = items
            obj_labels = None

        obj_points = obj_points.permute(0, 2, 1).contiguous()
        tensors_to_cuda = [obj_points, edge_indices, descriptor, batch_ids, obj_points_spatial]
        if obj_labels is not None:
            tensors_to_cuda.append(obj_labels)
        if edge_labels is not None:
            tensors_to_cuda.append(edge_labels)
        if atlas_embeddings is not None:
            tensors_to_cuda.append(atlas_embeddings)
        if atlas_valid_mask is not None:
            tensors_to_cuda.append(atlas_valid_mask)

        moved_tensors = self.cuda(*tensors_to_cuda)
        cursor = 0
        obj_points = moved_tensors[cursor]; cursor += 1
        edge_indices = moved_tensors[cursor]; cursor += 1
        descriptor = moved_tensors[cursor]; cursor += 1
        batch_ids = moved_tensors[cursor]; cursor += 1
        obj_points_spatial = moved_tensors[cursor]; cursor += 1
        if obj_labels is not None:
            obj_labels = moved_tensors[cursor]
            cursor += 1
        if edge_labels is not None:
            edge_labels = moved_tensors[cursor]
            cursor += 1
        if atlas_embeddings is not None:
            atlas_embeddings = moved_tensors[cursor]
            cursor += 1
        if atlas_valid_mask is not None:
            atlas_valid_mask = moved_tensors[cursor]

        return (
            obj_points,
            descriptor,
            edge_indices.long(),
            anchor_ids,
            batch_ids.long(),
            obj_points_spatial,
            cur_obj_texts,
            obj_labels,
            edge_labels,
            atlas_embeddings,
            atlas_valid_mask,
        )

    @torch.no_grad()
    def data_processing_train(self, items):
        obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids, _ = items
        obj_points = obj_points.permute(0, 2, 1).contiguous()
        obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = self.cuda(
            obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids
        )
        return obj_points, gt_class, gt_rel_cls, edge_indices.long(), descriptor, batch_ids.long()

    def train(self):
        train_loader = CustomDataLoader(
            config=self.config,
            dataset=self.dataset_train,
            batch_size=self.config.Batch_Size,
            num_workers=getattr(self.config, 'WORKERS', 4),
            drop_last=True,
            shuffle=True,
            collate_fn=collate_fn_mmg_diff,
        )

        start_epoch = 0
        init_weights_path = getattr(self.config, 'INIT_WEIGHTS_PATH', None)
        resume_path = getattr(self.config, 'RESUME_PATH', None)

        if init_weights_path and os.path.exists(init_weights_path):
            checkpoint = torch.load(init_weights_path, map_location=torch.device('cuda'))
            state_dict = normalize_state_dict_keys(extract_checkpoint_state_dict(checkpoint))
            self.model.load_state_dict(state_dict, strict=False)
            print("[Info] Loaded initialization weights.")
        elif resume_path and os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=torch.device('cuda'))
            state_dict = normalize_state_dict_keys(extract_checkpoint_state_dict(checkpoint))
            self.model.load_state_dict(state_dict, strict=False)
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            print(f"[Info] Resuming from epoch {start_epoch}.")
        elif resume_path:
            print(f"Warning: RESUME_PATH '{resume_path}' not found. Starting from scratch.")
            self.maybe_load_mask_encoder_init()
        else:
            self.maybe_load_mask_encoder_init()

        self.model.epoch = start_epoch
        self.model.train()

        direct_spatial_diffusion_only = bool(getattr(self.config, 'DIRECT_SPATIAL_DIFFUSION_ONLY', False))

        for epoch in range(start_epoch, self.config.MAX_EPOCHES):
            num_batches = len(train_loader)
            log_interval = max(1, int(getattr(self.config, 'LOG_INTERVAL', 100)))
            print(f'[Epoch {epoch + 1}/{self.config.MAX_EPOCHES}] Start training with {num_batches} steps.')
            log_sums = {'tot': 0.0, 'dif': 0.0}
            if direct_spatial_diffusion_only:
                log_sums['cd'] = 0.0
            else:
                log_sums.update({
                    'txt': 0.0,
                    'pos': 0.0,
                    'sca': 0.0,
                    'lay': 0.0,
                })
            log_count = 0

            for batch_idx, items in enumerate(train_loader):
                obj_points, descriptor, edge_indices, anchor_ids, batch_ids, obj_points_spatial, cur_obj_texts, obj_labels, edge_labels, atlas_embeddings, atlas_valid_mask = self.data_processing_train_pdiff(items)

                total_loss, diff_loss, _diff_aux_loss, text_contrastive_loss, position_loss, scale_loss, layout_recon_loss, total_metric, _edge_feat, _obj_feat = self.model(
                    obj_points.permute(0, 2, 1).contiguous(),
                    edge_indices,
                    obj_points_spatial,
                    descriptor=descriptor,
                    batch_ids=batch_ids,
                    anchor_id=anchor_ids,
                    istrain=True,
                    cur_obj_texts=cur_obj_texts,
                    obj_labels=obj_labels,
                    edge_labels=edge_labels,
                    atlas_embeddings=atlas_embeddings,
                    atlas_valid_mask=atlas_valid_mask,
                )

                global_step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Train/Total_Loss', total_loss.item(), global_step)
                self.writer.add_scalar('Train/Diff_Loss', diff_loss.item(), global_step)
                if direct_spatial_diffusion_only:
                    self.writer.add_scalar('Train/CD_Metric', total_metric, global_step)
                else:
                    self.writer.add_scalar('Train/Obj_Text_Contrastive_Loss', text_contrastive_loss.item(), global_step)
                    self.writer.add_scalar('Train/Position_Loss', position_loss.item(), global_step)
                    self.writer.add_scalar('Train/Scale_Loss', scale_loss.item(), global_step)
                    self.writer.add_scalar('Train/Layout_Recon_Loss', layout_recon_loss.item(), global_step)
                    self.writer.add_scalar('Train/Layout_Metric', total_metric, global_step)

                log_sums['tot'] += total_loss.item()
                log_sums['dif'] += diff_loss.item()
                if direct_spatial_diffusion_only:
                    log_sums['cd'] += total_metric
                else:
                    log_sums['txt'] += text_contrastive_loss.item()
                    log_sums['pos'] += position_loss.item()
                    log_sums['sca'] += scale_loss.item()
                    log_sums['lay'] += layout_recon_loss.item()
                log_count += 1

                if batch_idx == 0 or (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
                    avg_tot = log_sums['tot'] / log_count
                    avg_dif = log_sums['dif'] / log_count
                    start_step = batch_idx + 2 - log_count
                    if direct_spatial_diffusion_only:
                        avg_cd = log_sums['cd'] / log_count
                        print(
                            f"[Epoch {epoch + 1}/{self.config.MAX_EPOCHES}] "
                            f"steps {start_step}-{batch_idx + 1}/{num_batches} avg "
                            f"tot={avg_tot:.4f} "
                            f"dif={avg_dif:.4f} "
                            f"cd={avg_cd:.4f}"
                        )
                    else:
                        avg_txt = log_sums['txt'] / log_count
                        avg_pos = log_sums['pos'] / log_count
                        avg_sca = log_sums['sca'] / log_count
                        avg_lay = log_sums['lay'] / log_count
                        print(
                            f"[Epoch {epoch + 1}/{self.config.MAX_EPOCHES}] "
                            f"steps {start_step}-{batch_idx + 1}/{num_batches} avg "
                            f"tot={avg_tot:.4f} "
                            f"dif={avg_dif:.4f} "
                            f"txt={avg_txt:.4f} "
                            f"pos={avg_pos:.4f} "
                            f"sca={avg_sca:.4f} "
                            f"lay={avg_lay:.4f}"
                        )
                    for key in log_sums:
                        log_sums[key] = 0.0
                    log_count = 0

                self.backward(total_loss)

            self.model.epoch = epoch + 1
            completed_epoch = epoch + 1
            if completed_epoch % 10 == 0:
                save_path = os.path.join(self.save_dir, f'model_epoch_{completed_epoch}.pth')
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.lr_scheduler.state_dict(),
                    'loss': total_loss.item(),
                }
                torch.save(checkpoint, save_path)
                print(f'[Epoch {completed_epoch}] Saved checkpoint to {save_path}')

            if bool(getattr(self.config, 'VALIDATE_EVERY_N_EPOCH', True)):
                valid_interval = int(getattr(self.config, 'VALID_INTERVAL', 10))
                if valid_interval > 0 and completed_epoch % valid_interval == 0:
                    self.validation_for_cls(epoch=completed_epoch)

            if int(getattr(self.config, 'SAVE_SCENE_INTERVAL', 0)) > 0:
                scene_interval = int(getattr(self.config, 'SAVE_SCENE_INTERVAL', 0))
                if completed_epoch % scene_interval == 0:
                    self.save_scene_reconstruction(completed_epoch)

            self.run_reconstruction_diagnostics(completed_epoch)

    @torch.no_grad()
    def validation_for_cls(self, epoch=None):
        dataset_val = build_dataset_for_clustering(self.config)
        val_loader = CustomDataLoader(
            config=self.config,
            dataset=dataset_val,
            batch_size=self.config.Batch_Size,
            num_workers=0,
            drop_last=False,
            shuffle=False,
            collate_fn=collate_fn_mmg,
        )

        self.model.eval()
        all_edge_feats = []
        all_obj_feats = []
        all_gt_obj_cls = []
        all_gt_rel_cls = []

        for items in tqdm(val_loader, desc="Validation", dynamic_ncols=True):
            obj_points, gt_class, gt_rel_cls, edge_indices, descriptor, batch_ids = self.data_processing_train(items)
            gcn_edge_feature_3d, gcn_obj_feature_3d = self.model.forward_cls(
                obj_points.permute(0, 2, 1).contiguous(),
                edge_indices,
                descriptor=descriptor,
                batch_ids=batch_ids,
                istrain=False,
            )
            all_edge_feats.append(gcn_edge_feature_3d.detach().cpu())
            all_obj_feats.append(gcn_obj_feature_3d.detach().cpu())
            all_gt_obj_cls.append(gt_class.detach().cpu())
            all_gt_rel_cls.append(gt_rel_cls.detach().cpu())

        all_edge_feats = torch.cat(all_edge_feats, dim=0)
        all_obj_feats = torch.cat(all_obj_feats, dim=0)
        all_gt_obj_cls = torch.cat(all_gt_obj_cls, dim=0)
        all_gt_rel_cls = torch.cat(all_gt_rel_cls, dim=0)

        cm_save_dir = os.path.join(self.config.analysis_save_dir, "cm_save")
        os.makedirs(cm_save_dir, exist_ok=True)
        vis_save_dir = os.path.join(self.config.analysis_save_dir, "clustering_vis")
        os.makedirs(vis_save_dir, exist_ok=True)

        evaluate_and_plot_clustering(
            all_obj_feats,
            all_gt_obj_cls,
            save_path=os.path.join(cm_save_dir, f"cls_obj_{epoch}.png"),
            metric_prefix="val_obj",
        )
        evaluate_and_plot_clustering(
            all_edge_feats,
            all_gt_rel_cls,
            save_path=os.path.join(cm_save_dir, f"cls_edge_{epoch}.png"),
            metric_prefix="val_edge",
        )
        visualize_with_gt(
            all_obj_feats,
            all_gt_obj_cls,
            title_prefix="Object Features",
            save_path=os.path.join(vis_save_dir, f"object_features_{epoch}.png"),
        )
        visualize_with_gt(
            all_edge_feats,
            all_gt_rel_cls,
            title_prefix="Edge Features",
            save_path=os.path.join(vis_save_dir, f"edge_features_{epoch}.png"),
            ignore_zero_label=True,
        )
        self.model.train()

    def cuda(self, *args):
        return [item.to(self.config.DEVICE) for item in args]

    def save(self, epoch):
        self.model.save(epoch)

    def backward(self, loss):
        loss.backward()
        grad_clip_norm = float(getattr(self.config, 'GRAD_CLIP_NORM', 0.0))
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

    def _scene_recon_indices(self, epoch):
        dataset_size = len(self.dataset_train)
        if dataset_size == 0:
            return []

        sample_count = min(int(getattr(self.config, "SAVE_SCENE_SAMPLE_COUNT", 10)), dataset_size)
        rng = random.Random(int(getattr(self.config, "SEED", 0)) + int(epoch))
        return rng.sample(range(dataset_size), sample_count)

    def _diagnostic_indices(self):
        if self._diagnostic_sample_indices is not None:
            return self._diagnostic_sample_indices

        dataset_size = len(self.dataset_train)
        if dataset_size == 0:
            self._diagnostic_sample_indices = []
            return self._diagnostic_sample_indices

        sample_count = min(max(1, self.diagnostics_sample_count), dataset_size)
        rng = random.Random(int(getattr(self.config, "SEED", 0)) + 7919)
        self._diagnostic_sample_indices = rng.sample(range(dataset_size), sample_count)
        return self._diagnostic_sample_indices

    def _extract_diag_scalars(self, diag_outputs):
        scalars = {}
        for key, value in diag_outputs.items():
            if isinstance(value, bool):
                scalars[key] = float(value)
            elif isinstance(value, (int, float)):
                scalars[key] = float(value)
        return scalars

    def _summarize_diag_records(self, records):
        if not records:
            return {}

        metric_keys = [
            key for key, value in records[0].items()
            if isinstance(value, (int, float)) and key not in {"dataset_idx"}
        ]
        summary = {}
        for key in metric_keys:
            values = [float(record[key]) for record in records if key in record]
            if values:
                summary[key] = sum(values) / len(values)
        return summary

    @torch.no_grad()
    def run_reconstruction_diagnostics(self, epoch, infer_model=None):
        if not self.diagnostics_enable:
            return
        if self.diagnostics_interval <= 0 or epoch % self.diagnostics_interval != 0:
            return

        infer_model = infer_model or self.model
        sample_indices = self._diagnostic_indices()
        if not sample_indices:
            return

        output_dir = os.path.join(
            self.config.analysis_save_dir,
            "reconstruction_diagnostics",
            f"epoch_{epoch:03d}",
        )
        os.makedirs(output_dir, exist_ok=True)

        records = []
        infer_model.eval()
        try:
            for save_rank, dataset_idx in enumerate(sample_indices):
                sample = self.dataset_train[dataset_idx]
                scene_items = collate_fn_mmg_diff([sample])
                obj_points, descriptor, edge_indices, anchor_ids, batch_ids, obj_points_spatial, cur_obj_texts, obj_labels, edge_labels, atlas_embeddings, atlas_valid_mask = self.data_processing_train_pdiff(scene_items)
                diag_outputs = infer_model.collect_reconstruction_diagnostics(
                    obj_points.permute(0, 2, 1).contiguous(),
                    edge_indices,
                    descriptor=descriptor,
                    batch_ids=batch_ids,
                    anchor_id=anchor_ids,
                    obj_points_spatial=obj_points_spatial,
                    compare_unconditional=self.diagnostics_compare_unconditional,
                )

                scene_meta = None
                if hasattr(self.dataset_train, "samples_list") and dataset_idx < len(self.dataset_train.samples_list):
                    scene_meta = self.dataset_train.samples_list[dataset_idx]
                scene_id = scene_meta.get("scene_id", f"sample_{dataset_idx:05d}") if scene_meta else f"sample_{dataset_idx:05d}"

                if self.diagnostics_save_visuals:
                    target_points = diag_outputs["target_points"].detach()
                    visualize_scenes_plt_with_points(
                        diag_outputs["conditioned_one_step_points"].detach(),
                        target_points,
                        output_filename=os.path.join(output_dir, f"{save_rank:02d}_{scene_id}_one_step_cond.png"),
                    )
                    visualize_scenes_plt_with_points(
                        diag_outputs["conditioned_sample_points"].detach(),
                        target_points,
                        output_filename=os.path.join(output_dir, f"{save_rank:02d}_{scene_id}_sample_cond.png"),
                    )
                    if "unconditioned_one_step_points" in diag_outputs:
                        visualize_scenes_plt_with_points(
                            diag_outputs["unconditioned_one_step_points"].detach(),
                            target_points,
                            output_filename=os.path.join(output_dir, f"{save_rank:02d}_{scene_id}_one_step_uncond.png"),
                        )
                    if "unconditioned_sample_points" in diag_outputs:
                        visualize_scenes_plt_with_points(
                            diag_outputs["unconditioned_sample_points"].detach(),
                            target_points,
                            output_filename=os.path.join(output_dir, f"{save_rank:02d}_{scene_id}_sample_uncond.png"),
                        )

                record = {
                    "dataset_idx": int(dataset_idx),
                    "scene_id": scene_id,
                }
                record.update(self._extract_diag_scalars(diag_outputs))
                records.append(record)

            summary = self._summarize_diag_records(records)
            with open(os.path.join(output_dir, "records.json"), "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            if self.writer is not None:
                for key, value in summary.items():
                    self.writer.add_scalar(f'Diagnostics/{key}', value, epoch)

            if summary:
                ordered_keys = [
                    key for key in [
                        "conditioned_one_step_cd",
                        "conditioned_sample_cd",
                        "unconditioned_one_step_cd",
                        "unconditioned_sample_cd",
                        "condition_gain_one_step_cd",
                        "condition_gain_sample_cd",
                    ]
                    if key in summary
                ]
                summary_str = " ".join(f"{key}={summary[key]:.4f}" for key in ordered_keys)
                print(f"[Epoch {epoch}] Reconstruction diagnostics: {summary_str}")
        finally:
            infer_model.train()

    @torch.no_grad()
    def save_scene_reconstruction(self, epoch):
        sample_indices = self._scene_recon_indices(epoch)
        if not sample_indices:
            return

        output_dir = os.path.join(
            self.config.analysis_save_dir,
            "scene_reconstruction",
            f"epoch_{epoch:03d}",
        )
        os.makedirs(output_dir, exist_ok=True)

        self.model.eval()
        try:
            for save_rank, dataset_idx in enumerate(sample_indices):
                sample = self.dataset_train[dataset_idx]
                scene_items = collate_fn_mmg_diff([sample])
                obj_points, descriptor, edge_indices, anchor_ids, batch_ids, obj_points_spatial, cur_obj_texts, obj_labels, edge_labels, atlas_embeddings, atlas_valid_mask = self.data_processing_train_pdiff(scene_items)
                recon_outputs = self.model.reconstruct_scene_points(
                    obj_points.permute(0, 2, 1).contiguous(),
                    edge_indices,
                    descriptor=descriptor,
                    batch_ids=batch_ids,
                    anchor_id=anchor_ids,
                    obj_points_spatial=obj_points_spatial,
                )

                scene_meta = None
                if hasattr(self.dataset_train, "samples_list") and dataset_idx < len(self.dataset_train.samples_list):
                    scene_meta = self.dataset_train.samples_list[dataset_idx]
                scene_id = scene_meta.get("scene_id", f"sample_{dataset_idx:05d}") if scene_meta else f"sample_{dataset_idx:05d}"

                output_path = os.path.join(
                    output_dir,
                    f"{save_rank:02d}_{scene_id}.png",
                )
                visualize_scenes_plt_with_points(
                    recon_outputs["pred_layout_points"].detach(),
                    recon_outputs["target_layout_points"].detach(),
                    output_filename=output_path,
                )
        finally:
            self.model.train()
