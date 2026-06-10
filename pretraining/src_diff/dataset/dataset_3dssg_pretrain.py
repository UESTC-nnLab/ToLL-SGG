import os
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.utils.data as data

from src.dataset.dataset_3dssg import dataset_loading_3RScan, load_mesh
from src.utils import op_utils


class SSGPretrainDatasetGraph(data.Dataset):
    def __init__(
        self,
        splits: Sequence[str],
        multi_rel_outputs: bool,
        shuffle_objs: bool,
        use_rgb: bool,
        use_normal: bool,
        label_type: str,
        for_train: bool,
        max_edges: int = -1,
        root: str = None,
        root_3rscan: str = None,
        num_points: int = 128,
        num_points_union: int = 256,
    ):
        if not splits:
            raise ValueError("splits must not be empty")

        self.for_train = for_train
        self.root = root if root is not None else "/home/hyc/hyc_work/sceneGraph/Diff_SGG/data/3DSSG_subset"
        self.root_3rscan = root_3rscan if root_3rscan is not None else "/home/hyc/hyc_work/sceneGraph/SGG_dataset/3RScan"
        self.label_file = "labels.instances.align.annotated.v2.ply"
        self.label_type = label_type
        self.multi_rel_outputs = multi_rel_outputs
        self.shuffle_objs = shuffle_objs
        self.use_rgb = use_rgb
        self.use_normal = use_normal
        self.max_edges = max_edges
        self.num_points = num_points
        self.num_points_union = num_points_union

        self.classNames = None
        self.relationNames = None
        merged_data = {"scans": []}
        selected_scans = set()

        for split in splits:
            class_names, relation_names, split_data, split_selected_scans = dataset_loading_3RScan(
                self.root, self.root, split
            )
            if self.classNames is None:
                self.classNames = class_names
                self.relationNames = relation_names
            merged_data["scans"].extend(split_data["scans"])
            selected_scans.update(split_selected_scans)

        if self.multi_rel_outputs and self.relationNames and self.relationNames[0].lower() == "none":
            self.relationNames = self.relationNames[1:]

        self.relationship_json, self.objs_json, self.scans = self.read_relationship_json(merged_data, selected_scans)
        if len(self.scans) == 0:
            raise RuntimeError("No 3DSSG pretraining scans were loaded.")

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, index):
        scan_id = self.scans[index]
        scan_id_no_split = scan_id.rsplit("_", 1)[0]
        instance_to_label = self.objs_json[scan_id]

        scene_path = os.path.join(self.root_3rscan, scan_id_no_split)
        data = load_mesh(scene_path, self.label_file, self.use_rgb, self.use_normal)
        points = data["points"]
        instances = data["instances"]

        valid_instances = set(np.unique(instances).tolist())
        valid_instances.discard(0)

        all_nodes_cur = [obj_id for obj_id in sorted(instance_to_label.keys()) if obj_id in valid_instances]
        selected_rels = [
            rel
            for rel in self.relationship_json[scan_id]
            if int(rel[0]) in valid_instances and int(rel[1]) in valid_instances
        ]
        cur_obj_texts = [instance_to_label[obj_id] for obj_id in all_nodes_cur]

        (
            obj_points,
            obj_points_spatial,
            edge_indices,
            descriptor,
            obj_points_view2,
            descriptor_view2,
            gt_class,
            gt_rels,
        ) = self.data_preparation_multi_view(
            points,
            instances,
            self.num_points,
            self.num_points_union,
            all_nodes_cur,
            selected_rels,
            instance_to_label,
            self.classNames,
            padding=0.2,
        )

        while (edge_indices.numel() == 0 or gt_class.numel() == 0 or gt_rels.numel() == 0) and self.for_train:
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)

        anchor_index = int(np.random.randint(gt_class.shape[0])) if gt_class.numel() > 0 else 0

        return (
            obj_points,
            obj_points_spatial,
            descriptor,
            edge_indices,
            anchor_index,
            obj_points_view2,
            descriptor_view2,
            cur_obj_texts,
            gt_class,
            gt_rels,
        )

    def read_relationship_json(self, data, selected_scans: Iterable[str]):
        rel, objs, scans = {}, {}, []

        for scan_i in data["scans"]:
            if scan_i["scan"] == "fa79392f-7766-2d5c-869a-f5d6cfb62fc6" and self.label_file == "labels.instances.align.annotated.v2.ply":
                continue
            if scan_i["scan"] not in selected_scans:
                continue

            relationships_i = []
            for relationship in scan_i["relationships"]:
                relationships_i.append(relationship)

            objects_i = {}
            for obj_id, name in scan_i["objects"].items():
                objects_i[int(obj_id)] = name

            key = scan_i["scan"] + "_" + str(scan_i["split"])
            rel[key] = relationships_i
            objs[key] = objects_i
            scans.append(key)

        return rel, objs, scans

    def zero_mean(self, point):
        mean = torch.mean(point, dim=0)
        point -= mean.unsqueeze(0)
        return point

    def subs_rel_idx(self, arrangeidx, selected_rels):
        selected_rels = np.array(selected_rels, dtype=np.int64)
        for key in arrangeidx:
            selected_rels[np.where(selected_rels == key)] = arrangeidx[key]
        return selected_rels

    def apply_elastic_distortion(self, points, granularity=0.2, magnitude=0.05):
        min_coord = np.min(points, axis=0)
        max_coord = np.max(points, axis=0)
        dimensions = max_coord - min_coord
        max_dim = np.max(dimensions)

        num_centers = int(max(5, np.prod(dimensions) / (granularity ** 3)))
        num_centers = min(num_centers, 50)

        if len(points) > num_centers:
            center_indices = np.random.choice(len(points), num_centers, replace=False)
            centers = points[center_indices]
        else:
            centers = points

        vectors = np.random.normal(0, magnitude, (len(centers), 3))
        dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)

        sigma = max_dim * granularity
        weights = np.exp(-(dists ** 2) / (2 * sigma ** 2))
        deformation = np.dot(weights, vectors)
        deformation = deformation / (np.sum(weights, axis=1, keepdims=True) + 1e-8)
        return points + deformation

    def data_preparation_multi_view(
        self,
        points,
        instances,
        num_points,
        num_points_union,
        all_nodes_cur,
        selected_rels,
        instance_to_label,
        class_names,
        padding=0.2,
    ):
        del num_points_union
        del padding

        num_objects = len(all_nodes_cur)
        dim_point = points.shape[-1]

        obj_points = torch.zeros([num_objects, num_points, dim_point])
        obj_points_spatial = torch.zeros([num_objects, num_points, dim_point])
        descriptor = torch.zeros([num_objects, 11])
        obj_points_view2 = torch.zeros([num_objects, num_points, dim_point])
        descriptor_view2 = torch.zeros([num_objects, 11])
        gt_class = torch.zeros([num_objects], dtype=torch.long)

        arrangeidx = {}
        aug_scale = np.random.uniform(0.5, 2.0, (1, 3))
        aug_noise_sigma = 0.005
        aug_elastic_mag = 0.05
        view2_raw_points_buffer = []

        for i, instance_id in enumerate(all_nodes_cur):
            arrangeidx[instance_id] = i
            obj_pointset_raw = points[np.where(instances == instance_id)[0]]

            choice1 = np.random.choice(len(obj_pointset_raw), num_points, replace=True)
            obj_pointset_v1 = obj_pointset_raw[choice1, :]

            descriptor[i] = op_utils.gen_descriptor(torch.from_numpy(obj_pointset_v1)[:, :3])
            gt_class[i] = class_names.index(instance_to_label[instance_id])

            obj_pts_tensor_v1 = torch.from_numpy(obj_pointset_v1.astype(np.float32))
            obj_points_spatial[i] = obj_pts_tensor_v1.clone()
            obj_pts_tensor_v1[:, :3] = self.zero_mean(obj_pts_tensor_v1[:, :3])
            radius = torch.max(torch.sqrt(torch.sum(obj_pts_tensor_v1[:, :3] ** 2, dim=1)))
            if radius < 1e-6:
                radius = obj_pts_tensor_v1.new_tensor(1.0)
            obj_pts_tensor_v1[:, :3] = obj_pts_tensor_v1[:, :3] / radius
            obj_points[i] = obj_pts_tensor_v1

            choice2 = np.random.choice(len(obj_pointset_raw), num_points, replace=True)
            view2_raw_points_buffer.append(obj_pointset_raw[choice2, :].copy())

        if len(view2_raw_points_buffer) > 0:
            all_points_v2 = np.concatenate(view2_raw_points_buffer, axis=0)
            all_points_v2[:, :3] = all_points_v2[:, :3] * aug_scale
            all_points_v2[:, :3] = self.apply_elastic_distortion(all_points_v2[:, :3], magnitude=aug_elastic_mag)
            all_points_v2[:, :3] += np.random.normal(0, aug_noise_sigma, all_points_v2[:, :3].shape)

            current_idx = 0
            for i in range(num_objects):
                obj_pts_aug = all_points_v2[current_idx : current_idx + num_points, :]
                current_idx += num_points

                descriptor_view2[i] = op_utils.gen_descriptor(torch.from_numpy(obj_pts_aug)[:, :3])
                obj_pts_tensor_v2 = torch.from_numpy(obj_pts_aug.astype(np.float32))
                obj_pts_tensor_v2[:, :3] = self.zero_mean(obj_pts_tensor_v2[:, :3])
                radius = torch.max(torch.sqrt(torch.sum(obj_pts_tensor_v2[:, :3] ** 2, dim=1)))
                if radius < 1e-6:
                    radius = obj_pts_tensor_v2.new_tensor(1.0)
                obj_pts_tensor_v2[:, :3] = obj_pts_tensor_v2[:, :3] / radius
                obj_points_view2[i] = obj_pts_tensor_v2

        batch_size, num_sampled_points, _ = obj_points_spatial.shape
        obj_points_spatial = obj_points_spatial.view(batch_size * num_sampled_points, -1)
        obj_points_spatial[:, :3] = self.zero_mean(obj_points_spatial[:, :3])
        obj_points_spatial = obj_points_spatial.view(batch_size, num_sampled_points, -1)

        edge_pairs = [(int(edge[0]), int(edge[1])) for edge in selected_rels if int(edge[0]) in arrangeidx and int(edge[1]) in arrangeidx]
        if len(edge_pairs) == 0:
            edge_indices = torch.zeros((0, 2), dtype=torch.long)
            gt_rels = torch.zeros((0, len(self.relationNames)), dtype=torch.float32)
        else:
            edge_indices = torch.tensor(self.subs_rel_idx(arrangeidx, edge_pairs), dtype=torch.long)
            gt_rels = torch.zeros((len(edge_pairs), len(self.relationNames)), dtype=torch.float32)
            for rel_idx, rel in enumerate(selected_rels):
                src_id = int(rel[0])
                dst_id = int(rel[1])
                if src_id not in arrangeidx or dst_id not in arrangeidx:
                    continue
                rel_name = rel[3]
                if rel_name not in self.relationNames:
                    continue
                mapped_edge = self.subs_rel_idx(arrangeidx, [(src_id, dst_id)])[0]
                mapped_edge = tuple(int(v) for v in mapped_edge.tolist())
                edge_pos = edge_pairs.index((src_id, dst_id))
                rel_label_idx = self.relationNames.index(rel_name)
                gt_rels[edge_pos, rel_label_idx] = 1.0

        return (
            obj_points,
            obj_points_spatial,
            edge_indices,
            descriptor,
            obj_points_view2,
            descriptor_view2,
            gt_class,
            gt_rels,
        )
