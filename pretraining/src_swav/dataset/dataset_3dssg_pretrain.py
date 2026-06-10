import os
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.utils.data as data

from src.dataset.atlasnet_cache import fetch_object_embeddings, load_embedding_cache
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
        num_points: int = 1024,
        num_points_union: int = 256,
        atlas_embedding_path: str = None,
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
        self.min_object_points = max(1, self.num_points // 2)
        self.atlas_embedding_path = atlas_embedding_path
        self.atlas_embedding_cache = load_embedding_cache(atlas_embedding_path)

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
            (int(rel[0]), int(rel[1]))
            for rel in self.relationship_json[scan_id]
            if int(rel[0]) in valid_instances and int(rel[1]) in valid_instances
        ]
        (
            obj_points,
            obj_points_spatial,
            edge_indices,
            descriptor,
            obj_points_view2,
            descriptor_view2,
            gt_class,
            kept_nodes,
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

        cur_obj_texts = [instance_to_label[obj_id] for obj_id in kept_nodes]
        atlas_embeddings, atlas_valid_mask = fetch_object_embeddings(
            self.atlas_embedding_cache,
            scan_id,
            kept_nodes,
        )

        while (edge_indices.numel() == 0 or gt_class.numel() == 0) and self.for_train:
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
            atlas_embeddings,
            atlas_valid_mask,
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

    def _largest_connected_component(self, nodes, selected_rels):
        if len(nodes) <= 1:
            return list(nodes)

        adjacency = {node: set() for node in nodes}
        for src, dst in selected_rels:
            if src in adjacency and dst in adjacency:
                adjacency[src].add(dst)
                adjacency[dst].add(src)

        components = []
        visited = set()

        for node in nodes:
            if node in visited:
                continue

            stack = [node]
            visited.add(node)
            component = []

            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

            component_set = set(component)
            edge_count = sum(1 for src, dst in selected_rels if src in component_set and dst in component_set)
            components.append((sorted(component), edge_count))

        best_component, _ = max(
            components,
            key=lambda item: (len(item[0]), item[1], -min(item[0])),
        )
        return best_component

    def _filter_instances_for_connected_graph(self, points, instances, all_nodes_cur, selected_rels):
        eligible_nodes = []
        pointsets_by_instance = {}

        for instance_id in all_nodes_cur:
            obj_pointset_raw = points[np.where(instances == instance_id)[0]]
            if obj_pointset_raw.shape[0] >= self.min_object_points:
                eligible_nodes.append(instance_id)
                pointsets_by_instance[instance_id] = obj_pointset_raw

        if len(eligible_nodes) == 0:
            return [], [], {}

        eligible_node_set = set(eligible_nodes)
        filtered_rels = [
            (src, dst)
            for src, dst in selected_rels
            if src in eligible_node_set and dst in eligible_node_set
        ]

        kept_nodes = self._largest_connected_component(eligible_nodes, filtered_rels)
        kept_node_set = set(kept_nodes)
        kept_rels = [
            (src, dst)
            for src, dst in filtered_rels
            if src in kept_node_set and dst in kept_node_set
        ]
        kept_pointsets = {instance_id: pointsets_by_instance[instance_id] for instance_id in kept_nodes}

        return kept_nodes, kept_rels, kept_pointsets

    def _sample_or_upsample_pointset(self, obj_pointset_raw, target_num_points):
        num_raw_points = obj_pointset_raw.shape[0]
        point_dim = obj_pointset_raw.shape[1]

        if num_raw_points == target_num_points:
            return obj_pointset_raw.astype(np.float32, copy=True)

        if num_raw_points > target_num_points:
            choice = np.random.choice(num_raw_points, target_num_points, replace=False)
            return obj_pointset_raw[choice, :].astype(np.float32, copy=True)

        if num_raw_points < self.min_object_points:
            return None

        base_points = obj_pointset_raw.astype(np.float32, copy=True)
        xyz = base_points[:, :3]
        num_new_points = target_num_points - num_raw_points

        if num_raw_points <= 1:
            tiled = np.repeat(base_points, target_num_points, axis=0)
            return tiled.astype(np.float32, copy=False)

        pairwise_dists = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
        k = min(3, num_raw_points - 1)
        neighbor_indices = np.argsort(pairwise_dists, axis=1)[:, 1 : k + 1]

        if neighbor_indices.shape[1] == 0:
            choice = np.random.choice(num_raw_points, target_num_points, replace=True)
            return base_points[choice, :].astype(np.float32, copy=True)

        local_dists = pairwise_dists[np.arange(num_raw_points)[:, None], neighbor_indices]
        avg_local_dists = local_dists.mean(axis=1)

        if not np.all(np.isfinite(avg_local_dists)) or float(avg_local_dists.sum()) <= 1e-12:
            sampling_prob = None
        else:
            sampling_prob = avg_local_dists / avg_local_dists.sum()

        center_indices = np.random.choice(
            num_raw_points,
            size=num_new_points,
            replace=True,
            p=sampling_prob,
        )

        new_points = np.zeros((num_new_points, point_dim), dtype=np.float32)
        for new_idx, center_idx in enumerate(center_indices):
            center_point = base_points[center_idx]
            neighbors = neighbor_indices[center_idx]

            if len(neighbors) == 1:
                neighbor_point = base_points[neighbors[0]]
                alpha = np.random.rand()
                interpolated_point = (1.0 - alpha) * center_point + alpha * neighbor_point
            else:
                neighbor_point_1 = base_points[neighbors[0]]
                neighbor_point_2 = base_points[neighbors[1]]
                rand_1 = np.random.rand()
                rand_2 = np.random.rand()
                sqrt_rand_1 = np.sqrt(rand_1)
                interpolated_point = (
                    (1.0 - sqrt_rand_1) * center_point
                    + sqrt_rand_1 * (1.0 - rand_2) * neighbor_point_1
                    + sqrt_rand_1 * rand_2 * neighbor_point_2
                )

            new_points[new_idx] = interpolated_point

        upsampled = np.concatenate([base_points, new_points], axis=0)
        permutation = np.random.permutation(target_num_points)
        return upsampled[permutation].astype(np.float32, copy=False)

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

        all_nodes_cur, selected_rels, raw_pointsets = self._filter_instances_for_connected_graph(
            points, instances, all_nodes_cur, selected_rels
        )

        num_objects = len(all_nodes_cur)
        dim_point = points.shape[-1]

        if num_objects == 0:
            empty_points = torch.zeros([0, num_points, dim_point], dtype=torch.float32)
            empty_descriptor = torch.zeros([0, 11], dtype=torch.float32)
            empty_edges = torch.zeros((0, 2), dtype=torch.long)
            empty_class = torch.zeros([0], dtype=torch.long)
            return (
                empty_points,
                empty_points.clone(),
                empty_edges,
                empty_descriptor,
                empty_points.clone(),
                empty_descriptor.clone(),
                empty_class,
                [],
            )

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
            obj_pointset_raw = raw_pointsets[instance_id]

            obj_pointset_v1 = self._sample_or_upsample_pointset(obj_pointset_raw, num_points)
            obj_pointset_v2 = self._sample_or_upsample_pointset(obj_pointset_raw, num_points)

            if obj_pointset_v1 is None or obj_pointset_v2 is None:
                raise RuntimeError(
                    f"Instance {instance_id} passed filtering but failed point preparation with "
                    f"{len(obj_pointset_raw)} raw points."
                )

            descriptor[i] = op_utils.gen_descriptor(torch.from_numpy(obj_pointset_v1)[:, :3])
            gt_class[i] = class_names.index(instance_to_label[instance_id])

            obj_pts_tensor_v1 = torch.from_numpy(obj_pointset_v1.astype(np.float32))
            obj_points_spatial[i] = obj_pts_tensor_v1.clone()
            obj_pts_tensor_v1[:, :3] = self.zero_mean(obj_pts_tensor_v1[:, :3])
            obj_points[i] = obj_pts_tensor_v1

            view2_raw_points_buffer.append(obj_pointset_v2.copy())

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
                obj_points_view2[i] = obj_pts_tensor_v2

        batch_size, num_sampled_points, _ = obj_points_spatial.shape
        obj_points_spatial = obj_points_spatial.view(batch_size * num_sampled_points, -1)
        obj_points_spatial[:, :3] = self.zero_mean(obj_points_spatial[:, :3])
        obj_points_spatial = obj_points_spatial.view(batch_size, num_sampled_points, -1)

        if len(selected_rels) > 0:
            edge_indices = torch.from_numpy(self.subs_rel_idx(arrangeidx, selected_rels)).long()
        else:
            edge_indices = torch.zeros((0, 2), dtype=torch.long)

        return (
            obj_points,
            obj_points_spatial,
            edge_indices,
            descriptor,
            obj_points_view2,
            descriptor_view2,
            gt_class,
            all_nodes_cur,
        )
