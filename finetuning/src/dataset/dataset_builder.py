#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/hyc/hyc_work/sceneGraph/SGG_DIR')
from xml.dom.minidom import Node

from src.dataset.dataset_diffPoint import PdiffDatasetGraph
from src.dataset.dataset_3dssg import SSGDatasetGraph
from src.dataset.dataset_3dssg_pretrain import SSGPretrainDatasetGraph
def build_dataset(split,
                for_train,
                point_sample_num=1024,
                point_union_num=1024*2,
                root_ScanNet="/home/hyc/hyc_work/sceneGraph/ScanNet_1/ScanNet/scans",
                json_path="/home/hyc/hyc_work/sceneGraph/ScanNet_1/training_samples2.json",
                max_edges=-1):
    
    dataset = PdiffDatasetGraph(
                split,
                for_train,
                point_sample_num=point_sample_num,
                point_union_num=point_union_num,
                root_ScanNet=root_ScanNet,
                json_path=json_path,
                max_edges=max_edges
    )
    return dataset


def build_pretrain_dataset(config, for_train=True):
    pretrain_dataset = getattr(config, "PRETRAIN_DATASET", "scannet_subgraph").lower()

    if pretrain_dataset == "3dssg":
        dataset_cfg = getattr(config, "dataset", None)
        sg_model_cfg = getattr(config, "sg_model", None)
        splits = getattr(config, "PRETRAIN_SPLITS", ["train_scans", "validation_scans"])
        return SSGPretrainDatasetGraph(
            splits=splits,
            multi_rel_outputs=True,
            shuffle_objs=True,
            use_rgb=getattr(sg_model_cfg, "USE_RGB", False),
            use_normal=getattr(sg_model_cfg, "USE_NORMAL", False),
            label_type="3RScan160",
            for_train=for_train,
            max_edges=getattr(dataset_cfg, "max_edges", -1),
            root=getattr(dataset_cfg, "root", None),
            root_3rscan=getattr(dataset_cfg, "root_3rscan", None),
            num_points=getattr(dataset_cfg, "num_points", 128),
            num_points_union=getattr(dataset_cfg, "num_points_union", 256),
        )

    return build_dataset(
        "train_scannet",
        for_train,
        root_ScanNet=config.root_ScanNet,
        json_path=config.json_path,
    )

def build_dataset_for_clustering(config=None):

    root = None
    root_3rscan = None
    if config is not None and hasattr(config, 'dataset'):
        if hasattr(config.dataset, 'root'):
            root = config.dataset.root
        if hasattr(config.dataset, 'root_3rscan'):
            root_3rscan = config.dataset.root_3rscan

    dataset = SSGDatasetGraph(
        split='validation_scans', # 'train_scans'
        multi_rel_outputs=True,# True
        shuffle_objs=True, # True
        use_rgb=False, # False
        use_normal=False, # False
        label_type='3RScan160',
        for_train= True,
        max_edges = -1, # -1
        root=root,
        root_3rscan=root_3rscan,
    )
    return dataset

if __name__ == '__main__':
    import numpy as np

    dataset = build_dataset("train_scannet", True)
    obj_points, rel_points, descriptor, edge_indices, anchor_index,\
        obj_points_spatial, cur_obj_texts = dataset[4]
    print
