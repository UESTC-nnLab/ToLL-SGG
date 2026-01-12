#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/hyc/hyc_work/sceneGraph/SGG_DIR')
from xml.dom.minidom import Node

from src.dataset.dataset_diffPoint import PdiffDatasetGraph
from src.dataset.dataset_3dssg import SSGDatasetGraph
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

def build_dataset_for_clustering():

    dataset = SSGDatasetGraph(
        split='validation_scans', # 'train_scans'
        multi_rel_outputs=True,# True
        shuffle_objs=True, # True
        use_rgb=False, # False
        use_normal=False, # False
        label_type='3RScan160',
        for_train= True,
        max_edges = -1 # -1
    )
    return dataset

if __name__ == '__main__':
    import numpy as np

    dataset = build_dataset("train_scannet", True)
    obj_points, rel_points, descriptor, edge_indices, anchor_index,\
        obj_points_spatial, cur_obj_texts = dataset[4]
    print
