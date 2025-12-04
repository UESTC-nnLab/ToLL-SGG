#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/honsen/honsen/SceneGraph/SG_pretrain_diff')
from xml.dom.minidom import Node

from src.dataset.dataset_diffPoint import PdiffDatasetGraph
def build_dataset(split,
                for_train,
                point_sample_num=1024,
                point_union_num=1024*2,
                root_ScanNet="/home/honsen/tartan/ScanNet/scans",
                json_path="/home/honsen/tartan/ScanNet/training_samples2.json",
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


if __name__ == '__main__':
    import numpy as np

    dataset = build_dataset("train_scannet", True)
    obj_points, rel_points, descriptor, edge_indices = dataset[0]
    print
