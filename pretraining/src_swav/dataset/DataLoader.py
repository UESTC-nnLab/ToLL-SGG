from pydoc import describe
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
import numpy as np

class CustomSingleProcessDataLoaderIter(_SingleProcessDataLoaderIter):
    def __init__(self,loader):
        super().__init__(loader)
    def IndexIter(self):
        return self._sampler_iter
    
class CustomMultiProcessingDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self,loader):
        super().__init__(loader)
    def IndexIter(self):
        return self._sampler_iter


class CustomDataLoader(DataLoader):
    def __init__(self, config, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None):
        if worker_init_fn is None:
            worker_init_fn = self.init_fn
        super().__init__(dataset, batch_size, shuffle, sampler,
                 batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context)
        self.config = config
        
    def init_fn(self, worker_id):
        np.random.seed(self.config.SEED + worker_id)
        
    def __iter__(self):
        if self.num_workers == 0:
            return CustomSingleProcessDataLoaderIter(self)
        else:
            return CustomMultiProcessingDataLoaderIter(self)

def collate_fn_obj(batch):
    # batch
    
    name_list, instance2mask_list, obj_point_list, obj_label_list = [], [], [], []
    for i in batch:
        name_list.append(i[0])
        instance2mask_list.append(i[1])
        obj_point_list.append(i[2])
        obj_label_list.append(i[4])
    return name_list, instance2mask_list, torch.cat(obj_point_list, dim=0), torch.cat(obj_label_list, dim=0)

def collate_fn_rel(batch):
    # batch
    name_list, instance2mask_list, obj_label_list, rel_point_list, rel_label_list, edge_indices = [], [], [], [], [], []
    for i in batch:
        assert len(i) == 7
        name_list.append(i[0])
        instance2mask_list.append(i[1])
        obj_label_list.append(i[4])
        rel_point_list.append(i[3])
        rel_label_list.append(i[5])
        edge_indices.append(i[6])
    return name_list, instance2mask_list, torch.cat(obj_label_list, dim=0), torch.cat(rel_point_list, dim=0), torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0)

def collate_fn_obj_new(batch):
    # batch
    obj_point_list, obj_label_list = [], []
    for i in batch:
        obj_point_list.append(i[0])
        obj_label_list.append(i[2])
    return torch.cat(obj_point_list, dim=0), torch.cat(obj_label_list, dim=0)

def collate_fn_rel_new(batch):
    # batch
    rel_point_list, rel_label_list = [], []
    for i in batch:
        rel_point_list.append(i[1])
        rel_label_list.append(i[3])
    return torch.cat(rel_point_list, dim=0), torch.cat(rel_label_list, dim=0)


def collate_fn_all(batch):
    # batch
    obj_point_list, obj_label_list = [], []
    rel_point_list, rel_label_list = [], []
    edge_indices = []
    for i in batch:
        obj_point_list.append(i[0])
        obj_label_list.append(i[3])
        rel_point_list.append(i[2])
        rel_label_list.append(i[4])
        edge_indices.append(i[5])

    return torch.cat(obj_point_list, dim=0), torch.cat(obj_label_list, dim=0), torch.cat(rel_point_list, dim=0), torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0)

def collate_fn_all_des(batch):
    # batch
    obj_point_list, obj_label_list = [], []
    rel_label_list = []
    edge_indices, descriptor = [], []
    count = 0
    for i in batch:
        obj_point_list.append(i[0])
        obj_label_list.append(i[2])
        #rel_point_list.append(i[1])
        rel_label_list.append(i[3])
        edge_indices.append(i[4] + count)
        descriptor.append(i[5])
        # accumulate batch number to make edge_indices match correct object index
        count += i[0].shape[0]

    return torch.cat(obj_point_list, dim=0), torch.cat(obj_label_list, dim=0), torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0), torch.cat(descriptor, dim=0)

def collate_fn_all_2d(batch):
    # batch
    obj_point_list, obj_label_list, obj_2d_feats = [], [], []
    rel_label_list = []
    edge_indices, descriptor = [], []
    
    count = 0
    for i in batch:
        obj_point_list.append(i[0])
        obj_2d_feats.append(i[1])
        obj_label_list.append(i[3])
        #rel_point_list.append(i[2])
        rel_label_list.append(i[4])
        edge_indices.append(i[5] + count)
        descriptor.append(i[6])
        # accumulate batch number to make edge_indices match correct object index
        count += i[0].shape[0]

    return torch.cat(obj_point_list, dim=0), torch.cat(obj_2d_feats, dim=0), torch.cat(obj_label_list, dim=0), \
         torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0), torch.cat(descriptor, dim=0)

def collate_fn_det(batch):
    assert len(batch) == 1
    scene_points, obj_boxes, obj_labels, point_votes, point_votes_mask = [], [], [], [], []
    for i in range(len(batch)):
        scene_points.append(batch[i][0])
        obj_boxes.append(batch[i][1])
        obj_labels.append(batch[i][2])
        point_votes.append(batch[i][3])
        point_votes_mask.append(batch[i][4])
    
    scene_points = torch.stack(scene_points, dim=0)
    obj_boxes = torch.stack(obj_boxes, dim=0)
    obj_labels = torch.stack(obj_labels, dim=0)
    point_votes = torch.stack(point_votes, dim=0)
    point_votes_mask = torch.stack(point_votes_mask, dim=0)

    return scene_points, obj_boxes, obj_labels, point_votes, point_votes_mask


def collate_fn_mmg(batch):
    # batch
    obj_point_list, obj_label_list = [], []
    rel_label_list = []
    edge_indices, descriptor = [], []
    batch_ids = []
    scan_ids = []
    count = 0
    for i, b in enumerate(batch):
        obj_point_list.append(b[0])
        obj_label_list.append(b[1])
        rel_label_list.append(b[2])
        edge_indices.append(b[3] + count)
        descriptor.append(b[4])
        # accumulate batch number to make edge_indices match correct object index
        count += b[0].shape[0]
        # get batchs location
        batch_ids.append(torch.full((b[0].shape[0], 1), i))
        scan_ids.append(b[5])

    return torch.cat(obj_point_list, dim=0), torch.cat(obj_label_list, dim=0), \
         torch.cat(rel_label_list, dim=0), torch.cat(edge_indices, dim=0), \
         torch.cat(descriptor, dim=0), torch.cat(batch_ids, dim=0), scan_ids


def collate_fn_mmg_diff(batch):
    # batch
    cur_obj_texts = []
    obj_point_list = []
    edge_indices, descriptor = [], []
    batch_ids = []
    anchor_ids = []
    obj_points_spatial = []
    obj_points_view2 = []
    descriptor_view2 = []
    obj_class_labels = []
    atlas_embeddings = []
    atlas_valid_masks = []
    count = 0
    #  return obj_points, rel_points, descriptor, edge_indices
    for i, b in enumerate(batch):

        obj_point_list.append(b[0])
        obj_points_spatial.append(b[1])
        descriptor.append(b[2])
        edge_indices.append(b[3] + count)
        anchor_ids.append(b[4])
        obj_points_view2.append(b[5])
        descriptor_view2.append(b[6])
        cur_obj_texts+=b[7]
        if len(b) > 8 and b[8] is not None:
            obj_class_labels.append(b[8])
        if len(b) > 9 and b[9] is not None:
            atlas_embeddings.append(b[9])
        if len(b) > 10 and b[10] is not None:
            atlas_valid_masks.append(b[10])
        
        # accumulate batch number to make edge_indices match correct object index
        count += b[0].shape[0]

        # get batchs location
        batch_ids.append(torch.full((b[0].shape[0], 1), i))

    return  torch.cat(obj_point_list, dim=0), torch.cat(obj_points_spatial, dim=0),\
            torch.cat(descriptor, dim=0), torch.cat(edge_indices, dim=0), anchor_ids,\
            torch.cat(obj_points_view2, dim=0), torch.cat(descriptor_view2, dim=0),\
            cur_obj_texts, torch.cat(batch_ids, dim=0), \
            (torch.cat(obj_class_labels, dim=0) if len(obj_class_labels) > 0 else None), \
            (torch.cat(atlas_embeddings, dim=0) if len(atlas_embeddings) > 0 else None), \
            (torch.cat(atlas_valid_masks, dim=0) if len(atlas_valid_masks) > 0 else None)
