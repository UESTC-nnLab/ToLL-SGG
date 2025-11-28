#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:46:24 2020

@author: sc
"""
from torch_geometric.typing import Adj, Tensor
import torch
from torch_geometric.nn.conv import MessagePassing
from src.model.model_utils.networks_base import mySequential
from torch_geometric.utils import scatter
def MLP(channels: list, do_bn=False, on_last=False, drop_out=None):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    offset = 0 if on_last else 1
    for i in range(1, n):
        layers.append(
            torch.nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-offset):
            if do_bn:
                layers.append(torch.nn.BatchNorm1d(channels[i]))
            layers.append(torch.nn.ReLU())
            
            if drop_out is not None:
                layers.append(torch.nn.Dropout(drop_out))
    return mySequential(*layers)


def build_mlp(dim_list, activation='relu', do_bn=False,
              dropout=0, on_last=False):
   layers = []
   for i in range(len(dim_list) - 1):
     dim_in, dim_out = dim_list[i], dim_list[i + 1]
     layers.append(torch.nn.Linear(dim_in, dim_out))
     final_layer = (i == len(dim_list) - 2)
     if not final_layer or on_last:
       if do_bn:
         layers.append(torch.nn.BatchNorm1d(dim_out))
       if activation == 'relu':
         layers.append(torch.nn.ReLU())
       elif activation == 'leakyrelu':
         layers.append(torch.nn.LeakyReLU())
     if dropout > 0:
       layers.append(torch.nn.Dropout(p=dropout))
   return torch.nn.Sequential(*layers)


class Gen_Index(torch.nn.Module):
    """ Gathers the source and target node features for each edge. """
    def __init__(self):
        super().__init__()
        
    def forward(self, x: Tensor, edge_index: Adj) -> tuple[Tensor, Tensor]:
        """
        Args:
            x (Tensor): Node feature matrix with shape [num_nodes, num_features].
            edge_index (Adj): Graph connectivity with shape [2, num_edges].

        Returns:
            A tuple (x_i, x_j) containing:
            - x_i (Tensor): Features of target nodes for each edge.
            - x_j (Tensor): Features of source nodes for each edge.
        """
        # The first row of edge_index contains source node indices (j)
        # The second row contains target node indices (i)
        source_idx, target_idx = edge_index[:,0], edge_index[:,1]
        
        # Directly gather features using advanced indexing
        x_i = x[target_idx]
        x_j = x[source_idx]
        
        return x_i, x_j

class Aggre_Index(torch.nn.Module):
    """ Aggregates messages to their target nodes. """
    def __init__(self, aggr: str = 'add'):
        super().__init__()
        self.aggr = aggr
        
    def forward(self, x: Tensor, edge_index: Adj, dim_size: int) -> Tensor:
        """
        Args:
            x (Tensor): The source tensor of messages to aggregate (e.g., edge features),
                        with shape [num_edges, num_features].
            edge_index (Adj): Graph connectivity with shape [2, num_edges].
            dim_size (int): The number of nodes in the output tensor.

        Returns:
            Tensor: The aggregated node features with shape [dim_size, num_features].
        """
        # We aggregate messages onto the target nodes (i), which are in the second row.
        target_idx = edge_index[:,1]
        
        # Use the scatter utility to perform the aggregation
        return scatter(src=x, index=target_idx, dim=0, dim_size=dim_size, reduce=self.aggr)

if __name__ == '__main__':
    flow = 'source_to_target'
    # flow = 'target_to_source'
    g = Gen_Index(flow = flow)
    
    edge_index = torch.LongTensor([[0,1,2],
                                  [2,1,0]])
    x = torch.zeros([3,5])
    x[0,:] = 0
    x[1,:] = 1
    x[2,:] = 2
    x_i,x_j = g(x,edge_index)
    print('x_i',x_i)
    print('x_j',x_j)
    
    tmp = torch.zeros_like(x_i)
    tmp = torch.zeros([5,2])
    edge_index = torch.LongTensor([[0,1,2,1,0],
                                  [2,1,1,1,1]])
    for i in range(5):
        tmp[i] = -i
    aggr = Aggre_Index(flow=flow,aggr='max')
    xx = aggr(tmp, edge_index,dim_size=x.shape[0])
    print(x)
    print(xx)