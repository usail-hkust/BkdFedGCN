"""
    File to load dataset based on user control from main file
"""
import os
#os.chdir('../') # go to root folder of the pro
import sys
#sys.path.append('/home/nfs/federated_learning_jx/federated_learning/GNN_common/data')

import torch




def ogba_arxiv_data(dataset):
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph = dataset[0]

    num_nodes = graph.num_nodes

    # Add masks to graph
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    graph.train_mask = train_mask

    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val_idx] = True
    graph.val_mask = val_mask

    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True
    graph.test_mask = test_mask
    # extract indices of non-zero entries
    # indices = torch.nonzero(adj)

    # # create edge_index tensor by transposing the indices tensor
    # edge_index = indices.t()
    # graph.edge_index = edge_index
    return graph