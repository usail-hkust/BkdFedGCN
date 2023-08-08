"""
    File to load dataset based on user control from main file
"""
import os
#os.chdir('../') # go to root folder of the pro
import sys
#sys.path.append('/home/nfs/federated_learning_jx/federated_learning/GNN_common/data')

import torch
import numpy as np



def ogba_data(dataset):
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

def Amazon_data(dataset):


    graph = dataset[0]

    labels = graph.y.numpy()

    # num_classes = torch.unique(torch.from_numpy(labels)).size(0)
    # print("num classes", num_classes)

    dev_size = int(labels.shape[0] * 0.1)
    test_size = int(labels.shape[0] * 0.8)

    perm = np.random.permutation(labels.shape[0])
    test_index = perm[:test_size]
    dev_index = perm[test_size:test_size + dev_size]

    data_index = np.arange(labels.shape[0])
    test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)
    dev_mask = torch.tensor(np.in1d(data_index, dev_index), dtype=torch.bool)
    train_mask = ~(dev_mask + test_mask)
    test_mask = test_mask.reshape(1, -1).squeeze(0)
    val_mask = dev_mask.reshape(1, -1).squeeze(0)
    train_mask = train_mask.reshape(1, -1).squeeze(0)






   #graph.num_nodes = int(labels.shape[0])


    graph.train_mask = train_mask


    graph.val_mask = val_mask


    graph.test_mask = test_mask
    # extract indices of non-zero entries
    # indices = torch.nonzero(adj)

    # # create edge_index tensor by transposing the indices tensor
    # edge_index = indices.t()
    # graph.edge_index = edge_index

    return graph
def Coauthor_data(dataset):


    graph = dataset[0]

    labels = graph.y.numpy()
    dev_size = int(labels.shape[0] * 0.1)
    test_size = int(labels.shape[0] * 0.8)

    perm = np.random.permutation(labels.shape[0])
    test_index = perm[:test_size]
    dev_index = perm[test_size:test_size + dev_size]

    data_index = np.arange(labels.shape[0])
    test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)
    dev_mask = torch.tensor(np.in1d(data_index, dev_index), dtype=torch.bool)
    train_mask = ~(dev_mask + test_mask)
    test_mask = test_mask.reshape(1, -1).squeeze(0)
    val_mask = dev_mask.reshape(1, -1).squeeze(0)
    train_mask = train_mask.reshape(1, -1).squeeze(0)






   #graph.num_nodes = int(labels.shape[0])


    graph.train_mask = train_mask


    graph.val_mask = val_mask


    graph.test_mask = test_mask
    # extract indices of non-zero entries
    # indices = torch.nonzero(adj)

    # # create edge_index tensor by transposing the indices tensor
    # edge_index = indices.t()
    # graph.edge_index = edge_index

    return graph