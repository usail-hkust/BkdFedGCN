#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import utils
from copy import deepcopy
from torch_geometric.nn import GCNConv,GATConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
from Node_level_Models.helpers.func_utils import accuracy


# class GAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, heads):
#         super().__init__()
#         self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
#         # On the Pubmed dataset, use `heads` output heads in `conv2`.
#         self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
#                              concat=False, dropout=0.6)
#
#     def forward(self, x, edge_index):
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = F.elu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x


class GAT(nn.Module):

    def __init__(self, nfeat, nhid, nclass, heads=8,dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, self_loop=True, device=None):

        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GATConv(nfeat,nhid,heads,dropout=dropout)
        self.gc2 = GATConv(heads*nhid, nclass, concat=False, dropout=dropout)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None

        self.conv1 = GATConv(nfeat, nhid, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(heads*nhid, nclass, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index, edge_weight=None): 
        # x = F.dropout(x, p=self.dropout, training=self.training)    # optional
        # x = F.elu(self.gc1(x, edge_index, edge_weight))   # may apply later
        # #x = F.elu(self.gc1(x, edge_index))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, edge_index, edge_weight)
        # #x = self.gc2(x, edge_index)
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)


        return F.log_softmax(x,dim=1)

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self,global_model, features, edge_index, edge_weight, labels, idx_train,args, idx_val=None, train_iters=200, initialize=True, verbose=False):
        if initialize:
            self.initialize()
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features
        self.labels = torch.tensor(labels, dtype=torch.long)


        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            loss_train, loss_val, acc_train, acc_val =  self._train_with_val(global_model,self.labels, idx_train, idx_val, train_iters, verbose,args)
        return  loss_train, loss_val, acc_train, acc_val
    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        #optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            #loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output

    def _train_with_val(self, global_model,labels, idx_train, idx_val, train_iters, verbose, args):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = -10

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            #loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            if args.agg_method == "FedProx":
                # compute proximal_term
                proximal_term = 0.0
                for w, w_t in zip(self.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)

                loss_train = loss_train + (args.mu / 2) * proximal_term


            loss_train.backward()
            optimizer.step()



            self.eval()
            with torch.no_grad():
                output = self.forward(self.features, self.edge_index,self.edge_weight)
                #loss_val = F.nll_loss(output[idx_val], labels[idx_val])
                loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
                acc_val = accuracy(output[idx_val], labels[idx_val])
                acc_train = accuracy(output[idx_train], labels[idx_train])
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                print("acc_val: {:.4f}".format(acc_val))
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        return loss_train.item(), loss_val.item(), acc_train, acc_val

    def test(self, features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        acc_test = accuracy(output[idx_test], labels[idx_test])
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)
    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels,idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test]==labels[idx_test]).nonzero().flatten()   # return a tensor
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return acc_test,correct_nids
# %%
