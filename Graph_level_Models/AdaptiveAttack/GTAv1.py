from Node_level_Models.models.GCN import GCN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
import dgl
import random

import  numpy as np
import tqdm
class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=None,layer_norm_first=False,use_ln=False):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(layer-2):
            self.convs.append(GCNConv(nhid,nhid))
            self.lns.append(nn.LayerNorm(nhid))
        self.lns.append(nn.LayerNorm(nhid))
        self.gc2 = GCNConv(nhid, nhid)
        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None
        self.weight_decay = weight_decay

        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

        self.MLP_layer = MLPReadout(nhid, nclass)


    def forward(self, x, edge_index, edge_weight=None):
        if(self.layer_norm_first):
            x = self.lns[0](x)
        i=0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index,edge_weight))
            if self.use_ln:
                x = self.lns[i+1](x)
            i+=1
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index,edge_weight)
        x = torch.mean(x, dim=0)  # mean pooling
        x = self.MLP_layer(x)
        return x.view([-1, self.nclass])


# %%
class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input > thrd, torch.tensor(1.0, device=device, requires_grad=True),
                          torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None


class GraphTrojanNet(nn.Module):
    # In the furture, we may use a GNN model to generate backdoor
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum - 1):
            layers.append(nn.Linear(nfeat, nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        self.layers = nn.Sequential(*layers).to(device)

        self.feat = nn.Linear(nfeat, nout * nfeat)
        self.edge = nn.Linear(nfeat, int(nout * (nout - 1) / 2))
        self.device = device

    def forward(self, input, thrd):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """

        GW = GradWhere.apply
        h = self.layers(input)

        feat = self.feat(h)
        edge_weight = self.edge(h)
        # feat = GW(feat, thrd, self.device)
        edge_weight = GW(edge_weight, thrd, self.device)

        return feat, edge_weight


class HomoLoss(nn.Module):
    def __init__(self, args, device):
        super(HomoLoss, self).__init__()
        self.args = args
        self.device = device

    def forward(self, trigger_edge_index, trigger_edge_weights, x, thrd):
        trigger_edge_index = trigger_edge_index[:, trigger_edge_weights > 0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]], x[trigger_edge_index[1]])

        loss = torch.relu(thrd - edge_sims).mean()
        # print(edge_sims.min())
        return loss



class Backdoor:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.weights = None
        feature_dim = args.feature_dim
        num_class = args.num_class
        self.hidden = args.hidden
        self.trigger_index = self.get_trigger_index(args.trigger_size)
        # initial a shadow model
        self.shadow_model = GCN(nfeat=feature_dim,
                                nhid=self.args.hidden,
                                nclass=num_class,
                                dropout=0.0, device=self.device).to(self.device)
        # initalize a trojanNet to generate trigger
        self.trojan = GraphTrojanNet(self.device, feature_dim, args.trigger_size, layernum=2).to(self.device)



    def get_trigger_index(self, trigger_node_list):
        edge_list = []
        for j in range(len(trigger_node_list)):
            for k in range(j):
                edge_list.append([trigger_node_list[j], trigger_node_list[k]])
        edge_index = torch.tensor(edge_list, device=self.device).long().T
        return edge_index

    def get_trojan_edge(self, trigger_node_list):
        # edge_list = []
        #
        # for idx in idx_attach:
        #     edges = self.trigger_index.clone()
        #     edges[0, 0] = idx
        #     edges[1, 0] = start
        #     edges[:, 1:] = edges[:, 1:] + start
        #
        #     edge_list.append(edges)
        #     start += trigger_size
        # edge_index = torch.cat(edge_list, dim=1)

        # # to undirected
        # # row, col = edge_index
        # row = torch.cat([edge_index[0], edge_index[1]])
        # col = torch.cat([edge_index[1], edge_index[0]])
        # edge_index = torch.stack([row, col])
        edge_index = self.trigger_index(trigger_node_list).clone()
        return edge_index

    def inject_trigger(self,G_trigger,trigger_list,train_trigger_graphs):
        ######################################################################
        print("Start injecting trigger into the poisoned train datasets")
        for i, data in enumerate(tqdm(train_trigger_graphs)):
            for j in range(len(trigger_list[i]) - 1):
                for k in range(j + 1, len(trigger_list[i])):
                    if (data[0].has_edges_between(trigger_list[i][j], trigger_list[i][k]) or data[0].has_edges_between(
                            trigger_list[i][k], trigger_list[i][j])) \
                            and G_trigger.has_edge(j, k) is False:
                        ids = data[0].edge_ids(torch.tensor([trigger_list[i][j], trigger_list[i][k]]),
                                               torch.tensor([trigger_list[i][k], trigger_list[i][j]]))
                        data[0].remove_edges(ids)
                    elif (data[0].has_edges_between(trigger_list[i][j], trigger_list[i][k]) or data[
                        0].has_edges_between(trigger_list[i][k], trigger_list[i][j])) is False \
                            and G_trigger.has_edge(j, k):
                        data[0].add_edges(torch.tensor([trigger_list[i][j], trigger_list[i][k]]),
                                          torch.tensor([trigger_list[i][k], trigger_list[i][j]]))

        return data

    def fit(self, graphs, labels, idx_attach):

        args = self.args
        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.trojan.train()
        for i in range(args.trojan_epochs):
            for j in range(len(graphs)):
