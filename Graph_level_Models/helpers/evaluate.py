# -*- coding: utf-8 -*-

import time
import torch
import torchvision
from torch import nn, optim

def accuracy(scores, targets):
    #scores = scores.detach().argmax(dim=1)
    _, predicted = torch.max(scores.detach().data, 1)


    acc = ( predicted == targets).sum().item()
    return acc


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n



def gnn_evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    for batch_graphs, batch_labels in data_iter:
        net.eval()
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(torch.long)
        batch_labels = batch_labels.to(device)

        batch_scores = net.forward(batch_graphs, batch_x, batch_e)
        tmp_acc = accuracy(batch_scores, batch_labels)
        acc_sum += tmp_acc

        n += batch_labels.size(0)


    return acc_sum / n
