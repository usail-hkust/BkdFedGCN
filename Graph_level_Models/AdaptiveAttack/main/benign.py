import sys, os
sys.path.append(os.path.abspath('..'))

import time
import pickle
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from Graph_level_Models.AdaptiveAttack.utils.datareader import GraphData, DataReader
from Graph_level_Models.AdaptiveAttack.utils.datareader import TuDatasetstoGraph
from Graph_level_Models.AdaptiveAttack.utils.batch import collate_batch
from Graph_level_Models.AdaptiveAttack.model.gcn import GCN
#from Graph_level_Models.AdaptiveAttack.model.gat import GAT
#from Graph_level_Models.AdaptiveAttack.model.sage import GraphSAGE
#from Graph_level_Models.AdaptiveAttack.config import parse_args
import  tqdm

import argparse


def add_data_group(group):
    group.add_argument('--seed', type=int, default=123)
    group.add_argument('--dataset', type=str, default='AIDS', help="used dataset")
    group.add_argument('--data_path', type=str, default='../dataset', help="the directory used to save dataset")
    group.add_argument('--use_nlabel_asfeat', action='store_true', help="use node labels as (part of) node features")
    group.add_argument('--use_org_node_attr', action='store_true',
                       help="use node attributes as (part of) node features")
    group.add_argument('--use_degree_asfeat', action='store_true', help="use node degrees as (part of) node features")
    group.add_argument('--data_verbose', action='store_true', help="print detailed dataset info")
    group.add_argument('--save_data', action='store_true')


def add_model_group(group):
    group.add_argument('--model', type=str, default='gcn', help="used model")
    group.add_argument('--train_ratio', type=float, default=0.5, help="ratio of trainset from whole dataset")
    group.add_argument('--hidden_dim', nargs='+', default=[64, 16], type=int,
                       help='constrain how much products a vendor can have')
    group.add_argument('--num_head', type=int, default=2, help="GAT head number")

    group.add_argument('--batch_size', type=int, default=16)
    group.add_argument('--train_epochs', type=int, default=40)
    group.add_argument('--lr', type=float, default=0.01)
    group.add_argument('--lr_decay_steps', nargs='+', default=[25, 35], type=int)
    group.add_argument('--weight_decay', type=float, default=5e-4)
    group.add_argument('--dropout', type=float, default=0.5)
    group.add_argument('--train_verbose', action='store_true', help="print training details")
    group.add_argument('--log_every', type=int, default=1, help='print every x epoch')
    group.add_argument('--eval_every', type=int, default=5, help='evaluate every x epoch')

    group.add_argument('--clean_model_save_path', type=str, default='../save/model/clean')
    group.add_argument('--save_clean_model', action='store_true')


def add_atk_group(group):
    group.add_argument('--bkd_gratio_train', type=float, default=0.1, help="backdoor graph ratio in trainset")
    group.add_argument('--bkd_gratio_test', type=float, default=0.5, help="backdoor graph ratio in testset")
    group.add_argument('--bkd_num_pergraph', type=int, default=1, help="number of backdoor triggers per graph")
    group.add_argument('--bkd_size', type=int, default=5, help="number of nodes for each trigger")
    group.add_argument('--target_class', type=int, default=None, help="the targeted node/graph label")

    group.add_argument('--gtn_layernum', type=int, default=3, help="layer number of GraphTrojanNet")
    group.add_argument('--pn_rate', type=float, default=1,
                       help="ratio between trigger-embedded graphs (positive) and benign ones (negative)")
    group.add_argument('--gtn_input_type', type=str, default='2hop',
                       help="how to process org graphs before inputting to GTN")

    group.add_argument('--resample_steps', type=int, default=3, help="# iterations to re-select graph samples")
    group.add_argument('--bilevel_steps', type=int, default=4, help="# bi-level optimization iterations")
    group.add_argument('--gtn_lr', type=float, default=0.01)
    group.add_argument('--gtn_epochs', type=int, default=20, help="# attack epochs")
    group.add_argument('--topo_activation', type=str, default='sigmoid',
                       help="activation function for topology generator")
    group.add_argument('--feat_activation', type=str, default='relu', help="activation function for feature generator")
    group.add_argument('--topo_thrd', type=float, default=0.5, help="threshold for topology generator")
    group.add_argument('--feat_thrd', type=float, default=0,
                       help="threshold for feature generator (only useful for binary feature)")

    group.add_argument('--lambd', type=float, default=1, help="a hyperparameter to balance attack loss components")
    # group.add_argument('--atk_verbose', action='store_true', help="print attack details")
    group.add_argument('--save_bkd_model', action='store_true')
    group.add_argument('--bkd_model_save_path', type=str, default='../save/model/bkd')


def parse_args():
    gta_parser = argparse.ArgumentParser()
    data_group = gta_parser.add_argument_group(title="Data-related configuration")
    model_group = gta_parser.add_argument_group(title="Model-related configuration")
    atk_group = gta_parser.add_argument_group(title="Attack-related configuration")

    add_data_group(data_group)
    add_model_group(model_group)
    add_atk_group(atk_group)

    return gta_parser.parse_args()


def run(args, datasets, device):
    batch_size = 16
    lr, weight_decay = 0.01, 5e-4
    # parmater
    hidden_dim = [64, 16]
    dropout = 0.5
    lr_decay_steps =  [25, 35]
    train_epochs = 40

    datasets = TuDatasetstoGraph(datasets, args)
    datasets = GraphData(datasets)
    #print("datasets",datasets)
    loader = DataLoader(datasets,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate_batch)

    # prepare model
    in_dim = args.surrogate_num_features
    out_dim = args.surrogate_num_classes

    model = GCN(in_dim, out_dim, hidden_dim=hidden_dim, dropout=dropout)



    train_params = list(filter(lambda p: p.requires_grad, model.parameters()))


    # training
    loss_fn = F.cross_entropy

    optimizer = optim.Adam(train_params, lr=lr, weight_decay=weight_decay, betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(optimizer, lr_decay_steps, gamma=0.1)
    print("Start training surrogate model")
    model.to(device)
    for epoch in tqdm.tqdm(range(train_epochs)):
        model.train()
        start = time.time()
        train_loss, n_samples = 0, 0
        for batch_id, data in enumerate(loader):

            for i in range(len(data)):
                data[i] = data[i].to(device)
            # if args.use_cont_node_attr:
            #     data[0] = norm_features(data[0])

            optimizer.zero_grad()
            output = model(data)
            if len(output.shape)==1:
                output = output.unsqueeze(0)
            loss = loss_fn(output, data[4])
            loss.backward()
            optimizer.step()
            scheduler.step()

            time_iter = time.time() - start
            train_loss += loss.item() * len(output)
            n_samples += len(output)

        train_verbose = True
        log_every = 1

        if train_verbose and (epoch % log_every == 0 or epoch == train_epochs - 1):
            print('Surrogate Model Train Epoch: %d\tLoss: %.4f (avg: %.4f) \tsec/iter: %.2f' % (
                epoch + 1, loss.item(), train_loss / n_samples, time_iter / (batch_id + 1)))

    args.surrogate_gtn_layernum = 3
    args.surrogate_resample_steps = 3
    args.surrogate_bilevel_steps = 4
    # args.surrogate_topo_thrd = 0.5
    args.surrogate_gtn_epochs = 20
    args.surrogate_feat_thrd  = 0.0
    args.surrogate_topo_activation = 'sigmoid'
    args.surrogate_feat_activation = 'relu'
    args.gtn_input_type = '2hop'
    args.gtn_lr = 0.01
    return model,args


if __name__ == '__main__':
    args = parse_args()
    run(args)