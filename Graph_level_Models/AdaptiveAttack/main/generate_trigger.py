import sys, os

sys.path.append(os.path.abspath('..'))

import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from Graph_level_Models.AdaptiveAttack.utils.datareader import TuDatasetstoGraph
from Graph_level_Models.AdaptiveAttack.utils.bkdcdd import select_cdd_graphs, select_cdd_nodes
from Graph_level_Models.AdaptiveAttack.utils.mask import gen_mask, recover_mask
import Graph_level_Models.AdaptiveAttack.trojan.GTA as gta
from Graph_level_Models.AdaptiveAttack.trojan.input import gen_input
from Graph_level_Models.AdaptiveAttack.trojan.prop import train_model, evaluate
from Graph_level_Models.AdaptiveAttack.config import parse_args


def run(traindataset,testdataset,train_graph_idx,test_graph_idx,train_trigger_list,test_trigger_list, surrogate_model,args):
    traindataset = TuDatasetstoGraph(traindataset, args)
    testdataset = TuDatasetstoGraph(testdataset, args)
    traindataset_adj_len_list = [len(traindataset['adj_list'][i]) for i in range(len(traindataset['adj_list']))]
    testdataset_adj_len_list = [len(testdataset['adj_list'][i]) for i in range(len(testdataset['adj_list']))]
    max_train_adj = max(traindataset_adj_len_list)
    max_test_adj = max(testdataset_adj_len_list)

    max_nodes = max([max_train_adj,max_test_adj])

    # Initialize the surrogate model
    model = copy.deepcopy(surrogate_model).to(args.device)
    bkd_nid_groups_train = [[train_trigger_list[i]] for i in range(len(train_trigger_list))]
    bkd_nid_groups_test  = [[test_trigger_list[i]] for i in range(len(test_trigger_list))]
    # pick up initial candidates
    bkd_gids_train, bkd_nids_train, bkd_nid_groups_train = train_graph_idx, train_trigger_list,bkd_nid_groups_train
    bkd_gids_test, bkd_nids_test, bkd_nid_groups_test = test_graph_idx, test_trigger_list, bkd_nid_groups_test







    featdim = args.surrogate_num_features

    # init two generators for topo/feat
    toponet = gta.GraphTrojanNet(max_nodes, args.surrogate_gtn_layernum)
    featnet = gta.GraphTrojanNet(featdim, args.surrogate_gtn_layernum)


    all_set = [i for i in range(len(traindataset))]
    for rs_step in range(args.surrogate_resample_steps):  # for each step, choose different sample

        # randomly select new graph backdoor samples
        #bkd_gids_train, bkd_nids_train, bkd_nid_groups_train = self.bkd_cdd('train')

        # positive/negtive sample set
        pset = bkd_gids_train

        nset = list(set(all_set) - set(pset))


        # NOTE: for data that can only add perturbation on features, only init the topo value
        init_data_train = init_trigger(
            args, copy.deepcopy(traindataset), bkd_gids_train, bkd_nid_groups_train, 0.0, 0.0)
        bkd_data_train = copy.deepcopy(init_data_train)

        init_As = init_data_train['adj_list']
        nodenums = [len(adj) for adj in init_As]


        topomask_train, featmask_train = gen_mask(
            init_data_train, bkd_gids_train, bkd_nid_groups_train,max_nodes)
        Ainput_train, Xinput_train = gen_input(args, init_data_train, bkd_gids_train,max_nodes)

        for bi_step in range(args.surrogate_bilevel_steps):
            print("Resampling step %d, bi-level optimization step %d" % (rs_step, bi_step))

            toponet, featnet = gta.train_gtn(
                args, model, toponet, featnet,
                pset, nset, topomask_train, featmask_train,
                init_data_train, bkd_data_train, Ainput_train, Xinput_train)

            # get new backdoor datareader for training based on well-trained generators
            for gid in bkd_gids_train:
                rst_bkdA = toponet(
                    Ainput_train[gid], topomask_train[gid], args.surrogate_topo_thrd,
                    args.device, args.surrogate_topo_activation, 'topo')
                # rst_bkdA = recover_mask(nodenums[gid], topomask_train[gid], 'topo')
                # bkd_dr_train.data['adj_list'][gid] = torch.add(rst_bkdA, init_dr_train.data['adj_list'][gid])
                bkd_data_train['adj_list'][gid] = torch.add(
                    rst_bkdA[:nodenums[gid], :nodenums[gid]],
                    init_data_train['adj_list'][gid])

                rst_bkdX = featnet(
                    Xinput_train[gid], featmask_train[gid], args.surrogate_feat_thrd,
                    args.device, args.surrogate_feat_activation, 'feat')
                # rst_bkdX = recover_mask(nodenums[gid], featmask_train[gid], 'feat')
                # bkd_dr_train.data['features'][gid] = torch.add(rst_bkdX, init_dr_train.data['features'][gid])
                bkd_data_train['features'][gid] = torch.add(
                    rst_bkdX[:nodenums[gid]], init_data_train['features'][gid])

    # init test data
    # NOTE: for data that can only add perturbation on features, only init the topo value
    init_data_test = init_trigger(
        args, copy.deepcopy(testdataset), bkd_gids_test, bkd_nid_groups_test, 0.0, 0.0)
    bkd_data_test = copy.deepcopy(init_data_test)

    topomask_test, featmask_test = gen_mask(
        init_data_test, bkd_gids_test, bkd_nid_groups_test,max_nodes)
    Ainput_test, Xinput_test = gen_input(args, init_data_test, bkd_gids_test,max_nodes)

    init_As = init_data_test['adj_list']
    nodenums = [len(adj) for adj in init_As]
    # ----------------- Generate the Backdoor Attack Data -----------------#
    for gid in bkd_gids_test:
        SendtoCUDA(gid, [init_As, Ainput_test, topomask_test], args.device)    # only send the used graph items to cuda
        rst_bkdA = toponet(
            Ainput_test[gid], topomask_test[gid], args.surrogate_topo_thrd,
            args.device, args.surrogate_topo_activation, 'topo')
        # rst_bkdA = recover_mask(nodenums[gid], topomask_test[gid], 'topo')
        # bkd_dr_test.data['adj_list'][gid] = torch.add(rst_bkdA,
        #     torch.as_tensor(copy.deepcopy(init_dr_test.data['adj_list'][gid])))
        bkd_data_test['adj_list'][gid] = torch.add(
            rst_bkdA[:nodenums[gid], :nodenums[gid]],
            torch.as_tensor(copy.deepcopy(init_data_test['adj_list'][gid])))

        # rst_bkdX = featnet(
        #     Xinput_test[gid], featmask_test[gid], args.surrogate_feat_thrd,
        #     args.device, args.surrogate_feat_activation, 'feat')
        # # rst_bkdX = recover_mask(nodenums[gid], featmask_test[gid], 'feat')
        # # bkd_dr_test.data['features'][gid] = torch.add(
        # #     rst_bkdX, torch.as_tensor(copy.deepcopy(init_dr_test.data['features'][gid])))
        # bkd_data_test['features'][gid] = torch.add(
        #     rst_bkdX[:nodenums[gid]], torch.as_tensor(copy.deepcopy(init_data_test['features'][gid])))
        #

    return bkd_data_train['adj_list'], bkd_data_test['adj_list']

def bkd_cdd(self, subset: str):
    # - subset: 'train', 'test'
    # find graphs to add trigger (not modify now)
    bkd_gids = select_cdd_graphs(
        self.args, self.benign_dr.data['splits'][subset], self.benign_dr.data['adj_list'], subset)
    # find trigger nodes per graph
    # same sequence with selected backdoored graphs
    bkd_nids, bkd_nid_groups = select_cdd_nodes(
        self.args, bkd_gids, self.benign_dr.data['adj_list'])

    assert len(bkd_gids) == len(bkd_nids) == len(bkd_nid_groups)


    return bkd_gids, bkd_nids, bkd_nid_groups


def init_trigger(args, Data, bkd_gids: list, bkd_nid_groups: list, init_edge: float, init_feat: float):
    if init_feat == None:
        init_feat = - 1
        print('init feat == None, transferred into -1')

    # (in place) datareader trigger injection
    for i in tqdm(range(len(bkd_gids)), desc="initializing heuristic trigger..."):
        gid = bkd_gids[i]
        for group in bkd_nid_groups[i]:
            # change adj in-place
            src, dst = [], []
            for v1 in group:
                for v2 in group:
                    if v1 != v2:
                        src.append(v1)
                        dst.append(v2)
        a = np.array(Data['adj_list'][gid])
        a[src, dst] = init_edge
        Data['adj_list'][gid] = a.tolist()


        # change features in-place
        featdim = len(Data['features'][0][0])
        a = np.array(Data['features'][gid])
        a[group] = np.ones((len(group), featdim)) * init_feat
        Data['features'][gid] = a.tolist()

        # change graph labels
        assert args.target_label is not None
        Data['labels'][gid] = args.target_label

    return Data
#----------------------------------------------------------------
def SendtoCUDA(gid, items,device):
    """
    - items: a list of dict / full-graphs list,
             used as item[gid] in items
    - gid: int
    """

    for item in items:
        item[gid] = torch.as_tensor(item[gid], dtype=torch.float32).to(device)

if __name__ == '__main__':
    args = parse_args()
    attack = run(args)
