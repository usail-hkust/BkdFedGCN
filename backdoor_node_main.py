import numpy as np
import torch
import json
from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI


# from torch_geometric.loader import DataLoader
from Node_level_Models.helpers.func_utils import prune_unrelated_edge,prune_unrelated_edge_isolated
import scipy.sparse as sp
from Node_level_Models.helpers.config import args_parser


import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from Node_level_Models.helpers.func_utils import subgraph,get_split


from Node_level_Models.models.GTA import Backdoor
import Node_level_Models.helpers.selection_utils  as hs
from Node_level_Models.models.construct import model_construct
def main():

    args = args_parser()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #
    # with open(args.config) as f:
    #     config = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device_id)


    transform = T.Compose([T.NormalizeFeatures()])

    if(args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
        dataset = Planetoid(root='./data', \
                            name=args.dataset,\
                            transform=transform)
    elif(args.dataset == 'Flickr'):
        dataset = Flickr(root='./data/Flickr/', \
                        transform=transform)
    elif(args.dataset == 'Reddit2'):
        dataset = Reddit2(root='./data/Reddit2/', \
                        transform=transform)
    elif(args.dataset == 'ogbn-arxiv'):
        from ogb.nodeproppred import PygNodePropPredDataset
        # Download and process data at './dataset/ogbg_molhiv/'
        dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root='./data/')
        split_idx = dataset.get_idx_split()

    data = dataset[0].to(device)

    if(args.dataset == 'ogbn-arxiv'):
        nNode = data.x.shape[0]
        setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
        # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
        data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
        data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
        data.y = data.y.squeeze(1)
    # we build our own train test split

    data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)


    data.edge_index = to_undirected(data.edge_index)
    train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
    mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]




    # from kmeans_pytorch import kmeans, kmeans_predict


    # filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
    unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
    print("frac of unlabeled",len(unlabeled_idx)/data.num_nodes)

    size = args.vs_size #int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)
    print("#Attach Nodes:{}".format(size))
    # here is randomly select poison nodes from unlabeled nodes

    result_asr = []
    result_acc = []
    rs = np.random.RandomState(args.seed)
    seeds = rs.randint(1000,size=5)
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        args.seed = seed
        if(args.selection_method == 'none'):
            idx_attach = hs.obtain_attach_nodes(args,unlabeled_idx,size)
        elif(args.selection_method == 'cluster'):
            idx_attach = hs.cluster_distance_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
            idx_attach = torch.LongTensor(idx_attach).to(device)
        elif(args.selection_method == 'cluster_degree'):
            idx_attach = hs.cluster_degree_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
            idx_attach = torch.LongTensor(idx_attach).to(device)

        # In[10]:
        # train trigger generator
        model = Backdoor(args,device)
        model.fit(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
        poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned()

        if(args.defense_mode == 'prune'):
            poison_edge_index,poison_edge_weights = prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
            bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
        elif(args.defense_mode == 'isolate'):
            poison_edge_index,poison_edge_weights,rel_nodes = prune_unrelated_edge_isolated(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
            bkd_tn_nodes = torch.cat([idx_train,idx_attach]).tolist()
            bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
        else:
            bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
        print("precent of left attach nodes: {:.3f}"\
            .format(len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist()))/len(idx_attach)))
        #%%


        test_model = model_construct(args,args.test_model,data,device).to(device)
        for param_tensor in test_model.state_dict():
            print("param",param_tensor)

        test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)

        output = test_model(poison_x,poison_edge_index,poison_edge_weights)
        train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
        print("target class rate on Vs: {:.4f}".format(train_attach_rate))
        torch.cuda.empty_cache()
        #%%
        induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
        induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
        clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
        # test_model = test_model.cpu()

        print("accuracy on clean test nodes: {:.4f}".format(clean_acc))


        if(args.evaluate_mode == '1by1'):
            from torch_geometric.utils  import k_hop_subgraph
            overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(),induct_edge_weights.clone()
            asr = 0
            flip_asr = 0
            flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
            for i, idx in enumerate(idx_atk):
                idx=int(idx)
                sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [idx], num_hops = 2, edge_index = overall_induct_edge_index, relabel_nodes=True) # sub_mapping means the index of [idx] in sub)nodeset
                ori_node_idx = sub_induct_nodeset[sub_mapping]
                relabeled_node_idx = sub_mapping
                sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(device)
                # inject trigger on attack test nodes (idx_atk)'''
                with torch.no_grad():
                    induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(relabeled_node_idx,poison_x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights,device)
                    induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
                    # # do pruning in test datas'''
                    if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
                        induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device,large_graph=False)
                    # attack evaluation

                    output = test_model(induct_x,induct_edge_index,induct_edge_weights)
                    train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx]==args.target_class).float().mean()
                    # print("Node {}: {}, Origin Label: {}".format(i, idx, data.y[idx]))
                    # print("ASR: {:.4f}".format(train_attach_rate))
                    asr += train_attach_rate
                    if(data.y[idx] != args.target_class):
                        flip_asr += train_attach_rate
            asr = asr/(idx_atk.shape[0])
            flip_asr = flip_asr/(flip_idx_atk.shape[0])
            print("Overall ASR: {:.4f}".format(asr))
            print("Flip ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
        elif(args.evaluate_mode == 'overall'):
            # %% inject trigger on attack test nodes (idx_atk)'''
            induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
            induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
            # do pruning in test datas'''
            if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
                induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
            # attack evaluation

            # test_model = test_model.to(device)
            output = test_model(induct_x,induct_edge_index,induct_edge_weights)
            train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
            print("ASR: {:.4f}".format(train_attach_rate))
            flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
            flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
            print("Flip ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
            ca = test_model.test(induct_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
            print("CA: {:.4f}".format(ca))

        result_asr.append(float(asr))
        result_acc.append(float(clean_acc))

    print('The final ASR:{:.5f}, {:.5f}, Accuracy:{:.5f}, {:.5f}'\
                .format(np.average(result_asr),np.std(result_asr),np.average(result_acc),np.std(result_acc)))


if __name__ == '__main__':
    main()