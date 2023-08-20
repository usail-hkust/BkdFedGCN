import torch
import copy
import numpy as np
import  random
import torch_geometric.transforms as T
from torch_geometric.utils import scatter
from torch_geometric.datasets import Planetoid,Reddit2,Flickr,Reddit,Yelp
from torch_geometric.datasets import Coauthor, Amazon
from torch_geometric.utils import to_undirected

import Node_level_Models.helpers.selection_utils  as hs
from Node_level_Models.helpers.func_utils import subgraph,get_split
from Node_level_Models.helpers.split_graph_utils import split_Random, split_Louvain, split_Metis
from Node_level_Models.models.construct import model_construct
from Node_level_Models.helpers.func_utils import prune_unrelated_edge,prune_unrelated_edge_isolated
from Node_level_Models.data.datasets import  ogba_data,Amazon_data,Coauthor_data
from Node_level_Models.aggregators.aggregation import scaffold,init_control

def update_global(global_model, delta_models,args):
    state_dict = {}

    for name, param in global_model.state_dict().items():
        vs = []
        for client in delta_models.keys():
            vs.append(delta_models[client][name])
        vs = torch.stack(vs, dim=0)

        try:
            mean_value = vs.mean(dim=0)
            vs = param - args.glo_lr * mean_value
        except Exception:
            # for BN's cnt
            mean_value = (1.0 * vs).mean(dim=0).long()
            vs = param - args.glo_lr * mean_value
            vs = vs.long()

        state_dict[name] = vs

    global_model.load_state_dict(state_dict, strict=True)
    return global_model
def update_global_control(control, delta_controls):
    new_control = copy.deepcopy(control)
    for name, c in control.items():
        mean_ci = []
        for _, delta_control in delta_controls.items():
            mean_ci.append(delta_control[name])
        ci = torch.stack(mean_ci).mean(dim=0)
        new_control[name] = c - ci
    return new_control
def main(args, logger):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    Coauthor_list = ["Cs","Physics"]
    Amazon_list = ["computers","photo"]
    ##### DATA PREPARATION #####
    if (args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
        dataset = Planetoid(root='./data/', \
                            name=args.dataset, \
                            transform=T.LargestConnectedComponents())
    elif (args.dataset == 'Flickr'):
        dataset = Flickr(root='./data/Flickr/', \
                         transform=T.LargestConnectedComponents())
    elif (args.dataset == 'Reddit2'):
        dataset = Reddit2(root='./data/Reddit2/', \
                          transform=T.LargestConnectedComponents())
    elif (args.dataset == 'Reddit'):
        dataset = Reddit(root='./data/Reddit/', \
                          transform=T.LargestConnectedComponents())
    elif (args.dataset == 'Yelp'):
        dataset = Yelp(root='./data/Yelp/', \
                          transform=T.LargestConnectedComponents())
        # Convert one-hot encoded labels to integer labels
        labels = np.argmax(dataset.data.y.numpy(), axis=1) + 1

        # Create new data object with integer labels
        data = dataset.data
        data.y = torch.from_numpy(labels).reshape(-1, 1)
    elif (args.dataset == 'ogbn-arxiv'):
        from ogb.nodeproppred import PygNodePropPredDataset
        # Download and process data at './dataset/ogbg_molhiv/'
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data/')
    elif (args.dataset == 'ogbn-products'):
        from ogb.nodeproppred import PygNodePropPredDataset
        # Download and process data at './dataset/ogbg_molhiv/'
        dataset = PygNodePropPredDataset(name='ogbn-products', root='./data/')
    elif (args.dataset == 'ogbn-proteins'):
        from ogb.nodeproppred import PygNodePropPredDataset
        # Download and process data at './dataset/ogbg_molhiv/'
        dataset = PygNodePropPredDataset(name='ogbn-proteins', root='./data/')

    elif (args.dataset in Coauthor_list):
        dataset = Coauthor(root='./data/',name =args.dataset,  \
                          transform=T.NormalizeFeatures())
        print('datasets', dataset[0])
    elif (args.dataset in Amazon_list):
        dataset = Amazon(root='./data/',name =args.dataset,  \
                          transform=T.NormalizeFeatures())


    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    ogbn_data_list = ["ogbn-arxiv",'ogbn-products','ogbn-proteins']
    if args.dataset in ogbn_data_list:
        data = ogba_data(dataset)

    elif args.dataset in Amazon_list:
        data = Amazon_data(dataset)
        data.y = data.y.to(dtype=torch.long)
    elif args.dataset in Coauthor_list:
        data = Coauthor_data(dataset)
    else:
        data = dataset[0]  # Get the graph object.
    if args.dataset == 'ogbn-proteins':
        # Initialize features of nodes by aggregating edge features.
        row, col = data.edge_index
        data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')
        _, f_dim = data.x.size()
        print(f'ogbn-proteins Number of features: {f_dim}')
        print("data.y = data.y.to(torch.float)", data.y.shape)
    args.avg_degree = data.num_edges / data.num_nodes
    nclass = int(data.y.max() + 1)
    print("class", int(data.y.max() + 1))
    print('==============================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print('======================Start Splitting the Data========================================')
    if args.is_iid == "iid":
        client_data = split_Random(args, data)
    elif args.is_iid == "non-iid-louvain":
        client_data = split_Louvain(args, data)
    elif args.is_iid == "non-iid-Metis":
        client_data = split_Metis(args, data)
    else:
        raise NameError

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device_id)

    for i in range(args.num_workers):
        print(len(client_data[i]))


    #Create data objects for the new component-graphs

    #client_data = turn_to_pyg_data(client_graphs)
    print('======================Start Preparing the Data========================================')
    for i in range(args.num_workers):
        print("Client:{}".format(i))
        print(client_data[i])
        # Gather some statistics about the graph.
        print(f'Number of nodes: {client_data[i].num_nodes}')
        print(f'Number of edges: {client_data[i].num_edges}')
        print(f'Number of train: {client_data[i].train_mask.sum()}')
        print(f'Number of val: {client_data[i].val_mask.sum()}')
        print(f'Number of test: {client_data[i].test_mask.sum()}')
    #Create train, test masks
    client_train_edge_index = []

    client_edge_mask = []
    client_mask_edge_index = []
    client_unlabeled_idx = []
    client_idx_train, client_idx_val, client_idx_clean_test, client_idx_atk = [], [], [], []
    for k in range(len(client_data)):
        #client_data[k]= train_test_split(client_data[k], k, args.split)
        data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,client_data[k],device)
        client_idx_train.append(idx_train)
        client_idx_val.append(idx_val)
        client_idx_clean_test.append(idx_clean_test)
        client_idx_atk.append(idx_atk)

        edge_weight = torch.ones([data.edge_index.shape[1]], device=device, dtype=torch.float)
        data.edge_weight = edge_weight


        data.edge_index = to_undirected(data.edge_index)
        train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
        mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]
        client_data[k] = data
        client_train_edge_index.append(train_edge_index)
        client_edge_mask.append(edge_mask)
        client_mask_edge_index.append(mask_edge_index)
        # filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
        unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
        client_unlabeled_idx.append(unlabeled_idx)
    #### END OF DATA PREPARATION #####

    print('======================Start Preparing the Backdoor Attack========================================')
    # prepare for malicious attacker
    Backdoor_model_list = []
    heuristic_trigger_list = ["renyi","ws", "ba"]
    for i in range(args.num_mali):
        if args.trigger_type== "gta":
           from Node_level_Models.models.GTA import Backdoor
           Backdoor_model = Backdoor(args, device)
        elif args.trigger_type == "ugba":
            from Node_level_Models.models.backdoor import Backdoor
            Backdoor_model = Backdoor(args, device)
        elif args.trigger_type in heuristic_trigger_list:
            from Node_level_Models.models.Heuristic import Backdoor
            Backdoor_model = Backdoor(args, device)
        else:
            raise NameError
        Backdoor_model_list.append(Backdoor_model)


    print('======================Start Preparing the Trigger Posistion========================================')
    client_idx_attach = []
    for i in range(args.num_workers):
        size =  int((len(client_unlabeled_idx[i]))*args.poisoning_intensity)
        if (args.trigger_position == 'random'):
            idx_attach = hs.obtain_attach_nodes(args, client_unlabeled_idx[i], size)
            idx_attach = torch.LongTensor(idx_attach).to(device)
        elif (args.trigger_position == 'learn_cluster'):
            idx_attach = hs.cluster_distance_selection(args, client_data[i], client_idx_train[i], client_idx_val[i], client_idx_clean_test[i], client_unlabeled_idx[i],
                                                       client_train_edge_index[i], size, device)
            idx_attach = torch.LongTensor(idx_attach).to(device)
        elif (args.trigger_position == 'learn_cluster_degree'):
            idx_attach = hs.cluster_degree_selection(args, client_data[i], client_idx_train[i], client_idx_val[i], client_idx_clean_test[i], client_unlabeled_idx[i],
                                                       client_train_edge_index[i], size, device)
            idx_attach = torch.LongTensor(idx_attach).to(device)
        elif (args.trigger_position == 'degree'):
            idx_attach = hs.obtain_attach_nodes_degree(args, client_unlabeled_idx[i],client_data[i], size)
            idx_attach = torch.LongTensor(idx_attach).to(device)
        elif (args.trigger_position == 'cluster'):
            idx_attach = hs.obtain_attach_nodes_cluster(args, client_unlabeled_idx[i],client_data[i], size)
            idx_attach = torch.LongTensor(idx_attach).to(device)
        else:
            raise NameError
        client_idx_attach.append(idx_attach)

    print('======================Start Preparing the Posioned Datasets========================================')
    # construct the triggers
    client_poison_x, client_poison_edge_index, client_poison_edge_weights, client_poison_labels = [], [], [], []
    for i in range(args.num_mali):
        backdoor_model = Backdoor_model_list[i]
        backdoor_model.fit(client_data[i].x,client_train_edge_index[i], None, client_data[i].y, client_idx_train[i], client_idx_attach[i], client_unlabeled_idx[i])
        poison_x, poison_edge_index, poison_edge_weights, poison_labels = backdoor_model.get_poisoned()
        client_poison_x.append(poison_x)
        client_poison_edge_index.append(poison_edge_index)
        client_poison_edge_weights.append(poison_edge_weights)
        client_poison_labels.append(poison_labels)


    # data level defense
    client_bkd_tn_nodes = []
    for i in range(args.num_mali):
        if (args.defense_mode == 'prune'):
            poison_edge_index, poison_edge_weights = prune_unrelated_edge(args, client_poison_edge_index[i], client_poison_edge_weights[i],
                                                                          client_poison_x[i], device, large_graph=False)

            bkd_tn_nodes = torch.cat([client_idx_train[i], client_idx_attach[i]]).to(device)
        elif (args.defense_mode == 'isolate'):
            poison_edge_index, poison_edge_weights, rel_nodes = prune_unrelated_edge_isolated(args, client_poison_edge_index[i],
                                                                                              client_poison_edge_weights[i], client_poison_x[i],
                                                                                              device, large_graph=False)
            bkd_tn_nodes = torch.cat([client_idx_train[i], client_idx_attach[i]]).tolist()
            bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
        else:
            poison_edge_weights = client_poison_edge_weights[i]
            poison_edge_index = client_poison_edge_index[i]
            bkd_tn_nodes = torch.cat([client_idx_train[i].to(device), client_idx_attach[i].to(device)])
        print("precent of left attach nodes: {:.3f}" \
              .format(len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist())) / len(idx_attach)))

        client_poison_edge_index[i] = poison_edge_index
        client_poison_edge_weights[i] = poison_edge_weights
        client_bkd_tn_nodes.append(bkd_tn_nodes)

    optimizer_list = []
    print('======================Start Preparing the Models========================================')
    # Initialize clients
    model_list = []
    for i in range(args.num_workers):
        test_model = model_construct(args, args.model, data, device,nclass).to(device)
        model_list.append(test_model)

    # Initialize the sever model
    global_model = model_construct(args, args.model, data, device,nclass).to(device)

    random.seed(args.seed)
    #rs = random.sample(range(0,args.num_clients),args.num_malicious)

    rs = [i for i in range(args.num_mali)]
    #print("+++++++++++++ Federated Node Classification +++++++++++++")
    #args.federated_rounds = epoch, the inner iteration normly is set to 1.
    print("rs",rs)
    args.epoch_backdoor = int(args.epoch_backdoor * args.epochs)
    print('======================Start Training Model========================================')


    # control variates
    server_control = init_control(global_model, device)

    client_controls = {
        id: init_control(global_model, device)
        for id, client in enumerate(model_list)
    }
    #print(" client",client_controls)
    for epoch in range(args.epochs):
        client_induct_edge_index = []
        client_induct_edge_weights = []

        # worker results
        worker_results = {}
        for i in range(args.num_workers):
            worker_results[f"client_{i}"] = {"train_loss": None, "train_acc": None, "val_loss": None, "val_acc": None}

        delta_models = {}
        delta_controls = {}
        if epoch >= args.epoch_backdoor:
            for j in range(args.num_workers):
                if j in rs:
                    loss_train, loss_val, acc_train, acc_val,\
                    client_control, delta_control, delta_model = scaffold(global_model,
                                                                          server_control,
                                                                          client_controls[j],
                                                                          model = model_list[j],
                                                                          features = client_poison_x[j].to(device),
                                                                          edge_index = client_poison_edge_index[j].to(device),
                                                                          edge_weight = client_poison_edge_weights[j].to(device),
                                                                          labels = client_poison_labels[j].to(device),
                                                                          idx_train = client_bkd_tn_nodes[j].to(device),
                                                                          args = args,
                                                                          idx_val=client_idx_val[j].to(device),
                                                                          train_iters=args.inner_epochs)

                    client_controls[j] = copy.deepcopy(client_control)
                    delta_models[j] = copy.deepcopy(delta_model)
                    delta_controls[j] = copy.deepcopy(delta_control)

                    print("Malicious client: {} ,Acc train: {:.4f}, Acc val: {:.4f}".format(j,acc_train,acc_val))

                    induct_edge_index = torch.cat([client_poison_edge_index[j].to(device), client_mask_edge_index[j].to(device)], dim=1)
                    induct_edge_weights = torch.cat(
                        [client_poison_edge_weights[j], torch.ones([client_mask_edge_index[j].shape[1]], dtype=torch.float, device=device)])


                    # clean_acc = model_list[j].test(client_poison_x[j].to(device), induct_edge_index.to(device),
                    #                                induct_edge_weights.to(device), client_data[j].y.to(device),
                    #                                client_idx_clean_test[j].to(device))
                else:
                    #client_train_edge_index
                    train_edge_weights = torch.ones([client_train_edge_index[j].shape[1]]).to(device)


                    loss_train, loss_val, acc_train, acc_val,\
                    client_control, delta_control, delta_model = scaffold(global_model,
                                                                          server_control,
                                                                          client_controls[j],
                                                                          model = model_list[j],
                                                                          features = client_data[j].x.to(device),
                                                                          edge_index = client_train_edge_index[j].to(device),
                                                                          edge_weight = train_edge_weights.to(device),
                                                                          labels = client_data[j].y.to(device),
                                                                          idx_train = client_idx_train[j].to(device),
                                                                          args = args,
                                                                          idx_val= client_idx_val[j].to(device),
                                                                          train_iters=args.inner_epochs)

                    client_controls[j] = copy.deepcopy(client_control)
                    delta_models[j] = copy.deepcopy(delta_model)
                    delta_controls[j] = copy.deepcopy(delta_control)


                    print("Clean client: {} ,Acc train: {:.4f}, Acc val: {:.4f}".format(j, acc_train, acc_val))


                    induct_x, induct_edge_index, induct_edge_weights = client_data[j].x, client_data[j].edge_index, client_data[j].edge_weight

                # save worker results
                for ele in worker_results[f"client_{j}"]:
                    if ele == "train_loss":
                        worker_results[f"client_{j}"][ele] = loss_train
                    elif ele == "train_acc":
                        worker_results[f"client_{j}"][ele] = acc_train
                    elif ele == "val_loss":
                        worker_results[f"client_{j}"][ele] = loss_val
                    elif ele == "val_acc":
                        worker_results[f"client_{j}"][ele] = acc_val

                client_induct_edge_index.append(induct_edge_index)
                client_induct_edge_weights.append(induct_edge_weights)

            # wandb logger
            logger.log(worker_results)
        else:
            for j in range(args.num_workers):

                train_edge_weights = torch.ones([client_train_edge_index[j].shape[1]]).to(device)

                loss_train, loss_val, acc_train, acc_val, \
                client_control, delta_control, delta_model = scaffold(global_model,
                                                                      server_control,
                                                                      client_controls[j],
                                                                      model=model_list[j],
                                                                      features=client_data[j].x.to(device),
                                                                      edge_index=client_train_edge_index[j].to(device),
                                                                      edge_weight=train_edge_weights.to(device),
                                                                      labels=client_data[j].y.to(device),
                                                                      idx_train=client_idx_train[j].to(device),
                                                                      args=args,
                                                                      idx_val=client_idx_val[j].to(device),
                                                                      train_iters=args.inner_epochs)

                client_controls[j] = copy.deepcopy(client_control)
                delta_models[j] = copy.deepcopy(delta_model)
                delta_controls[j] = copy.deepcopy(delta_control)









                print("Clean client: {} ,Acc train: {:.4f}, Acc val: {:.4f}".format(j, acc_train, acc_val))
                induct_x, induct_edge_index, induct_edge_weights = client_data[j].x, client_data[j].edge_index, client_data[j].edge_weight
                # clean_acc = model_list[j].test(client_data[j].x.to(device), client_data[j].edge_index.to(device),
                #                                client_data[j].edge_weight.to(device), client_data[j].y.to(device),
                #                                client_idx_clean_test[j].to(device))

                # save worker results
                for ele in worker_results[f"client_{j}"]:
                    if ele == "train_loss":
                        worker_results[f"client_{j}"][ele] = loss_train
                    elif ele == "train_acc":
                        worker_results[f"client_{j}"][ele] = acc_train
                    elif ele == "val_loss":
                        worker_results[f"client_{j}"][ele] = loss_val
                    elif ele == "val_acc":
                        worker_results[f"client_{j}"][ele] = acc_val

                client_induct_edge_index.append(induct_edge_index)
                client_induct_edge_weights.append(induct_edge_weights)

            # wandb logger
            logger.log(worker_results)


        global_model = update_global(global_model, delta_models, args)
        new_control = update_global_control(
            control=server_control,
            delta_controls=delta_controls,
        )
        server_control = copy.deepcopy(new_control)

    overall_performance = []
    overall_malicious_train_attach_rate = []
    overall_malicious_train_flip_asr = []
    for i in range(args.num_workers):
        if i in rs:
            idx_atk = client_idx_atk[i]
            induct_x, induct_edge_index, induct_edge_weights = Backdoor_model_list[i].inject_trigger(client_idx_atk[i], client_poison_x[i], client_induct_edge_index[i],
                                                                                    client_induct_edge_weights[i], device)

            output = model_list[i](induct_x, induct_edge_index, induct_edge_weights)
            train_attach_rate = (output.argmax(dim=1)[idx_atk] == args.target_class).float().mean()
            print("ASR: {:.4f}".format(train_attach_rate))
            overall_malicious_train_attach_rate.append(train_attach_rate.cpu().numpy())
            idx_atk = idx_atk.to(device)
            data.y = data.y.to(device)

            flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
            flip_asr = (output.argmax(dim=1)[flip_idx_atk] == args.target_class).float().mean()
            print("Flip ASR: {:.4f}/{} nodes".format(flip_asr, flip_idx_atk.shape[0]))
            overall_malicious_train_flip_asr.append(flip_asr.cpu().numpy())
        else:
            # %% inject trigger on attack test nodes (idx_atk)'
            induct_x, induct_edge_index, induct_edge_weights = client_data[i].x, client_data[i].edge_index, client_data[i].edge_weight





        Accuracy = test_model.test(induct_x.to(device), induct_edge_index.to(device), induct_edge_weights.to(device), client_data[i].y.to(device), client_idx_clean_test[i].to(device))
        print("Client: {}, Accuracy: {:.4f}".format(i,Accuracy))
        overall_performance.append(Accuracy)
        print(overall_malicious_train_attach_rate)


    transfer_attack_success_rate_list = []
    if args.num_workers-args.num_mali <= 0:
        average_transfer_attack_success_rate = -10000.0
    else:
        for i in range(args.num_mali):
            for j in range(args.num_workers - args.num_mali):
                idx_atk = client_idx_atk[i]
                induct_x, induct_edge_index, induct_edge_weights = Backdoor_model_list[i].inject_trigger(
                    client_idx_atk[i], client_poison_x[i], client_induct_edge_index[i],
                    client_induct_edge_weights[i], device)

                output = model_list[args.num_mali+j](induct_x, induct_edge_index, induct_edge_weights)
                train_attach_rate = (output.argmax(dim=1)[idx_atk] == args.target_class).float().mean()
                # print("ASR: {:.4f}".format(train_attach_rate))
                # overall_malicious_train_attach_rate.append(train_attach_rate.cpu().numpy())

                print('Clean client %d with  trigger %d: %.3f' % (args.num_mali+j, i, train_attach_rate))
                transfer_attack_success_rate_list.append(train_attach_rate.cpu().numpy())
        average_transfer_attack_success_rate = np.mean(np.array(transfer_attack_success_rate_list))
    print("Malicious client: {}".format(rs))
    print("Average ASR: {:.4f}".format(np.array(overall_malicious_train_attach_rate).sum() / args.num_mali))
    print("Flip ASR: {:.4f}".format(np.array(overall_malicious_train_flip_asr).sum()/ args.num_mali))
    print("Transfer ASR: {:.4f} ".format(average_transfer_attack_success_rate))
    print("Average Performance on clean test set: {:.4f}".format(np.array(overall_performance).sum() / args.num_workers))
    average_overall_performance =  np.array(overall_performance).sum() / args.num_workers
    average_ASR = np.array(overall_malicious_train_attach_rate).sum() / args.num_mali
    average_Flip_ASR = np.array(overall_malicious_train_flip_asr).sum()/ args.num_mali
    return average_overall_performance, average_ASR, average_Flip_ASR, average_transfer_attack_success_rate

if __name__ == '__main__':
    main()