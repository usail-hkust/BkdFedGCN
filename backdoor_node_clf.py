import torch

from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI,Reddit,Yelp
import torch_geometric.transforms as T
import numpy as np
import os
import time
import Node_level_Models.helpers.selection_utils  as hs

from Node_level_Models.helpers.func_utils import subgraph,get_split
from torch_geometric.utils import to_undirected
#Split Graph and creating client datasets
from helpers.split_graph_utils import split_Random, split_Louvain
from Node_level_Models.models.construct import model_construct
from Node_level_Models.helpers.func_utils import prune_unrelated_edge,prune_unrelated_edge_isolated
import  random

def main(args, logger):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


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
        split_idx = dataset.get_idx_split()



    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the graph object.
    print("data.y",data.y)
    args.avg_degree = data.num_edges / data.num_nodes
    print(data)
    print('==============================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')

    if args.is_iid == "iid":
        client_data = split_Random(args, data)
    elif args.is_iid == "non-iid-louvain":
        client_data = split_Louvain(args, data)
    else:
        raise NameError

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device_id)
    #Split Graph and creating client datasets

    #client_data = split_communities(data, args.clients)
    #client_data = split_Random(args,data)


    print("client_graphs",client_data)

    for i in range(args.num_workers):
        print(len(client_data[i]))


    #Create data objects for the new component-graphs

    #client_data = turn_to_pyg_data(client_graphs)

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




    # prepare for malicious attacker
    Backdoor_model_list = []
    heuristic_trigger_list = ["renyi","ws", "ba", "rr", "gta"]
    for i in range(args.num_mali):
        if args.trigger_type== "gta":
           from Node_level_Models.models.GTA import Backdoor
           Backdoor_model = Backdoor(args, device)
        elif args.trigger_type == "adaptive":
            from Node_level_Models.models.GTA import Backdoor
            Backdoor_model = Backdoor(args, device)
        elif args.trigger_type in heuristic_trigger_list:
            from Node_level_Models.models.Heuristic import Backdoor
            Backdoor_model = Backdoor(args, device)
        else:
            raise NameError
        Backdoor_model_list.append(Backdoor_model)

    # prepare for backdoor injected node index
    #size = args.vs_size

    client_idx_attach = []
    for i in range(args.num_workers):
        size =  int((len(client_unlabeled_idx[i]))*args.poisoning_intensity)
        if (args.trigger_position == 'random'):
            idx_attach = hs.obtain_attach_nodes(args, client_unlabeled_idx[i], size)
            idx_attach = torch.LongTensor(idx_attach).to(device)
        elif (args.trigger_position == 'cluster'):
            idx_attach = hs.cluster_distance_selection(args, client_data[i], client_idx_train[i], client_idx_val[i], client_idx_clean_test[i], client_unlabeled_idx[i],
                                                       client_train_edge_index[i], size, device)
            idx_attach = torch.LongTensor(idx_attach).to(device)
        elif (args.trigger_position == 'cluster_degree'):
            idx_attach = hs.cluster_degree_selection(args, client_data[i], client_idx_train[i], client_idx_val[i], client_idx_clean_test[i], client_unlabeled_idx[i],
                                                       client_train_edge_index[i], size, device)
            idx_attach = torch.LongTensor(idx_attach).to(device)
        else:
            raise NameError
        client_idx_attach.append(idx_attach)


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

    # Initialize clients
    model_list = []
    for i in range(args.num_workers):
        test_model = model_construct(args, args.test_model, data, device).to(device)
        model_list.append(test_model)

    # Initialize the sever model
    severe_model = model_construct(args, args.test_model, data, device).to(device)

    random.seed(args.seed)
    #rs = random.sample(range(0,args.num_clients),args.num_malicious)

    rs = [i for i in range(args.num_mali)]
    #print("+++++++++++++ Federated Node Classification +++++++++++++")
    #args.federated_rounds = epoch, the inner iteration normly is set to 1.
    print("rs",rs)
    args.epoch_backdoor = int(args.epoch_backdoor * args.epochs)
    for epoch in range(args.epochs):
        client_induct_edge_index = []
        client_induct_edge_weights = []

        # worker results
        worker_results = {}
        for i in range(args.num_workers):
            worker_results[f"client_{i}"] = {"train_loss": None, "train_acc": None, "val_loss": None, "val_acc": None}

        if epoch >= args.epoch_backdoor:
            for j in range(args.num_workers):
                if j in rs:
                    loss_train, loss_val, acc_train, acc_val = model_list[j].fit(client_poison_x[j].to(device),
                                                                                 client_poison_edge_index[j].to(device),
                                                                                 client_poison_edge_weights[j].to(device),
                                                                                 client_poison_labels[j].to(device),
                                                                                 client_bkd_tn_nodes[j].to(device),
                                                                                 client_idx_val[j].to(device),
                                                                                 train_iters=args.inner_epochs, verbose=False)

                    output = model_list[j](client_poison_x[j].to(device), client_poison_edge_index[j].to(device), client_poison_edge_weights[j].to(device))
                    train_attach_rate = (output.argmax(dim=1)[idx_attach] == args.target_class).float().mean()
                    print("malicious client: {} ,target class rate on Vs: {:.4f}".format(j,train_attach_rate))
                    induct_edge_index = torch.cat([client_poison_edge_index[j].to(device), client_mask_edge_index[j].to(device)], dim=1)
                    induct_edge_weights = torch.cat(
                        [client_poison_edge_weights[j], torch.ones([client_mask_edge_index[j].shape[1]], dtype=torch.float, device=device)])

                    clean_acc = model_list[j].test(client_poison_x[j].to(device), induct_edge_index.to(device),
                                                   induct_edge_weights.to(device), client_data[j].y.to(device),
                                                   client_idx_clean_test[j].to(device))
                else:
                    loss_train, loss_val, acc_train, acc_val = model_list[j].fit(client_data[j].x.to(device),
                                                                                 client_data[j].edge_index.to(device),
                                                                                 client_data[j].edge_weight.to(device),
                                                                                 client_data[j].y.to(device),
                                                                                 client_idx_train[j].to(device),
                                                                                 client_idx_val[j].to(device),
                                                                                 train_iters=args.inner_epochs,
                                                                                 verbose=False)

                    induct_x, induct_edge_index, induct_edge_weights = client_data[j].x, client_data[j].edge_index, client_data[j].edge_weight
                    clean_acc = model_list[j].test(client_data[j].x.to(device), client_data[j].edge_index.to(device),
                                                   client_data[j].edge_weight.to(device), client_data[j].y.to(device),
                                                   client_idx_clean_test[j].to(device))

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
                loss_train, loss_val, acc_train, acc_val = model_list[j].fit(client_data[j].x.to(device),
                                                                             client_data[j].edge_index.to(device),
                                                                             client_data[j].edge_weight.to(device),
                                                                             client_data[j].y.to(device),
                                                                             client_idx_train[j].to(device),
                                                                             client_idx_val[j].to(device),
                                                                             train_iters=args.inner_epochs,
                                                                             verbose=False)

                induct_x, induct_edge_index, induct_edge_weights = client_data[j].x, client_data[j].edge_index, client_data[j].edge_weight
                clean_acc = model_list[j].test(client_data[j].x.to(device), client_data[j].edge_index.to(device),
                                               client_data[j].edge_weight.to(device), client_data[j].y.to(device),
                                               client_idx_clean_test[j].to(device))

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

            print("accuracy on clean test nodes: {:.4f}".format(clean_acc))

            # Server Aggregation
            Sub_model_list = random.sample(model_list, args.num_sample_submodels)
            for param_tensor in Sub_model_list[0].state_dict():
                avg = (sum(c.state_dict()[param_tensor] for c in Sub_model_list)) / len(Sub_model_list)
                # Update the global
                severe_model.state_dict()[param_tensor].copy_(avg)
                # Send global to the local
                for cl in model_list:
                    cl.state_dict()[param_tensor].copy_(avg)

    # Test performance

    overall_performance = []
    overall_malicious_train_attach_rate = []
    overall_malicious_train_flip_asr = []
    for i in range(args.num_workers):
        if i in rs:

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
    print("Average Performance on clean test set: {:.4f}".format(np.array(overall_performance).sum() / args.num_workers))
    average_overall_performance =  np.array(overall_performance).sum() / args.num_workers
    average_ASR = np.array(overall_malicious_train_attach_rate).sum() / args.num_mali
    average_Flip_ASR = np.array(overall_performance).sum() / args.num_workers
    return average_overall_performance, average_ASR, average_Flip_ASR, average_transfer_attack_success_rate

if __name__ == '__main__':
    main()