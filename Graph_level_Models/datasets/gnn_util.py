import networkx as nx
import random
import torch
import pickle
from tqdm import tqdm
import os
import numpy as np
import copy
import dgl
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset

class TriggerDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]  # graphs
        self.graph_labels = lists[1] # labels

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])

def transform_dataset(trainset, testset, avg_nodes, args):
    train_untarget_idx = []
    for i in range(len(trainset)):
        if trainset[i][1].item() != args.target_label:
            train_untarget_idx.append(i)

    train_untarget_graphs = [copy.deepcopy(graph) for graph in trainset if graph[1].item() != args.target_label]
    train_labels = [graph[1] for graph in trainset]
    num_classes = torch.max(torch.tensor(train_labels)).item() + 1

    tmp_graphs = []
    tmp_idx = []
    num_trigger_nodes = int(avg_nodes * args.frac_of_avg) # avg_nodes is the average number of all grpah's nodes
    for idx, graph in enumerate(train_untarget_graphs):
        if graph[0].num_nodes() > num_trigger_nodes:
            tmp_graphs.append(graph)
            tmp_idx.append(train_untarget_idx[idx])
    n_trigger_graphs = int(args.poisoning_intensity * len(trainset))
    final_idx = []
    if n_trigger_graphs <= len(tmp_graphs):
        train_trigger_graphs = tmp_graphs[:n_trigger_graphs]
        final_idx = tmp_idx[:n_trigger_graphs]

    else:
        train_trigger_graphs = tmp_graphs
        final_idx = tmp_idx

    ##############################################################################################
    print("Start generating trigger position by {}".format(args.trigger_position))
    default_min_num_trigger_nodes = 3
    if num_trigger_nodes < default_min_num_trigger_nodes:
        num_trigger_nodes = default_min_num_trigger_nodes

    #Randomly choose the trigger
    trigger_list = []


    if args.trigger_position == "random":
        for data in train_trigger_graphs:
            # print("data[0].nodes().tolist()",len(data[0].nodes().tolist()))
            # print("num trigger nodes", num_trigger_nodes)
            if len(data[0].nodes().tolist()) < num_trigger_nodes:
                trigger_num = data[0].nodes().tolist()
            else:
                trigger_num = random.sample(data[0].nodes().tolist(), num_trigger_nodes)
            trigger_list.append(trigger_num)
    elif args.trigger_position == "degree":
        for data in train_trigger_graphs:
            #  transfer data to Network graph
            g = dgl.to_networkx(data[0].cpu())
            # sort according to degree
            degree_dict = dict(g.degree())
            sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
            trigger_num = sorted_nodes[:num_trigger_nodes]
            trigger_list.append(trigger_num)
    elif args.trigger_position == "cluster":
        for data in train_trigger_graphs:
            #  transfer data to Network graph
            g = dgl.to_networkx(data[0].cpu())
            #  sort according to cluster
            simple_g = nx.Graph(g)
            clustering_dict = nx.clustering(simple_g,weight='weight')
            sorted_nodes = sorted(clustering_dict, key=clustering_dict.get, reverse=True)

            trigger_num = sorted_nodes[:num_trigger_nodes]
            trigger_list.append(trigger_num)
    else:
        raise NameError

    ######################################################################
    print("Start preparing for the poisoned test datasets")
    test_changed_graphs = [copy.deepcopy(graph) for graph in testset if graph[1].item() != args.target_label]
    delete_test_changed_graphs = []
    test_changed_graphs_final = []
    for graph in test_changed_graphs:
        if graph[0].num_nodes() < num_trigger_nodes:
            delete_test_changed_graphs.append(graph)
    for graph in test_changed_graphs:
        if graph not in delete_test_changed_graphs:
            test_changed_graphs_final.append(graph)
    test_changed_graphs = test_changed_graphs_final
    print("The number of test changed graphs is: %d"%len(test_changed_graphs_final))
    test_trigger_list = []
    test_graph_idx = []
    for gid,graph in enumerate(test_changed_graphs):
        trigger_idx = random.sample(graph[0].nodes().tolist(), num_trigger_nodes)
        test_trigger_list.append(trigger_idx)
        test_graph_idx.append(int(gid))


    ######################################################################
    print("Start generating trigger by {}".format(args.trigger_type))

    if args.trigger_type == "renyi":

        G_trigger = nx.erdos_renyi_graph(num_trigger_nodes, args.density, directed=False)
        if G_trigger.edges():
            G_trigger = G_trigger
        else:
            G_trigger = nx.erdos_renyi_graph(num_trigger_nodes, 1.0, directed=False)

    elif args.trigger_type == "ws":
        G_trigger = nx.watts_strogatz_graph(num_trigger_nodes, args.avg_degree, args.density)
    elif args.trigger_type == "ba":
        if args.avg_degree >= num_trigger_nodes:
            args.avg_degree = num_trigger_nodes - 1
        # n: int Number of nodes
        # m: int Number of edges to attach from a new node to existing nodes
        G_trigger = nx.random_graphs.barabasi_albert_graph(n= num_trigger_nodes, m= args.avg_degree)
    elif args.trigger_type == "rr":
        #d int The degree of each node.
        # n integer The number of nodes.The value of must be even.
        if args.avg_degree >= num_trigger_nodes:
            args.avg_degree = num_trigger_nodes - 1
        if num_trigger_nodes % 2 != 0:
            num_trigger_nodes +=1
        G_trigger = nx.random_graphs.random_regular_graph(d = args.avg_degree, n = num_trigger_nodes)     # generate a regular graph which has 20 nodes & each node has 3 neghbour nodes.

    elif args.trigger_type == "gta":
        # adaptive method for generate the triggers, each poisoned graph have a specific trigger.
        # testing
        from Graph_level_Models.AdaptiveAttack.main.benign import  run as surrogate_model_run
        from Graph_level_Models.AdaptiveAttack.main.generate_trigger import run as run_generate_trigger
        surrogate_model,gta_args = surrogate_model_run(args,trainset,args.device)
        train_trigger_list = trigger_list
        bkd_poisoned_adj_train, bkd_poisoned_adj_test = run_generate_trigger(trainset,test_changed_graphs,final_idx,test_graph_idx,train_trigger_list,test_trigger_list, surrogate_model,args)
    else:
        raise NameError

    if args.trigger_type == "gta":
        ############################Adaptive trigger##########################################
        print("Start injecting trigger into the poisoned train datasets")
        for i in range(len(final_idx)):
            data = trainset[i][0]
            edge_index =  adj_to_edge_index(bkd_poisoned_adj_train[i])
            data = data.to(args.device)
            data.remove_edges(torch.arange(data.number_of_edges()).to(args.device))
            # Replace current edge_index with new edge_index
            data.add_edges(edge_index[0].to(args.device), edge_index[1].to(args.device))
        ######################################################################
        print("Start injecting trigger into the poisoned test datasets")
        for i in range(len(test_changed_graphs)):
            data = test_changed_graphs[i][0]
            edge_index =  adj_to_edge_index(bkd_poisoned_adj_test[i])
            data = data.to(args.device)
            data.remove_edges(torch.arange(data.number_of_edges()).to(args.device))
            # Replace current edge_index with new edge_index
            data.add_edges(edge_index[0].to(args.device), edge_index[1].to(args.device))
        G_trigger = None

    else:
        ############################Heuristic trigger##########################################
        print("Start injecting trigger into the poisoned train datasets")
        for  i, data in enumerate(tqdm(train_trigger_graphs)):
            for j in range(len(trigger_list[i])-1):
                for k in range(j+1, len(trigger_list[i])):
                    if (data[0].has_edges_between(trigger_list[i][j], trigger_list[i][k]) or data[0].has_edges_between(trigger_list[i][k], trigger_list[i][j])) \
                        and G_trigger.has_edge(j, k) is False:
                        ids = data[0].edge_ids(torch.tensor([trigger_list[i][j], trigger_list[i][k]]), torch.tensor([trigger_list[i][k], trigger_list[i][j]]))
                        data[0].remove_edges(ids)
                    elif (data[0].has_edges_between(trigger_list[i][j], trigger_list[i][k]) or data[0].has_edges_between(trigger_list[i][k], trigger_list[i][j])) is False \
                        and G_trigger.has_edge(j, k):
                        data[0].add_edges(torch.tensor([trigger_list[i][j], trigger_list[i][k]]), torch.tensor([trigger_list[i][k], trigger_list[i][j]]))

        ######################################################################
        print("Start injecting trigger into the poisoned test datasets")
        # evaluation: randomly inject the trigger into the graph
        for ith, graph in tqdm(enumerate(test_changed_graphs)):
            trigger_idx = test_trigger_list[ith]
            for i in range(len(trigger_idx)-1):
                for j in range(i+1, len(trigger_idx)):
                    if (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) \
                        and G_trigger.has_edge(i, j) is False:
                        ids = graph[0].edge_ids(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
                        graph[0].remove_edges(ids)
                    elif (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) is False \
                        and G_trigger.has_edge(i, j):
                        graph[0].add_edges(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))


    graphs = [data[0] for data in train_trigger_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(train_trigger_graphs))]
    train_trigger_graphs = DGLFormDataset(graphs, labels)



    graphs = [data[0] for data in test_changed_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(test_changed_graphs))]
    test_trigger_graphs = DGLFormDataset(graphs, labels)

    #### Construct the clean data
    test_clean_data = [copy.deepcopy(graph) for graph in testset]

    test_clean_graphs = [data[0] for data in test_clean_data]
    test_clean_labels = [data[1] for data in test_clean_data]
    test_clean_data = DGLFormDataset(test_clean_graphs, test_clean_labels)
    #### Construct the unchaged data and changed data into the same datsets [unchanged data, changed data]
    test_unchanged_data = [copy.deepcopy(graph) for graph in testset if graph[1].item() == args.target_label]
    test_unchanged_graphs = [data[0] for data in test_unchanged_data]
    test_unchanged_labels = [data[1] for data in test_unchanged_data]
    test_unchanged_data = DGLFormDataset(test_unchanged_graphs, test_unchanged_labels)


    return train_trigger_graphs, test_trigger_graphs, G_trigger, final_idx, test_clean_data, test_unchanged_data

def transform_dataset_same_local_trigger(trainset, testset, avg_nodes, args, G_trigger):
    train_untarget_idx = []
    for i in range(len(trainset)):
        if trainset[i][1].item() != args.target_label:
            train_untarget_idx.append(i)

    train_untarget_graphs = [copy.deepcopy(graph) for graph in trainset if graph[1].item() != args.target_label]
    tmp_graphs = []
    tmp_idx = []
    num_trigger_nodes = int(avg_nodes * args.frac_of_avg)
    for idx, graph in enumerate(train_untarget_graphs):
        if graph[0].num_nodes() > num_trigger_nodes:
            tmp_graphs.append(graph)
            tmp_idx.append(train_untarget_idx[idx])
    n_trigger_graphs = int(args.poisoning_intensity*len(trainset))
    final_idx = []
    if n_trigger_graphs <= len(tmp_graphs):
        train_trigger_graphs = tmp_graphs[:n_trigger_graphs]
        final_idx = tmp_idx[:n_trigger_graphs]

    else:
        train_trigger_graphs = tmp_graphs
        final_idx = tmp_idx
    trigger_list = []
    for data in train_trigger_graphs:
        trigger_num = random.sample(data[0].nodes().tolist(), num_trigger_nodes)
        trigger_list.append(trigger_num)

    for  i, data in enumerate(train_trigger_graphs):
        for j in range(len(trigger_list[i])-1):
            for k in range(j+1, len(trigger_list[i])):
                if (data[0].has_edges_between(trigger_list[i][j], trigger_list[i][k]) or data[0].has_edges_between(trigger_list[i][k], trigger_list[i][j])) \
                    and G_trigger.has_edge(j, k) is False:
                    ids = data[0].edge_ids(torch.tensor([trigger_list[i][j], trigger_list[i][k]]), torch.tensor([trigger_list[i][k], trigger_list[i][j]]))
                    data[0].remove_edges(ids)
                elif (data[0].has_edges_between(trigger_list[i][j], trigger_list[i][k]) or data[0].has_edges_between(trigger_list[i][k], trigger_list[i][j])) is False \
                    and G_trigger.has_edge(j, k):
                    data[0].add_edges(torch.tensor([trigger_list[i][j], trigger_list[i][k]]), torch.tensor([trigger_list[i][k], trigger_list[i][j]]))
    ## rebuild data with target label
    graphs = [data[0] for data in train_trigger_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(train_trigger_graphs))]
    train_trigger_graphs = DGLFormDataset(graphs, labels)



    test_changed_graphs = [copy.deepcopy(graph) for graph in testset if graph[1].item() != args.target_label]


    delete_test_changed_graphs = []
    test_changed_graphs_final = []
    for graph in test_changed_graphs:
        if graph[0].num_nodes() < num_trigger_nodes:
            delete_test_changed_graphs.append(graph)
    for graph in test_changed_graphs:
        if graph not in delete_test_changed_graphs:
            test_changed_graphs_final.append(graph)
    test_changed_graphs = test_changed_graphs_final
    print("num_of_test_changed_graphs is: %d"%len(test_changed_graphs_final))
    for graph in test_changed_graphs:
        trigger_idx = random.sample(graph[0].nodes().tolist(), num_trigger_nodes)
        for i in range(len(trigger_idx)-1):
            for j in range(i+1, len(trigger_idx)):
                if (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) \
                    and G_trigger.has_edge(i, j) is False:
                    ids = graph[0].edge_ids(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
                    graph[0].remove_edges(ids)
                elif (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) is False \
                    and G_trigger.has_edge(i, j):
                    graph[0].add_edges(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
    graphs = [data[0] for data in test_changed_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(test_changed_graphs))]
    test_trigger_graphs = DGLFormDataset(graphs, labels)
    #### Construct the clean data
    test_clean_graphs = [copy.deepcopy(graph) for graph in testset]
    test_clean_graphs = [data[0] for data in test_clean_graphs]
    test_clean_labels = [torch.tensor([data[1]]) for data in test_clean_graphs]
    test_clean_data = DGLFormDataset(test_clean_graphs, test_clean_labels)
    #### Construct the unchaged data and changed data into the same datsets [unchanged data, changed data]
    test_unchanged_graphs = [copy.deepcopy(graph) for graph in testset if graph[1].item() == args.target_label]
    test_unchanged_graphs = [data[0] for data in test_unchanged_graphs]
    test_unchanged_labels = [torch.tensor([data[1]]) for data in test_unchanged_graphs]

    test_poison_graphs = graphs + test_unchanged_graphs
    test_poison_labels = labels + test_unchanged_labels
    test_poison_data = DGLFormDataset(test_poison_graphs, test_poison_labels)

    return train_trigger_graphs, test_trigger_graphs, final_idx, test_clean_data, test_poison_data

def inject_global_trigger_test(testset, avg_nodes, args, triggers):
    test_changed_graphs = [copy.deepcopy(graph) for graph in testset if graph[1].item() != args.target_label]
    
    num_mali = len(triggers)
    num_trigger_nodes = int(avg_nodes * args.frac_of_avg) * num_mali
    delete_test_changed_graphs = []
    test_changed_graphs_final = []
    for graph in test_changed_graphs:
        if graph[0].num_nodes() < num_trigger_nodes:
            delete_test_changed_graphs.append(graph)
    for graph in test_changed_graphs:
        if graph not in delete_test_changed_graphs:
            test_changed_graphs_final.append(graph)
    test_changed_graphs = test_changed_graphs_final
    if len(test_changed_graphs) == 0:
        raise ValueError('num_trigger_nodes are larger than all the subgraphs!!! Please resize the num_mali and frac_of_avg')
    print("num_of_test_changed_graphs is: %d"%len(test_changed_graphs_final))
    each_trigger_nodes = int(avg_nodes * args.frac_of_avg)
    for graph in test_changed_graphs:
        trigger_idx = random.sample(graph[0].nodes().tolist(), num_trigger_nodes)
        for idx, trigger in enumerate(triggers):
            start = each_trigger_nodes * idx
            for i in range(start, start+each_trigger_nodes-1):
                for j in range(i+1, start+each_trigger_nodes):
                    if (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) \
                        and trigger.has_edge(i, j) is False:
                        ids = graph[0].edge_ids(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
                        graph[0].remove_edges(ids)
                    elif (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) is False \
                        and trigger.has_edge(i, j):
                        graph[0].add_edges(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
    graphs = [data[0] for data in test_changed_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(test_changed_graphs))]
    test_trigger_graphs = DGLFormDataset(graphs, labels)
    return test_trigger_graphs

def inject_global_trigger_train(trainset, avg_nodes, args, triggers):
    train_untarget_idx = []
    for i in range(len(trainset)):
        if trainset[i][1].item() != args.target_label:
            train_untarget_idx.append(i)
   
    train_untarget_graphs = [copy.deepcopy(graph) for graph in trainset if graph[1].item() != args.target_label]
    tmp_graphs = []
    tmp_idx = []
    num_mali = len(triggers)
    num_trigger_nodes = int(avg_nodes * args.frac_of_avg) * num_mali

    for idx, graph in enumerate(train_untarget_graphs):
        if graph[0].num_nodes() > num_trigger_nodes:
            tmp_graphs.append(graph)
            tmp_idx.append(train_untarget_idx[idx])

    n_trigger_graphs = int(args.poisoning_intensity*len(trainset))
    final_idx = []
    if n_trigger_graphs <= len(tmp_graphs):
        train_trigger_graphs = tmp_graphs[:n_trigger_graphs]
        final_idx = tmp_idx[:n_trigger_graphs]
    else:
        train_trigger_graphs = tmp_graphs
        final_idx = tmp_idx
    print("num_of_train_trigger_graphs is: %d"%len(train_trigger_graphs))
    each_trigger_nodes = int(avg_nodes * args.frac_of_avg)
    for graph in train_trigger_graphs:
        trigger_idx = random.sample(graph[0].nodes().tolist(), num_trigger_nodes)
        for idx, trigger in enumerate(triggers):
            start = each_trigger_nodes * idx
            for i in range(start, start+each_trigger_nodes-1):
                for j in range(i+1, start+each_trigger_nodes):
                    if (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) \
                        and trigger.has_edge(i, j) is False:
                        ids = graph[0].edge_ids(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
                        graph[0].remove_edges(ids)
                    elif (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) is False \
                        and trigger.has_edge(i, j):
                        graph[0].add_edges(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
    graphs = [data[0] for data in train_trigger_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(train_trigger_graphs))]
    train_trigger_graphs = DGLFormDataset(graphs, labels)
    return train_trigger_graphs, final_idx


def save_object(obj, filename):
    savedir = os.path.split(filename)[0]
    if not os.path.exists(savedir):
        os.makedirs(savedir)
  
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_pkl(filename):
    with open(filename, 'rb') as input:
        graphs = pickle.load(input)
    return graphs

def check_graph_type(dataset):
    graph = dataset.train[0][0]
    edges = graph.edges()
    edges_0 = edges[0].tolist()
    edges_1 = edges[1].tolist()
    count = 0
    for i in range(len(edges_0)):
        for j in range(i, len(edges_0)):
            if edges_0[j] == edges_1[i] and edges_1[j] == edges_0[i]:
                count += 2
    if count == len(edges_0):
        flag = True
    else:
        flag = False
    return flag

def p_degree_non_iid_split(trainset, args, num_classes):
    #sort trainset
    sorted_trainset = []
    for i in range(num_classes):
        indices = [idx for idx in range(len(trainset)) if trainset[idx][1] == i]
        tmp = [trainset[j] for j in indices]
        print("len tmp",len(tmp))
        sorted_trainset.append(tmp)

    p = args.p_degree
    #split data for every class
    # if num_classes == 2:
    #     p = 0.7
    # else:
    #     p = 0.5
    length_list = []
    for i in range(num_classes):
        n = len(sorted_trainset[i])
                                                                                                                                                                                                                                                                    
        p_list = [((1-p)*num_classes)/((num_classes-1)*args.num_workers)] * args.num_workers

        if i*args.num_workers % num_classes != 0:
            start_idx = int(i*args.num_workers/num_classes) + 1
            p_list[start_idx-1] = ((1-p)*num_classes)/((num_classes-1)*args.num_workers)*(i*args.num_workers/num_classes-start_idx+1) + \
                p*num_classes/args.num_workers * (start_idx - i*args.num_workers/num_classes)
        else:
            start_idx = int(i*args.num_workers/num_classes)

        if (i+1)*args.num_workers % num_classes != 0:
            end_idx = int((i+1)*args.num_workers/num_classes)
            p_list[end_idx] = p*num_classes/args.num_workers * ((i+1)*args.num_workers/num_classes-end_idx) + \
                ((1-p)*num_classes)/((num_classes-1)*args.num_workers)*(1 - (i+1)*args.num_workers/num_classes + end_idx)
        else:
            end_idx = int(start_idx + args.num_workers/num_classes)
        
        for k in range(start_idx, end_idx):
            p_list[k] = p*num_classes/args.num_workers



        length = [pro * n for pro in p_list]
        length = [int(e) for e in length]
        if sum(length) > n:
            length = (np.array(length) - int( (sum(length) - n)/args.num_workers ) -1).tolist()
        length_list.append(length)

    partition = []
    for i in range(args.num_workers):
        dataset = []
        for j in range(num_classes):
            start_idx = sum(length_list[j][:i])
            end_idx = start_idx + length_list[j][i]

            dataset += [sorted_trainset[j][k] for k in range(start_idx, end_idx)]
        partition.append(dataset)
    return partition


def num_noniid_split(dataset, args,min_num,max_num):
    """
    Sample non-I.I.D client data from dataset
    -> Different clients can hold vastly different amounts of data
    :param dataset:
    :param num_users:
    :return:
    """
    num_dataset = len(dataset)
    idx = np.arange(num_dataset)
    dict_users = {i: list() for i in range(args.num_workers)}

    random_num_size = np.random.randint(min_num, max_num + 1, size=args.num_workers)
    print(f"Total number of datasets owned by clients : {sum(random_num_size)}")

    # total dataset should be larger or equal to sum of splitted dataset.
    assert num_dataset >= sum(random_num_size)

    # divide and assign
    partition = []
    for i, rand_num in enumerate(random_num_size):
        rand_set = set(np.random.choice(idx, rand_num, replace=False))
        idx = list(set(idx) - rand_set)
        dict_users[i] = rand_set
        #my_dict = {val: idx for idx, val in enumerate(rand_set)}
        #indices = [my_dict[val] for val in rand_set]
        # print("rand_set",rand_set)
        # print("indices", list(rand_set))
        # print("dataset", dataset[0])
        partition.append([dataset[i] for i in rand_set])
    return partition





def split_dataset(args, dataset):
    """

    Parameters
    ----------
    args: ags for datasets
    dataset: TUDatasets [graph,labels]

    Returns
    -------
    participation data for each clients:
    [train_client_0,train_client_1,...,test_client_0,test_client_1,...]
    """
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #########################################
    num_classes = torch.max(dataset.all.graph_labels).item() + 1
    dataset_all = dataset.train[0] + dataset.val[0] + dataset.test[0]

    graph_sizes = []
    for data in dataset_all:
        graph_sizes.append(data[0].num_nodes())
    graph_sizes.sort()
    n = int(0.3*len(graph_sizes))
    graph_size_normal = graph_sizes[n:len(graph_sizes)-n]
    count = 0
    for size in graph_size_normal:
        count += size
    avg_nodes = count / len(graph_size_normal)
    avg_nodes = round(avg_nodes)
    ###################precious version all the cleient has the same test datasets
    # total_size = len(dataset_all)
    # test_size = int(total_size/(4*args.num_workers+1)) # train size : test size = 4 : 1
    # train_size = total_size - test_size
    # client_num = int(train_size/args.num_workers)
    # length = [client_num]*(args.num_workers-1)
    #
    # length.append(train_size-(args.num_workers-1)*client_num)
    #
    # length.append(test_size)
    #####################changed each client has a different test data different from the precious version that each version has the same testdata
    #length: [client_1_train, client_2_train,...,client_1_test_,client_3_test,...]

    total_size = len(dataset_all)
    test_size = int(total_size/(4*args.num_workers+1*args.num_workers)) # train size : test size = 4 : 1
    train_size = total_size - test_size*args.num_workers
    client_num = int(train_size/args.num_workers)
    length = [client_num]*(args.num_workers-1)
    length.append(train_size-(args.num_workers-1)*client_num)
    for i in range(args.num_workers-1):
        length.append(test_size)
    length.append(total_size - train_size - test_size*(args.num_workers-1))


    ##################################
    # return the adverage degree of nodes among all graphs
    sum_avg_degree = 0
    for data in dataset_all:
        # Get the degree of each node
        degrees = data[0].in_degrees()
        # Calculate the average degree
        avg_degree = degrees.float().mean().item()
        sum_avg_degree += avg_degree
    args.avg_degree = int(sum_avg_degree / len(graph_sizes))
    if args.is_iid == "iid":
        # iid splitq
        partition_data = random_split(dataset_all, length) # split training data and test data
    elif args.is_iid == "p-degree-non-iid":
        # p-degree-non-iid: Local Model Poisoning Attacks to Byzantine-Robust Federated Learning
        # non-iid split
        total_size = len(dataset_all)
        test_size = int(total_size / (4 * args.num_workers + 1 * args.num_workers))  # train size : test size = 4 : 1
        total_train_size = total_size - test_size * args.num_workers
        total_test_size = test_size * args.num_workers
        length = [total_train_size, total_test_size]
        trainset, testset = random_split(dataset_all, length)
        train_partition_data = p_degree_non_iid_split(trainset, args, num_classes)
        test_partition_data = p_degree_non_iid_split(testset, args, num_classes)
        for k in range(len(test_partition_data)):
            train_partition_data.append(test_partition_data[k])
        partition_data = train_partition_data
    elif args.is_iid == "num-non-iid":
        # p-degree-non-iid: Local Model Poisoning Attacks to Byzantine-Robust Federated Learning
        # non-iid split
        total_size = len(dataset_all)
        test_size = int(total_size / (4 * args.num_workers + 1 * args.num_workers))  # train size : test size = 4 : 1
        total_train_size = total_size - test_size * args.num_workers
        total_test_size = test_size * args.num_workers
        length = [total_train_size, total_test_size]
        trainset, testset = random_split(dataset_all, length)
        train_min_num = int(0.6 * (total_train_size / args.num_workers))
        train_max_num = int(0.9 * (total_train_size / args.num_workers))
        test_min_num = int(0.6 * (test_size))
        test_max_num = int(0.9 * (test_size))
        train_partition_data = num_noniid_split(trainset, args, min_num= train_min_num, max_num= train_max_num)
        test_partition_data = num_noniid_split(testset, args, min_num= test_min_num, max_num= test_max_num)
        for k in range(len(test_partition_data)):
            train_partition_data.append(test_partition_data[k])
        partition_data = train_partition_data
    else:
        raise  NameError

    return partition_data, avg_nodes

def adj_to_edge_index(adj):
    N = adj.shape[0]  # get number of nodes

    # find non-zero entries in the adjacency matrix
    rows, cols = torch.where(adj != 0)

    # create edge index tensor
    edge_index = torch.stack([rows, cols], dim=0)

    self_loops = False
    # add self-loops (if desired)
    if self_loops:
        self_loop_index = torch.arange(N)
        edge_index = torch.cat([edge_index, torch.stack([self_loop_index, self_loop_index])], dim=1)
    return edge_index