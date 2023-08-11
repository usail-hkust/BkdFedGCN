from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import community as community_louvain
import random
from tqdm import tqdm
import metispy as metis

import torch_geometric
def split_communities(data, clients):
    G = to_networkx(data, to_undirected=True, node_attrs=['x', 'y'])
    communities = sorted(nx.community.asyn_fluidc(G, clients, max_iter=5000, seed=12345))

    node_groups = []
    for com in communities:
        node_groups.append(list(com))
    list_of_clients = []

    for i in range(clients):
        list_of_clients.append(from_networkx(G.subgraph(node_groups[i]).copy()))

    return list_of_clients

def split_Metis(args,data):
    """
    original code link： https://github.com/alibaba/FederatedScope/blob/fe1806b36b4629bb0057e84912d5f42a79f4461d/federatedscope/core/splitters/graph/random_splitter.py#L14
    :param args: args.overlapping_rate(float):Additional samples of overlapping data, \
            eg. ``'0.4'``;
                    args.drop_edge(float): Drop edges (drop_edge / client_num) for each \
            client within overlapping part.
    :param data:
    :param clients:
    :return:
    """
    args.drop_edge = 0
    ovlap = args.overlapping_rate
    drop_edge = args.drop_edge
    client_num = args.num_workers

    sampling_rate = (np.ones(client_num) -
                          ovlap) / client_num

    data.index_orig = torch.arange(data.num_nodes)


    print("Graph to Networkx")
    G = to_networkx(
        data,
        node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
        to_undirected=True)
    print("Setting node attributes")
    nx.set_node_attributes(G,
                           dict([(nid, nid)
                                 for nid in range(nx.number_of_nodes(G))]),
                           name="index_orig")
    print("Calculating  partition")
    client_node_idx = {idx: [] for idx in range(client_num)}

    n_cuts, membership = metis.part_graph(G, client_num)
    indices = []
    for i in range(client_num):
        client_indices = np.where(np.array(membership) == i)[0]
        indices.append(client_indices)
    indices = np.concatenate(indices)



    sum_rate = 0
    for idx, rate in enumerate(sampling_rate):
        client_node_idx[idx] = indices[round(sum_rate *
                                             data.num_nodes):round(
            (sum_rate + rate) *
            data.num_nodes)]
        sum_rate += rate

    if ovlap:
        ovlap_nodes = indices[round(sum_rate * data.num_nodes):]
        for idx in client_node_idx:
            client_node_idx[idx] = np.concatenate(
                (client_node_idx[idx], ovlap_nodes))

    # Drop_edge index for each client
    if drop_edge:
        ovlap_graph = nx.Graph(nx.subgraph(G, ovlap_nodes))
        ovlap_edge_ind = np.random.permutation(
            ovlap_graph.number_of_edges())
        drop_all = ovlap_edge_ind[:round(ovlap_graph.number_of_edges() *
                                         drop_edge)]
        drop_client = [
            drop_all[s:s + round(len(drop_all) / client_num)]
            for s in range(0, len(drop_all),
                           round(len(drop_all) / client_num))
        ]

    graphs = []
    for owner in client_node_idx:
        nodes = client_node_idx[owner]
        sub_g = nx.Graph(nx.subgraph(G, nodes))
        if drop_edge:
            sub_g.remove_edges_from(
                np.array(ovlap_graph.edges)[drop_client[owner]])
        graphs.append(from_networkx(sub_g))

    return graphs



def split_Random(args,data):
    """
    original code link： https://github.com/alibaba/FederatedScope/blob/fe1806b36b4629bb0057e84912d5f42a79f4461d/federatedscope/core/splitters/graph/random_splitter.py#L14
    :param args: args.overlapping_rate(float):Additional samples of overlapping data, \
            eg. ``'0.4'``;
                    args.drop_edge(float): Drop edges (drop_edge / client_num) for each \
            client within overlapping part.
    :param data:
    :param clients:
    :return:
    """
    args.drop_edge = 0
    ovlap = args.overlapping_rate
    drop_edge = args.drop_edge
    client_num = args.num_workers

    sampling_rate = (np.ones(client_num) -
                          ovlap) / client_num

    data.index_orig = torch.arange(data.num_nodes)


    print("Graph to Networkx")
    G = to_networkx(
        data,
        node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
        to_undirected=True)
    print("Setting node attributes")
    nx.set_node_attributes(G,
                           dict([(nid, nid)
                                 for nid in range(nx.number_of_nodes(G))]),
                           name="index_orig")
    print("Calculating  partition")
    client_node_idx = {idx: [] for idx in range(client_num)}
    indices = np.random.permutation(data.num_nodes)

    sum_rate = 0
    for idx, rate in enumerate(sampling_rate):
        client_node_idx[idx] = indices[round(sum_rate *
                                             data.num_nodes):round(
            (sum_rate + rate) *
            data.num_nodes)]
        sum_rate += rate

    if ovlap:
        ovlap_nodes = indices[round(sum_rate * data.num_nodes):]
        for idx in client_node_idx:
            client_node_idx[idx] = np.concatenate(
                (client_node_idx[idx], ovlap_nodes))

    # Drop_edge index for each client
    if drop_edge:
        ovlap_graph = nx.Graph(nx.subgraph(G, ovlap_nodes))
        ovlap_edge_ind = np.random.permutation(
            ovlap_graph.number_of_edges())
        drop_all = ovlap_edge_ind[:round(ovlap_graph.number_of_edges() *
                                         drop_edge)]
        drop_client = [
            drop_all[s:s + round(len(drop_all) / client_num)]
            for s in range(0, len(drop_all),
                           round(len(drop_all) / client_num))
        ]

    graphs = []
    for owner in client_node_idx:
        nodes = client_node_idx[owner]
        sub_g = nx.Graph(nx.subgraph(G, nodes))
        if drop_edge:
            sub_g.remove_edges_from(
                np.array(ovlap_graph.edges)[drop_client[owner]])
        graphs.append(from_networkx(sub_g))

    return graphs

def split_Louvain(args,data):
    """
    original code link： https://github.com/alibaba/FederatedScope/blob/fe1806b36b4629bb0057e84912d5f42a79f4461d/federatedscope/core/splitters/graph/random_splitter.py#L14
    :param args: args.overlapping_rate(float):Additional samples of overlapping data, \
            eg. ``'0.4'``;
                    args.drop_edge(float): Drop edges (drop_edge / client_num) for each \
            client within overlapping part.
    :param data:
    :param clients:
    :return:
    """
    args.delta = 40

    delta= args.delta
    client_num = args.num_workers
    data.index_orig = torch.arange(data.num_nodes)
    print("Graph to Networkx")
    # G = to_networkx(
    #     data,
    #     node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
    #     to_undirected=True)

    node_attrs = ['x', 'y', 'train_mask', 'val_mask', 'test_mask']


    G = to_networkx(data, node_attrs=node_attrs, to_undirected=True)
    #partition = community_louvain.best_partition(G)
    #Large_data_list = ['Reddit','Reddit2','Yelp','Flickr']
    Large_data_list = ['Reddit']
    print("Setting node attributes")
    nx.set_node_attributes(G,
                           dict([(nid, nid)
                                 for nid in tqdm(range(nx.number_of_nodes(G)))]),
                           name="index_orig")




    # with tqdm(desc="Calculating community partition", total= total) as pbar:
    #     partition = community_louvain.best_partition(G)
    #     pbar.update(1)
    print("Calculating community partition")
    if args.dataset in Large_data_list:
        partition = community_louvain.best_partition(G,resolution = 0.1)
    else:
        partition = community_louvain.best_partition(G)


    cluster2node = {}
    for node in partition:
        cluster = partition[node]
        if cluster not in cluster2node:
            cluster2node[cluster] = [node]
        else:
            cluster2node[cluster].append(node)

    max_len = len(G) // client_num - delta
    max_len_client = len(G) // client_num

    tmp_cluster2node = {}
    for cluster in cluster2node:
        while len(cluster2node[cluster]) > max_len:
            tmp_cluster = cluster2node[cluster][:max_len]
            tmp_cluster2node[len(cluster2node) + len(tmp_cluster2node) +
                             1] = tmp_cluster
            cluster2node[cluster] = cluster2node[cluster][max_len:]
    cluster2node.update(tmp_cluster2node)

    orderedc2n = (zip(cluster2node.keys(), cluster2node.values()))
    orderedc2n = sorted(orderedc2n, key=lambda x: len(x[1]), reverse=True)

    client_node_idx = {idx: [] for idx in range(client_num)}
    idx = 0
    for (cluster, node_list) in orderedc2n:
        while len(node_list) + len(
                client_node_idx[idx]) > max_len_client + delta:
            idx = (idx + 1) % client_num
        client_node_idx[idx] += node_list
        idx = (idx + 1) % client_num

    graphs = []
    for owner in client_node_idx:
        nodes = client_node_idx[owner]
        graphs.append(from_networkx(nx.subgraph(G, nodes)))

    return graphs















def turn_to_pyg_data(client_graphs):
    client_data = []
    for i in range(len(client_graphs)):
        client_data.append(from_networkx(client_graphs[i]))

    return client_data


def train_test_split(data, client_id, split_percentage):
    mask = torch.randn((data.num_nodes)) < split_percentage
    nmask = torch.logical_not(mask)
    train_mask = mask
    test_mask = nmask
    data.train_mask = train_mask
    data.test_mask = test_mask
    return data


def trainer(model, optimizer, criterion, data):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    #print(data.x.shape)
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return loss, test_acc


def tester(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc


def tester2(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    f1 = f1_score(y_true=data.y[data.test_mask], y_pred=pred[data.test_mask], average='macro', zero_division=1)
    precision = precision_score(y_true=data.y[data.test_mask], y_pred=pred[data.test_mask], average='macro',
                                zero_division=1)
    recall = recall_score(y_true=data.y[data.test_mask], y_pred=pred[data.test_mask], average='macro', zero_division=1)
    return test_acc, f1, precision, recall


class EarlyStopping:
    def __init__(self, patience=20, change=0., path='euclid_model', mode='minimize'):
        """
        patience: Waiting threshold for val loss to improve.
        change: Minimum change in the model's quality.
        path: Path for saving the model to.
        """
        self.patience = patience
        self.change = change
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
        self.mode = mode

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.change and self.mode == "minimize":
            self.counter += 1

            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif score > self.best_score + self.change and self.mode == "maximize":
            self.counter += 1

            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0
