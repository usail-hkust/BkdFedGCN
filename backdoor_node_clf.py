import torch
import matplotlib.pyplot as plt
import argparse
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import numpy as np
import os
import time
import Node_level_Models.helpers.selection_utils  as hs
#Run Federated Experiment
#from model import GCN, SAGE
#from federated_node_classification import run_federated_node_classification
from Node_level_Models.helpers.func_utils import subgraph,get_split
from torch_geometric.utils import to_undirected
#Split Graph and creating client datasets
from helpers.split_graph_utils import split_communities,split_Random, split_Louvain,turn_to_pyg_data,train_test_split
from Node_level_Models.models.construct import model_construct
from Node_level_Models.helpers.func_utils import prune_unrelated_edge,prune_unrelated_edge_isolated
import  random
# get the start time
st = time.time()
torch.manual_seed(12345)
np.random.seed(12345)
torch.cuda.manual_seed_all(12345)

parser = argparse.ArgumentParser(description='Insert Arguments')
parser.add_argument("--seed", type=int, default=10, help="seed")
parser.add_argument("--num_clients", type=int, default=5, help="number of clients")
parser.add_argument("--num_sample_submodels", type=int, default=4, help="num of clients randomly selected to participate in Federated Learning")
parser.add_argument("--hidden_channels", type=int, default=32, help="size of GNN hidden layer")
parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate for training")
parser.add_argument("--epochs", type=int, default=1, help="epochs for training")
parser.add_argument('--device_id', type=int, default=0,  # ["iid","non-iid"]
                    help='device id')
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN', 'GAT', 'GraphSage', 'GIN'])
parser.add_argument('--dataset', type=str, default='Cora',
                    help='Dataset',
                    choices=['Cora', 'Citeseer', 'Pubmed', 'PPI', 'Flickr', 'ogbn-arxiv', 'Reddit', 'Reddit2',
                             'Yelp'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=0.5)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--federated_rounds', type=int, default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int, default=400, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int, default=1, help='Number of inner')
# backdoor setting
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--trigger_size', type=int, default=3,
                    help='tirgger_size')
parser.add_argument('--vs_size', type=int, default=4,
                    help="ratio of poisoning nodes relative to the full graph")
# defense setting
parser.add_argument('--defense_mode', type=str, default="none",
                    choices=['prune', 'isolate', 'none'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.2,
                    help="Threshold of prunning edges")
parser.add_argument('--homo_loss_weight', type=float, default=0,
                    help="Weight of optimize similarity loss")
# attack setting
parser.add_argument('--dis_weight', type=float, default=1,
                    help="Weight of cluster distance")
parser.add_argument('--selection_method', type=str, default='none',
                    choices=['loss', 'conf', 'cluster', 'none', 'cluster_degree'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN', 'GAT', 'GraphSage', 'GIN'],
                    help='Model used to attack')
parser.add_argument('--evaluate_mode', type=str, default='1by1',
                    choices=['overall', '1by1'],
                    help='Model used to attack')
parser.add_argument('--backdoor', type=str, default='GTA',
                    choices=['GTA', '1by1'],
                    help='Generate the trigger methods')

# federated setting
parser.add_argument('--num_malicious', type=int, default=5,
                    help="number of malicious attacker")
parser.add_argument("--split_method", type=str, default="random", help="split the graph into the clients: random is randomly split, louvain is the community detection method")

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)



##### DATA PREPARATION #####
#Import and Examine Dataset
if args.dataset.lower() == 'pubmed':
    dataset = Planetoid(root='./data', name='PubMed', transform=T.LargestConnectedComponents())
elif args.dataset.lower() == 'cora':
    dataset = Planetoid(root='./data', name='Cora', transform=T.LargestConnectedComponents())
elif args.dataset.lower() == 'citeseer':
    dataset = Planetoid(root='./data', name='CiteSeer',transform=T.LargestConnectedComponents())
else:
    print("No such dataset!")
    exit()

print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the graph object.

print(data)
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')

if args.split_method == "random":
    client_data = split_Random(args, data)
elif args.split_method == "louvain":
    client_data = split_Louvain(args, data)
else:
    raise NameError

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.device_id)
#Split Graph and creating client datasets

#client_data = split_communities(data, args.clients)
#client_data = split_Random(args,data)


print("client_graphs",client_data)

for i in range(args.num_clients):
    print(len(client_data[i]))


#Create data objects for the new component-graphs

#client_data = turn_to_pyg_data(client_graphs)

for i in range(args.num_clients):
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
for i in range(args.num_malicious):
    if args.backdoor == "GTA":
       from Node_level_Models.models.GTA import Backdoor
       Backdoor_model = Backdoor(args, device)
    elif args.backdoor == "Adaptive":
        from Node_level_Models.models.GTA import Backdoor
        Backdoor_model = Backdoor(args, device)
    else:
        raise NameError
    Backdoor_model_list.append(Backdoor_model)

# prepare for backdoor injected node index
size = args.vs_size #int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio) num_trigger_nodes
client_idx_attach = []
for i in range(args.num_clients):
    if (args.selection_method == 'none'):
        idx_attach = hs.obtain_attach_nodes(args, client_unlabeled_idx[i], size)
        idx_attach = torch.LongTensor(idx_attach).to(device)
    elif (args.selection_method == 'cluster'):
        idx_attach = hs.cluster_distance_selection(args, client_data[i], client_idx_train[i], client_idx_val[i], client_idx_clean_test[i], client_unlabeled_idx[i],
                                                   client_train_edge_index[i], size, device)
        idx_attach = torch.LongTensor(idx_attach).to(device)
    elif (args.selection_method == 'cluster_degree'):
        idx_attach = hs.cluster_degree_selection(args, client_data[i], client_idx_train[i], client_idx_val[i], client_idx_clean_test[i], client_unlabeled_idx[i],
                                                   client_train_edge_index[i], size, device)
        idx_attach = torch.LongTensor(idx_attach).to(device)
    else:
        raise NameError
    client_idx_attach.append(idx_attach)


# construct the triggers
client_poison_x, client_poison_edge_index, client_poison_edge_weights, client_poison_labels = [], [], [], []
for i in range(args.num_malicious):
    backdoor_model = Backdoor_model_list[i]
    backdoor_model.fit(client_data[i].x,client_train_edge_index[i], None, client_data[i].y, client_idx_train[i], client_idx_attach[i], client_unlabeled_idx[i])
    poison_x, poison_edge_index, poison_edge_weights, poison_labels = backdoor_model.get_poisoned()
    client_poison_x.append(poison_x)
    client_poison_edge_index.append(poison_edge_index)
    client_poison_edge_weights.append(poison_edge_weights)
    client_poison_labels.append(poison_labels)


# data level defense
client_bkd_tn_nodes = []
for i in range(args.num_malicious):
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
model_list = []
optimizer_list = []

# Initialize clients
model_list = []
for i in range(args.num_clients):
    test_model = model_construct(args, args.test_model, data, device).to(device)
    model_list.append(test_model)

# Initialize the sever model
severe_model = model_construct(args, args.test_model, data, device).to(device)

random.seed(args.seed)
#rs = random.sample(range(0,args.num_clients),args.num_malicious)

rs = [i for i in range(args.num_malicious)]
#print("+++++++++++++ Federated Node Classification +++++++++++++")
#args.federated_rounds = epoch, the inner iteration normly is set to 1.
print("rs",rs)
for i in range(args.federated_rounds+1):
    client_induct_edge_index = []
    client_induct_edge_weights = []
    for j in range(args.num_clients):
        if j in rs:

            model_list[j].fit(client_poison_x[j].to(device), client_poison_edge_index[j].to(device), client_poison_edge_weights[j].to(device), client_poison_labels[j].to(device), client_bkd_tn_nodes[j].to(device), client_idx_val[j].to(device),
                           train_iters=args.epochs, verbose=False)

            output = model_list[j](client_poison_x[j].to(device), client_poison_edge_index[j].to(device), client_poison_edge_weights[j].to(device))
            train_attach_rate = (output.argmax(dim=1)[idx_attach] == args.target_class).float().mean()
            print("malicious client: {} ,target class rate on Vs: {:.4f}".format(j,train_attach_rate))
            induct_edge_index = torch.cat([client_poison_edge_index[j].to(device), client_mask_edge_index[j].to(device)], dim=1)
            induct_edge_weights = torch.cat(
                [client_poison_edge_weights[j], torch.ones([client_mask_edge_index[j].shape[1]], dtype=torch.float, device=device)])

            clean_acc = model_list[j].test(client_poison_x[j].to(device), induct_edge_index.to(device),
                                           induct_edge_weights.to(device), client_data[j].y.to(device), client_idx_clean_test[j].to(device))
        else:
            #fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False):
            model_list[j].fit(client_data[j].x.to(device), client_data[j].edge_index.to(device), client_data[j].edge_weight.to(device), client_data[j].y.to(device), client_idx_train[j].to(device), client_idx_val[j].to(device),
                           train_iters=args.epochs, verbose=False)
            induct_x, induct_edge_index, induct_edge_weights = client_data[j].x, client_data[j].edge_index, client_data[
                j].edge_weight
            clean_acc = model_list[j].test(client_data[j].x.to(device), client_data[j].edge_index.to(device),
                                           client_data[j].edge_weight.to(device), client_data[j].y.to(device), client_idx_clean_test[j].to(device))
        client_induct_edge_index.append(induct_edge_index)
        client_induct_edge_weights.append(induct_edge_weights)



        # test_model = test_model.cpu()

        print("accuracy on clean test nodes: {:.4f}".format(clean_acc))

        # Server Aggregation
        Sub_model_list = random.sample(model_list, args.num_sample_submodels)
        for param_tensor in Sub_model_list[0].state_dict():
            avg = (sum(c.state_dict()[param_tensor] for c in Sub_model_list)) / len(Sub_model_list)
            severe_model.state_dict()[param_tensor].copy_(avg)
            for cl in model_list:
                cl.state_dict()[param_tensor].copy_(avg)

# Test performance

overall_performance = []
overall_malicious_train_attach_rate = []
overall_malicious_train_flip_asr = []
for i in range(args.num_clients):
    if i in rs:
        # inject trigger on attack test nodes (idx_atk)'''
        induct_x, induct_edge_index, induct_edge_weights = Backdoor_model_list[i].inject_trigger(client_idx_atk[i], client_poison_x[i], client_induct_edge_index[i],
                                                                                client_induct_edge_weights[i], device)

        # if (args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
        #     induct_edge_index, induct_edge_weights = prune_unrelated_edge(args, induct_edge_index, induct_edge_weights,
        #                                                                   induct_x, device)
        # attack evaluation
        # test_model = test_model.to(device)
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
        # %% inject trigger on attack test nodes (idx_atk)'''
        induct_x, induct_edge_index, induct_edge_weights = client_data[i].x, client_data[i].edge_index, client_data[i].edge_weight





    ca = test_model.test(induct_x.to(device), induct_edge_index.to(device), induct_edge_weights.to(device), client_data[i].y.to(device), client_idx_clean_test[i].to(device))
    print("Client: {}, CA: {:.4f}".format(i,ca))
    overall_performance.append(ca)
    print(overall_malicious_train_attach_rate)
print("Malicious client: {}".format(rs))
print("Average ASR: {:.4f}".format(np.array(overall_malicious_train_attach_rate).sum() / args.num_malicious))
print("Flip ASR: {:.4f}".format(np.array(overall_malicious_train_flip_asr).sum()/ args.num_malicious))
print("Average Performance on clean test set: {:.4f}".format(np.array(overall_performance).sum() / args.num_clients))



