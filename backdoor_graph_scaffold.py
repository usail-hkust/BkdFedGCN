import  random
import torch
from torch import nn
import json
import os
import time
import numpy as np
import copy
from torch.utils.data import DataLoader
from Graph_level_Models.helpers.config import args_parser
from Graph_level_Models.datasets.gnn_util import  transform_dataset,  split_dataset
from Graph_level_Models.datasets.TUs import TUsDataset
from Graph_level_Models.nets.TUs_graph_classification.load_net import gnn_model
from Graph_level_Models.helpers.evaluate import gnn_evaluate_accuracy
from Graph_level_Models.defenses.defense import foolsgold
from Graph_level_Models.trainer.workerbase  import WorkerBase
from Graph_level_Models.aggregators.aggregation import fed_avg,fed_opt
from Graph_level_Models.aggregators.aggregation import scaffold,init_control

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


def server_robust_agg(args, grad):  ## server aggregation
    grad_in = np.array(grad).reshape((args.num_workers, -1)).mean(axis=0)
    return grad_in.tolist()


class ClearDenseClient(WorkerBase):
    def __init__(self, client_id, model, loss_func, train_iter, attack_iter, test_iter, config, optimizer, device,
                 grad_stub, args, scheduler):
        super(ClearDenseClient, self).__init__(model=model, loss_func=loss_func, train_iter=train_iter,
                                               attack_iter=attack_iter, test_iter=test_iter, config=config,
                                               optimizer=optimizer, device=device)
        self.client_id = client_id
        self.grad_stub = None
        self.args = args
        self.scheduler = scheduler

    def update(self):
        pass


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


def main(args, logger):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    with open(args.config) as f:
        config = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device_id)
    dataset = TUsDataset(args)
    args.device = device
    collate = dataset.collate
    MODEL_NAME = config['model']
    net_params = config['net_params']
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()

    net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]

    num_classes = torch.max(dataset.all.graph_labels).item() + 1
    net_params['n_classes'] = num_classes
    net_params['dropout'] = args.dropout
    args.epoch_backdoor = int(args.epoch_backdoor * args.epochs)

    model = gnn_model(MODEL_NAME, net_params)
    #global_model = gnn_model(MODEL_NAME, net_params)


    # Initialize clients
    model_list = []
    for i in range(args.num_workers):
        test_model = gnn_model(MODEL_NAME, net_params)
        model_list.append(test_model)

    # Initialize the sever model
    global_model = gnn_model(MODEL_NAME, net_params)







    client = []

    # logger data
    loss_func = nn.CrossEntropyLoss()
    # Load data
    partition, avg_nodes = split_dataset(args, dataset)
    drop_last = True if MODEL_NAME == 'DiffPool' else False
    triggers = []

    all_workers_clean_test_list = []
    for i in range(args.num_workers):
        local_model = copy.deepcopy(model)
        local_model = local_model.to(device)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)




        train_dataset = partition[i]
        test_dataset = partition[args.num_workers + i]

        print("Client %d training data num: %d" % (i, len(train_dataset)))
        print("Client %d testing data num: %d" % (i, len(test_dataset)))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  drop_last=drop_last,
                                  collate_fn=dataset.collate)
        attack_loader = None
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 drop_last=drop_last,
                                 collate_fn=dataset.collate)
        all_workers_clean_test_list.append(test_loader)
        client.append(ClearDenseClient(client_id=i, model=local_model, loss_func=loss_func, train_iter=train_loader,
                                       attack_iter=attack_loader, test_iter=test_loader, config=config,
                                       optimizer=optimizer, device=device, grad_stub=None, args=args,
                                       scheduler=scheduler))
    # check model memory address
    for i in range(args.num_workers):
        add_m = id(client[i].model)
        add_o = id(client[i].optimizer)
        print('model {} address: {}'.format(i, add_m))
        print('optimizer {} address: {}'.format(i, add_o))
    # prepare backdoor local backdoor dataset
    train_loader_list = []
    attack_loader_list = []
    test_clean_loader_list = []
    test_unchanged_loader_list = []

    for i in range(args.num_mali):
        train_trigger_graphs, test_trigger_graphs, G_trigger, final_idx, test_clean_data, test_unchanged_data = transform_dataset(partition[i], partition[args.num_workers+i],
                                                                                            avg_nodes, args)
        #triggers.append(G_trigger)
        tmp_graphs = [partition[i][idx] for idx in range(len(partition[i])) if idx not in final_idx]
        train_dataset = train_trigger_graphs + tmp_graphs
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  drop_last=drop_last,
                                  collate_fn=dataset.collate)

        # only trigger data in test dataloader
        attack_loader = DataLoader(test_trigger_graphs, batch_size=args.batch_size, shuffle=False,
                                   drop_last=drop_last,
                                   collate_fn=dataset.collate)
        # only clean data in test dataloader
        test_clean_loader = DataLoader(test_clean_data, batch_size=args.batch_size, shuffle=False,
                                   drop_last=drop_last,
                                   collate_fn=dataset.collate)
        # only unchanged data in test dataloader
        test_unchanged_loader = DataLoader(test_unchanged_data, batch_size=args.batch_size, shuffle=False,
                                   drop_last=drop_last,
                                   collate_fn=dataset.collate)


        train_loader_list.append(train_loader)
        attack_loader_list.append(attack_loader)

        test_clean_loader_list.append(test_clean_loader)
        test_unchanged_loader_list.append(test_unchanged_loader)




    # control variates
    server_control = init_control(global_model, device)

    client_controls = {
        id: init_control(global_model, device)
        for id, client in enumerate(model_list)
    }



    weight_history = []
    delta_models = {}
    delta_controls = {}
    for epoch in range(args.epochs):
        print('epoch:', epoch)

        # worker results
        worker_results = {}
        for i in range(args.num_workers):
            worker_results[f"client_{i}"] = {"train_loss": None, "train_acc": None, "test_loss": None, "test_acc": None}

        if epoch >= args.epoch_backdoor:
            # malicious clients start backdoor attack
            for i in range(0, args.num_mali):
                client[i].train_iter = train_loader_list[i]
                client[i].attack_iter = attack_loader_list[i]



        for i in range(args.num_workers):
            # att_list = []
            # train_loss, train_acc, test_loss, test_acc = client[i].gnn_train()
            # different_clients_test_accuracy_local_trigger.append(test_acc)
            # client[i].scheduler.step()
            train_loss, test_loss, train_acc, test_acc,\
            client_control, delta_control, delta_model = scaffold(global_model=global_model,
                                                                  server_control=server_control,
                                                                  client_control=client_controls[i],
                                                                  model= model_list[i],
                                                                  train_iter= client[i].train_iter,
                                                                  test_iter= client[i].test_iter,
                                                                  device=device,
                                                                  args=args)

            client_controls[i] = copy.deepcopy(client_control)
            delta_models[i] = copy.deepcopy(delta_model)
            delta_controls[i] = copy.deepcopy(delta_control)

            print('Client %d, loss %.4f, train acc %.3f, test loss %.4f, test acc %.3f'
                  % (i, train_loss, train_acc, test_loss, test_acc))


            # save worker results
            for ele in worker_results[f"client_{i}"]:
                if ele == "train_loss":
                    worker_results[f"client_{i}"][ele] = train_loss
                elif ele == "train_acc":
                    worker_results[f"client_{i}"][ele] = train_acc
                elif ele == "test_loss":
                    worker_results[f"client_{i}"][ele] = test_loss
                elif ele == "test_acc":
                    worker_results[f"client_{i}"][ele] = test_acc

            for j in range(len(triggers)):
                tmp_acc = gnn_evaluate_accuracy(attack_loader_list[j], model_list[i])
                print('Client %d with local trigger %d: %.3f' % (i, j, tmp_acc))
                #att_list.append(tmp_acc)


        # wandb logger
        logger.log(worker_results)

        #selected_clients = random.sample(client, args.num_selected_models)
        # if there is a defense applied
        global_model = update_global(global_model, delta_models, args)
        new_control = update_global_control(
            control=server_control,
            delta_controls=delta_controls,
        )
        server_control = copy.deepcopy(new_control)




    # average all the workers
    all_clean_acc_list = []
    for i in range(args.num_workers):
        tmp_acc = gnn_evaluate_accuracy(all_workers_clean_test_list[i], model_list[i])
        print('Client %d with clean accuracy: %.3f' % (i,  tmp_acc))
        all_clean_acc_list.append(tmp_acc)

    average_all_clean_acc = np.mean(np.array(all_clean_acc_list))


    local_attack_success_rate_list = []
    for i in range(args.num_mali):
        tmp_acc = gnn_evaluate_accuracy(attack_loader_list[i], model_list[i])
        print('Malicious client %d with local trigger, attack success rate: %.4f' % (i, tmp_acc))
        local_attack_success_rate_list.append(tmp_acc)
    average_local_attack_success_rate_acc = np.mean(np.array(local_attack_success_rate_list))


    local_clean_acc_list = []
    for i in range(args.num_mali):
        tmp_acc = gnn_evaluate_accuracy(test_clean_loader_list[i], model_list[i])
        print('Malicious client %d with clean data, clean accuracy: %.4f' % (i, tmp_acc))

        local_clean_acc_list.append(tmp_acc)
    average_local_clean_acc = np.mean(np.array(local_clean_acc_list))

    local_unchanged_acc_list = []
    for i in range(args.num_mali):
        tmp_acc = gnn_evaluate_accuracy(test_unchanged_loader_list[i], model_list[i])
        print('Malicious client %d with unchanged data, the unchanged clean accuracy: %.3f' % (i, tmp_acc))
        local_unchanged_acc_list.append(tmp_acc)
    average_local_unchanged_acc = np.mean(np.array(local_unchanged_acc_list))

    transfer_attack_success_rate_list = []
    if args.num_workers-args.num_mali <= 0:
        average_transfer_attack_success_rate = -10000.0
    else:
        for i in range(args.num_mali):
            for j in range(args.num_workers - args.num_mali):
                tmp_acc = gnn_evaluate_accuracy(attack_loader_list[i], model_list[args.num_mali+j])
                print('Clean client %d with  trigger %d: %.3f' % (args.num_mali+j, i, tmp_acc))
                transfer_attack_success_rate_list.append(tmp_acc)
        average_transfer_attack_success_rate = np.mean(np.array(transfer_attack_success_rate_list))

    return average_all_clean_acc, average_local_attack_success_rate_acc, average_local_clean_acc,average_local_unchanged_acc, average_transfer_attack_success_rate




if __name__ == '__main__':
    main()