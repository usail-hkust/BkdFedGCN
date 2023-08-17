from trainer.workerbase  import WorkerBase
import torch
from torch import nn
import json
import os
from Graph_level_Models.helpers.config import args_parser
from Graph_level_Models.datasets.gnn_util import    split_dataset
from Graph_level_Models.datasets.TUs import TUsDataset
from Graph_level_Models.nets.TUs_graph_classification.load_net import gnn_model
from Graph_level_Models.helpers.evaluate import gnn_evaluate_accuracy
import numpy as np
from torch.utils.data import DataLoader
import copy


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

    model = gnn_model(MODEL_NAME, net_params)

    # print("Target Model:\n{}".format(model))
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
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
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




    weight_history = []
    for epoch in range(args.epochs):
        print('epoch:', epoch)

        # worker results
        worker_results = {}
        for i in range(args.num_workers):
            worker_results[f"client_{i}"] = {"train_loss": None, "train_acc": None, "test_loss": None, "test_acc": None}

        for i in range(args.num_workers):
            att_list = []
            train_loss, train_acc, test_loss, test_acc = client[i].gnn_train()

            client[i].scheduler.step()
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



        # wandb logger
        logger.log(worker_results)

        weights = []
        for i in range(args.num_workers):
            weights.append(client[i].get_weights())
            weight_history.append(client[i].get_weights())
        #print('len weights',len(weights[0]))

        # Aggregation in the server to get the global model

        result = server_robust_agg(args, weights)

        for i in range(args.num_workers):
            client[i].set_weights(weights=result)
            client[i].upgrade()

        # evaluate the global model: test_acc
        test_acc = gnn_evaluate_accuracy(client[0].test_iter, client[0].model)
        print('Global Test Acc: %.3f' % test_acc)



    # clean accuracy
    # average all the workers
    all_clean_acc_list = []
    for i in range(args.num_workers):
        tmp_acc = gnn_evaluate_accuracy(all_workers_clean_test_list[i], client[i].model)
        print('Client %d with clean accuracy: %.3f' % (i,  tmp_acc))
        all_clean_acc_list.append(tmp_acc)

    average_all_clean_acc = np.mean(np.array(all_clean_acc_list))




    return average_all_clean_acc




if __name__ == '__main__':
    main()