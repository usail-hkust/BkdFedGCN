from trainer.workerbase  import WorkerBase
import torch
from torch import nn
import json
import os
from Graph_level_Models.helpers.config import args_parser
from Graph_level_Models.datasets.gnn_util import  transform_dataset, inject_global_trigger_test, save_object, split_dataset
from Graph_level_Models.datasets.TUs import TUsDataset
from Graph_level_Models.nets.TUs_graph_classification.load_net import gnn_model
from Graph_level_Models.helpers.evaluate import gnn_evaluate_accuracy
import time
import numpy as np
from torch.utils.data import DataLoader
import copy
from defense import foolsgold

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
    # prepare backdoor local backdoor dataset
    train_loader_list = []
    attack_loader_list = []
    test_clean_loader_list = []
    test_unchanged_loader_list = []

    for i in range(args.num_mali):
        train_trigger_graphs, test_trigger_graphs, G_trigger, final_idx, test_clean_data, test_unchanged_data = transform_dataset(partition[i], partition[args.num_workers+i],
                                                                                            avg_nodes, args)
        triggers.append(G_trigger)
        tmp_graphs = [partition[i][idx] for idx in range(len(partition[i])) if idx not in final_idx]
        train_dataset = train_trigger_graphs + tmp_graphs
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  drop_last=drop_last,
                                  collate_fn=dataset.collate)

        # only trigger data
        attack_loader = DataLoader(test_trigger_graphs, batch_size=args.batch_size, shuffle=True,
                                   drop_last=drop_last,
                                   collate_fn=dataset.collate)
        # only clean data
        test_clean_loader = DataLoader(test_clean_data, batch_size=args.batch_size, shuffle=True,
                                   drop_last=drop_last,
                                   collate_fn=dataset.collate)
        # only unchanged data
        test_unchanged_loader = DataLoader(test_unchanged_data, batch_size=args.batch_size, shuffle=True,
                                   drop_last=drop_last,
                                   collate_fn=dataset.collate)


        train_loader_list.append(train_loader)
        attack_loader_list.append(attack_loader)

        test_clean_loader_list.append(test_clean_loader)
        test_unchanged_loader_list.append(test_unchanged_loader)

    # save global trigger in order to implement centrilized backoor attack
    if args.num_mali > 0:
        filename = "./Data/global_trigger/%d/%s_%s_%d_%d_%d_%.2f_%.2f_%.2f_%s" \
                   % (args.seed, MODEL_NAME, config['dataset'], args.num_workers, args.num_mali, args.epoch_backdoor,
                      args.frac_of_avg, args.poisoning_intensity, args.density, args.trigger_type) + '.pkl'
        path = os.path.split(filename)[0]
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        # save the global trigger that is then used in the centralized backdoor attack
        save_object(triggers, filename)
        print('The global trigger is saved successfully!')
        test_global_trigger = inject_global_trigger_test(partition[-1], avg_nodes, args, triggers)
        test_global_trigger_load = DataLoader(test_global_trigger, batch_size=args.batch_size, shuffle=True,
                                              drop_last=drop_last,
                                              collate_fn=dataset.collate)
    acc_record = [0]
    counts = 0
    weight_history = []
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

        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        different_clients_test_accuracy_local_trigger = []
        different_clients_test_accuracy_global_trigger = []
        for i in range(args.num_workers):
            att_list = []
            train_loss, train_acc, test_loss, test_acc = client[i].gnn_train()
            different_clients_test_accuracy_local_trigger.append(test_acc)
            client[i].scheduler.step()
            global_att = gnn_evaluate_accuracy(test_global_trigger_load, client[i].model)
            different_clients_test_accuracy_global_trigger.append(global_att)
            print('Client %d, loss %.4f, train acc %.3f, test loss %.4f, test acc %.3f'
                  % (i, train_loss, train_acc, test_loss, test_acc))
            print('Client %d with global trigger: %.3f' % (i, global_att))

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
                tmp_acc = gnn_evaluate_accuracy(attack_loader_list[j], client[i].model)
                print('Client %d with local trigger %d: %.3f' % (i, j, tmp_acc))
                att_list.append(tmp_acc)
            if not args.filename == "":
                save_path = os.path.join(args.filename, str(args.seed), config['model'] + '_' + args.dataset + \
                                         '_%d_%d_%.2f_%.2f_%.2f_%s' % (
                                         args.num_workers, args.num_mali, args.frac_of_avg, args.poisoning_intensity,
                                         args.density,args.trigger_type) + '_%d.txt' % i)
                path = os.path.split(save_path)[0]
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)

                with open(save_path, 'a') as f:
                    f.write('%.3f %.3f %.3f %.3f %.3f ' % (train_loss, train_acc, test_loss, test_acc, global_att))
                    for i in range(len(triggers)):
                        f.write('%.3f' % att_list[i])
                        f.write(' ')
                    f.write('\n')

        # wandb logger
        logger.log(worker_results)

        weights = []
        for i in range(args.num_workers):
            weights.append(client[i].get_weights())
            weight_history.append(client[i].get_weights())
        #print('len weights',len(weights[0]))

        # Aggregation in the server to get the global model
        # if there is a defense applied
        if args.defense == 'foolsgold':
            result, weight_history, alpha = foolsgold(args, weight_history, weights)
            save_path = os.path.join("./Results/alpha/DBA", str(args.seed), MODEL_NAME + '_' + args.dataset + \
                                     '_%d_%d_%.2f_%.2f_%.2f_%s' % (
                                     args.num_workers, args.num_mali, args.frac_of_avg, args.poisoning_intensity,
                                     args.density,args.trigger_type) + '_alpha.txt')
            path = os.path.split(save_path)[0]
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            with open(save_path, 'a') as f:
                for i in range(args.num_workers):
                    f.write("%.3f" % (alpha[i]))
                    f.write(' ')
                f.write("\n")
        else:
            result = server_robust_agg(args, weights)

        for i in range(args.num_workers):
            client[i].set_weights(weights=result)
            client[i].upgrade()

        # evaluate the global model: test_acc
        test_acc = gnn_evaluate_accuracy(client[0].test_iter, client[0].model)
        print('Global Test Acc: %.3f' % test_acc)
        if not args.filename == "":
            save_path = os.path.join(args.filename, str(args.seed),
                                     MODEL_NAME + '_' + args.dataset + '_%d_%d_%.2f_%.2f_%.2f_%s' \
                                     % (args.num_workers, args.num_mali, args.frac_of_avg, args.poisoning_intensity,
                                        args.density,args.trigger_type) + '_global_test.txt')
            path = os.path.split(save_path)[0]
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)

            with open(save_path, 'a') as f:
                f.write("%.3f" % (test_acc))
                f.write("\n")

        # inject triggers into the testing data
        if args.num_mali > 0 and epoch >= args.epoch_backdoor:
            local_att_acc = []
            global_att_acc = gnn_evaluate_accuracy(test_global_trigger_load, client[0].model)
            print('Global model with global trigger: %.3f' % global_att_acc)
            for i in range(args.num_mali):
                tmp_acc = gnn_evaluate_accuracy(attack_loader_list[i], client[0].model)
                print('Global model with local trigger %d: %.3f' % (i, tmp_acc))
                local_att_acc.append(tmp_acc)

            if not args.filename == "":
                save_path = os.path.join(args.filename, str(args.seed),
                                         MODEL_NAME + '_' + args.dataset + '_%d_%d_%.2f_%.2f_%.2f_%s' \
                                         % (args.num_workers, args.num_mali, args.frac_of_avg, args.poisoning_intensity,
                                            args.density,args.trigger_type) + '_global_attack.txt')
                path = os.path.split(save_path)[0]
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
                with open(save_path, 'a') as f:
                    f.write("%.3f" % (global_att_acc))
                    f.write(' ')
                    for i in range(args.num_mali):
                        f.write("%.3f" % (local_att_acc[i]))
                        f.write(' ')
                    f.write('\n')


    # clean accuracy , poison accuracy, attack success rate
    # average all the workers
    all_clean_acc_list = []
    for i in range(args.num_workers):
        tmp_acc = gnn_evaluate_accuracy(all_workers_clean_test_list[i], client[i].model)
        print('Client %d with clean accuracy: %.3f' % (i,  tmp_acc))
        all_clean_acc_list.append(tmp_acc)

    average_all_clean_acc = np.mean(np.array(all_clean_acc_list))


    local_attack_success_rate_list = []
    for i in range(args.num_mali):
        tmp_acc = gnn_evaluate_accuracy(attack_loader_list[i], client[i].model)
        print('local model:%d with local trigger: %.3f' % (i, tmp_acc))
        local_attack_success_rate_list.append(tmp_acc)
    average_local_attack_success_rate_acc = np.mean(np.array(local_attack_success_rate_list))


    local_clean_acc_list = []
    for i in range(args.num_mali):
        tmp_acc = gnn_evaluate_accuracy(test_clean_loader_list[i], client[i].model)
        print('local model:%d with clean data: %.3f' % (i, tmp_acc))

        local_clean_acc_list.append(tmp_acc)
    average_local_clean_acc = np.mean(np.array(local_clean_acc_list))

    local_unchanged_acc_list = []
    for i in range(args.num_mali):
        tmp_acc = gnn_evaluate_accuracy(test_unchanged_loader_list[i], client[i].model)
        print('local model:%d with unchanged: %.3f' % (i, tmp_acc))
        local_unchanged_acc_list.append(tmp_acc)
    average_local_unchanged_acc = np.mean(np.array(local_unchanged_acc_list))


    return average_all_clean_acc, average_local_attack_success_rate_acc, average_local_clean_acc,average_local_unchanged_acc




if __name__ == '__main__':
    main()