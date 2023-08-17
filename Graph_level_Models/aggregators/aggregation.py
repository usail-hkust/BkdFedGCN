import  random
import torch
import torch.nn as nn
import torch.nn.functional as F
from Graph_level_Models.helpers.metrics import accuracy_TU as accuracy
from copy import deepcopy
import copy




def fed_avg(severe_model,local_clients,args):
    #selected_models = random.sample(model_list, args.num_selected_models)
    for param_tensor in local_clients[0].model.state_dict():
        avg = (sum(c.model.state_dict()[param_tensor] for c in local_clients)) / len(local_clients)
        # Update the global
        severe_model.state_dict()[param_tensor].copy_(avg)
        # Send global to the local
        # for cl in model_list:
        #     cl.state_dict()[param_tensor].copy_(avg)
    return severe_model
def _initialize_global_optimizer(model, args):
    # global optimizer
    if args.glo_optimizer == "SGD":
        # similar as FedAvgM
        global_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.glo_lr,
            momentum=0.9,
            weight_decay=0.0
        )
    elif args.glo_optimizer == "Adam":
        global_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.glo_lr,
            betas=(0.9, 0.999),
            weight_decay=0.0
        )
    else:
        raise ValueError("No such glo_optimizer: {}".format(
            args.glo_optimizer
        ))
    return global_optimizer
def fed_opt(global_model,local_models,args):
    #local_models = random.sample(model_list, args.num_selected_models)
    global_optimizer = _initialize_global_optimizer(
        model=global_model, args=args
    )
    mean_state_dict = {}

    for name, param in global_model.state_dict().items():
        vs = []
        for id,client in enumerate(local_models):
            vs.append(local_models[id].model.state_dict()[name])
        vs = torch.stack(vs, dim=0)

        try:
            mean_value = vs.mean(dim=0)
        except Exception:
            # for BN's cnt
            mean_value = (1.0 * vs).mean(dim=0).long()
        mean_state_dict[name] = mean_value

    # zero_grad
    global_optimizer.zero_grad()
    global_optimizer_state = global_optimizer.state_dict()

    # new_model
    new_model = copy.deepcopy(global_model)
    new_model.load_state_dict(mean_state_dict, strict=True)

    # set global_model gradient
    with torch.no_grad():
        for param, new_param in zip(
                global_model.parameters(), new_model.parameters()
        ):
            param.grad = param.data - new_param.data

    # replace some non-parameters's state dict
    state_dict = global_model.state_dict()
    for name in dict(global_model.named_parameters()).keys():
        mean_state_dict[name] = state_dict[name]
    global_model.load_state_dict(mean_state_dict, strict=True)

    # optimization
    global_optimizer = _initialize_global_optimizer(
        global_model, args
    )
    global_optimizer.load_state_dict(global_optimizer_state)
    global_optimizer.step()

    return global_model
########################################################################
# def init_control(model,device):
#     """ a dict type: {name: params}
#     """
#     control = {
#         name: torch.zeros_like(
#             p.data
#         ).to(device) for name, p in model.state_dict().items()
#     }
#     return control

def init_control(model, device):
    """ a dict type: {name: params} """
    control = {k: torch.zeros_like(v.data).to(device) for k, v in model.named_parameters()}
    return control
def get_delta_model(model0, model1):
    """ return a dict: {name: params}
    """
    state_dict = {}
    for name, param0 in model0.named_parameters():
        param1 = model1.state_dict()[name]
        state_dict[name] = param0.detach() - param1.detach()
    return state_dict


class ScaffoldOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(
            lr=lr, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def step(self, server_control, client_control, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        #
        # for group in self.param_groups:
        #     for p, c, ci in zip(group['params'], server_control.values(), client_control.values()):
        #         if p.grad is None:
        #             continue
        #         print("p.grad.data",p.grad.data.shape)
        #         print("c.data", c.data.shape)
        #         print("ci.data", ci.data.shape)
        #         dp = p.grad.data + c.data - ci.data
        #         p.data = p.data - dp.data * group['lr']
        for group in self.param_groups:
            for p, c, ci in zip(group['params'], server_control.values(), client_control.values()):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group['lr']


def update_local(model,server_control, client_control, global_model,train_iter,test_iter, device,
                 args):

    glo_model = copy.deepcopy(global_model)

    optimizer = ScaffoldOptimizer(
        model.parameters(),
        lr=args.scal_lr,
        weight_decay=args.weight_decay
    )

    for _ in range(args.local_steps):
        train_l_sum, train_acc_sum, train_n, train_batch_count = 0.0, 0.0, 0, 0
        for batch_graphs, batch_labels in  train_iter:
            model.train()
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
            batch_e = batch_graphs.edata['feat'].to(device)

            batch_labels = batch_labels.to(torch.long)
            batch_labels = batch_labels.to(device)

            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            l = model.loss(batch_scores, batch_labels)

            optimizer.zero_grad()
            l.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            optimizer.step(
                server_control=server_control,
                client_control=client_control
            )
            train_l_sum += l.cpu().item()
            train_acc_sum += accuracy(batch_scores, batch_labels)
            train_n += batch_labels.size(0)
            train_batch_count += 1

            model.eval()
            with torch.no_grad():
                test_l_sum, test_acc_sum, test_n, test_batch_count = 0.0, 0.0, 0, 0
                for batch_graphs, batch_labels in test_iter:
                    batch_graphs = batch_graphs.to(device)
                    batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
                    batch_e = batch_graphs.edata['feat'].to(device)

                    batch_labels = batch_labels.to(torch.long)
                    batch_labels = batch_labels.to(device)

                    batch_scores = model.forward(batch_graphs, batch_x, batch_e)
                    l = model.loss(batch_scores, batch_labels)


                    test_l_sum += l.cpu().item()
                    test_acc_sum += accuracy(batch_scores, batch_labels)
                    test_n += batch_labels.size(0)
                    test_batch_count += 1




        delta_model = get_delta_model(glo_model, model)


    local_steps = args.local_steps

    return delta_model, local_steps,train_l_sum / train_batch_count, train_acc_sum / train_n, test_l_sum / test_batch_count, test_acc_sum / train_n


def update_local_control(delta_model, server_control,
        client_control, steps, lr):

    new_control = copy.deepcopy(client_control)
    delta_control = copy.deepcopy(client_control)

    for name in delta_model.keys():
        c = server_control[name]
        ci = client_control[name]
        delta = delta_model[name]

        new_ci = ci.data - c.data + delta.data / (steps * lr)
        new_control[name].data = new_ci
        delta_control[name].data = ci.data - new_ci
    return new_control, delta_control


def scaffold(global_model,server_control,client_control,model,
                 train_iter, test_iter, device,
                 args):

    global_model = global_model.to(device)
    model = model.to(device)
    # update local with control variates / ScaffoldOptimizer
    delta_model, local_steps,loss_train,  acc_train,loss_val, acc_val = update_local(model, server_control, client_control, global_model, train_iter,
                                                                                     test_iter, device,args)

    client_control, delta_control = update_local_control(
        delta_model=delta_model,
        server_control=server_control,
        client_control=client_control,
        steps=local_steps,
        lr=args.lr,
    )


    return loss_train, loss_val, acc_train, acc_val,client_control, delta_control, delta_model