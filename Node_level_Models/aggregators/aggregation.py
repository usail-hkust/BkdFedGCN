import torch
import  random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Node_level_Models.helpers.func_utils import accuracy
from copy import deepcopy
from torch_geometric.nn import GCNConv
import copy




def fed_avg(severe_model,model_list,args):
    Sub_model_list = random.sample(model_list, args.num_sample_submodels)
    for param_tensor in Sub_model_list[0].state_dict():
        avg = (sum(c.state_dict()[param_tensor] for c in Sub_model_list)) / len(Sub_model_list)
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
def fed_opt(global_model,model_list,args):
    local_models = random.sample(model_list, args.num_sample_submodels)
    global_optimizer = _initialize_global_optimizer(
        model=global_model, args=args
    )
    mean_state_dict = {}

    for name, param in global_model.state_dict().items():
        vs = []
        for id,client in enumerate(local_models):
            vs.append(local_models[id].state_dict()[name])
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
def init_control(model,device):
    """ a dict type: {name: params}
    """
    control = {
        name: torch.zeros_like(
            p.data
        ).to(device) for name, p in model.state_dict().items()
    }
    return control
def get_delta_model(model0, model1):
    """ return a dict: {name: params}
    """
    state_dict = {}
    for name, param0 in model0.state_dict().items():
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

        ng = len(self.param_groups[0]["params"])
        names = list(server_control.keys())


        # t = 0
        # for group in self.param_groups:
        #     for p in group["params"]:
        #         if p.grad is None:
        #             continue
        #
        #         c = server_control[names[t]]
        #         ci = client_control[names[t]]
        #
        #         print(names[t], p.shape, c.shape, ci.shape)
        #         d_p = p.grad.data + c.data - ci.data
        #         p.data = p.data - d_p.data * group["lr"]
        #         t += 1
        # assert t == ng
        for group in self.param_groups:
            for p, c, ci in zip(group['params'], server_control.values(), client_control.values()):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group['lr']

        return loss

def update_local(model,server_control, client_control, global_model,
                 features, edge_index, edge_weight, labels, idx_train,
                 args, idx_val=None, train_iters=200):

    glo_model = copy.deepcopy(global_model)

    optimizer = ScaffoldOptimizer(
        model.parameters(),
        lr=args.scal_lr,
        weight_decay=args.weight_decay
    )

    best_loss_val = 100
    best_acc_val = 0

    for i in range(train_iters):
        model.train()


        output = model.forward(features, edge_index, edge_weight)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])

        optimizer.zero_grad()
        loss_train.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm
        )
        optimizer.step(
            server_control=server_control,
            client_control=client_control
        )


        model.eval()
        with torch.no_grad():
            output = model.forward(features, edge_index, edge_weight)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            acc_train = accuracy(output[idx_train], labels[idx_train])


        if acc_val > best_acc_val:
            best_acc_val = acc_val

            weights = deepcopy(model.state_dict())
            model.load_state_dict(weights)




    delta_model = get_delta_model(glo_model, model)


    local_steps = train_iters

    return delta_model, local_steps,loss_train.item(), loss_val.item(), acc_train, acc_val


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
                 features, edge_index, edge_weight, labels, idx_train,
                 args, idx_val=None, train_iters=200):


    # # control variates
    # server_control = init_control(global_model)
    #
    # client_controls = {
    #     client: init_control(global_model)
    #     for id, client in enumerate(model_list)
    # }

    # update local with control variates / ScaffoldOptimizer
    delta_model, local_steps,loss_train, loss_val, acc_train, acc_val = update_local(
        model, server_control, client_control, global_model,
        features, edge_index, edge_weight, labels, idx_train,
        args, idx_val=idx_val, train_iters=train_iters
    )


    client_control, delta_control = update_local_control(
        delta_model=delta_model,
        server_control=server_control,
        client_control=client_control,
        steps=local_steps,
        lr=args.lr,
    )


    return loss_train, loss_val, acc_train, acc_val,client_control, delta_control, delta_model