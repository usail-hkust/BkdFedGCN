import  random
import torch
import torch.nn as nn
import torch.nn.functional as F
from Node_level_Models.helpers.func_utils import accuracy
from copy import deepcopy
import copy




def fed_avg(severe_model,local_models,args):
    #selected_models = random.sample(model_list, args.num_selected_models)
    for param_tensor in local_models[0].state_dict():
        avg = (sum(c.state_dict()[param_tensor] for c in local_models)) / len(local_models)
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

######################################defense ########################
def fed_median(global_model,client_models, args):
    """
    Implementation of median refers to `Byzantine-robust distributed
    learning: Towards optimal statistical rates`
    [Yin et al., 2018]
    (http://proceedings.mlr.press/v80/yin18a/yin18a.pdf)

    It computes the coordinate-wise median of recieved updates from clients

    The code is adapted from https://github.com/alibaba/FederatedScope/blob/master/federatedscope/core/aggregators/median_aggregator.py
    """
    client_parameters = [model.parameters() for model in client_models]
    for global_param, *client_params in zip(global_model.parameters(),
                                            *client_parameters):
        temp = torch.stack(client_params, dim=0)
        temp_pos, _ = torch.median(temp, dim=0)
        temp_neg, _ = torch.median(-temp, dim=0)
        new_temp = (temp_pos - temp_neg) / 2
        global_param.data = new_temp
    return global_model
###################################### fed_trimmedmean ##############################################################
def fed_trimmedmean(global_model,client_models, args):
    """
    Implementation of median refer to `Byzantine-robust distributed
    learning: Towards optimal statistical rates`
    [Yin et al., 2018]
    (http://proceedings.mlr.press/v80/yin18a/yin18a.pdf)

    The code is adapted from https://github.com/alibaba/FederatedScope/blob/master/federatedscope/core/aggregators/trimmedmean_aggregator.py
    """


    client_parameters = [model.parameters() for model in client_models]
    excluded_ratio = args.excluded_ratio
    excluded_num = int(len(client_models) * excluded_ratio)
    for global_param, *client_params in zip(global_model.parameters(),
                                            *client_parameters):
        temp = torch.stack(client_params, dim=0)
        pos_largest, _ = torch.topk(temp, excluded_num, dim=0)
        neg_smallest, _ = torch.topk(-temp, excluded_num, dim=0)
        new_stacked = torch.cat([temp, -pos_largest, neg_smallest], dim=0).sum(dim=0).float()
        new_stacked /= len(temp) - 2 * excluded_num
        global_param.data = new_stacked
    return global_model

###################################### fed_multi_krum ##############################################################

def _calculate_score( models,args):
    """
    Calculate Krum scores
    """
    byzantine_node_num = args.num_mali
    model_num = len(models)
    closest_num = model_num - byzantine_node_num - 2

    distance_matrix = torch.zeros(model_num, model_num)
    for index_a in range(model_num):
        for index_b in range(index_a, model_num):
            if index_a == index_b:
                distance_matrix[index_a, index_b] = float('inf')
            else:
                distance_matrix[index_a, index_b] = distance_matrix[
                    index_b, index_a] = _calculate_distance(
                    models[index_a], models[index_b])

    sorted_distance = torch.sort(distance_matrix)[0]
    krum_scores = torch.sum(sorted_distance[:, :closest_num], axis=-1)
    return krum_scores

def _calculate_distance(model_a, model_b):
    """
    Calculate the Euclidean distance between two given model parameter lists
    """
    distance = 0.0
    #model_a_params,model_b_params = model_a.parameters(), model_b.parameters()
    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        distance += torch.dist(param_a.data, param_b.data, p=2)

    return distance
def fed_multi_krum(global_model, client_models, args):
    federate_ignore_weight = False
    byzantine_node_num = args.num_mali
    client_num = len(client_models)
    agg_num = args.agg_num
    assert 2 * byzantine_node_num + 2 < client_num, \
        "it should be satisfied that 2*byzantine_node_num + 2 < client_num"
    # each_model: (sample_size, model_para)
    #models_para = [model.parameters() for model in client_models]
    krum_scores = _calculate_score(client_models, args)
    index_order = torch.sort(krum_scores)[1].numpy()
    reliable_models = list()
    reliable_client_train_loaders = []
    for number, index in enumerate(index_order):
        if number < agg_num:
            reliable_models.append(client_models[index])


    client_parameters = [model.parameters() for model in reliable_models]
    if  federate_ignore_weight:
        weights = torch.as_tensor([len(train_loader) for train_loader in reliable_client_train_loaders])
        weights = weights / weights.sum()
    else:
        weights = torch.as_tensor([1 for _ in range(len(reliable_models))])
        weights = weights / weights.sum()

    for model_parameter in zip(global_model.parameters(), *client_parameters):
        global_parameter = model_parameter[0]
        client_parameter = [client_parameter.data * weight for client_parameter, weight in
                            zip(model_parameter[1:], weights)]
        client_parameter = torch.stack(client_parameter, dim=0).sum(dim=0)
        global_parameter.data = client_parameter

    return global_model
###################################### fed_bulyan ##############################################################
def fed_bulyan(global_model, client_models, args):
    """
    Implementation of Bulyan refers to `The Hidden Vulnerability
    of Distributed Learning in Byzantium`
    [Mhamdi et al., 2018]
    (http://proceedings.mlr.press/v80/mhamdi18a/mhamdi18a.pdf)

    It combines the MultiKrum aggregator and the treamedmean aggregator
    """

    agg_num = args.agg_num
    byzantine_node_num = args.num_mali
    client_num = len(client_models)
    assert 2 * byzantine_node_num + 2 < client_num, \
        "it should be satisfied that 2*byzantine_node_num + 2 < client_num"
    # assert 4 * byzantine_node_num + 3 <= client_num, \
    #     "it should be satisfied that 4 * byzantine_node_num + 3 <= client_num"

    # models_para = [model.parameters() for model in client_models]


    krum_scores = _calculate_score(client_models, args)
    index_order = torch.sort(krum_scores)[1].numpy()
    reliable_models = []
    #reliable_client_train_loaders = []
    for number, index in enumerate(index_order):
        if number < agg_num:
            reliable_models.append(client_models[index])


    client_parameters = [model.parameters() for model in reliable_models]


    '''
    Sort parameter for each coordinate of the rest \theta reliable
    local models, and find \gamma (gamma<\theta-2*self.byzantine_num)
    parameters closest to the median to perform averaging
    '''
    excluded_num = args.excluded_num

    for global_param, *client_params in zip(global_model.parameters(),
                                            *client_parameters):
        temp = torch.stack(client_params, dim=0)
        pos_largest, _ = torch.topk(temp, excluded_num, dim=0)
        neg_smallest, _ = torch.topk(-temp, excluded_num, dim=0)
        new_stacked = torch.cat([temp, -pos_largest, neg_smallest], dim=0).sum(dim=0).float()
        new_stacked /= len(temp) - 2 * excluded_num
        global_param.data = new_stacked

    return global_model