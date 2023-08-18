import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='Insert Arguments')
    parser.add_argument("--seed", type=int, default=10, help="seed")
    parser.add_argument("--num_workers", type=int, default=5, help="number of clients")
    parser.add_argument("--num_selected_models", type=int, default=5, help="num of clients randomly selected to participate in Federated Learning")
    parser.add_argument("--hidden_channels", type=int, default=32, help="size of GNN hidden layer")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate for training")
    parser.add_argument("--inner_epochs", type=int, default=1, help="epochs for training")
    parser.add_argument('--device_id', type=int, default=0,  # ["iid","non-iid"]
                        help='device id')
    parser.add_argument('--model', type=str, default='GraphSage', help='model',
                        choices=['GCN', 'GAT', 'GraphSage', 'GIN'])
    parser.add_argument('--dataset', type=str, default='Reddit',
                        help='Dataset',
                        choices=['Cora', 'Citeseer', 'Pubmed', 'Flickr', 'ogbn-arxiv', 'Reddit', 'Reddit2',
                                 'Yelp',"Cs","Physics","computers","photo",'ogbn-products','ogbn-proteins'])
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
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train benign and backdoor model.')
    parser.add_argument('--trojan_epochs', type=int, default=400, help='Number of epochs to train trigger generator.')
    parser.add_argument('--inner', type=int, default=1, help='Number of inner')

    # backdoor setting
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--trigger_size', type=int, default=3,
                        choices=[3,4,5,6,7,8,9,10],
                        help='tirgger_size')
    parser.add_argument('--vs_size', type=int, default=4,
                        help="ratio of poisoning nodes relative to the full graph")
    parser.add_argument('--poisoning_intensity', type=float, default=0.1,
                        help="ratio of poisoning nodes relative to the full graph")
    parser.add_argument('--density', type=float, default=0.8, help='density of the edge in the generated trigger')
    # defense setting
    parser.add_argument('--defense_mode', type=str, default="none",
                        choices=['prune', 'isolate', 'none'],
                        help="Mode of defense")
    parser.add_argument('--prune_thr', type=float, default=0.2,
                        help="Threshold of prunning edges")

    # attack setting
    parser.add_argument('--dis_weight', type=float, default=1,
                        help="Weight of cluster distance")
    parser.add_argument('--trigger_position', type=str, default='random',
                        choices=[ 'learn_cluster', 'random', 'learn_cluster_degree',"degree","cluster"],
                        help='Method to select idx_attach for training trojan model')
    parser.add_argument('--evaluate_mode', type=str, default='1by1',
                        choices=['overall', '1by1'],
                        help='Model used to attack')
    parser.add_argument('--trigger_type', type=str, default='renyi',
                        choices=["renyi","ws", "ba", "gta","ugba"],
                        help='Generate the trigger methods')
    parser.add_argument('--degree', type=int, default=3,
                        help='The degree of trigger type')
    parser.add_argument('--target_loss_weight', type=float, default=1,
                        help="Weight of optimize outter trigger generator in ugba")
    parser.add_argument('--homo_loss_weight', type=float, default=100,
                        help="Weight of optimize similarity loss in ugba")
    parser.add_argument('--homo_boost_thrd', type=float, default=0.8,
                        help="Threshold of increase similarity in ugba")
    # federated setting
    parser.add_argument('--num_mali', type=int, default=1,
                        help="number of malicious attacker")
    parser.add_argument('--overlapping_rate', type=float, default=0.0, choices=[0.0,0.1,0.2,0.3,0.4,0.5],
                        help="Additional samples of overlapping data")
    parser.add_argument("--is_iid", type=str, default="iid", choices=["iid", "non-iid-louvain",'non-iid-Metis'],
                        help="split the graph into the clients: random is randomly split, louvain is the community detection method")
    parser.add_argument('--epoch_backdoor', type=float, default= 0.0, choices=[0.0,0.05,0.1,0.2,0.3,0.4,0.5], help='from which epoch the malicious clients start backdoor attack')
    parser.add_argument('--proj_name', type=str, default="BkdFedGCN-rebuttal", help='wandb logger project name')
    # semi-settings
    parser.add_argument('--ratio_training', type=float, default=0.4, help='labels of ratio of training')
    parser.add_argument('--ratio_val', type=float, default=0.1, help='labels of ratio of val')
    parser.add_argument('--ratio_testing', type=float, default=0.2, help='labels of ratio of testing')
    #other federated algoritm  settings
    parser.add_argument('--agg_method', type=str, default="FedAvg",
                        help='Federated Algorithms')
    parser.add_argument('--mu', type=float, default=0.01, help='proximal term constant')
    parser.add_argument('--glo_optimizer', type=str, default="Adam", help='the optimizer of global model in FedOPT')
    parser.add_argument('--glo_lr', type=float, default=3e-4, help='the learning rate  of global model in FedOPT')
    parser.add_argument('--max_grad_norm', type=float, default=100.0, help='max grad norm')
    parser.add_argument('--scal_lr', type=float, default=0.01, help='the learning rate  of global model in FedOPT')

    # other federated algorithm  settings (defense)
    parser.add_argument('--agg_num', type=int, default=1, help='aggregation number for multi-krum and bulyan')
    parser.add_argument('--excluded_num', type=int, default=1, help='excluded number for  bulyan')
    parser.add_argument('--excluded_ratio', type=float, default=0.2, help='excluded number for  bulyan')

    args = parser.parse_args()
    return args
