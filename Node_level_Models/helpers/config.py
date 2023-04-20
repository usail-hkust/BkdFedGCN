import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='Insert Arguments')
    parser.add_argument("--seed", type=int, default=10, help="seed")
    parser.add_argument("--num_clients", type=int, default=5, help="number of clients")
    parser.add_argument("--num_sample_submodels", type=int, default=5,
                        help="num of clients randomly selected to participate in Federated Learning")
    parser.add_argument("--hidden_channels", type=int, default=32, help="size of GNN hidden layer")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate for training")
    parser.add_argument("--inner_epochs", type=int, default=1, help="epochs for training")
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
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train benign and backdoor model.')
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
    parser.add_argument('--trigger_position', type=str, default='random',
                        choices=['loss', 'conf', 'cluster', 'none', 'cluster_degree'],
                        help='Method to select idx_attach for training trojan model (none means randomly select)')
    parser.add_argument('--test_model', type=str, default='GCN',
                        choices=['GCN', 'GAT', 'GraphSage', 'GIN'],
                        help='Model used to attack')
    parser.add_argument('--evaluate_mode', type=str, default='1by1',
                        choices=['overall', '1by1'],
                        help='Model used to attack')
    parser.add_argument('--trigger_type', type=str, default='renyi',
                        choices=["renyi","ws", "ba", "rr", "gta","adaptive"],
                        help='Generate the trigger methods')

    # federated setting
    parser.add_argument('--num_malicious', type=int, default=1,
                        help="number of malicious attacker")
    parser.add_argument("--split_method", type=str, default="random",
                        help="split the graph into the clients: random is randomly split, louvain is the community detection method")

    args = parser.parse_args()
    return args
