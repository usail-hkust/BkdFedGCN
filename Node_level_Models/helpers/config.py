import argparse

def args_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug mode')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=10, help='Random seed.')
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
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train benign and backdoor model.')
    parser.add_argument('--trojan_epochs', type=int, default=400, help='Number of epochs to train trigger generator.')
    parser.add_argument('--inner', type=int, default=1, help='Number of inner')
    # backdoor setting
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--trigger_size', type=int, default=3,
                        help='tirgger_size')
    parser.add_argument('--vs_size', type=int, default=40,
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




   # other settings





    # federated arguments
    parser.add_argument('--num_workers', type=int, default=10, help="number of clients in total")
    parser.add_argument('--batch_size', type=int, default=128, help="local batch size")
    parser.add_argument('--step_size', type=int, default=100, help="step size")
    parser.add_argument('--gamma', type=float, default=0.7, help="gamma")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")
    parser.add_argument('--defense', type=str, default='None', help='whethere perform a defense, e.g., foolsgold, flame')

    # argument for backdoor attack in GNN model
    #parser.add_argument('--dataset', type=str, default="NCI1", help='name of dataset')
    parser.add_argument('--datadir', type=str, default="./Data", help='path to save the dataset')
    parser.add_argument('--config', help="Please give a config.json file with model and training details")
    parser.add_argument('--target_label', type=int, default=0, help='target label of the poisoned dataset')
    parser.add_argument('--poisoning_intensity', type=float, default=0.2, help='frac of training dataset to be injected trigger')
    parser.add_argument('--frac_of_avg', type=float, default=0.2, help='frac of avg nodes to be injected the trigger')
    parser.add_argument('--density', type=float, default=0.8, help='density of the edge in the generated trigger')
    parser.add_argument('--num_mali', type=int, default=3, help="number of malicious clients")
    parser.add_argument('--avg_degree', type=int, default=3,
                        help="number of average node degree for the parapamter to generate the trigger: watts_strogatz_graph, barabasi_albert_graph, random_regular_graph")
    parser.add_argument('--trigger_type', type=str, default="renyi", # ["renyi","ws",'ba','rr']
                        help='trigger graph generated by erdos_renyi_graph, watts_strogatz_graph, barabasi_albert_graph, random_regular_graph.')
    parser.add_argument('--is_iid', type=str, default= "iid", # ["iid","non-iid"]
                        help='"iid" stands for "independent and identically distributed, "non-iid" stands for "independent and identically distributed.')
    parser.add_argument('--device_id', type=int, default= 0, # ["iid","non-iid"]
                        help='device id')
    parser.add_argument('--filename', type = str, default = "", help='path of output file(save results)')
    parser.add_argument('--epoch_backdoor', type=int, default=0, help='from which epoch the malicious clients start backdoor attack')


    args = parser.parse_args()
    return args
