import argparse

def args_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # federated arguments
    parser.add_argument('--num_workers', type=int, default=10, help="number of clients in total")
    parser.add_argument('--batch_size', type=int, default=128, help="local batch size")
    parser.add_argument('--epochs', type=int, default=1000, help="training epochs")
    parser.add_argument('--lr', type=float, default=7e-4, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="weight decay")
    parser.add_argument('--step_size', type=int, default=100, help="step size")
    parser.add_argument('--gamma', type=float, default=0.7, help="gamma")
    parser.add_argument('--dropout', type=float, default=0.0, help="drop out")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")
    parser.add_argument('--defense', type=str, default='None', help='whethere perform a defense, e.g., foolsgold, flame')

    # argument for backdoor attack in GNN model
    parser.add_argument('--dataset', type=str, default="NCI1", help='name of dataset')
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
    parser.add_argument('--trigger_position', type=str, default="random", # ["random"]
                        help='Locations in a graph (subgraph) where a trigger is inserted')
    parser.add_argument('--is_iid', type=str, default= "iid", # ["iid","non-iid"]
                        help='"iid" stands for "independent and identically distributed, "non-iid" stands for "independent and identically distributed.')
    parser.add_argument('--device_id', type=int, default= 0, # ["iid","non-iid"]
                        help='device id')
    parser.add_argument('--filename', type = str, default = "", help='path of output file(save results)')
    parser.add_argument('--epoch_backdoor', type=int, default=0, help='from which epoch the malicious clients start backdoor attack')
    parser.add_argument('--seed', type=int, default=0, help='0-9')
    parser.add_argument('--proj_name', type=str, default="playground_proj", help='wandb logger project name')
    parser.add_argument('--exp_name', type=str, default="playground_exp", help='wandb logger experiment name')

    args = parser.parse_args()
    return args
