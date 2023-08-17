from Graph_level_Models.helpers.config import args_parser

from Graph_level_Models.helpers.metrics_utils import log_test_results
import numpy as np
import json
import wandb

args = args_parser()
rs = np.random.RandomState(args.seed)
seeds = rs.randint(1000, size=5)


project_name = [args.proj_name, args.proj_name+ "debug"]
proj_name = project_name[0]

def main(args):
    with open(args.config) as f:
        config = json.load(f)
    model_name = config['model']
    if args.defense == "scaffold":
        from backdoor_graph_scaffold import main as backdoor_main
    else:
        from backdoor_graph_clf import main as backdoor_main


    fed_name = "Fed_alg-"+args.defense + "_"
    # 'data-{}_model-{}_IID-{}_num_workers-{}_num_mali-{}_epoch_backdoor-{}_frac_of_avg-{}_trigger_type-{}_trigger_position-{}_poisoning_intensity-{}'
    file_name = fed_name +'D-{}_M-{}_IID-{}_NW-{}_NM-{}_EB-{}_FA-{}_TPye-{}_TPo-{}_PI-{}'.format(
        args.dataset,
        model_name,
        args.is_iid,
        args.num_workers,
        args.num_mali,
        args.epoch_backdoor,
        args.frac_of_avg,
        args.trigger_type,
        args.trigger_position,
        args.poisoning_intensity)

    average_all_clean_acc_list, average_local_attack_success_rate_acc_list, average_local_clean_acc_list = [], [], []
    results_table = []
    metric_list = []
    average_local_unchanged_acc_list = []
    average_transfer_attack_success_rate_list = []
    for i in range(len(seeds)):
        args.seed = seeds[i]

        # wandb init
        logger = wandb.init(
            #entity="hkust-gz",
            project=proj_name,
            group=file_name,
            name=f"round_{i}",
            config=args,
        )

        average_all_clean_acc, average_local_attack_success_rate_acc, average_local_clean_acc, average_local_unchanged_acc,average_transfer_attack_success_rate = backdoor_main(args, logger)
        results_table.append([average_all_clean_acc, average_local_attack_success_rate_acc, average_local_clean_acc, average_local_unchanged_acc,average_transfer_attack_success_rate])
        logger.log({"average_all_clean_acc": average_all_clean_acc,
                    "average_local_attack_success_rate_acc": average_local_attack_success_rate_acc,
                    "average_local_clean_acc": average_local_clean_acc,
                    "average_local_unchanged_acc": average_local_unchanged_acc,
                    "average_transfer_attack_success_rate":average_transfer_attack_success_rate})

        average_all_clean_acc_list.append(average_all_clean_acc)
        average_local_attack_success_rate_acc_list.append(average_local_attack_success_rate_acc)
        average_local_clean_acc_list.append(average_local_clean_acc)
        average_local_unchanged_acc_list.append(average_local_unchanged_acc)
        average_transfer_attack_success_rate_list.append(average_transfer_attack_success_rate)
        # end the logger
        wandb.finish()

    # wandb table logger init
    columns = ["average_all_clean_acc", "average_local_attack_success_rate_acc", "average_local_clean_acc", "average_local_unchanged_acc","average_transfer_attack_success_rate"]
    logger_table = wandb.Table(columns=columns, data=results_table)
    table_logger = wandb.init(
        #entity="hkust-gz",
        project=proj_name,
        group=file_name,
        name=f"exp_results",
        config=args,
    )
    table_logger.log({"results": logger_table})
    wandb.finish()

    mean_average_all_clean_acc, mean_average_local_attack_success_rate_acc, mean_average_local_clean_acc = np.mean(np.array(average_all_clean_acc_list)),\
                                                                                                           np.mean(np.array(average_local_attack_success_rate_acc_list)),\
                                                                                                           np.mean(np.array(average_local_clean_acc_list))

    std_average_all_clean_acc, std_average_local_attack_success_rate_acc, std_average_local_clean_acc = np.std(np.array(average_all_clean_acc_list)),\
                                                                                                        np.std(np.array(average_local_attack_success_rate_acc_list)),\
                                                                                                        np.std(np.array(average_local_clean_acc_list))

    mean_average_local_unchanged_acc, std_average_local_unchanged_acc = np.mean(np.array(average_local_unchanged_acc_list)), np.std(np.array(average_local_unchanged_acc_list))
    mean_average_transfer_attack_success_rate, std_average_transfer_attack_success_rate = np.mean(np.array(average_transfer_attack_success_rate_list)), np.std(np.array(average_transfer_attack_success_rate_list))



    header = ['dataset', 'model', "mean_average_all_clean_acc",
              "std_average_all_clean_acc", "mean_average_local_attack_success_rate_acc", "std_average_local_attack_success_rate_acc",
              "mean_average_local_clean_acc", "std_average_local_clean_acc",
              "mean_average_local_unchanged_acc","std_average_local_unchanged_acc","mean_average_transfer_attack_success_rate","std_average_transfer_attack_success_rate"]
    paths = "./checkpoints/Graph/"

    metric_list.append(args.dataset)
    metric_list.append(model_name)
    metric_list.append(mean_average_all_clean_acc)
    metric_list.append(std_average_all_clean_acc)

    metric_list.append(mean_average_local_attack_success_rate_acc)
    metric_list.append(std_average_local_attack_success_rate_acc)

    metric_list.append(mean_average_local_clean_acc)
    metric_list.append(std_average_local_clean_acc)


    metric_list.append(mean_average_local_unchanged_acc)
    metric_list.append(std_average_local_unchanged_acc)

    metric_list.append(mean_average_transfer_attack_success_rate)
    metric_list.append(std_average_transfer_attack_success_rate)
    paths = paths + "data-{}/".format(args.dataset) + "model-{}/".format(model_name) + file_name
    log_test_results(paths, header, file_name)
    log_test_results(paths, metric_list, file_name)

if __name__ == '__main__':
    args = args_parser()
    main(args)
    # models = ["GCN", "GAT", "GraphSage"]
    # args.is_iid = "p-degree-non-iid"
    # args.dataset = "COLORS-3"
    # for m in models:
    #     args.config = f"./Graph_level_Models/configs/TUS/TUs_graph_classification_{m}_{args.dataset}_100k.json"
    #     main(args)
