from Graph_level_Models.helpers.config import args_parser
from backdoor_graph_clf import  main as backdoor_main
from helpers.metrics_utils import log_test_results
import  numpy as np
import json
import wandb

args = args_parser()
rs = np.random.RandomState(args.seed)
seeds = rs.randint(1000,size=5)


def main(args):

    average_all_clean_acc_list, average_local_attack_success_rate_acc_list, average_local_clean_acc_list = [], [], []
    metric_list = []
    average_local_unchanged_acc_list = []
    for i in range(len(seeds)):
        args.seed = seeds[i]
        with open(args.config) as f:
            config = json.load(f)
        model_name = config['model']
        #'data-{}_model-{}_IID-{}_num_workers-{}_num_mali-{}_epoch_backdoor-{}_frac_of_avg-{}_trigger_type-{}_trigger_position-{}_poisoning_intensity-{}'
        file_name = 'D-{}_M-{}_IID-{}_NW-{}_NM-{}_EB-{}_FA-{}_TPye-{}_TPo-{}_PI-{}'.format(
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


        # wandb init
        logger = wandb.init(
            entity="hkust-gz",
            project=args.proj_name,
            group=file_name,
            name=f"round_{i}",
            config=args,
        )


        average_all_clean_acc, average_local_attack_success_rate_acc, average_local_clean_acc,average_local_unchanged_acc = backdoor_main(args, logger)
        average_all_clean_acc_list.append(average_all_clean_acc)
        average_local_attack_success_rate_acc_list.append(average_local_attack_success_rate_acc)
        average_local_clean_acc_list.append(average_local_clean_acc)

        average_local_unchanged_acc_list.append(average_local_unchanged_acc)
        # end the logger



    mean_average_all_clean_acc, mean_average_local_attack_success_rate_acc, mean_average_local_clean_acc = np.mean(np.array(average_all_clean_acc_list)),\
                                                                                                                                                                 np.mean(np.array(average_local_attack_success_rate_acc_list)),\
                                                                                                                                                                 np.mean(np.array(average_local_clean_acc_list))
    std_average_all_clean_acc, std_average_local_attack_success_rate_acc, std_average_local_clean_acc = np.std(np.array(average_all_clean_acc_list)),\
                                                                                                                                                                 np.std(np.array(average_local_attack_success_rate_acc_list)),\
                                                                                                                                                                 np.std(np.array(average_local_clean_acc_list))

    mean_average_local_unchanged_acc, std_average_local_unchanged_acc = np.mean(np.array(average_local_unchanged_acc_list)), np.std(np.array(average_local_unchanged_acc_list))

    header = ['dataset', 'model', "mean_average_all_clean_acc",
              "std_average_all_clean_acc", "mean_average_local_attack_success_rate_acc", "std_average_local_attack_success_rate_acc",
              "mean_average_local_clean_acc", "std_average_local_clean_acc",
              "mean_average_local_unchanged_acc","std_average_local_unchanged_acc"]
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
    paths = paths + "data-{}/".format(args.dataset) + "model-{}/".format(model_name) + file_name
    log_test_results(paths, header, file_name)
    log_test_results(paths, metric_list, file_name)
    logger.finish()

if __name__ == '__main__':
    args = args_parser()
    main(args)