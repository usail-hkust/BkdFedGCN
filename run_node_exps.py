from Node_level_Models.configs.config import args_parser
from Node_level_Models.helpers.metrics_utils import log_test_results


import numpy as np
import wandb

args = args_parser()
rs = np.random.RandomState(args.seed)
seeds = rs.randint(1000, size=5)


project_name = [args.proj_name, args.proj_name+ "debug"]
proj_name = project_name[0]

def main(args):
    model_name = args.model
    # 'data-{}_model-{}_IID-{}_num_workers-{}_num_mali-{}_epoch_backdoor-{}_frac_of_avg-{}_trigger_type-{}_trigger_position-{}_poisoning_intensity-{}'
    Alg_name = "Alg-" +args.agg_method
    file_name = Alg_name + 'D-{}_M-{}_IID-{}_NW-{}_NM-{}_EB-{}_TS-{}_TPye-{}_TPo-{}_PI-{}_OR-{}'.format(
        args.dataset,
        model_name,
        args.is_iid,
        args.num_workers,
        args.num_mali,
        args.epoch_backdoor,
        args.trigger_size,
        args.trigger_type,
        args.trigger_position,
        args.poisoning_intensity,
        args.overlapping_rate)

    average_overall_performance_list, average_ASR_list, average_Flip_ASR_list, average_transfer_attack_success_rate_list = [], [], [], []
    results_table = []
    metric_list = []
    if args.agg_method == "scaffold":
        from backdoor_node_clf_scaffold import main as backdoor_main
    else:
        from backdoor_node_clf import main as backdoor_main

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

        average_overall_performance, average_ASR, average_Flip_ASR, average_transfer_attack_success_rate = backdoor_main(args, logger)
        results_table.append([average_overall_performance, average_ASR, average_Flip_ASR, average_transfer_attack_success_rate])
        logger.log({"average_overall_performance": average_overall_performance,
                    "average_ASR": average_ASR,
                    "average_Flip_ASR": average_Flip_ASR,
                    "average_transfer_attack_success_rate": average_transfer_attack_success_rate})

        average_overall_performance_list.append(average_overall_performance)
        average_ASR_list.append(average_ASR)
        average_Flip_ASR_list.append(average_Flip_ASR)
        average_transfer_attack_success_rate_list.append(average_transfer_attack_success_rate)
        # end the logger
        wandb.finish()

    # wandb table logger init
    columns = ["average_overall_performance","average_ASR","average_Flip_ASR","average_transfer_attack_success_rate"]
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

    mean_average_overall_performance, mean_average_ASR, mean_average_Flip_ASR, mean_average_transfer_attack_success_rate = np.mean(np.array(average_overall_performance_list)),\
                                                                                                           np.mean(np.array(average_ASR_list)),\
                                                                                                           np.mean(np.array(average_Flip_ASR_list)), \
                                                                                                           np.mean(np.array(average_transfer_attack_success_rate_list))

    std_average_overall_performance, std_average_ASR, std_average_Flip_ASR, std_average_transfer_attack_success_rate = np.std(np.array(average_overall_performance_list)),\
                                                                                                           np.std(np.array(average_ASR_list)),\
                                                                                                           np.std(np.array(average_Flip_ASR_list)), \
                                                                                                           np.std(np.array(average_transfer_attack_success_rate_list))


    header = ['dataset', 'model', "mean_average_overall_performance",
              "std_average_overall_performance", "mean_average_ASR", "std_average_ASR",
              "mean_average_Flip_ASR", "std_average_Flip_ASR",
              "mean_average_local_unchanged_acc","std_average_transfer_attack_success_rate"]
    paths = "./checkpoints/Node/"

    metric_list.append(args.dataset)
    metric_list.append(model_name)
    metric_list.append(mean_average_overall_performance)
    metric_list.append(std_average_overall_performance)

    metric_list.append(mean_average_ASR)
    metric_list.append(std_average_ASR)

    metric_list.append(mean_average_Flip_ASR)
    metric_list.append(std_average_Flip_ASR)


    metric_list.append(mean_average_transfer_attack_success_rate)
    metric_list.append(std_average_transfer_attack_success_rate)


    paths = paths + "data-{}/".format(args.dataset) + "model-{}/".format(model_name) + file_name
    log_test_results(paths, header, file_name)
    log_test_results(paths, metric_list, file_name)

if __name__ == '__main__':
    args = args_parser()
    main(args)
