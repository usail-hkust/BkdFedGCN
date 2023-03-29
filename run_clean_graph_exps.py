from Graph_level_Models.helpers.config import args_parser
from clean_graph_clf import  main as clean_main
from helpers.metrics_utils import log_test_results
import numpy as np
import json
import wandb

args = args_parser()
rs = np.random.RandomState(args.seed)
seeds = rs.randint(1000, size=5)

project_name = [args.proj_name, args.proj_name+ "debug"]
proj_name = project_name[1]
def main(args):
    with open(args.config) as f:
        config = json.load(f)
    model_name = config['model']
    # 'data-{}_model-{}_IID-{}_num_workers-{}_num_mali-{}_epoch_backdoor-{}_frac_of_avg-{}_trigger_type-{}_trigger_position-{}_poisoning_intensity-{}'
    args.exp_class = "clean"
    file_name = 'Exp-Class-{}_D-{}_M-{}_IID-{}_NW-{}'.format(args.exp_class,
        args.dataset,
        model_name,
        args.is_iid,
        args.num_workers)

    average_all_clean_acc_list= []
    results_table = []
    metric_list = []
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

        average_all_clean_acc = clean_main(args, logger)
        results_table.append([average_all_clean_acc])
        logger.log({"average_all_clean_acc": average_all_clean_acc})

        average_all_clean_acc_list.append(average_all_clean_acc)

        # end the logger
        wandb.finish()

    # wandb table logger init
    columns = ["average_all_clean_acc"]
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

    mean_average_all_clean_acc = np.mean(np.array(average_all_clean_acc_list))

    std_average_all_clean_acc = np.std(np.array(average_all_clean_acc_list))



    header = ['dataset', 'model', "mean_average_all_clean_acc",
              "std_average_all_clean_acc"]
    paths = "./checkpoints/Graph/"

    metric_list.append(args.dataset)
    metric_list.append(model_name)
    metric_list.append(mean_average_all_clean_acc)
    metric_list.append(std_average_all_clean_acc)


    paths = paths + "data-{}/".format(args.dataset) + "model-{}/".format(model_name) + file_name
    log_test_results(paths, header, file_name)
    log_test_results(paths, metric_list, file_name)

if __name__ == '__main__':
    args = args_parser()
    main(args)
