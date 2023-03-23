from Graph_level_Models.helpers.config import args_parser
from backdoor_graph_clf import  main as backdoor_main
from helpers.metrics_utils import log_test_results
import  numpy as np
import json
import wandb

args = args_parser()
rs = np.random.RandomState(args.seed)
seeds = rs.randint(1000,size=5)
def log_test_csv(args, file_name, method):
    metric_list = []
    data_set = args.dataset
    metric_list.append(data_set)
    model_name = args.backbone
    metric_list.append(model_name)


    method_name = method
    metric_list.append(method_name)


    log_test_results(args.model_dir, metric_list, file_name)


def main(args):
    local_model_local_trigger_lists, local_model_global_trigger_lists, global_model_local_trigger_lists, global_model_global_trigger_lists = [],[],[],[]

    metric_list = []
    for i in range(len(seeds)):
        args.seed = seeds[i]

        # wandb init
        logger = wandb.init(
            entity="hkust-gz",
            project="test",
            group=args.exp_name,
            name=f"round_{i}",
            config=args,
        )

        local_model_metric, global_model_metric = backdoor_main(args, logger)

        # end the logger
        logger.finish()

        different_clients_test_accuracy_local_trigger = local_model_metric[0]
        different_clients_test_accuracy_global_trigger = local_model_metric[1]
        average_clients_local_trigger_accuracy, average_clients_global_trigger_accuracy = np.mean(np.array(different_clients_test_accuracy_local_trigger)), np.std(np.array(different_clients_test_accuracy_global_trigger))
        local_model_local_trigger_lists.append(average_clients_local_trigger_accuracy)
        local_model_global_trigger_lists.append(average_clients_global_trigger_accuracy)


        local_att_acc = global_model_metric[0]
        average_local_att_acc = np.mean(np.array(local_att_acc))
        global_att_acc = global_model_metric[1]
        global_model_local_trigger_lists.append(average_local_att_acc)
        global_model_global_trigger_lists.append(global_att_acc)

    mean_local_model_local_trigger, mean_local_model_global_trigger, mean_global_model_local_trigger, mean_global_model_global_trigger = np.mean(np.array(local_model_local_trigger_lists)),\
                                                                                                                                                                 np.mean(np.array(local_model_global_trigger_lists)),\
                                                                                                                                                                 np.mean(np.array(global_model_local_trigger_lists)),\
                                                                                                                                                                 np.mean(np.array(global_model_global_trigger_lists))
    std_local_model_local_trigger, std_local_model_global_trigger, std_global_model_local_trigger, std_global_model_global_trigger = np.std(np.array(local_model_local_trigger_lists)), \
                                                                                                                                                             np.std(np.array(local_model_global_trigger_lists)), \
                                                                                                                                                             np.std(np.array(global_model_local_trigger_lists)), \
                                                                                                                                                      np.std(np.array(global_model_global_trigger_lists))
    with open(args.config) as f:
        config = json.load(f)
    header = ['dataset', 'model', 'mean_local_model_local_trigger', 'std_local_model_local_trigger']
    paths = "./checkpoints/Graph/"
    model_name = config['model']
    file_name = 'data-{}_model-{}_IID-{}_num_workers-{}_num_mali-{}_epoch_backdoor-{}_frac_of_avg-{}_trigger_type-{}_trigger_position-{}_poisoning_intensity-{}'.format(args.dataset,
                                                                                                                                                                        model_name,
                                                                                                                                                                        args.is_iid,
                                                                                                                                                                        args.num_workers,
                                                                                                                                                                        args.num_mali,
                                                                                                                                                                        args.epoch_backdoor,
                                                                                                                                                                        args.frac_of_avg,
                                                                                                                                                                        args.trigger_type,
                                                                                                                                                                        args.trigger_position,
                                                                                                                                                                        args.poisoning_intensity)

    log_test_results(paths, header, file_name)
    metric_list.append(args.dataset)
    metric_list.append(model_name)
    metric_list.append(mean_local_model_local_trigger)
    metric_list.append(std_local_model_local_trigger)
    log_test_results(paths, metric_list, file_name)


if __name__ == '__main__':
    args = args_parser()
    main(args)