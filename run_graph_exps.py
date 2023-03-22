from Graph_level_Models.helpers.config import args_parser
from backdoor_graph_clf import  main as backdoor_main
import  numpy as np

args = args_parser()
rs = np.random.RandomState(args.seed)
seeds = rs.randint(1000,size=5)

def main():
    local_model_local_trigger_lists, local_model_global_trigger_lists, global_model_local_trigger_lists, global_model_global_trigger_lists = [],[],[],[]
    for i in range(len(seeds)):
        args.seed = seeds[i]
        local_model_metric, global_model_metric, args = backdoor_main(args)
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


    paths = '.data'


if __name__ == '__main__':
    main()