import subprocess as sub
import os


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'run_graph_exps.py')


# python run_graph_exps.py --dataset NCI1 \
#                          --config ./Graph_level_Models/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json \
#                          --is_iid iid\
#                          --num_workers 5\
#                          --num_mali 1\
#                          --epoch_backdoor 0\
#                          --frac_of_avg 0.1\
#                          --trigger_type renyi\
#                          --trigger_position random\
#                          --poisoning_intensity 0.1\
#                          --filename ./checkpoints/Graph \
#                          --device_id 0
#                          --epochs 10


#Model_list = ["./Graph_level_Models/configs/TUS/TUs_graph_classification_GCN_TRIANGLES_100k.json"]#,
Model_list = [ "./Graph_level_Models/configs/TUS/TUs_graph_classification_GAT_TRIANGLES_100k.json"]#,
              # "./Graph_level_Models/configs/TUS/TUs_graph_classification_GraphSage_TRIANGLES_100k.json"]
dataset = "TRIANGLES"
frac_of_avg_list = [0.1,0.2,0.3,0.4,0.5]
IID_list = ["p-degree-non-iid", "num-non-iid"]
trigger_type_list = [ "ws", "ba", "rr", "gta"]
poisoning_intensity_list = [0.2,0.3,0.4,0.5]
trigger_position_list = ["degree", "cluster"]
epoch_backdoor_list = [0.3,0.4,0.5]
num_mali_list = [2,3,4,5]
iid = "iid"
num_workers = 5
num_mali = 1
epoch_backdoor = 0
trigger_type = "random"
poisoning_intensity = 0.1
checkpoints_filename = "./checkpoints/Graph"
device_id = 1



Current_exp_name = "poisoning_intensity"

Experiment_list = ["epoch_backdoor","frac_of_avg","trigger_type","trigger_position","poisoning_intensity"]
for kk in range(len(Experiment_list)):
    Current_exp_name = Experiment_list[kk]
    if Current_exp_name == "iid":
        print("Starting Running Graph Backdoor Attack on Federated Experiments for --iid")
        for i in range(len(Model_list)):
            model = Model_list[i]
            for j in range(len(IID_list)):
                iid = IID_list[j]
                print( '--dataset', f'{dataset}', '--config', f'{model}',"--is_iid",f'{iid}', "--device_id",f'{device_id}')
                sub.call(["python", filename, '--dataset', f'{dataset}', '--config', f'{model}',"--is_iid",f'{iid}', "--device_id",f'{device_id}'])
    elif Current_exp_name == "num_mali":
        print("Starting Running Graph Backdoor Attack on Federated Experiments for --num_mali")
        for i in range(len(Model_list)):
            model = Model_list[i]
            for j in range(len(num_mali_list)):
                num_mali = num_mali_list[j]
                print( '--dataset', f'{dataset}', '--config', f'{model}',"--num_mali",f'{num_mali}', "--device_id",f'{device_id}')
                sub.call(["python", filename, '--dataset', f'{dataset}', '--config', f'{model}',"--num_mali",f'{num_mali}', "--device_id",f'{device_id}'])
    elif Current_exp_name == "epoch_backdoor":
        print("Starting Running Graph Backdoor Attack on Federated Experiments for --epoch_backdoor")
        for i in range(len(Model_list)):
            model = Model_list[i]
            for j in range(len(epoch_backdoor_list)):
                epoch_backdoor = epoch_backdoor_list[j]
                print( '--dataset', f'{dataset}', '--config', f'{model}',"--epoch_backdoor",f'{epoch_backdoor}', "--device_id",f'{device_id}')
                sub.call(["python", filename, '--dataset', f'{dataset}', '--config', f'{model}',"--epoch_backdoor",f'{epoch_backdoor}', "--device_id",f'{device_id}'])

    elif Current_exp_name == "frac_of_avg":
        # Experiments 1
        print("Starting Running Graph Backdoor Attack on Federated Experiments for --frac_of_avg")
        for i in range(len(Model_list)):
            model = Model_list[i]
            for j in range(len(frac_of_avg_list)):
                frac_of_avg = frac_of_avg_list[j]
                print('--dataset', f'{dataset}', '--config', f'{model}', "--frac_of_avg",f'{frac_of_avg}', "--device_id",f'{device_id}')
                sub.call(["python", filename, '--dataset', f'{dataset}', '--config', f'{model}',"--is_iid",f'{iid}', "--frac_of_avg",f'{frac_of_avg}', "--device_id",f'{device_id}'])
    elif Current_exp_name == "trigger_type":
        # Experiments 2
        print("Starting Running Graph Backdoor Attack on Federated Experiments for --trigger_type")
        for i in range(len(Model_list)):
            model = Model_list[i]
            for j in range(len(trigger_type_list)):
                trigger_type = trigger_type_list[j]
                print('--dataset', f'{dataset}', '--config', f'{model}',"--is_iid",f'{iid}', "--trigger_type",f'{trigger_type}', "--device_id",f'{device_id}')
                sub.call(["python", filename, '--dataset', f'{dataset}', '--config', f'{model}',"--is_iid",f'{iid}', "--trigger_type",f'{trigger_type}', "--device_id",f'{device_id}'])
    elif Current_exp_name == "trigger_position":
        # Experiments 2
        print("Starting Running Graph Backdoor Attack on Federated Experiments for --trigger_position")
        for i in range(len(Model_list)):
            model = Model_list[i]
            for j in range(len(trigger_position_list)):
                trigger_position= trigger_position_list[j]
                print('--dataset', f'{dataset}', '--config', f'{model}',"--is_iid",f'{iid}', "--trigger_position",f'{trigger_position}', "--device_id",f'{device_id}')
                sub.call(["python", filename, '--dataset', f'{dataset}', '--config', f'{model}',"--is_iid",f'{iid}', "--trigger_position",f'{trigger_position}', "--device_id",f'{device_id}'])

    elif Current_exp_name == "poisoning_intensity":
        # Experiments 2
        print("Starting Running Graph Backdoor Attack on Federated Experiments for --poisoning_intensity")
        for i in range(len(Model_list)):
            model = Model_list[i]
            for j in range(len(poisoning_intensity_list)):
                poisoning_intensity = poisoning_intensity_list[j]
                print('--dataset', f'{dataset}', '--config', f'{model}', "--poisoning_intensity",f'{poisoning_intensity}', "--device_id",f'{device_id}')
                sub.call(["python", filename, '--dataset', f'{dataset}', '--config', f'{model}',"--is_iid",f'{iid}', "--poisoning_intensity",f'{poisoning_intensity}', "--device_id",f'{device_id}'])

    else:
        raise NameError
