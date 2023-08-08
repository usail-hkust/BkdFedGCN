import subprocess as sub
import os


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'run_node_exps.py')


# python run_node_exps.py  --model GCN\
#                          --dataset Citeseer\
#                          --is_iid iid\
#                          --num_workers 5\
#                          --num_mali 1\
#                          --epoch_backdoor 0.0\
#                          --trigger_size 3\
#                          --trigger_type renyi\
#                          --trigger_position degree\
#                          --poisoning_intensity 0.1\
#                          --overlapping_rate 0.0\
#                          --device_id 1 \
#                          --epochs 20


#Model_list = ["GCN"]
Model_list = ["GAT"]#,
# Model_list = ["GraphSage"]
dataset = "Yelp"
trigger_size_list = [4,5,6,7,8,9,10]
IID_list = ["iid", "non-iid-louvain"]
trigger_type_list = ["renyi","ws", "ba", "gta","ugba"]
poisoning_intensity_list = [0.2,0.3,0.4,0.5]
trigger_position_list = ["degree", "cluster"]
epoch_backdoor_list = [0.1,0.2,0.3,0.4,0.5]
overlapping_rate_list = [0.1,0.2,0.3,0.4,0.5]
num_mali_list = [2,3,4,5]
iid = "iid"
num_workers = 5
num_mali = 1
epoch_backdoor = 0
trigger_type = "random"
poisoning_intensity = 0.1
checkpoints_filename = "./checkpoints/Graph"
device_id = 0
Coauthor_list = ["Cs", "Physics"]
Amazon_list = ["computers", "photo"]
if dataset  in Amazon_list:
    epochs = 2000
if dataset  in Coauthor_list:
    epochs = 1000
epochs = 2000
Current_exp_name = "poisoning_intensity"

Experiment_list = ["iid","num_mali","epoch_backdoor","overlapping_rate","trigger_size","trigger_type","trigger_position","poisoning_intensity"]
for kk in range(len(Experiment_list)):
    Current_exp_name = Experiment_list[kk]
    if Current_exp_name == "iid":
        print("Starting Running Graph Backdoor Attack on Federated Experiments for --iid")
        for i in range(len(Model_list)):
            model = Model_list[i]
            for j in range(len(IID_list)):
                iid = IID_list[j]
                print("--model", f'{model}','--dataset', f'{dataset}',"--is_iid",f'{iid}', "--device_id",f'{device_id}',"--epochs",f'{epochs}')
                sub.call(["python", filename,"--model", f'{model}','--dataset', f'{dataset}',"--is_iid",f'{iid}', "--device_id",f'{device_id}',"--epochs",f'{epochs}'])
    elif Current_exp_name == "num_mali":
        print("Starting Running Graph Backdoor Attack on Federated Experiments for --num_mali")
        for i in range(len(Model_list)):
            model = Model_list[i]
            for j in range(len(num_mali_list)):
                num_mali = num_mali_list[j]
                print("--model", f'{model}', '--dataset', f'{dataset}',"--num_mali",f'{num_mali}', "--device_id",f'{device_id}',"--epochs",f'{epochs}')
                sub.call(["python", filename,"--model", f'{model}', '--dataset', f'{dataset}',"--num_mali",f'{num_mali}', "--device_id",f'{device_id}',"--epochs",f'{epochs}'])
    elif Current_exp_name == "epoch_backdoor":
        print("Starting Running Graph Backdoor Attack on Federated Experiments for --epoch_backdoor")
        for i in range(len(Model_list)):
            model = Model_list[i]
            for j in range(len(epoch_backdoor_list)):
                epoch_backdoor = epoch_backdoor_list[j]
                print("--model", f'{model}', '--dataset', f'{dataset}',"--epoch_backdoor",f'{epoch_backdoor}', "--device_id",f'{device_id}',"--epochs",f'{epochs}')
                sub.call(["python", filename,"--model", f'{model}', '--dataset', f'{dataset}',"--epoch_backdoor",f'{epoch_backdoor}', "--device_id",f'{device_id}',"--epochs",f'{epochs}'])
    elif Current_exp_name == "overlapping_rate":
        print("Starting Running Graph Backdoor Attack on Federated Experiments for --overlapping_rate")
        for i in range(len(Model_list)):
            model = Model_list[i]
            for j in range(len(overlapping_rate_list)):
                overlapping_rate = overlapping_rate_list[j]
                print("--model", f'{model}', '--dataset', f'{dataset}',"--overlapping_rate",f'{overlapping_rate}', "--device_id",f'{device_id}',"--epochs",f'{epochs}')
                sub.call(["python", filename,"--model", f'{model}', '--dataset', f'{dataset}',"--overlapping_rate",f'{overlapping_rate}', "--device_id",f'{device_id}',"--epochs",f'{epochs}'])

    elif Current_exp_name == "trigger_size":
        print("Starting Running Graph Backdoor Attack on Federated Experiments for --trigger_size")
        for i in range(len(Model_list)):
            model = Model_list[i]
            for j in range(len(trigger_size_list)):
                trigger_size = trigger_size_list[j]
                print("--model", f'{model}', '--dataset', f'{dataset}',"--is_iid",f'{iid}', "--trigger_size",f'{trigger_size}', "--device_id",f'{device_id}',"--epochs",f'{epochs}')
                sub.call(["python", filename,"--model", f'{model}', '--dataset', f'{dataset}',"--is_iid",f'{iid}', "--trigger_size",f'{trigger_size}', "--device_id",f'{device_id}',"--epochs",f'{epochs}'])
    elif Current_exp_name == "trigger_type":
        print("Starting Running Graph Backdoor Attack on Federated Experiments for --trigger_type")
        for i in range(len(Model_list)):
            model = Model_list[i]
            for j in range(len(trigger_type_list)):
                trigger_type = trigger_type_list[j]
                print("--model", f'{model}', '--dataset', f'{dataset}',"--is_iid",f'{iid}', "--trigger_type",f'{trigger_type}', "--device_id",f'{device_id}',"--epochs",f'{epochs}')
                sub.call(["python", filename,"--model", f'{model}', '--dataset', f'{dataset}',"--is_iid",f'{iid}', "--trigger_type",f'{trigger_type}', "--device_id",f'{device_id}',"--epochs",f'{epochs}'])
    elif Current_exp_name == "trigger_position":
        print("Starting Running Graph Backdoor Attack on Federated Experiments for --trigger_position")
        for i in range(len(Model_list)):
            model = Model_list[i]
            for j in range(len(trigger_position_list)):
                trigger_position= trigger_position_list[j]
                print("--model", f'{model}', '--dataset', f'{dataset}',"--is_iid",f'{iid}', "--trigger_position",f'{trigger_position}', "--device_id",f'{device_id}',"--epochs",f'{epochs}')
                sub.call(["python", filename,"--model", f'{model}', '--dataset', f'{dataset}',"--is_iid",f'{iid}', "--trigger_position",f'{trigger_position}', "--device_id",f'{device_id}',"--epochs",f'{epochs}'])

    elif Current_exp_name == "poisoning_intensity":
        print("Starting Running Graph Backdoor Attack on Federated Experiments for --poisoning_intensity")
        for i in range(len(Model_list)):
            model = Model_list[i]
            for j in range(len(poisoning_intensity_list)):
                poisoning_intensity = poisoning_intensity_list[j]
                print("--model", f'{model}', '--dataset', f'{dataset}',"--is_iid",f'{iid}', "--poisoning_intensity",f'{poisoning_intensity}', "--device_id",f'{device_id}',"--epochs",f'{epochs}')
                sub.call(["python", filename,"--model", f'{model}', '--dataset', f'{dataset}',"--is_iid",f'{iid}', "--poisoning_intensity",f'{poisoning_intensity}', "--device_id",f'{device_id}',"--epochs",f'{epochs}'])

    else:
        raise NameError
