# Benchmarking the Impact of Multi-Component Backdoor Attacks on Federated Graph Learning for Node and Graph Classification Tasks

This study presents a benchmark analysis of the impact of multi-component backdoor attacks on federated graph learning for both node and graph classification tasks. The aim is to explore the effects of these attacks on various components of the learning process and provide insights into their potential impact on model performance.

[Backdoor Attack on Federated Graph Learning ]() (xx 2023)



## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:

> Node Level (TODO)
- OS: Ubuntu 18.04.4


> Graph Level (TODO)
- torch>=1.9.0
- torchvision>=0.10.0
- numpy>=1.23.2
- dgl>=0.9.1
- networkx>=2.4
- hdbscan==0.8.28
- joblib==1.1.0

## Acknowledgement
The codes are modifed based on [Xu et al. 2020](https://github.com/xujing1994/bkd_fedgnn) and [Dai et al. 2020](https://github.com/ventr1c/UGBA)
Note that  the authors demonstrate how the node level backdoor attack can be adapted to the federated graph learning settings.

## Threat Model
We consider the most widely studied setting:
- **L-inf norm constraint with the maximal epsilon be 8/255 on CIFAR-10**.
- **No accessibility to additional data, neither labeled nor unlabeled**.
- **Utilize the PGD-AT framework in [Madry et al. 2018](https://arxiv.org/abs/1706.06083)**.
- **TO DO**.

## Dataset
We consider the most widely studied datasets:
- **Node level:**
- **Graph level:** `NCI1`, `PROTEINS_full`, `TRIANGLES`


## GCN Model
We consider the most widely studied GCN models:
- **GCN**.
- **GAT**.
- **GraphSAGE**.



## Backdoor attack on  Graph Classification in Federated Graph Learning 
###  Train a clean Federated GNN model
```
python TO DO
```

###  Backdoor attack  in Federated GNNs
```
python TO DO
```

###  Uncovering the Use of Multiple Components in Backdoor Attacks on Federated Graph Neural Networks: Insights from Graph Classification Experiments



|        | Component            | Paramater                                                                             | Control                 | Default Value | Choice                           |
|--------|----------------------|---------------------------------------------------------------------------------------|-------------------------|---------------|----------------------------------|
| Server |  IID & Non-IID       | Independent and identically distributed & Non Independent and identically distributed | `--is_iid`              | `"iid"`       | `"iid"`, `"non-iid"`             |
|        | Number of Workers    | The number of normal worker                                                           | `--num_workers`         | `5`           | `5`                              |
|        | Number of Malicious  | The number of malicious attacker                                                      | `--num_mali`            | `1`           | `1`,`2`,`3`,`4`,`5`              |
|        | Start Backdoor Time  | The time at which a backdoor is first conducted by an attacker.                       | `--epoch_backdoor`      | `0`           | TODO                             |
| Client | Trigger Size         | The size of a trigger (the number of trigger's nodes)                                 | `--frac_of_avg`         | `0.1`         | `0.1`,`0.2`,`0.3`,`0.4`,`0.4`    |
|        | Trigger Type         | The specific type of trigger type                                                     | `--trigger_type`        | `"reny"`      | `"reny"`,`"ws"`, `"ba"`, `"rr"`  |
|        | Trigger Position     | Locations in a graph (subgraph) where a trigger  is inserted                          | `--trigger_position`    | `"random"`    | `"random"`                       |
|        | Poisoning Rate       | Percentage of training data that has been  poisoned                                   | `--poisoning_intensity` | `0.1`         | `0.1`, `0.2`, `0.3`, `0.4`,`0.5` |


- **Model**: `GCN`, `GAT`, `GraphSAGE`
- **Dataset**: `NCI1`, `PROTEINS_full`, `TRIANGLES`
- **Optimizer**: Adam with default hyperparameters
- **Total epoch**: `1000`
- **Batch size**: `128`
- **Learning rate**: `7e-4`


running command for training:
```python
python run_graph_exps.py --dataset NCI1 \
                         --model GCN \
                         --config ./Graph_level_Models/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json \
                         --is_iid iid\
                         --num_workers 5\
                         --num_mali 1\
                         --epoch_backdoor 0\
                         --frac_of_avg 0.1\
                         --trigger_type reny\
                         --trigger_position random\
                         --poisoning_intensity 0.1\
                         --filename ./Checkpoints/Graph \
                         --device_id 0
```

> Each experiment was repeated 5 times with a different seed each time

> Refer:
> Each experiment was repeated 10 times with a different seed each time


### Component Candidates (need to change according to the experiments)
Importance rate: *Critical*; *Useful*; *Insignificance*

- **Early stopping w.r.t. training epochs** (*Critical*).
Early stopping w.r.t. training epochs was first introduced in the [code of TRADES](https://github.com/yaodongyu/TRADES), and was later thoroughly studied by [Rice et al., 2020](https://arxiv.org/abs/2002.11569). Due to its effectiveness, we regard this trick as a default choice.

- **Early stopping w.r.t. attack intensity** (*Useful*). Early stopping w.r.t. attack iterations was studied by [Wang et al. 2019](proceedings.mlr.press/v97/wang19i/wang19i.pdf) and [Zhang et al. 2020](https://arxiv.org/abs/2002.11242). Here we exploit the strategy of the later one, where the authors show that this trick can promote clean accuracy. The relevant flags include `--earlystopPGD` indicates whether apply this trick, while '--earlystopPGDepoch1' and '--earlystopPGDepoch2' separately indicate the epoch to increase the tolerence t by one, as detailed in [Zhang et al. 2020](https://arxiv.org/abs/2002.11242). (*Note that early stopping attack intensity may degrade worst-case robustness under strong attacks*)

- **Warmup w.r.t. learning rate** (*Insignificance*). Warmup w.r.t. learning rate was found useful for [FastAT](https://arxiv.org/abs/2001.03994), while [Rice et al., 2020](https://arxiv.org/abs/2002.11569) found that piecewise decay schedule is more compatible with early stop w.r.t. training epochs. The relevant flags include `--warmup_lr` indicates whether apply this trick, while `--warmup_lr_epoch` indicates the end epoch of the gradually increase of learning rate.

- **Warmup w.r.t. epsilon** (*Insignificance*). [Qin et al. 2019](https://arxiv.org/abs/1907.02610) use warmup w.r.t. epsilon in their implementation, where the epsilon gradually increase from 0 to 8/255 in the first 15 epochs. Similarly, the relevant flags include `--warmup_eps` indicates whether apply this trick, while `--warmup_eps_epoch` indicates the end epoch of the gradually increase of epsilon.

- **Batch size** (*Insignificance*). The typical batch size used for CIFAR-10 is 128 in the adversarial setting. In the meanwhile, [Xie et al. 2019](https://arxiv.org/pdf/1812.03411.pdf) apply a large batch size of 4096 to perform adversarial training on ImageNet, where the model is distributed on 128 GPUs and has quite robust performance. The relevant flag is `--batch-size`. According to [Goyal et al. 2017](https://arxiv.org/abs/1706.02677), we take bs=128 and lr=0.1 as a basis, and scale the lr when we use larger batch size, e.g., bs=256 and lr=0.2.

- **Label smoothing** (*Useful*). Label smoothing is advocated by [Shafahi et al. 2019](https://arxiv.org/abs/1910.11585) to mimic the adversarial training procedure. The relevant flags include `--labelsmooth` indicates whether apply this trick, while `--labelsmoothvalue` indicates the degree of smoothing applied on the label vectors. When `--labelsmoothvalue=0`, there is no label smoothing applied. (*Note that only moderate label smoothing (~0.2) is helpful, while exccessive label smoothing (>0.3) could be harmful, as observed in [Jiang et al. 2020](https://arxiv.org/abs/2006.13726)*)

- **Optimizer** (*Insignificance*). Most of the AT methods apply SGD with momentum as the optimizer. In other cases, [Carmon et al. 2019](https://arxiv.org/abs/1905.13736) apply SGD with Nesterov, and [Rice et al., 2020](https://arxiv.org/abs/2002.11569) apply Adam for cyclic learning rate schedule. The relevant flag is `--optimizer`, which include common optimizers implemented by official Pytorch API and recently proposed gradient centralization trick by [Yong et al. 2020](https://arxiv.org/abs/2004.01461).

- **Weight decay** (*Critical*). The values of weight decay used in previous AT methods mainly fall into `1e-4` (e.g., [Wang et al. 2019](proceedings.mlr.press/v97/wang19i/wang19i.pdf)), `2e-4` (e.g., [Madry et al. 2018](https://arxiv.org/abs/1706.06083)), and `5e-4` (e.g., [Rice et al., 2020](https://arxiv.org/abs/2002.11569)). We find that slightly different values of weight decay could largely affect the robustness of the adversarially trained models.

- **Activation function** (*Useful*). As shown in [Xie et al., 2020a](https://arxiv.org/pdf/2006.14536.pdf), the smooth alternatives of `ReLU`, including `Softplus` and `GELU` can promote the performance of adversarial training. The relevant flags are `--activation` to choose the activation, and `--softplus_beta` to set the beta for Softplus. Other hyperparameters are used by default in the code.

- **BN mode** (*Useful*). TRADES applies eval mode of BN when crafting adversarial examples during training, while PGD-AT methods implemented by [Madry et al. 2018](https://arxiv.org/abs/1706.06083) or [Rice et al., 2020](https://arxiv.org/abs/2002.11569) use train mode of BN to craft training adversarial examples. As indicated by [Xie et al., 2020b](https://arxiv.org/pdf/1906.03787.pdf), properly dealing with BN layers is critical to obtain a well-performed adversarially trained model, while train mode of BN during multi-step PGD process may blur the distribution. 


### NCI1 (need to change according to the experiments)
|paper           | Architecture | clean         | AA |
|---|:---:|:---:|:---:|
| **OURS (TRADES)**[[Checkpoint](http://ml.cs.tsinghua.edu.cn/~xiaoyang/downloads/bag_of_tricks/wide20_trades_eps8_tricks.pt)] | WRN-34-20| 86.43 | 54.39 |
| **OURS (TRADES)**[[Checkpoint](http://ml.cs.tsinghua.edu.cn/~xiaoyang/downloads/bag_of_tricks/wide10_trades_eps8_tricks.pt)] | WRN-34-10| 85.48 | 53.80 |
| [(Pang et al., 2020)](https://arxiv.org/abs/2002.08619) | WRN-34-20| 85.14 | 53.74 |
| [(Zhang et al., 2020)](https://arxiv.org/abs/2002.11242)| WRN-34-10| 84.52 | 53.51 |
| [(Rice et al., 2020)](https://arxiv.org/abs/2002.11569) | WRN-34-20| 85.34 | 53.35 |


## Backdoor attack on  Node Classification in Federated Graph Learning 
###  Train a clean Federated GNN model
```
python TO DO
```

###  Backdoor attack  in Federated GNNs
```
python TO DO
```

###  Multi-components in Backdoor attack  in Federated GNNs: Graph Classification experiments



|        | Component            | Paramater                                                                             | Control                 | Default Value | Choice                           |
|--------|----------------------|---------------------------------------------------------------------------------------|-------------------------|---------------|----------------------------------|
| Server |  IID & Non-IID       | Independent and identically distributed & Non Independent and identically distributed | `--is_iid`              | `"iid"`       | `"iid"`, `"non-iid"`             |
|        | Number of Workers    | The number of normal worker                                                           | `--num_workers`         | `5`           | `5`                              |
|        | Number of Malicious  | The number of malicious attacker                                                      | `--num_mali`            | `1`           | `1`,`2`,`3`,`4`,`5`              |
|        | Start Backdoor Time  | The time at which a backdoor is first conducted by an attacker.                       | `--epoch_backdoor`      | `0`           | TODO                             |
| Client | Trigger Size         | The size of a trigger (the number of trigger's nodes)                                 | `--frac_of_avg`         | `0.1`         | `0.1`,`0.2`,`0.3`,`0.4`,`0.4`    |
|        | Trigger Type         | The specific type of trigger type                                                     | `--trigger_type`        | `"reny"`      | `"reny"`,`"ws"`, `"ba"`, `"rr"`  |
|        | Trigger Position     | Locations in a graph (subgraph) where a trigger  is inserted                          | `--trigger_position`    | `"random"`    | `"random"`                       |
|        | Poisoning Rate       | Percentage of training data that has been  poisoned                                   | `--poisoning_intensity` | `0.1`         | `0.1`, `0.2`, `0.3`, `0.4`,`0.5` |


- **Model**: `GCN`, `GAT`, `GraphSAGE`
- **Dataset**: `NCI1`, `PROTEINS_full`, `TRIANGLES`
- **Optimizer**: Adam with default hyperparameters
- **Total epoch**: `1000`
- **Batch size**: `128`
- **Learning rate**: `7e-4`


running command for training:
```python
python run_graph_exps.py --dataset NCI1 \
                         --model GCN \
                         --config ./Graph_level_Models/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json \
                         --is_iid iid\
                         --num_workers 5\
                         --num_mali 1\
                         --epoch_backdoor 0\
                         --frac_of_avg 0.1\
                         --trigger_type reny\
                         --trigger_position random\
                         --poisoning_intensity 0.1\
                         --filename ./Checkpoints/Graph \
                         --device_id 0
```

> Each experiment was repeated 5 times with a different seed each time

> Refer:
> Each experiment was repeated 10 times with a different seed each time


### Component Candidates (need to change according to the experiments)
Importance rate: *Critical*; *Useful*; *Insignificance*

- **Early stopping w.r.t. training epochs** (*Critical*).
Early stopping w.r.t. training epochs was first introduced in the [code of TRADES](https://github.com/yaodongyu/TRADES), and was later thoroughly studied by [Rice et al., 2020](https://arxiv.org/abs/2002.11569). Due to its effectiveness, we regard this trick as a default choice.

- **Early stopping w.r.t. attack intensity** (*Useful*). Early stopping w.r.t. attack iterations was studied by [Wang et al. 2019](proceedings.mlr.press/v97/wang19i/wang19i.pdf) and [Zhang et al. 2020](https://arxiv.org/abs/2002.11242). Here we exploit the strategy of the later one, where the authors show that this trick can promote clean accuracy. The relevant flags include `--earlystopPGD` indicates whether apply this trick, while '--earlystopPGDepoch1' and '--earlystopPGDepoch2' separately indicate the epoch to increase the tolerence t by one, as detailed in [Zhang et al. 2020](https://arxiv.org/abs/2002.11242). (*Note that early stopping attack intensity may degrade worst-case robustness under strong attacks*)

- **Warmup w.r.t. learning rate** (*Insignificance*). Warmup w.r.t. learning rate was found useful for [FastAT](https://arxiv.org/abs/2001.03994), while [Rice et al., 2020](https://arxiv.org/abs/2002.11569) found that piecewise decay schedule is more compatible with early stop w.r.t. training epochs. The relevant flags include `--warmup_lr` indicates whether apply this trick, while `--warmup_lr_epoch` indicates the end epoch of the gradually increase of learning rate.

- **Warmup w.r.t. epsilon** (*Insignificance*). [Qin et al. 2019](https://arxiv.org/abs/1907.02610) use warmup w.r.t. epsilon in their implementation, where the epsilon gradually increase from 0 to 8/255 in the first 15 epochs. Similarly, the relevant flags include `--warmup_eps` indicates whether apply this trick, while `--warmup_eps_epoch` indicates the end epoch of the gradually increase of epsilon.

- **Batch size** (*Insignificance*). The typical batch size used for CIFAR-10 is 128 in the adversarial setting. In the meanwhile, [Xie et al. 2019](https://arxiv.org/pdf/1812.03411.pdf) apply a large batch size of 4096 to perform adversarial training on ImageNet, where the model is distributed on 128 GPUs and has quite robust performance. The relevant flag is `--batch-size`. According to [Goyal et al. 2017](https://arxiv.org/abs/1706.02677), we take bs=128 and lr=0.1 as a basis, and scale the lr when we use larger batch size, e.g., bs=256 and lr=0.2.

- **Label smoothing** (*Useful*). Label smoothing is advocated by [Shafahi et al. 2019](https://arxiv.org/abs/1910.11585) to mimic the adversarial training procedure. The relevant flags include `--labelsmooth` indicates whether apply this trick, while `--labelsmoothvalue` indicates the degree of smoothing applied on the label vectors. When `--labelsmoothvalue=0`, there is no label smoothing applied. (*Note that only moderate label smoothing (~0.2) is helpful, while exccessive label smoothing (>0.3) could be harmful, as observed in [Jiang et al. 2020](https://arxiv.org/abs/2006.13726)*)

- **Optimizer** (*Insignificance*). Most of the AT methods apply SGD with momentum as the optimizer. In other cases, [Carmon et al. 2019](https://arxiv.org/abs/1905.13736) apply SGD with Nesterov, and [Rice et al., 2020](https://arxiv.org/abs/2002.11569) apply Adam for cyclic learning rate schedule. The relevant flag is `--optimizer`, which include common optimizers implemented by official Pytorch API and recently proposed gradient centralization trick by [Yong et al. 2020](https://arxiv.org/abs/2004.01461).

- **Weight decay** (*Critical*). The values of weight decay used in previous AT methods mainly fall into `1e-4` (e.g., [Wang et al. 2019](proceedings.mlr.press/v97/wang19i/wang19i.pdf)), `2e-4` (e.g., [Madry et al. 2018](https://arxiv.org/abs/1706.06083)), and `5e-4` (e.g., [Rice et al., 2020](https://arxiv.org/abs/2002.11569)). We find that slightly different values of weight decay could largely affect the robustness of the adversarially trained models.

- **Activation function** (*Useful*). As shown in [Xie et al., 2020a](https://arxiv.org/pdf/2006.14536.pdf), the smooth alternatives of `ReLU`, including `Softplus` and `GELU` can promote the performance of adversarial training. The relevant flags are `--activation` to choose the activation, and `--softplus_beta` to set the beta for Softplus. Other hyperparameters are used by default in the code.

- **BN mode** (*Useful*). TRADES applies eval mode of BN when crafting adversarial examples during training, while PGD-AT methods implemented by [Madry et al. 2018](https://arxiv.org/abs/1706.06083) or [Rice et al., 2020](https://arxiv.org/abs/2002.11569) use train mode of BN to craft training adversarial examples. As indicated by [Xie et al., 2020b](https://arxiv.org/pdf/1906.03787.pdf), properly dealing with BN layers is critical to obtain a well-performed adversarially trained model, while train mode of BN during multi-step PGD process may blur the distribution. 


### NCI1 (need to change according to the experiments)
|paper           | Architecture | clean         | AA |
|---|:---:|:---:|:---:|
| **OURS (TRADES)**[[Checkpoint](http://ml.cs.tsinghua.edu.cn/~xiaoyang/downloads/bag_of_tricks/wide20_trades_eps8_tricks.pt)] | WRN-34-20| 86.43 | 54.39 |
| **OURS (TRADES)**[[Checkpoint](http://ml.cs.tsinghua.edu.cn/~xiaoyang/downloads/bag_of_tricks/wide10_trades_eps8_tricks.pt)] | WRN-34-10| 85.48 | 53.80 |
| [(Pang et al., 2020)](https://arxiv.org/abs/2002.08619) | WRN-34-20| 85.14 | 53.74 |
| [(Zhang et al., 2020)](https://arxiv.org/abs/2002.11242)| WRN-34-10| 84.52 | 53.51 |
| [(Rice et al., 2020)](https://arxiv.org/abs/2002.11569) | WRN-34-20| 85.34 | 53.35 |














## References
If you find the code useful for your research, please consider citing
```bib
@inproceedings{fan2022ASTFA,
 author =  {Fan LIU, Hao LIU, Wenzhao JIANG},
 title = {Practical Adversarial Attacks on Spatiotemporal
Traffic Forecasting Models},
 booktitle = {In Proceedings of the Thirty-sixth Annual Conference on Neural Information Processing Systems (NeurIPS)},
 year = {2022}
 }
```

and/or our related works


```bib
@inproceedings{fan2022ASTFA,
 author =  {Fan LIU, Hao LIU, Wenzhao JIANG},
 title = {Practical Adversarial Attacks on Spatiotemporal
Traffic Forecasting Models},
 booktitle = {In Proceedings of the Thirty-sixth Annual Conference on Neural Information Processing Systems (NeurIPS)},
 year = {2022}
 }
```
