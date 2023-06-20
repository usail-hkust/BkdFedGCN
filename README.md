# Bkd-FedGNN: A Benchmark for Classification Backdoor Attacks on Federated Graph Neural Networks

This study presents a benchmark analysis of the impact of multi-component backdoor attacks on federated graph learning for both node and graph classification tasks. The aim is to explore the effects of these attacks on various components of the learning process and provide insights into their potential impact on model performance.

Graph neural networks (GNNs) enhance model generalizability and leverage large-scale graph datasets by incorporating the message-passing mechanism, but their practical application faces data privacy challenges that hinder data sharing. To overcome this, federated GNNs combine GNNs with federated learning (FL), enabling machine learning systems to be trained without direct access to sensitive data. However, federated learning's distributed nature introduces vulnerabilities, particularly backdoor attacks resulting from privacy issues. The exploration of graph backdoor attacks on federated GNNs has revealed vulnerabilities in these systems. However, due to the complex settings in federated learning, graph backdoor attacks have not been fully explored. This lack of exploration is attributed to insufficient benchmark coverage and inadequate analysis of critical factors with graph backdoor attacks on federated GNNs.
To address these limitations, we propose a benchmark, Bkd-FedGNN, for graph backdoor attacks on federated GNNs. In detail,    we provide a unified framework for classification  backdoor attacks on federated GNNs, encompassing both node-level and graph-level classification tasks. This framework decomposes the graph backdoor attack into trigger generation and trigger injection steps, extending the node-level backdoor attack to the federated GNNs setting. In addition, we thoroughly investigate the impact of multiple critical factors on graph backdoor attacks in federated GNNs. These factors are categorized into global-level and local-level factors, including data distribution, the number of malicious attackers, attack time, overlapping rate, trigger size, trigger type, trigger position, and poisoning rate.

![Framework](/figs/Architechture.png)

## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:

- torch>=1.9.0
- torchvision>=0.10.0
- numpy>=1.23.2
- dgl>=0.9.1
- networkx>=2.4
- hdbscan==0.8.28
- joblib==1.1.0



## Threat Model
We consider the most widely studied setting:
### Attack Objective: 
- Assuming there are a total of $K$ clients, with $M$ ($M \leq K$) of them being malicious, each malicious attacker independently conducts the backdoor attack on their own models.  The primary goal of a backdoor attack is to manipulate the model in such a way that it misclassifies specific pre-defined labels (known as target labels) only within the poisoned data samples. It is important to ensure that the model's accuracy remains unaffected when processing clean data. 

### Attack Knowledge: 
- In this setting, we assume that the malicious attacker has complete knowledge of their own training data. They have the capability to generate triggers. It should be noted that this scenario is quite practical since the clients have full control over their own data. 

### Attacker Capability: 

The malicious client has the ability to inject triggers into the training datasets, but this capability is limited within predetermined constraints such as trigger size and poisoned data rate. The intention is to contaminate the training datasets. However, the malicious client lacks the ability to manipulate the server-side aggregation process or interfere with other clients' training processes and models.



## Dataset
We consider the most widely studied datasets:
- **Node level:** `Cora`, `Citeseer`, `CS`,`Physics`, `Photo`, `Computers`
- **Graph level:** Molecules: `AIDS`,`NCI1` Bioinformatics: `PROTEINS_full`,`DD`, `ENZYMES`  Synthetic: `TRIANGLES`


## GCN Model
We consider the most widely studied GCN models:
- **GCN**.
- **GAT**.
- **GraphSAGE**.



## Graph Backdoor attacks on  Node Classification in Federated GNNs


###  Backdoor attack  in Federated GNNs
```
python run_node_exps.py  --model GCN\
                         --dataset Cora\
                         --is_iid iid\
                         --num_workers 5\
                         --num_mali 1\
                         --epoch_backdoor 0\
                         --trigger_size 3\
                         --trigger_type renyi\
                         --trigger_position random\
                         --poisoning_intensity 0.1\
                         --overlapping_rate 0.0
```

###  Multi Factors in Backdoor attack  in Federated GNNs: Graph Classification experiments


|        | Component            | Paramater                                                                             | Control                 | Default Value | Choice                           |
|--------|----------------------|---------------------------------------------------------------------------------------|-------------------------|---------------|----------------------------------|
| Server |  IID & Non-IID       | Independent and identically distributed & Non Independent and identically distributed | `--is_iid`              | `iid`       | `iid`, `non-iid-louvain`             |
|        | Number of Workers    | The number of normal worker                                                           | `--num_workers`         | `5`           | `5`                              |
|        | Number of Malicious  | The number of malicious attacker                                                      | `--num_mali`            | `1`           | `1`,`2`,`3`,`4`,`5`              |
|        | Start Backdoor Time  | The time at which a backdoor is first conducted by an attacker.                       | `--epoch_backdoor`      | `0`           | `0`,`0.1`,`0.2`,`0.3`,`0.4`，`0.5` |
|        | Cross Nodes  | the proportion of data that overlaps between adjacent subsets when splitting a dataset into client_num subsets for federated learningzsa                       | `--overlapping_rate`      | `0`           | `0`,`0.1`,`0.2`,`0.3`,`0.4`，`0.5` |
| Client | Trigger Size         | The size of a trigger (the number of trigger's nodes)                                 | `--trigger_size`         | `3`         | `3`,`4`,`5`,`6`,`7`,`8`,`9`,`10`    |
|        | Trigger Type         | The specific type of trigger type                                                     | `--trigger_type`        | `renyi`      | `renyi`,`ws`, `ba`,`gta`,`ugba`|
|        | Trigger Position     | Locations in a graph (subgraph) where a trigger  is inserted                          | `--trigger_position`    | `random`    | `random` ,`cluster`,`cluster_degree`            |
|        | Poisoning Rate       | Percentage of training data that has been  poisoned                                   | `--poisoning_intensity` | `0.1`         | `0.1`, `0.2`, `0.3`, `0.4`,`0.5` |


- **Model**: `GCN`, `GAT`, `GraphSAGE`
- **Dataset**: `Cora`, `Citeseer`, `CS`,`Physics`, `Photo`, `Computers`
- **Optimizer**: Adam with default hyperparameters
- **Total epoch**: `200`
- **Learning rate**: `0.01`

 
> Each experiment was repeated 5 times with a different seed each time




## Backdoor attack on  Graph Classification in Federated Graph Learning 




###  Multiple Factors in Backdoor Attacks on Federated Graph Neural Networks: Insights from Graph Classification Experiments


|        | Component            | Paramater                                                                             | Control                 | Default Value | Choice                           |
|--------|----------------------|---------------------------------------------------------------------------------------|-------------------------|---------------|----------------------------------|
| Server |  IID & Non-IID       | Independent and identically distributed & Non Independent and identically distributed | `--is_iid`              | `iid`       | `iid`, `p-degree-non-iid`, `num-non-iid`|
|        | Number of Workers    | The number of normal worker                                                           | `--num_workers`         | `5`           | `5`                              |
|        | Number of Malicious  | The number of malicious attacker                                                      | `--num_mali`            | `1`           | `1`,`2`,`3`,`4`,`5`              |
|        | Start Backdoor Time  | The time at which a backdoor is first conducted by an attacker.                       | `--epoch_backdoor`      | `0`           | int[(`0.1`,`0.2`,`0.3`,`0.4`,`0.5`) * 1000 ]|
| Client | Trigger Size         | The size of a trigger (the number of trigger's nodes)                                 | `--frac_of_avg`         | `0.1`         | `0.1`,`0.2`,`0.3`,`0.4`,`0.5`    |
|        | Trigger Type         | The specific type of trigger type                                                     | `--trigger_type`        | `renyi`       | `renyi`,`ws`, `ba`, `rr`, `gta`        |
|        | Trigger Position     | Locations in a graph (subgraph) where a trigger  is inserted                          | `--trigger_position`    | `random`      | `random`,`degree`,`cluster`       |
|        | Poisoning Rate       | Percentage of training data that has been  poisoned                                   | `--poisoning_intensity` | `0.1`         | `0.1`, `0.2`, `0.3`, `0.4`,`0.5` |

Other paramaters

- **Model**: `GCN`, `GAT`, `GraphSAGE`
- **Dataset**: Molecules: `AIDS`,`NCI1` Bioinformatics: `PROTEINS_full`,`DD`, `ENZYMES`  Synthetic:  `COLORS-3`
- **Optimizer**: Adam with default hyperparameters
- **Total epoch**: `1000`
- **Batch size**: `128`
- **Learning rate**: `7e-4`

####  Train a clean Federated GNN model
```
python run_clean_graph_exps.py --dataset NCI1 \
                         --config ./Graph_level_Models/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json \
                         --is_iid iid\
                         --num_workers 5\

```


####  Backdoor attack  in Federated GNNs
running command for training:
```python
python run_graph_exps.py --dataset NCI1 \
                         --config ./Graph_level_Models/configs/TUS/TUs_graph_classification_GCN_NCI1_100k.json \
                         --is_iid iid\
                         --num_workers 5\
                         --num_mali 1\
                         --epoch_backdoor 0\
                         --frac_of_avg 0.1\
                         --trigger_type renyi\
                         --trigger_position random\
                         --poisoning_intensity 0.1\
                         --filename ./checkpoints/Graph \
                         --device_id 0
```

> Each experiment was repeated 5 times with a different seed each time








## References
If you find the code useful for your research, please consider citing
```bib
 @inproceedings{fan2023BkdFedGNN,
  author    = {Fan LIU and
               Siqi LAI and
               Yansong NING and 
               Hao LIU},
  title     = {Bkd-FedGNN: A Benchmark for Classification Backdoor Attacks on Federated Graph Neural Network},
  booktitle = {Arxiv},
  pages     = {},
  doi       = {10.13140/RG.2.2.34558.15681},
  publisher = {Arxiv},
  year      = {2023},
  timestamp = {}
}
```

and/or our related works

```bib
 @inproceedings{fan2023RbDAT,
  author    = {Fan LIU and
               Weijia ZHANG and
               Hao LIU},
  title     = {Robust Spatiotemporal Traffic Forecasting with Reinforced  Dynamic Adversarial Training},
  booktitle = {Proceedings of the 29th {ACM} {SIGKDD} International Conference on
               Knowledge Discovery and Data Mining, {KDD} 2023, Long Beach, CA, USA, August 6–10, 2023},
  pages     = {},
  publisher = {{ACM}},
  year      = {2023},
  timestamp = {}
}
```


```bib
@inproceedings{fan2022ASTFA,
 author =  {Fan LIU, Hao LIU, Wenzhao JIANG},
 title = {Practical Adversarial Attacks on Spatiotemporal
Traffic Forecasting Models},
 booktitle = {In Proceedings of the Thirty-sixth Annual Conference on Neural Information Processing Systems (NeurIPS)},
 year = {2022}
 }
```






## Acknowledgement
The codes are modifed based on [Xu et al. 2020](https://github.com/xujing1994/bkd_fedgnn) and [Dai et al. 2020](https://github.com/ventr1c/UGBA)
To the best of our knowledge, our work is the first to extend the node-level backdoor attack to the federated GNNs setting

