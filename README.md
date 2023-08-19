# Bkd-FedGNN: A Benchmark for Classification Backdoor Attacks on Federated Graph Neural Networks [[pdf](https://arxiv.org/abs/2306.10351)]

This study presents a benchmark analysis of the impact of multi-component backdoor attacks on federated graph learning for both node and graph classification tasks. The aim is to explore the effects of these attacks on various components of the learning process and provide insights into their potential impact on model performance.

Graph neural networks (GNNs) enhance model generalizability and leverage large-scale graph datasets by incorporating the message-passing mechanism, but their practical application faces data privacy challenges that hinder data sharing. To overcome this, federated GNNs combine GNNs with federated learning (FL), enabling machine learning systems to be trained without direct access to sensitive data. However, federated learning's distributed nature introduces vulnerabilities, particularly backdoor attacks resulting from privacy issues. The exploration of graph backdoor attacks on federated GNNs has revealed vulnerabilities in these systems. However, due to the complex settings in federated learning, graph backdoor attacks have not been fully explored. This lack of exploration is attributed to insufficient benchmark coverage and inadequate analysis of critical factors with graph backdoor attacks on federated GNNs.
To address these limitations, we propose a benchmark, Bkd-FedGNN, for graph backdoor attacks on federated GNNs. In detail,    we provide a unified framework for classification  backdoor attacks on federated GNNs, encompassing both node-level and graph-level classification tasks. This framework decomposes the graph backdoor attack into trigger generation and trigger injection steps, extending the node-level backdoor attack to the federated GNNs setting. In addition, we thoroughly investigate the impact of multiple critical factors on graph backdoor attacks in federated GNNs. These factors are categorized into global-level and local-level factors, including data distribution, the number of malicious attackers, attack time, overlapping rate, trigger size, trigger type, trigger position, and poisoning rate.

![Framework](/figs/Architechture.png)




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
- **Graph level:** `AIDS`,`NCI1`,`PROTEINS_full`,`DD`, `ENZYMES` ,  `COLORS-3`


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


###  Backdoor attack  in Federated GNNs on other federated algorithms and defense methods

```
--agg_method FedOpt, FedProx,scaffold, fed_trimmedmean, fedMedian, fed_krum，
fed_multi_krum, fed_bulyan
```

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
                         --overlapping_rate 0.0 \
                         --agg_method FedOpt
```



###  Multi Factors in Backdoor attack  in Federated GNNs: Graph Classification experiments


|        | Component            | Paramater                                                                             | Control                 | Default Value | Choice                           |
|--------|----------------------|---------------------------------------------------------------------------------------|-------------------------|---------------|----------------------------------|
| Server |  IID & Non-IID       | Independent and identically distributed & Non Independent and identically distributed | `--is_iid`              | `iid`       | `iid`, `non-iid-louvain`, `non-iid-Metis`     |
|        | Number of Workers    | The number of normal worker                                                           | `--num_workers`         | `5`           | `5`                              |
|        | Number of Malicious  | The number of malicious attacker                                                      | `--num_mali`            | `1`           | `1`,`2`,`3`,`4`,`5`              |
|        | Attack Time  | The time at which a backdoor is first conducted by an attacker.                       | `--epoch_backdoor`      | `0`           | `0`,`0.1`,`0.2`,`0.3`,`0.4`，`0.5` |
|        | Overlapping Rate  | the proportion of data that overlaps between adjacent subsets when splitting a dataset into client_num subsets for federated learningzsa                       | `--overlapping_rate`      | `0`           | `0`,`0.1`,`0.2`,`0.3`,`0.4`，`0.5` |
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
|        | Attack Time  | The time at which a backdoor is first conducted by an attacker.                       | `--epoch_backdoor`      | `0`           | int[(`0.1`,`0.2`,`0.3`,`0.4`,`0.5`) * 1000 ]|
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


####  Backdoor attack  in Federated GNNs  on other  federated algorithms
running command for training:
```python 
--defense fedavg, fedopt, fedprox, fed_trimmedmean, fedMedian, fed_krum，fed_multi_krum, fed_bulyan
```

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
@misc{liu2023bkdfedgnn,
      title={Bkd-FedGNN: A Benchmark for Classification Backdoor Attacks on Federated Graph Neural Network}, 
      author={Fan Liu and Siqi Lai and Yansong Ning and Hao Liu},
      year={2023},
      eprint={2306.10351},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
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
@inproceedings{fan2022AdvST,
 author = {LIU, Fan and Liu, Hao and Jiang, Wenzhao},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {19035--19047},
 publisher = {Curran Associates, Inc.},
 title = {Practical Adversarial Attacks on Spatiotemporal Traffic Forecasting Models},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/79081c95482707d2db390542614e29cd-Paper-Conference.pdf},
 volume = {35},
 year = {2022}
}


```






## Acknowledgement
The codes are modifed based on [Xu et al. 2020](https://github.com/xujing1994/bkd_fedgnn) and [Dai et al. 2020](https://github.com/ventr1c/UGBA)
To the best of our knowledge, our work is the first to extend the node-level backdoor attack to the federated GNNs setting


## Install metis


```python
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
gunzip metis-5.1.0.tar.gz
tar -xvf metis-5.1.0.tar
rm metis-5.1.0.tar
cd metis-5.1.0
make config shared=1
make install
export METIS_DLL=/usr/local/lib/libmetis.so

pip3 install metis-python
```

## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:

- appdirs==1.4.4
- brotlipy==0.7.0
- cachetools==5.3.0
- certifi==2022.12.7
- cffi==1.15.0
- chardet==5.1.0
- charset-normalizer==3.0.1
- click==8.1.3
- contourpy==1.0.7
- cryptography==38.0.4
- cycler==0.11.0
- Cython==0.29.33
- dgl==1.0.1+cu116
- docker-pycreds==0.4.0
- Flask==2.2.3
- fonttools==4.39.0
- gitdb==4.0.10
- GitPython==3.1.31
- hdbscan==0.8.28
- idna==3.4
- importlib-metadata==6.0.0
- importlib-resources==5.12.0
- itsdangerous==2.1.2
- Jinja2==3.1.2
- joblib==1.1.0
- kiwisolver==1.4.4
- MarkupSafe==2.1.2
- matplotlib==3.7.1
- mkl-fft==1.3.1
- mkl-random==1.2.2
- mkl-service==2.4.0
- networkx==3.0
- numpy==1.24.2
- nvidia-cublas-cu11==11.10.3.66
- nvidia-cuda-nvrtc-cu11==11.7.99
- nvidia-cuda-runtime-cu11==11.7.99
- nvidia-cudnn-cu11==8.5.0.96
- nvidia-ml-py==11.525.84
- nvitop==1.0.0
- packaging==23.0
- pathtools==0.1.2
- Pillow==9.4.0
- pip==22.3.1
- protobuf==4.22.3
- psutil==5.9.4
- pycparser==2.21
- pyg-lib==0.1.0+pt113cu117
- pyOpenSSL==22.0.0
- pyparsing==3.0.9
- PySocks==1.7.1
- python-dateutil==2.8.2
- python-louvain==0.16
- PyYAML==6.0
- requests==2.28.2
- scikit-learn==1.1.3
- scikit-learn-extra==0.2.0
- scipy==1.10.1
- sentry-sdk==1.20.0
- setproctitle==1.3.2
- setuptools==65.6.3
- six==1.16.0
- smmap==5.0.0
- termcolor==2.2.0
- threadpoolctl==3.1.0
- torch==1.13.1
- torch-cluster==1.6.0+pt113cu117
- torch-geometric==2.2.0
- torch-scatter==2.1.0+pt113cu117
- torch-sparse==0.6.16+pt113cu117
- torch-spline-conv==1.2.1+pt113cu117
- torchvision==0.14.1
- tqdm==4.64.1
- typing_extensions==4.5.0
- urllib3==1.26.14
- wandb==0.15.2
- Werkzeug==2.2.3
- wheel==0.38.4
- zipp==3.15.0
