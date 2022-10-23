# gcl
1. Install the relevant packages/ set the environment as in environment.yml file.
2. Run run.sh for continual learning


 # Requirements:
 1. Set conda env name as 'py38'.  Python version:  3.8.12
 2. Pytorch version:  1.8.0  ( conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch )
 3. Pytorch-geometric version: 2.0.1  ( conda install pyg -c pyg -c conda-forge )

 Disclaimer: This code is inspired from [here](https://github.com/GraphNAS/GraphNAS). For the comparative baselines, refer [here](https://github.com/hhliu79/TWP).

 Cite our paper to use this code:

```yaml
@inproceedings{10.1145/3511808.3557427,
author = {Rakaraddi, Appan and Siew Kei, Lam and Pratama, Mahardhika and de Carvalho, Marcus},
title = {Reinforced Continual Learning for Graphs},
year = {2022},
isbn = {9781450392365},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3511808.3557427},
doi = {10.1145/3511808.3557427},
abstract = {Graph Neural Networks (GNNs) have become the backbone for a myriad of tasks pertaining to graphs and similar topological data structures. While many works have been established in domains related to node and graph classification/regression tasks, they mostly deal with a single task. Continual learning on graphs is largely unexplored and existing graph continual learning approaches are limited to the task-incremental learning scenarios. This paper proposes a graph continual learning strategy that combines the architecture-based and memory-based approaches. The structural learning strategy is driven by reinforcement learning, where a controller network is trained in such a way to determine an optimal number of nodes to be added/pruned from the base network when new tasks are observed, thus assuring sufficient network capacities. The parameter learning strategy is underpinned by the concept of Dark Experience replay method to cope with the catastrophic forgetting problem. Our approach is numerically validated with several graph continual learning benchmark problems in both task-incremental learning and class-incremental learning settings. Compared to recently published works, our approach demonstrates improved performance in both the settings. The implementation code can be found at https://github.com/codexhammer/gcl.},
booktitle = {Proceedings of the 31st ACM International Conference on Information &amp; Knowledge Management},
pages = {1666–1674},
numpages = {9},
keywords = {graph neural networks, continual learning, reinforcement learning},
location = {Atlanta, GA, USA},
series = {CIKM '22}
}
```