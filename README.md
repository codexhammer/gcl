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
 @misc{https://doi.org/10.48550/arxiv.2209.01556,
  doi = {10.48550/ARXIV.2209.01556},
  
  url = {https://arxiv.org/abs/2209.01556},
  
  author = {Rakaraddi, Appan and Lam, Siew Kei and Pratama, Mahardhika and De Carvalho, Marcus},
    
  title = {Reinforced Continual Learning for Graphs},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```