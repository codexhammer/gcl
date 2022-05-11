#!/bin/bash

code=main.py
gpu_id=1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py38
python $code --dataset Cora --abl 1 --setting task --mp_nn gcn --gpu_id $gpu_id
python $code --dataset Citeseer --abl 1 --setting task --mp_nn gcn --gpu_id $gpu_id
python $code --dataset Cora --abl 2 --setting task --mp_nn gcn --gpu_id $gpu_id
python $code --dataset Citeseer --abl 2 --setting task --mp_nn gcn --gpu_id $gpu_id

python $code --dataset Cora --abl 1 --setting class --mp_nn gcn --gpu_id $gpu_id
python $code --dataset Citeseer --abl 1 --setting class --mp_nn gcn --gpu_id $gpu_id
python $code --dataset Cora --abl 2 --setting class --mp_nn gcn --gpu_id $gpu_id
python $code --dataset Citeseer --abl 2 --setting class --mp_nn gcn --gpu_id $gpu_id