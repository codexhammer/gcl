#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate py38
code=main.py
dataset="Cora"
gpu_id=1
python $code --dataset $dataset --setting task --controller_max_step 4 --mp_nn gat --gpu_id $gpu_id
python $code --dataset $dataset --setting task --controller_max_step 4 --mp_nn sg --gpu_id $gpu_id
python $code --dataset $dataset --setting class --controller_max_step 4 --mp_nn gat --gpu_id $gpu_id
python $code --dataset $dataset --setting class --controller_max_step 4 --mp_nn sg --gpu_id $gpu_id

dataset="Citeseer"
python $code --dataset $dataset --setting task --controller_max_step 4 --mp_nn gat --gpu_id $gpu_id
python $code --dataset $dataset --setting task --controller_max_step 4 --mp_nn sg --gpu_id $gpu_id
python $code --dataset $dataset --setting class --controller_max_step 4 --mp_nn gat --gpu_id $gpu_id
python $code --dataset $dataset --setting class --controller_max_step 4 --mp_nn sg --gpu_id $gpu_id
