#!/bin/bash

code=main.py
dataset=CoraFull
gpu_id=0
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py38
# python $code --dataset $dataset --setting task --mp_nn gat --controller_max_step 4 --gpu_id $gpu_id
# python $code --dataset $dataset --setting class --mp_nn gat --controller_max_step 4 --gpu_id $gpu_id
python $code --dataset $dataset --setting task --mp_nn sg --controller_max_step 4 --gpu_id $gpu_id
python $code --dataset $dataset --setting class --mp_nn sg --controller_max_step 4 --gpu_id $gpu_id