#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate py38
code=main.py
dataset="Citeseer"
gpu_id=0
cuda= false
python $code --dataset $dataset --setting task --controller_max_step 1 --cuda false
python $code --dataset $dataset --setting task --controller_max_step 2 --cuda false
python $code --dataset $dataset --setting task --controller_max_step 3 --cuda false
python $code --dataset $dataset --setting task --controller_max_step 4 --cuda false
python $code --dataset $dataset --setting class --controller_max_step 1 --cuda false
python $code --dataset $dataset --setting class --controller_max_step 2 --cuda false
python $code --dataset $dataset --setting class --controller_max_step 3 --cuda false
python $code --dataset $dataset --setting class --controller_max_step 4 --cuda false