#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate py38
code=main.py
dataset="Citeseer"
gpu_id=0
cuda= false
python $code --dataset $dataset --setting task --controller_max_step 4 --cuda false
python $code --dataset $dataset --setting class --controller_max_step 4 --cuda false
dataset="Cora"
python $code --dataset $dataset --setting task --controller_max_step 4 --cuda false
python $code --dataset $dataset --setting class --controller_max_step 4 --cuda false
dataset="Computers"
python $code --dataset $dataset --setting task --controller_max_step 4 --cuda false
python $code --dataset $dataset --setting class --controller_max_step 4 --cuda false
dataset="CoraFull"
python $code --dataset $dataset --setting task --controller_max_step 4 --cuda false
python $code --dataset $dataset --setting class --controller_max_step 4 --cuda false