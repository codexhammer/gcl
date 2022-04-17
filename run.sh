#!/bin/bash

code=main.py
seed=1234
conda activate py38
python $code --random_seed $seed --abl 0
python $code --random_seed $seed --abl 1
python $code --random_seed $seed --abl 2
python $code --random_seed $seed --abl 3