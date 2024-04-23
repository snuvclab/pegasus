#!/bin/zsh
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate pegasus

CUDA_VISIBLE_DEVICES=$1 python demo_gsam.py --base_path $2 --input $3 --output $4 --text_prompt $5 --white_bg