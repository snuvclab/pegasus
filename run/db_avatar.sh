#!/bin/zsh
set -e
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate pegasus
cd $HOME/GitHub/pegasus/code
device=4
export CUDA_VISIBLE_DEVICES=$device

ntfy done python scripts/exp_runner.py --conf ./confs/db_avatar.conf --nepoch 63
ntfy done python scripts/exp_runner.py --conf ./confs/db_avatar.conf --is_eval
