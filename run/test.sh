#!/bin/zsh
set -e
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate pegasus

cd $HOME/GitHub/pegasus/code
device=4
export CUDA_VISIBLE_DEVICES=$device

python scripts/exp_runner.py --conf ./confs/pegasus.conf --is_eval