#!/bin/zsh
set -e
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate pegasus

cd $HOME/GitHub/pegasus/code
device=4,5,6,7
export CUDA_VISIBLE_DEVICES=$device

python -m torch.distributed.launch --nproc_per_node=4 scripts/exp_runner.py --conf ./confs/pegasus_init.conf --nepoch 11
python -m torch.distributed.launch --nproc_per_node=4 scripts/exp_runner.py --conf ./confs/pegasus.conf