#!/bin/zsh
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate

env_name="pegasus"
conda remove -y -n $env_name --all
set -e

conda create -y -n $env_name python=3.9
conda activate $env_name

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
export FORCE_CUDA=1
cd $HOME/GitHub/
git clone https://github.com/HyunsooCha/pytorch3d_pointavatar.git
cd $HOME/GitHub/pytorch3d_pointavatar
git switch point-avatar
pip install -e .
cd $HOME/GitHub/$env_name
pip install -r ./scripts/requirement.txt
python -m pip install -e ./preprocess/Grounded-Segment-Anything/segment_anything
python -m pip install -e ./preprocess/Grounded-Segment-Anything/GroundingDINO