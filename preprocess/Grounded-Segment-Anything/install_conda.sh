#!/bin/zsh
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate

conda remove -y -n gsam --all
set -e
conda create -y -n gsam python=3.9
conda activate gsam
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
# export CUDA_HOME=/path/to/cuda-11.3/

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
pip install diffusers transformers accelerate scipy safetensors natsort
