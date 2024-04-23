#!/bin/bash
#######################################################
# Things that you should modify
device=4
object_name='original'
path=$HOME/'GitHub/pegasus/data/datasets'
source_human='Hyunsoo_Cha_Orange'
target_humans='Hyunsoo_Cha_Orange'
########################################################
set -e
subject_name='synthetic_db_'$source_human
video_folder=$path/$subject_name
video_names=''
for target in $target_humans
do
    if [[ -z $video_names ]]; then
        video_names="${object_name}_${target}.mp4"
    else
        video_names="$video_names ${object_name}_${target}.mp4"
    fi
done
path_preprocess=$HOME/'GitHub/pegasus/preprocess/'
path_modnet=$path_preprocess/'MODNet'
modnet_ckpt_path=$path_modnet/'pretrained/modnet_webcam_portrait_matting.ckpt'
export CUDA_VISIBLE_DEVICES=$device
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate pegasus
mkdir -p $video_folder
mkdir -p $video_folder/$subject_name
echo "[INFO] copy source dataset."
for video in $video_names
do
  echo $video
  IFS='.' read -r -a array <<< $video
  cp -r $path/original_db/original_db/$source_human $video_folder/$subject_name/"${array[0]}"
done
echo "[INFO] segment object."
cd $path_preprocess
for video in $video_names
do
  video_path=$video_folder/$video
  echo $video
  IFS='.' read -r -a array <<< $video
  python object_masking_by_bisenet.py --base_path $video_folder/$subject_name/"${array[0]}" --input_dir image --output_dir mask_object --label hair --kernel_size 0
done
echo "[INFO] make blacklist to exclude the training dataset."
cd $path_preprocess
for video in $video_names
do
  echo $video
  IFS='.' read -r -a array <<< $video
  base_path=$video_folder/$subject_name/"${array[0]}"
  python blacklist.py --base_path $video_folder/$subject_name/ --parent_directories "${array[0]}" --sub_directories "mask mask_object" --reference_directory image
done