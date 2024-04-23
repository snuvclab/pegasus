#!/bin/bash
#######################################################
# Things that you should modify
device=4
object_name='nose'
path=$HOME/'GitHub/pegasus/data/datasets'
source_human='Hyunsoo_Cha_Orange'
target_humans='Chloe_Grace_Moretz' # John_Krasinski
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
  mv $video_folder/$subject_name/"${array[0]}"/image $video_folder/$subject_name/"${array[0]}"/image_source
  mv $video_folder/$subject_name/"${array[0]}"/mask $video_folder/$subject_name/"${array[0]}"/mask_source
done
echo "[INFO] copy target rendered images to synthetic database."
for video in $video_names
do
  IFS='.' read -r -a array <<< $video
  target=${video#*${object_name}_}
  target=${target%.mp4}
  cp -r "$HOME/GitHub/pegasus/data/experiments/original_db/original_db/${target}/eval/${source_human}/epoch_"*"_default_rendering_${source_human}/rgb_erode_dilate" "$video_folder/$subject_name/"${array[0]}"/image_rendering"
  cp -r "$HOME/GitHub/pegasus/data/experiments/original_db/original_db/${target}/eval/${source_human}/epoch_"*"_default_rendering_${source_human}/normal_erode_dilate" "$video_folder/$subject_name/"${array[0]}"/normal_rendering"
done
echo "[INFO] segment object."
cd $path_preprocess
for video in $video_names
do
  video_path=$video_folder/$video
  echo $video
  IFS='.' read -r -a array <<< $video
  python object_masking_by_bisenet.py --base_path $video_folder/$subject_name/"${array[0]}" --input_dir image_source --output_dir mask_object --label $object_name --kernel_size 0
  python object_masking_by_bisenet.py --base_path $video_folder/$subject_name/"${array[0]}" --input_dir image_rendering --output_dir mask_object_rendering --label $object_name --kernel_size 0
done
echo "[INFO] make blacklist to exclude the noisy images for poisson blending."
cd $path_preprocess
for video in $video_names
do
  echo $video
  IFS='.' read -r -a array <<< $video
  base_path=$video_folder/$subject_name/"${array[0]}"
  python blacklist.py --base_path $video_folder/$subject_name/ --parent_directories "${array[0]}" --sub_directories "image mask mask_object image_rendering" --reference_directory image_source
done
echo "[INFO] run blending source to target"
cd $path_preprocess
for video in $video_names
do
  echo $video
  IFS='.' read -r -a array <<< $video
  base_path=$video_folder/$subject_name/"${array[0]}"
  python blending_src2tgt.py --base_path $base_path --blend_type poisson --image_original_dir image_source --image_rendering_dir image_rendering --mask_object_dir mask_object --mask_object_rgb_dir mask_object_rgb --mask_object_rendering_dir mask_object_rendering --start_frame 0 --save_dir image
done
echo "[INFO] mask generation for image."
cd $path_modnet
for video in $video_names
do
  echo $video
  IFS='.' read -r -a array <<< $video
  base_path=$video_folder/$subject_name/"${array[0]}"
  python demo/image_matting/colab/inference.py --input-path $base_path/'image_rendering' --output-path $base_path/'mask' --ckpt-path $modnet_ckpt_path
done
echo "[INFO] make blacklist to exclude the training dataset."
cd $path_preprocess
for video in $video_names
do
  echo $video
  IFS='.' read -r -a array <<< $video
  base_path=$video_folder/$subject_name/"${array[0]}"
  python blacklist.py --base_path $video_folder/$subject_name/ --parent_directories "${array[0]}" --sub_directories "image mask mask_object image_rendering" --reference_directory image_source
done