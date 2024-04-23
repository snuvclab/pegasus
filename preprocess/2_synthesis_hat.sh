#!/bin/bash
#######################################################
# Things that you should modify
device=4
object_name='hat'
path=$HOME/'GitHub/pegasus/data/datasets'
source_human='Hyunsoo_Cha_Orange'
target_humans='Chloe_Grace_Moretz John_Krasinski'
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
path_gsam=$path_preprocess/'Grounded-Segment-Anything'
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
echo "[INFO] segment object utilizing GSAM."
cd $path_gsam
for video in $video_names
do
  video_path=$video_folder/$video
  echo $video
  IFS='.' read -r -a array <<< $video
  ./run_gsam.sh $device $video_folder/$subject_name/"${array}"/ image_rendering mask_object $object_name
done
echo "[INFO] hair removal by stable diffusion."
cd $path_preprocess
for video in $video_names
do
  IFS='.' read -r -a array <<< $video
  echo $video
  python hair_removal_by_diffusion.py --base_path $video_folder/$subject_name/"${array}" --input_dir image_source --output_dir image_bald
done
echo "[INFO] image blending."
cd $path_preprocess
for video in $video_names
do
  video_path=$video_folder/$video
  echo $video
  IFS='.' read -r -a array <<< $video
  python image_blending.py --base_path $video_folder/$subject_name/"${array}"/ --image_bald_dir image_bald --image_rendering_dir image_rendering --mask_hat_dir mask_object --mask_hat_rgb_dir mask_object_rgb --output_image_dir image
done
echo "[INFO] mask generation for bald from image."
cd $path_modnet
for video in $video_names
do
  echo $video
  IFS='.' read -r -a array <<< $video
  python demo/image_matting/colab/inference.py --input-path $video_folder/$subject_name/"${array[0]}"/'image_bald' --output-path $video_folder/$subject_name/"${array[0]}"/'mask_bald' --ckpt-path $modnet_ckpt_path
done
echo "[INFO] mask generation for synthesis image."
cd $path_preprocess
for video in $video_names
do
  echo $video
  IFS='.' read -r -a array <<< $video
  python mask_synthesis.py --base_path $video_folder/$subject_name/"${array[0]}" --mask_object_dir mask_object --mask_bald_dir mask_bald --mask_save_dir mask
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
echo "[INFO] run blacklist segment failure"
cd $path_preprocess
for video in $video_names
do
  echo $video
  IFS='.' read -r -a array <<< $video
  base_path=$video_folder/$subject_name/"${array[0]}"
  python blacklist_segment_failure.py --base_path $video_folder/$subject_name --subject_name "${array[0]}" --image_dir image --image_rendering_dir image_rendering --image_original_dir image_source --mask_dir mask_source
done