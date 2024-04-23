#!/bin/bash
set -e
#######################################################
# Things you need to modify
device=5
subject_name='original_db'
path=$HOME/'GitHub/pegasus/data/datasets'
video_names='Yura1.mp4 Yura2.mp4'
resize=512
fps=1
# fx, fy, cx, cy in pixels, need to adjust with resizing and cropping
fx=1539.67462
fy=1508.93280
cx=261.442628
cy=253.231895
global_trans=False # NOTE The case where 'False' is needed is when the camera needs to move back and forth. If not, and both the camera and the user are fixed, it is much better to set it to 'True'.
########################################################
video_folder=$path/$subject_name
home_preprocess=$HOME/'GitHub/pegasus/preprocess'
path_modnet=$home_preprocess'/MODNet'
path_deca=$home_preprocess'/DECA'
path_parser=$home_preprocess'/face-parsing.PyTorch'
export CUDA_VISIBLE_DEVICES=$device
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate
conda activate pegasus
echo "[INFO] background/foreground segmentation"
cd $path_modnet
for video in $video_names
do
  video_path=$video_folder/$video
  echo $video
  IFS='.' read -r -a array <<< $video
  mkdir -p $video_folder/$subject_name/"${array[0]}"
  python demo/video_matting/custom/run.py --video $video_folder/"${array[0]}.mp4" --result-type matte --fps $fps
done
echo "[INFO] save the images and masks with ffmpeg"
cd $home_preprocess
for video in $video_names
do
  video_path=$video_folder/$video
  echo $video
  IFS='.' read -r -a array <<< $video
  echo $video_folder/$subject_name/"${array[0]}"/"image"
  mkdir -p $video_folder/$subject_name/"${array[0]}"/"image"
  ffmpeg -i $video_folder/"${array[0]}.mp4" -q:v 2 $video_folder/$subject_name/"${array[0]}"/"image"/"%d.png"
  mkdir -p $video_folder/$subject_name/"${array[0]}"/"mask"
  ffmpeg -i $video_folder/"${array[0]}_matte.mp4" -q:v 2 $video_folder/$subject_name/"${array[0]}"/"mask"/"%d.png"
done
echo "[INFO] DECA FLAME parameter estimation"
cd $path_deca
for video in $video_names
do
  echo $video
  IFS='.' read -r -a array <<< $video
  mkdir -p $video_folder/$subject_name/"${array[0]}"/"deca"
  python demos/demo_reconstruct.py -i $video_folder/$subject_name/"${array[0]}"/image --savefolder $video_folder/$subject_name/"${array[0]}"/"deca" --saveCode True --saveVis False --sample_step 1  --render_orig False
done
echo "[INFO] face alignment landmark detector"
cd $home_preprocess
for video in $video_names
do
  echo $video
  IFS='.' read -r -a array <<< $video
  python keypoint_detector.py --path $video_folder/$subject_name/"${array[0]}"
done
echo "[INFO] iris segmentation with fdlite"
cd $home_preprocess
for video in $video_names
do
  video_path=$video_folder/$video
  echo $video
  IFS='.' read -r -a array <<< $video
  python iris.py --path $video_folder/$subject_name/"${array}"
done
cd $path_deca
for video in $video_names
do
  IFS='.' read -r -a array <<< $(basename $video)
  echo $video
  python optimize.py --path $video_folder/$subject_name/"${array}" --cx $cx --cy $cy --fx $fx --fy $fy --size $resize --global_trans $global_trans --save_name 'flame_params_independent'
done
echo "[INFO] semantic segmentation with face parsing"
cd $path_parser
for video in $video_names
do
  video_path=$video_folder/$video
  echo $video
  IFS='.' read -r -a array <<< $video
  python test.py --dspth $video_folder/$subject_name/"${array}"/image --respth $video_folder/$subject_name/"${array}"/semantic
done