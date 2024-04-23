#!/bin/bash
# This code is heavily based on DECA https://github.com/yfeng95/DECA/blob/master/fetch_data.sh

home_pegasus=$HOME/GitHub/pegasus
home_deca=$home_pegasus/preprocess/DECA

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

echo -e "\nBefore you continue, you must register at https://flame.is.tue.mpg.de/ and agree to the FLAME license terms."
read -p "Username (FLAME):" username
read -p "Password (FLAME):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p $home_deca/data

echo -e "\nDownloading FLAME..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' -O $home_deca/'data/FLAME2020.zip' --no-check-certificate --continue
unzip $home_deca/data/FLAME2020.zip -d $home_deca/data/FLAME2020
cp $home_deca/data/FLAME2020/generic_model.pkl $home_deca/data
mkdir -p $home_pegasus/code/flame/FLAME2020
mv $home_deca/data/FLAME2020/generic_model.pkl $home_pegasus/code/flame/FLAME2020/
rm -rf $home_deca/data/FLAME2020
rm $home_deca/data/FLAME2020.zip

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O $home_pegasus/preprocess/Grounded-Segment-Anything/pretrain/sam_vit_h_4b8939.pth