#!/bin/zsh
# Get the hostname
hostname=$(hostname)
if [[ "$hostname" == "vclabserver5" ]]; then
  echo "Hostname is vclabserver5"
  set -e
  ln -s /media/ssd1/hyunsoocha/original_db $HOME/GitHub/pegasus/data/datasets/original_db
  ln -s /media/ssd2/hyunsoocha/GitHub/pegasus_dev/data/datasets/total_composition_Guy $HOME/GitHub/pegasus/data/datasets/total_composition_Guy
fi