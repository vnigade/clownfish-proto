#!/bin/bash

# Model serving end-points
: ${edge_host:="localhost"}
: ${edge_port:="10001"}
: ${cloud_host:=""}
: ${cloud_port:="10001"}

# dataset config
: ${dataset="PKUMMD"}
: ${dataset_annotation="pkummd_cross_subject.json"}
: ${workers:=0}
: ${dataset_root:="$HOME"}

# Window config
: ${window_size:=16}
: ${window_stride:=4}

# Local model
: ${local_model:="resnet-18"}
: ${local_image_size:=224}

# Fusion method config
: ${fusion_method:="exponential_smoothing"}
: ${sim_method:="fix_ma"}
: ${siminet_path:="${dataset_root}/datasets/${dataset}/model_ckpt/siminet_${local_model}_window_${window_size}_${window_stride}_size_${local_image_size}_epoch_99.pth"}
# : ${filter_interval:=6}
filter_intervals=(6 6 8 6)

# Select program
: ${PROG:="main.py"}

# Network config
: ${LATENCY:="0ms"}
BW=(0.0 7.5 5 0.0)

: ${password:=""}
: ${output_prefix:="output"}

function sudo_cmd() {
  echo "sudo command: $*"
  echo "${password}" | sudo -S "$@"
}

# signal handler
function stop() {
  kill -TERM ${fusion_pid} &>/dev/null
  sudo kill -TERM ${shaper_pid} &>/dev/null  
  sudo ./scripts/shaper.sh clear eth0
}
trap "stop" SIGHUP SIGINT SIGTERM

# start bandwidth shaping
# export LATENCY="${LATENCY}"
# sudo -E ./scripts/shaper.sh start eth0 25Mbit

for (( iter=0; iter<4; iter++ ))
do
  bw=${BW[${iter}]}
  if [ $iter -eq 0 ] || [ $iter -eq 3 ]; then
      echo "No shaping for iteration 1:"
  else
      echo "Bandwidth shaping: $bw" | tee ${output_prefix}_bw_${bw}_${iter}.diff
      sudo_cmd ./scripts/shaper.sh start eth0 ${bw}Mbit
  fi

  filter_interval=${filter_intervals[${iter}]}
  echo "Using filter interval: ${filter_interval}"
  # Start the fusion
  python3 ${PROG} --dataset ${dataset} --dataset_path ${dataset_root}/datasets/${dataset}/rgb_frames/val_bw/${iter} --dataset_annotation ${dataset_root}/datasets/${dataset}/splits/${dataset_annotation} --edge_host ${edge_host} --edge_port ${edge_port} --cloud_host "${cloud_host}" --cloud_port ${cloud_port} --fusion_method ${fusion_method} --sim_method ${sim_method} --siminet_path ${siminet_path} --workers ${workers} --window_size ${window_size} --window_stride ${window_stride} --filter_interval ${filter_interval} >> ${output_prefix}_bw_${bw}_${iter}.diff # & # --enable_stats &

  mv ./accuracy.log accuracy_bw_${bw}_${iter}.log
  cat ./fps.log >> fps_bw_${bw}_${iter}.log
  sudo_cmd ./scripts/shaper.sh clear eth0
done

# clean bandwidth shaping
stop
