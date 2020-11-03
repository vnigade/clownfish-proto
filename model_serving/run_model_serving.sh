#!/bin/bash
: ${sample_duration:=16}
: ${sample_size:=224}
: ${model:=""}
: ${port_number:=10001}
: ${num_workers:=4}
: ${dataset_root:="$HOME/datasets/PKUMMD/"}

export PYTHONPATH="${PYTHONPATH}:../"

# Resnet-18
if [ "${model}" == "resnet-18" ]; then
echo "Starting resnet-18 model..."
python3 main.py --model resnet --resnet_shortcut A --modality RGB --resume_path ${dataset_root}/model_ckpt/resnet-18_window_${sample_duration}_size_${sample_size}.pth --sample_duration ${sample_duration} --sample_size ${sample_size} --num_classes 51 --model_depth 18 --port_number ${port_number} --workers ${num_workers} # --enable_stats

# Resnext-101
elif [ "${model}" == "resnext-101" ]; then
echo "Starting resnext-101 model..."
python3 main.py --model resnext --resnet_shortcut B --resnext_cardinality 32 --modality RGB --resume_path ${dataset_root}/model_ckpt/resnext-101_window_${sample_duration}_size_${sample_size}.pth --sample_duration ${sample_duration} --sample_size ${sample_size} --num_classes 51 --model_depth 101 --port_number ${port_number} --workers ${num_workers} # --enable_stats
fi

