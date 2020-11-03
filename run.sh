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
: ${filter_interval:=6}

# Select program
: ${PROG:="main.py"}

export PYTHONPATH="${PYTHONPATH}:./"

python3 ${PROG} --dataset ${dataset} --dataset_path ${dataset_root}/datasets/${dataset}/rgb_frames/val_bw/all/ --dataset_annotation ${dataset_root}/datasets/${dataset}/splits/${dataset_annotation} --edge_host ${edge_host} --edge_port ${edge_port} --cloud_host "${cloud_host}" --cloud_port ${cloud_port} --fusion_method ${fusion_method} --sim_method ${sim_method} --siminet_path ${siminet_path} --workers ${workers} --window_size ${window_size} --window_stride ${window_stride} --filter_interval ${filter_interval} # --enable_stats
