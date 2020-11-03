# Download model checkpoints for PKUMMD datasets
#!/bin/bash
: ${data_dir:="$HOME/datasets/PKUMMD"}

function gd_download() {
    [ -e $2 ] || mkdir -p $2
    dst="$2/$3"
    echo "Downloading file ${dst}"
    python3 google_drive.py $1 "$dst"
    tar -xvf "$dst" -C $2 > /dev/null
    rm "$dst"
}

# class file and annotation file of PKUMMD dataset
dst_dir="${data_dir}"
file="splits.tar.gz"
gd_download 1m8tpx8iSqpzalIRg6t_wS9F-tjc7fdBo "$dst_dir" ${file}

# resnet-18 model
dst_dir="${data_dir}/model_ckpt/"
file="resnet-18_window_16_size_224.pth.tar.gz"
gd_download 1xVkPZAoB2w3dCYGXOu7itLdFUq0Ko3hk "$dst_dir" ${file}

# resnext-101 model
dst_dir="${data_dir}/model_ckpt/"
file="resnext-101_window_16_size_224.pth.tar.gz"
gd_download 1hteWluXoXOZneTL3g_oub6DqIGF5C56e "$dst_dir" ${file}

# siminet model trained on local model
dst_dir="${data_dir}/model_ckpt/"
file="siminet_resnet-18_window_16_4_size_224_epoch_99.pth.tar.gz"
gd_download 1SfXvBX6WX7D0r94SwK4OXeY8iWCKInuZ "$dst_dir" ${file}

# mobilenet-early-discard model
dst_dir="${data_dir}/model_ckpt/"
file="mobilenet-early-discard_window_16_size_112.pth.tar.gz"
gd_download 1MN0mNSvwaoL2rRC9CG91Fck4JnDwAlC4 "$dst_dir" ${file}
