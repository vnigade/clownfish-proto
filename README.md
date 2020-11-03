# Description
This repository contains the implementation of the Clownfish prototype. You can run this code on an edge node, where required packages can be installed easily, and on a remote cloud node.

# Software Requirements
The code is tested with the following installed software packages.

1. Python 3.6+
2. Pytorch 1.4.0
3. Scikit-learn 0.23.0
5. Request 2.24.0       # Needed for downloading the data from Google drive.
6. Grpc 1.27.0
7. Protobuf 3.0.0
8. tc, netem            # Needed for traffic shaping experiments, we borrowed scripts from [AWStream](https://github.com/awstream/awstream)

# Getting started
## Clone this repository and submodules
```shell
$ git submodule update --init
$ export PYTHONPATH="/absolute/path/to/clownfish-proto"
```

## Generate python gRPC code from proto files
```shell
$ cd model_serving/protos
$ ./generate_proto.sh
```

## Download dataset and model checkpoints
* PKU-MMD videos

See instructions [here](https://github.com/vnigade/clownfish-3D-ResNets) on how to download PKU-MMD video data.

* PKU-MMD split files and model checkpoints
```shell
$ ./download_data.sh
```

Assume/Create the structure of dataset directories in the following way:

```misc
~/
  dataset/
    PKUMMD/
      rgb_frames/
        val/0291-L/image_*****.jpg            # 132 Validation videos and JPEG frames for cross subject evaluation
               .../image_*****.jpg
        val_bw/0/
              0291-L/ 0292-M/ 0293-R/         # three videos used for no shaping. Here, create symlinks.
        val_bw/1/
              0294-L/ 0295-M/ 0296-R/         # three videos used for 7.5Mbps
        val_bw/2/
              0297-L/ 0298-M/ 0299-R/         # three videos used for 5Mbps  
        val_bw/3/
              0300-L/ 0301-M/ 0302-R/         # three videos used for no shaping 
        val_bw/all/
              0291-L/ ... 0302-R/             # all 12 videos used
        val_early_discard/0316-R/
      models/
      model_ckpt/
        resnet-18_window_16_size_224.pth
        resnext-101_window_16_size_224.pth
        siminet_resnet-18_window_16_4_size_224_epoch_99.pth
      splits/pkummd_cross_subject.json
      scores_dump/
```

## Issues
1. `ImportError: cannot import name 'pre_act_resnet'` 

To fix this issue, run the following command from the root of this repo.
```shell
$ sed -i 's/from\ models/from\ .models/g' ResNets_3D_PyTorch/model.py
```

# How to run

This repository has three branches,
1. `main`, which is used to experiment with Clownfish, i.e., the fusion between edge and cloud 
2. `async`, which is used to experiment with remote cloud. It contains an asynchronous implementation of cloud-only solution which is used to compare in a fair manner with Clownfish for low bandwidth scenario
3. `early-discard`, which is used to experiment with EarlyDiscard solution

## Run Clownfish solution

Checkout the `main` branch on the remote (cloud) and the local (edge) node.
```shell
$ git checkout master
$ git submodule update --init
```

Run model serving with the `ResNext-101` model on the remote node (let's consider IP address to be 192.168.1.1) at port 10001.
```shell
$ cd model_serving
$ sample_duration=16 sample_size=224 model="resnext-101" ./run_model_serving.sh
```

Run model serving with the `ResNet-18` model on the edge node (let's consider IP address to be 192.168.1.2) at port 10001.
```shell
$ cd model_serving
$ sample_duration=16 sample_size=224 model="resnet-18" ./run_model_serving.sh
```

1. Run fusion with `SimiNet` as a similarity method
```shell
$ sim_method="siminet" edge_host="localhost" edge_port=10001 cloud_port=10001 cloud_host="192.168.1.1" window_size=16 window_stride=4 ./run.sh
```
2. Run fusion for latency scenarios
```shell
$ LATENCY=150ms sudo -E ./scripts/shaper.sh start eth0 100Mbit
$ sim_method="siminet" edge_host="localhost" edge_port=10001 cloud_port=10001 cloud_host="192.168.1.1" window_size=16 window_stride=4 ./run.sh
$ sudo ./scripts/shaper.sh clear eth0
```

3. Run fusion for bandwidth scenarios

We used 12 videos for this experiment. Check the [structure](#assume/create-the-structure-of-dataset-directories-in-the-following-way:) of dataset directories for the videos list. If you want to run for your list of videos then please check [run_bw_shaping.sh](./run_bw_shaping.sh). Change the `SINK` variable in [scripts/shaper.sh](./scripts/shaper.sh) to your remote address, i.e., 192.168.1.1.
```shell
$ password="your_sudo_passwd" sim_method="siminet" edge_host="localhost" edge_port=10001 cloud_port=10001 cloud_host="192.168.1.1" window_size=16 window_stride=4 ./run_bw_shaping.sh
```

For every run, the accuracy and system throughput is saved in files `accuracy*.log` and `fps*.log`, respectively. If you have enabled stats then you can find it in `*.stats` files.

## Run Cloud-only solution
Checkout the `async` branch on the remote (cloud) and the local (edge) node.
```shell
$ git checkout async
$ git submodule update --init
```

Run model serving with the `ResNext-101` model on the remote node at port 10001.
```shell
$ cd model_serving
$ sample_duration=16 sample_size=224 model="resnext-101" ./run_model_serving.sh
```

1. Run a normal scenario
```shell
$ cloud_port=10001 cloud_host="192.168.1.1" window_size=16 window_stride=4 ./run.sh
```

Note that the above run applies a moving average on cloud results. Therefore, the accuracy of this run would be higher than that of non-moving average cloud results.

2. Run for bandwidth scenarios
```shell
$ password="your_sudo_passwd" cloud_port=10001 cloud_host="192.168.1.1"  window_size=16 window_stride=4 ./run_bw_shaping.sh
```

## Run EarlyDiscard, i.e., the filtering approach presented in [this](http://elijah.cs.cmu.edu/DOCS/drone2018-CAMERA-READY.pdf) paper.

Checkout the `early-discard` branch on the remote (cloud) and the local (edge) node.
```shell
$ git checkout early-discard
$ git submodule update --init
```

Run model serving with the `ResNext-101` model on the remote node at port 10001.
```shell
$ cd model_serving
$ sample_duration=16 sample_size=224 model="resnext-101" ./run_model_serving.sh
```

Run early-discard model (`3D-MobileNet`) on the edge node. Note that the sample size is 112.
```shell
$ cd model_serving
$ sample_duration=16 sample_size=112 model="mobilenet-early-discard" ./run_model_serving.sh
```

1. Run for normal scenario
```shell
$ edge_host="localhost" edge_port=10001 cloud_port=10001 cloud_host="192.168.1.1" window_size=16 window_stride=4 ./run.sh
```

2. Run a bandwidth scenarios. Here, early-discard model is run on the edge node.
```shell
$ password="your_sudo_password" edge_host="localhost" edge_port=10001 cloud_port=10001 cloud_host="192.168.1.1" window_size=16 window_stride=4 ./run_bw_shaping.sh
```

For more options, please do check `run*.sh` and `opts.py` script files.

# Note
This repository is a cleaned-up version of the code used for Clownfish experiments. If you find any bug or issue in this code (or, in the paper) then please let us know.
