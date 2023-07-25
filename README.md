# SteerNeRF: Accelerating NeRF Rendering via Smooth Viewpoint Trajectory (CVPR 2023)

This is the code for our CVPR 2023 paper: [SteerNeRF: Accelerating NeRF Rendering via Smooth Viewpoint Trajectory](https://jasonlsc.github.io/SteerNeRF/)

## Installation

This implementation has **strict** requirements due to dependencies on other libraries. If you encounter installation problem due to hardware/software mismatch, feel free to email me at [jasonlisicheng@zju.edu.cn](jasonlisicheng@zju.edu.cn).

### Hardware

* OS: Ubuntu 20.04
* NVIDIA GPU with Compute Compatibility >= 86 and memory > 12GB (Tested with RTX 3090), CUDA 11.3 (might work with older version)
* 32GB RAM (in order to load full size images)

### Software

* Python>=3.8 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n steernerf python=3.8` to create a conda environment and activate it by `conda activate steernerf`)
* Python libraries
    * Install `pytorch>=1.11.0` by `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113`
    * Install `torch-scatter` following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation)
    * Install `tinycudann` following their [instruction](https://github.com/NVlabs/tiny-cuda-nn#requirements) (compilation and pytorch extension)
    * Install `apex` following their [instruction](https://github.com/NVIDIA/apex#linux)
    * Install core requirements by `pip install -r requirements.txt`

* Cuda extension: Upgrade `pip` to >= 22.1 and run `pip install models/csrc/` 
* TensorRT tools:
    * Download `TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz` from [NVIDIA official website](https://developer.nvidia.com/nvidia-tensorrt-download)
    * Add environment variable in `~/.bashrc`
      * `export LD_LIBRARY_PATH=<Path of TensorRT>:$LD_LIBRARY_PATH`
      * e.g. `export LD_LIBRARY_PATH=~/TensorRT-8.4.3.1/lib:$LD_LIBRARY_PATH`
    * Install `TensorRT wheel` by `cd TensorRT-8.4.3.1/python && python3 -m pip install tensorrt-8.4.3.1-cp38-none-linux_x86_64.whl`
    * Install `uff wheel` by `cd TensorRT-8.4.3.1/uff && python3 -m pip install uff-0.6.9-py2.py3-none-any.whl`
    * Install `graphsurgeon wheel` by `cd TensorRT-8.4.3.1/graphsurgeon && python3 -m pip install graphsurgeon-0.4.6-py2.py3-none-any.whl`
    * Install `onnx-graphsurgeon wheel` by `cd TensorRT-8.4.3.1/onnx_graphsurgeon && python3 -m pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl`
    * Install `onnxsim` by `pip install onnxsim`

## Dataset
Download dataset from the [link](https://pan.baidu.com/s/1gxJPb3TmH_QlhXv4s1Zm9A?pwd=30zc) (Passcode:30zc)
Put the zip file in the path `./data_cfg`, then unzip.
Images and corresponding poses of train/test split and pre-defined rendering trajectory are in the corresponding subdirectory.

## Training
Use shell files in `./scripts` to train models.

## Checkpoints
Download ckeckpoint files from the [link](https://pan.baidu.com/s/1gxJPb3TmH_QlhXv4s1Zm9A?pwd=30zc) (Passcode:30zc), put the zip file in the path `./checkpoints`, and unzip.
Parameters for neural feature fields, parameters for neural renderer and converted TensorRT Engine of neural renderer are in the corresponding subdirectory.

## TensorRT conversion

### Step 1: conversion to ONNX format

Use `convert.sh` to convert U-net model from pytorch format into ONNX format. You need to change `export NAME=Family` in shell script.

Then, you will get corresponding onnx file in `./ckpts/nsvf/{scene_name}`.

### Step 2: conversion to TensorRT engine

Use `trt.sh` to convert ONNX file into TensorRT engine file.

Then, you will get corresponding TensorRT engine in `./ckpts/nsvf/{scene_name}`.

## Rendering on a smooth trajectory

Use `render_traj.sh` to render a sequence of frames and save the video under `./output`. 
The operation of saving frames make it take longer time.
So if you want to get accurate runtime breakdown for real-time rendering, you could delete the input parameter `--save_traj_img`.

## Acknowledgement

Our code is huge influenced by a third party python implementation of Instant-NGP, [ngp_pl](https://github.com/kwea123/ngp_pl).

If you find this code useful, please consider citing:

```tex
@InProceedings{Li_2023_CVPR,
    author    = {Li, Sicheng and Li, Hao and Wang, Yue and Liao, Yiyi and Yu, Lu},
    title     = {SteerNeRF: Accelerating NeRF Rendering via Smooth Viewpoint Trajectory},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {20701-20711}
}
```

