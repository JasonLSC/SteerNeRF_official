#!/bin/bash
### README ###
# The bash is used to setup a complete pipeline


export CUDA_VISIBLE_DEVICES=7

# **The folder** contains GT and pseudo GT: 1100 ref. image and 3300 input pose
export ROOT_DIR=/work/Users/lisicheng/Dataset/nerf_sr_data/TanksAndTemple

python train.py \
    --root_dir $ROOT_DIR/Barn \
    --exp_name Barn \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --training_stage NeRF_pretrain