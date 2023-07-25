#!/bin/bash
### README ###
# The bash is used to setup a pretraining

export CUDA_VISIBLE_DEVICES=0

# **The folder** contains GT and pseudo GT: 1100 ref. image and 3300 input pose
export ROOT_DIR=./data_cfg/Synthetic_NeRF

python train.py \
    --root_dir $ROOT_DIR/Chair \
    --exp_name Chair \
    --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --training_stage NeRF_pretrain \
    --val_only --ckpt_path /work/Users/lisicheng/Code/SteerNeRF_public/ckpts/nsvf/Chair/NeRF_Pretrain/epoch=19_slim_feat.ckpt