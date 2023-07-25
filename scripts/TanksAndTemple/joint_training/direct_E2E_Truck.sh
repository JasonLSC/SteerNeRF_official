#!/bin/bash
### README ###
# The bash is used to setup a complete pipeline
# Use best lr setting: "-lr 3e-3 --lr_SR 5e-4" to train

export CUDA_VISIBLE_DEVICES=0
export ROOT_DIR=./data_cfg/TanksAndTemple

python train.py \
    --root_dir $ROOT_DIR/Truck \
    --exp_name Truck \
    --super_sampling --super_sampling_factor 4 \
    --feature_training \
    --frame_num 1 \
    --sr_model_type unet_feature_slim_multiframe \
    --num_epochs 300 --batch_size 16384 --lr 3e-3 --lr_SR 1e-4 --eval_lpips \
    --training_stage End2End \
    --complete_pipeline --direct_E2E \
    --patch_size 128


