export CUDA_VISIBLE_DEVICES=6

# **The folder** contains GT and pseudo GT: 1100 ref. image and 3300 input pose
#export ROOT_DIR=/work/Users/lisicheng/Dataset/nerf_sr_data/TanksAndTemple/  /home/hli/Code/steernerf_1.0/
#export ROOT_DIR=/work/Users/lisicheng/Dataset/nerf_sr_data/Synthetic_NeRF/

export ROOT_DIR=./data_cfg/Synthetic_NeRF

export NAME=Chair
python train.py \
    --root_dir $ROOT_DIR/$NAME \
    --exp_name $NAME \
    --super_sampling --super_sampling_factor 4 \
    --feature_training \
    --frame_num 3 \
    --sr_model_type unet_feature_slim_multiframe \
    --num_epochs 300 --batch_size 16384 --lr 3e-3 --lr_SR 1e-4 --eval_lpips \
    --training_stage End2End \
    --complete_pipeline --direct_E2E \
    --val_only --TRT_enable \
    --TRT_engine_file ./ckpts/nsvf/$NAME/End2End_direct/unet_int8_$NAME.engine \
    --ckpt_path ./ckpts/nsvf/$NAME/End2End_direct/epoch=299_slim_feat.ckpt
    # --save_traj_img

