# trt_transform
export SCENE=Chair
export prefix=/work/Users/lisicheng/Code/SteerNeRF_public/ckpts/nsvf/$SCENE/End2End_direct

onnxsim $prefix/unet_$SCENE.onnx $prefix/unet_sim_$SCENE.onnx

# change to the dir of TensorRT installation
cd /work/Users/lisicheng/Code/TensorRT-8.4.3.1/bin

./trtexec \
    --onnx=$prefix/unet_sim_$SCENE.onnx \
    --saveEngine=$prefix/unet_fp16_$SCENE.engine \
    --device=7 \
    --fp16

./trtexec \
    --onnx=$prefix/unet_sim_$SCENE.onnx \
    --saveEngine=$prefix/unet_int8_$SCENE.engine \
    --device=7 \
    --int8
