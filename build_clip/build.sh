#!/bin/bash

cd ../clip
export PYTHONPATH=$PYTHONPATH:/workspace/clip/cn_clip
# 当前位于clip源代码目录
echo "cur path: `pwd`"

echo "prepare filesystem"

mkdir -p /content/output/pretrained_weights
mkdir -p /content/output/deploy

echo "pt model -> onnx model"

# python cn_clip/deploy/pytorch_to_onnx.py \
#          --model-arch ViT-B-16 \
#          --pytorch-ckpt-path /content/output/pretrained_weights/clip_cn_vit-b-16.pt \
#          --save-onnx-path /content/output/deploy/vit-b-16  \
#          --convert-text --convert-vision

echo "done onnx model"
ls -alh /content/output/deploy

python /workspace/env/check_onnx.py


echo "onnx model -> trt model"
python cn_clip/deploy/onnx_to_tensorrt.py \
       --model-arch ViT-B-16 \
       --convert-text \
       --text-onnx-path /content/output/deploy/vit-b-16.txt.fp16.onnx \
       --save-tensorrt-path /content/output/deploy/vit-b-16 \
       --batch-size 128\
       --fp16
python cn_clip/deploy/onnx_to_tensorrt.py \
       --model-arch ViT-B-16 \
       --convert-vision \
       --vision-onnx-path /content/output/deploy/vit-b-16.img.fp16.onnx \
       --save-tensorrt-path /content/output/deploy/vit-b-16 \
       --batch-size 32\
       --fp16
echo "done tensorrt model"
ls -alh /content/output/deploy

python /workspace/env/check_trt.py


python3 cn_clip/deploy/speed_benchmark.py \
        --model-arch ViT-B-16 \
        --pytorch-ckpt /content/output/pretrained_weights/clip_cn_vit-b-16.pt \
        --pytorch-precision fp16 \
        --onnx-image-model /content/output/deploy/vit-b-16.img.fp16.onnx \
        --onnx-text-model /content/output/deploy/vit-b-16.txt.fp16.onnx \
        --tensorrt-image-model /content/output/deploy/vit-b-16.img.fp16.trt \
        --tensorrt-text-model /content/output/deploy/vit-b-16.txt.fp16.trt
