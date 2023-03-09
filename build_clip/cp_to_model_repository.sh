#!/bin/bash

MODEL_REPOSITORY_DIR=/data00/home/heguangfu/model_repository
MODEL_OUTPUT=/data00/home/heguangfu/output


cp $MODEL_OUTPUT/deploy/vit-b-16.img.fp16.trt $MODEL_REPOSITORY_DIR/clip_vitb16_img/1/model.plan
cp $MODEL_OUTPUT/deploy/vit-b-16.txt.fp16.trt $MODEL_REPOSITORY_DIR/clip_vitb16_txt/1/model.plan

echo "origin model md5:"
md5sum $MODEL_OUTPUT/deploy/*.trt

echo "dest model md5"
md5sum $MODEL_REPOSITORY_DIR/clip_vitb16_img/1/model.plan
md5sum $MODEL_REPOSITORY_DIR/clip_vitb16_txt/1/model.plan
