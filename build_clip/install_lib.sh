#!/bin/bash

echo "install onnx dependencies..."

python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# TensorRT版本TensorRT[本镜像默认自带]
# onnx版本1.13.0，onnxruntime-gpu版本1.13.1，onnxmltools版本1.11.1

pip install onnx==1.13.0 onnxruntime-gpu==1.13.1 onnxmltools==1.11.1


echo "install cn_clip"

cd ../clip; pip install -e .
export PYTHONPATH=$PYTHONPATH:/workspace/clip/cn_clip
# 当前位于clip源代码目录
echo "cur path: `pwd`"

