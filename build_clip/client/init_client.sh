#!/bin/bash

docker run --name=trt_client -it \
           --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
           -v /data00/home/heguangfu/build_clip/client:/workspace \
           nvcr.io/nvidia/tritonserver:23.01-py3-sdk bash
