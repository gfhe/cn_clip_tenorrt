#!/bin/bash

docker run --name=trt_dev -it \
           --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
           -v /data00/home/heguangfu/build_clip:/workspace/env \
           -v /data00/home/heguangfu/source/Chinese-CLIP:/workspace/clip \
           -v /data00/home/heguangfu/output:/content/output \
           nvcr.io/nvidia/pytorch:23.01-py3 bash
