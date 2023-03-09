# 完成必要的import（下文省略）
import onnxruntime
from PIL import Image
import numpy as np
import torch
import argparse
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from cn_clip.clip.utils import _MODELS, _MODEL_INFO, _download, available_models, create_model, image_transform

# 载入ONNX图像侧模型（**请替换${DATAPATH}为实际的路径**）
img_sess_options = onnxruntime.SessionOptions()
img_run_options = onnxruntime.RunOptions()
img_run_options.log_severity_level = 2
img_onnx_model_path="/content/output/deploy/vit-b-16.img.fp16.onnx"
img_session = onnxruntime.InferenceSession(img_onnx_model_path,
                                        sess_options=img_sess_options,
                                        providers=["CUDAExecutionProvider"])

# 预处理图片
model_arch = "ViT-B-16" # 这里我们使用的是ViT-B-16规模，其他规模请对应修改
preprocess = image_transform(_MODEL_INFO[model_arch]['input_resolution'])
# 示例皮卡丘图片，预处理后得到[1, 3, 分辨率, 分辨率]尺寸的Torch Tensor
image = preprocess(Image.open("pokemon.jpeg")).unsqueeze(0)
image = torch.vstack([image, image])

# 用ONNX模型计算图像侧特征
image_features = img_session.run(["unnorm_image_features"], {"image": image.cpu().numpy()})[0] # 未归一化的图像特征
image_features = torch.tensor(image_features)
image_features /= image_features.norm(dim=-1, keepdim=True) # 归一化后的Chinese-CLIP图像特征，用于下游任务
print(image_features.shape) # Torch Tensor shape: [1, 特征向量维度]


# 载入ONNX文本侧模型（**请替换${DATAPATH}为实际的路径**）
txt_sess_options = onnxruntime.SessionOptions()
txt_run_options = onnxruntime.RunOptions()
txt_run_options.log_severity_level = 2
txt_onnx_model_path="/content/output/deploy/vit-b-16.txt.fp16.onnx"
txt_session = onnxruntime.InferenceSession(txt_onnx_model_path,
                                        sess_options=txt_sess_options,
                                        providers=["CUDAExecutionProvider"])

# 为4条输入文本进行分词。序列长度指定为52，需要和转换ONNX模型时保持一致（参见转换时的context-length参数）
text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"], context_length=52) 

# 用ONNX模型依次计算文本侧特征
text_features = []
for i in range(len(text)):
    one_text = np.expand_dims(text[i].cpu().numpy(),axis=0)
    text_feature = txt_session.run(["unnorm_text_features"], {"text":one_text})[0] # 未归一化的文本特征
    text_feature = torch.tensor(text_feature)
    text_features.append(text_feature)
text_features = torch.squeeze(torch.stack(text_features),dim=1) # 4个特征向量stack到一起
text_features = text_features / text_features.norm(dim=1, keepdim=True) # 归一化后的Chinese-CLIP文本特征，用于下游任务
print(text_features.shape) # Torch Tensor shape: [4, 特征向量维度]


# 内积后softmax
# 注意在内积计算时，由于对比学习训练时有temperature的概念
# 需要乘上模型logit_scale.exp()，我们的预训练模型logit_scale均为4.6052，所以这里乘以100
# 对于用户自己的ckpt，请使用torch.load载入后，查看ckpt['state_dict']['module.logit_scale']或ckpt['state_dict']['logit_scale']
logits_per_image = 100 * image_features @ text_features.t()
print(logits_per_image.softmax(dim=-1)) # 图文相似概率: [[1.2252e-03, 5.2874e-02, 6.7116e-04, 9.4523e-01]]
