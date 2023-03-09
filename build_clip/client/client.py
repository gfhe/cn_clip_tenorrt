# from PIL import Image
import numpy as np
# from torchvision import transforms
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype

#import cn_clip.clip as clip
#from cn_clip.clip.utils import image_transform

triton_client = httpclient.InferenceServerClient(url="10.208.62.27:8000")


# 事先处理好的token：clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"], context_length=52) 
text = np.array([[ 101, 3345, 2225, 7991,  102,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0]], dtype=np.int32)
text_input = httpclient.InferInput("text", text.shape, datatype="INT32")
text_input.set_data_from_numpy(text, binary_data=False)

text_output = httpclient.InferRequestedOutput("unnorm_text_features")

# Querying the server
results = triton_client.infer(model_name="clip_vitb16_txt", inputs=[text_input], outputs=[text_output])
test_feature = results.as_numpy('unnorm_text_features')


print(test_feature.shape)
test_feature /= np.linalg.norm(x=test_feature, ord=2,  axis=1, keepdims=True)

print(test_feature)


print(np.sum(test_feature))
