from PIL import Image
import numpy as np
from torchvision import transforms
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype

import cn_clip.clip as clip
from cn_clip.clip.utils import image_transform

triton_client = httpclient.InferenceServerClient(url="localhost:8000")

text = torch.vstack([clip.tokenize(["杰尼龟"])])
text_input = httpclient.InferInput("text", text.shape, datatype="INT32")
text_input.

image = image_transform(Image.open(pokemon.jpeg)).unsqueeze(0)
image_input  = httpclient.InferInput("image", image.shape, datatype="FP32")
image_input.set_data_from_numpy(image, binary_data=True)

test_output = httpclient.InferRequestedOutput("output", binary_data=True, class_count=1000)

# Querying the server
results = triton_client.infer(model_name="resnet50", inputs=[test_input], outputs=[test_output])
test_output_fin = results.as_numpy('output')
