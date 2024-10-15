import torch
import onnxruntime
import numpy as np
import cv2
from time import time

session = onnxruntime.InferenceSession(
    '/home/zhanghanbo/projects/RCAN/flagged/RCAN_x4_l-dy.onnx',
    providers=[("CUDAExecutionProvider", {
        "cudnn_conv_algo_search": "DEFAULT",
        'device_id': 1
    })])

# 读取并预处理图像
img = cv2.imread(
    "/home/zhanghanbo/projects/RCAN/DIV2K/DIV2K_train_LR_bicubic/X4/0001x4.png"
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def inference(img):
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)

    t0 = time()
    result = session.run([output_name], {input_name: img})
    print(time() - t0)
    output = result[0]
    output_img = output.squeeze(0)
    output_img = output_img.transpose((1, 2, 0))
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)

    cv2.imwrite("output_image.png", output_img)


inference(img)
inference(img)
img = np.resize(img, (1000, 1000, 3))
inference(img)
inference(img)