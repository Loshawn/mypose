import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
import onnxruntime
import time

example_images = [
    "flagged/gradio/example/0814x4.png",
    "flagged/gradio/example/0826x4.png",
    "flagged/gradio/example/0834x4.png",
    "flagged/gradio/example/0837x4.png",
    "flagged/gradio/example/0840x4.png",
    "x2.png"
]


def Super_resolution_img_onnx(img):
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)

    time0 = time.time()
    result = session.run([output_name], {input_name: img})
    print(time.time() - time0)

    output = result[0]
    output_img = output.squeeze(0)
    output_img = output_img.transpose((1, 2, 0))
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)
    return output_img


def Super_resolution_large(img, block_size=1024, extend=24, device="cuda:1"):
    block_img_size = block_size - extend
    H, W = img.shape[:2]
    n_h = (H + block_img_size - 1) // block_img_size
    n_w = (W + block_img_size - 1) // block_img_size

    output = np.zeros((H * 4, W * 4, 3))

    for h in range(n_h):
        for w in range(n_w):
            if h != n_h - 1:
                h_start = h * block_img_size
                h_end = h_start + block_size

                h_start_block = 0
                h_end_block = block_img_size * 4

                h_start_sr = h_start * 4
                h_end_sr = h_start_sr + block_img_size * 4

            else:
                h_start = H - 1 - block_size
                h_end = H - 1

                h_start_block = -1 - block_img_size * 4
                h_end_block = -1

                h_start_sr = 4 * H - 1 - block_img_size * 4
                h_end_sr = 4 * H - 1

            if w != n_w - 1:
                w_start = w * block_img_size
                w_end = w_start + block_size

                w_start_block = 0
                w_end_block = block_img_size * 4

                w_start_sr = w_start * 4
                w_end_sr = w_start_sr + block_img_size * 4

            else:
                w_start = W - 1 - block_size
                w_end = W - 1

                w_start_block = -1 - block_img_size * 4
                w_end_block = -1

                w_start_sr = 4 * W - 1 - block_img_size * 4
                w_end_sr = 4 * W - 1

            img_block = img[h_start:h_end, w_start:w_end, :]

            out_block = Super_resolution_img_onnx(img_block)[
                h_start_block:h_end_block, w_start_block:w_end_block, :]

            cv2.imwrite(f"flagged/gradio/result/sr{h}_{w}.png",
                        cv2.cvtColor(out_block, cv2.COLOR_RGB2BGR))
            output[h_start_sr:h_end_sr, w_start_sr:w_end_sr, :] = out_block

    return np.uint8(output[:H * 4 - 1, :W * 4 - 1, :])


def process_image(img):
    img_lr = img
    h, w = img.shape[:2]
    cv2.imwrite("flagged/gradio/result/img_low.png",
                cv2.cvtColor(img_lr, cv2.COLOR_RGB2BGR))

    if h < 1024 or w < 1024:
        img_sr = Super_resolution_img_onnx(img_lr)

    else:
        img_sr = Super_resolution_large(img_lr)

    cv2.imwrite("flagged/gradio/result/img_sr.png",
                cv2.cvtColor(img_sr, cv2.COLOR_RGB2BGR))

    bl = h // 10
    window_lr = img_lr[h // 2 - bl:h // 2 + bl, w // 2 - bl:w // 2 + bl, :]
    window_lr = cv2.resize(window_lr, (bl * 8, bl * 8),
                           interpolation=cv2.INTER_LINEAR)

    # 提取超分辨率窗口
    window_sr = img_sr[4 * h // 2 - bl * 4:4 * h // 2 + bl * 4,
                       4 * w // 2 - bl * 4:4 * w // 2 + bl * 4, :]

    window = np.hstack((window_lr, window_sr))
    window[:, bl * 8 - 1:bl * 8 + 2, :] = (0, 0, 0)
    cv2.imwrite("flagged/gradio/result/img_window.png",
                cv2.cvtColor(window, cv2.COLOR_RGB2BGR))

    session.set_providers([("CUDAExecutionProvider", {
        "cudnn_conv_algo_search": "DEFAULT",
        'device_id': 1,
    })])

    print('-------------------------------')

    return "flagged/gradio/result/img_sr.png", "flagged/gradio/result/img_window.png"


if __name__ == "__main__":
    session = onnxruntime.InferenceSession(
        '/home/zhanghanbo/projects/RCAN/flagged/RCAN_x4.onnx',
        providers=[("CUDAExecutionProvider", {
            "cudnn_conv_algo_search": "DEFAULT",
            'device_id': 1,
        })])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 预热
    session.run([output_name],
                {input_name: np.zeros((1, 3, 224, 224)).astype(np.float32)})

    # 创建 Gradio 接口
    iface = gr.Interface(
        fn=process_image,  # 处理图像的函数
        inputs=gr.Image(type="numpy"),  # 输入组件
        outputs=[
            gr.Image(type="numpy", label="SR-x4"),
            gr.Image(type="numpy", label="Contrast")
        ],
        title="图像超分辨率",
        description="还在为看动漫分辨率发愁吗? 试试图像超分辨率, 1080P 变 8K !!",
        examples=example_images)

    # 启动接口
    iface.launch(share=False)
