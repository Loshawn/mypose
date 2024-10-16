import sys
import os
from time import time

dir_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(dir_root)

import models
from models.utils.keypoints_from_heatmaps import keypoints_from_heatmaps

from src.utils import parameter, util
from torchvision import transforms
import cv2
import numpy as np


def inference(model, image_path, confidence_threshold=0.5):
    print(f"Original image : {os.path.abspath(image_path)}\n")
    image_names = os.path.splitext(os.path.basename(image_path))
    result_path = os.path.abspath(
        os.path.join(os.path.dirname(image_path),
                     f"{image_names[0]}_result{image_names[1]}"))
    img_ori = cv2.imread(image_path)
    org_h, org_w = img_ori.shape[:2]

    img = cv2.resize(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB), (192, 256))

    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),  # 标准化处理
    ])
    img = transform(img).unsqueeze(0)
    print("Begin inference ...")
    t0 = time()
    heatmaps = model(img).detach().cpu().numpy()
    t1 = time() - t0
    print("Inference done.")
    print(f"Inference time : {t1} s\n")

    points, prob = keypoints_from_heatmaps(
        heatmaps=heatmaps,
        center=np.array([[org_w // 2, org_h // 2]]),
        scale=np.array([[org_w / 200, org_h / 200]]),
        unbiased=True,
        use_udp=True)

    points = np.concatenate([points[:, :, ::-1], prob], axis=2)
    circle_size = max(1, min(img_ori.shape[:2]) // 150)

    image = img_ori.copy()

    for point in points:
        for pt in point:
            if pt[2] > confidence_threshold:
                image = cv2.circle(image, (int(pt[1]), int(pt[0])),
                                   circle_size, (255, 0, 0), -1)

    cv2.imwrite(result_path, image)
    print(f"Image with keypoints is saved as '{result_path}'")


if __name__ == "__main__":
    # r"C:\Users\Loshawn\Documents\PyFiles\DeepLearning\mypose\exp\model_210.pth"
    args = util.init_config(parameter.args)
    model = models.build_model(args)
    model = models.load_model(
        model,
        args.pretrain,
        device="cuda")
    model.eval()
    inference(model, args.image_path)
