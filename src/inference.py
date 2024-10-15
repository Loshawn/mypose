import models
from models.utils.keypoints_from_heatmaps import keypoints_from_heatmaps
from utils import parameter, util
import torch
import cv2
import numpy as np

args = util.init_config(parameter.args)

model = models.build_model(args)
model = models.load_model(model,"exp/ViTPose/log_2024-10-15-20-02/model/model_2.pth",device="cuda:7")


img = cv2.imread("test/000000000885.jpg")
img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (192,256))
org_w, org_h = img.shape[:2]

print(img.shape)
img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to("cuda:7")
print(img_tensor.shape)
print("Begin inference ...")
heatmaps = model(img_tensor).detach().cpu().numpy()
print("Inference done.")
points, prob = keypoints_from_heatmaps(heatmaps=heatmaps, center=np.array([[org_w//2, org_h//2]]), scale=np.array([[org_w, org_h]]),
                                        unbiased=True, use_udp=True)
points = np.concatenate([points[:, :, ::-1], prob], axis=2)

print(points)
