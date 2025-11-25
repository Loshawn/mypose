import os
import json
import numpy as np
from xtcocotools.coco import COCO
import cv2
from tqdm import tqdm

class WholebodyFootKeypointsExtractor:
    def __init__(self, root, image_set, output_dir):
        """
        初始化脚部关键点提取器

        :param root: 数据集根目录
        :param image_set: 数据集划分（例如 train2017 或 test2017）
        """
        self.root = root
        self.output_dir = output_dir
        self.image_set = image_set
        self.coco = COCO(self._get_ann_file_keypoint())
        self.num_joints = 133  # COCO WholeBody中共有133个关键点
        self.foot_joint_ids = [17, 18, 19, 20, 21, 22]  # 脚部关键点的索引，17-22

    def _get_ann_file_keypoint(self):
        """
        返回COCO注释文件路径

        :return: COCO注释文件路径
        """
        prefix = "coco_wholebody" if "test" not in self.image_set else "image_info"
        return os.path.join(
            self.root, "annotations", prefix + "_" +
            str(self.image_set).replace("2017", "") + "_v1.0" + ".json"
        )

    def image_path_from_index(self, index):
        file_name = "%012d.jpg" % index
        prefix = "test2017" if "test" in self.image_set else self.image_set
        data_name = prefix
        image_path = os.path.join(self.root, data_name, file_name)
        return image_path

    def extract_foot_keypoints(self):
        """
        从COCO数据集中提取所有图像的脚部关键点

        :return: 脚部关键点的列表
        """
        foot_info = []
        image_ids = self.coco.getImgIds()
        for img_id in tqdm(image_ids):
            img_anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=False))
            total_foot_kpts = []
            for ann in img_anns:
                if ann['category_id'] == 1 and ann["area"] > 0 and sum(ann["foot_kpts"]) > 0:
                    total_foot_kpts.append(ann["foot_kpts"])


            if total_foot_kpts == []:
                continue
            else:
                info = {
                    "img_path" : self.image_path_from_index(img_id),
                    "img_name" : f"{img_id:012d}.jpg",
                    "foot_kpts" : total_foot_kpts
                }

                foot_info.append(info)

        with open(f'{self.output_dir}/data.json', 'w', encoding='utf-8') as f:
            json.dump(foot_info, f, ensure_ascii=False, indent=4)

        self.foot_info = foot_info

    def draw_kpts(self):
        print("Drawing ...")
        for info in tqdm(self.foot_info):
            img = cv2.imread(info["img_path"])

            for kpts in info["foot_kpts"]:
                kpt = np.array(kpts).reshape(-1, 3).astype(int)
                for x, y, v in kpt:
                    if v > 0:
                        cv2.circle(img, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
            cv2.imwrite(f"{self.output_dir}/{info['img_name']}", img)
            
    def forward(self):
        self.extract_foot_keypoints()
        self.draw_kpts()


dataset_root = '/home/zhhb/Projects/mmpose/data/coco'
image_set = 'train2017'
output_dir = 'data/foot'

foot_extractor = WholebodyFootKeypointsExtractor(dataset_root, image_set, output_dir)

foot_extractor.forward()

