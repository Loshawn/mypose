import os
import sys
import pickle
import random

import cv2
import json_tricks as json
import numpy as np
from pycocotools.coco import COCO
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

from .util import fliplr_joints, affine_transform, get_affine_transform

import numpy as np

class WholeBodyDataset(Dataset):

    def __init__(self, args, split):

        super(WholeBodyDataset, self).__init__()

        self.args = args
        # 数据集相关路径和参数
        self.is_train = True if split == "train" else False

        self.scale_prob = args.scale  # 是否使用缩放增强
        self.flip_prob = args.flip  # 翻转概率
        self.rotate_prob = args.rotate  # 旋转概率
        self.rotation_factor = 45  # 最大旋转角度
        self.heatmap_sigma = 3  # 热图的高斯分布sigma值
        self.use_different_joints_weight = True

        # 图片和标注路径
        self.annotation_path = args.data[split]["ann_file"]

        # 图像尺寸和热图尺寸设置
        self.image_size = args.data_cfg["image_size"]
        self.aspect_ratio = args.data_cfg["image_size"][
            0] * 1.0 / args.data_cfg["image_size"][1]  # 图像宽高比
        self.heatmap_size = (int(args.data_cfg["image_size"][0] / 4),
                             int(args.data_cfg["image_size"][1] / 4))  # 热图宽高
        
        self.heatmap_type = 'gaussian'  # 热图类型为高斯分布
        self.pixel_std = 200  # 图像像素标准偏差，用于变换

        # 关节点信息
        self.num_joints = 133  # 关节点数量
        self.num_joints_half_body = 8  # 半身关节点数量

        # 翻转关节点对和身体部位划分
        self.flip_body = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                          [13, 14], [15, 16]]
        self.flip_foot = [[17, 20], [18, 21], [19, 22]]
        self.flip_face = [[23, 39], [24, 38], [25, 37], [26, 36], [27, 35],
                          [28, 34], [29, 33], [30, 32], [40, 49], [41, 48],
                          [42, 47], [43, 46], [44, 45], [54, 58], [55, 57],
                          [59, 68], [60, 67], [61, 66], [62, 65], [63, 70],
                          [64, 69], [71, 77], [72, 76], [73, 75], [78, 82],
                          [79, 81], [83, 87], [84, 86], [88, 90]]
        self.flip_hand = [[91, 112], [92, 113], [93, 114], [94,
                                                            115], [95, 116],
                          [96, 117], [97, 118], [98, 119], [99, 120],
                          [100, 121], [101, 122], [102, 123], [103, 124],
                          [104, 125], [105, 126], [106, 127], [107, 128],
                          [108, 129], [109, 130], [110, 131], [111, 132]]

        self.flip_pairs = self.flip_body + self.flip_foot + self.flip_face + self.flip_hand

        self.upper_body_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 上半身关节点
        self.lower_body_ids = [11, 12, 13, 14, 15, 16]  # 下半身关节点

        # self.joints_weight = np.array([
        #     1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2,
        #     1.5, 1.5
        # ]).reshape((self.num_joints, 1))  # 关节点权重

        self.joints_weight = np.concatenate((
            np.full(7, 1),  # 1~7
            np.full(2, 1.2),  # 8~9
            np.full(2, 1.5),  # 10~11
            np.full(2, 1),  # 12~13
            np.full(2, 1.2),  # 14~15
            np.full(2, 1.5),  # 16~17
            np.full(6, 1.5),  # 18~23
            np.full(68, 1),  # 24~91
            np.full(42, 1.5)  # 92~133
        )).reshape((self.num_joints, 1))

        # 图像转换与归一化
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),  # 标准化处理
        ])

        # 加载COCO数据集
        self.coco = COCO(self.annotation_path)  # 使用COCO API加载标注文件

        # 创建一个注释列表及对应的图像（每个图像可以包含多个检测结果）

        self.data = []  # 初始化数据列表

        # 加载COCO每张图像的注释
        for imgId in tqdm(self.coco.getImgIds(), desc="准备图像和注释 ... "):

            ann_ids = self.coco.getAnnIds(imgIds=imgId,
                                          iscrowd=False)  # 获取注释ID
            img = self.coco.loadImgs(imgId)[0]  # 加载图像

            objs = self.coco.loadAnns(ann_ids)  # 加载注释对象

            # 清洗边界框
            valid_objs = []  # 存储有效的对象
            for obj in objs:
                obj["keypoints"] = obj["keypoints"] + obj["foot_kpts"] + obj[
                    "face_kpts"] + obj["lefthand_kpts"] + obj["righthand_kpts"]
                # 跳过非人类对象（这不应该发生）
                if obj['category_id'] != 1:
                    continue

                # 忽略没有关键点注释的对象
                if max(obj['keypoints']) == 0:
                    continue

                # 获取边界框
                x, y, w, h = obj['bbox']
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((img['width'] - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((img['height'] - 1, y1 + np.max((0, h - 1))))

                # 只使用有效的边界框
                if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                    obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                    valid_objs.append(obj)

            objs = valid_objs

            # 对于每个注释，将格式化的注释添加到self.data
            for obj in objs:
                joints = np.zeros((self.num_joints, 2), dtype=np.float32)
                joints_visibility = np.ones((self.num_joints, 2),
                                            dtype=np.float32)

                joints[:, 0] = obj['keypoints'][0::3]  # 提取 x 坐标
                joints[:, 1] = obj['keypoints'][1::3]  # 提取 y 坐标

                # 使用切片和向量化操作来计算可见性
                t_vis = np.clip(obj['keypoints'][2::3], 0,
                                1).astype(int)  # 提取并限制可见性
                joints_visibility[:, 0] = t_vis  # 设置可见性
                joints_visibility[:, 1] = t_vis

                center, scale = self._box2cs(obj['clean_bbox'][:4])

                self.data.append({
                    'imgId': imgId,
                    'annId': obj['id'],
                    'imgPath':
                    f"{args.data[split]['img_prefix']}/{imgId:012d}.jpg",
                    # 'imgPath': os.path.join(self.data_root, self.data_version, 'id_%s.jpg' % imgId),
                    'center': center,
                    'scale': scale,
                    'joints': joints,
                    'joints_visibility': joints_visibility,
                })

        # 完成检查是否需要准备数据 -> 我们不应该准备数据
        print('\nCOCO 数据集加载完成！')

        # 默认值
        self.bbox_thre = 1.0  # 边界框阈值
        self.image_thre = 0.0  # 图像阈值
        self.in_vis_thre = 0.2  # 可见性阈值
        self.nms_thre = 1.0  # 非极大值抑制阈值
        self.oks_thre = 0.9  # 关键点的OKS阈值

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 从数据集中获取索引对应的关节数据
        joints_data = self.data[index].copy()

        # 加载图像
        try:
            image = cv2.imread(joints_data['imgPath'],
                               flags=3)  # 使用 OpenCV 读取图像
            image = cv2.cvtColor(image,
                                 cv2.COLOR_BGR2RGB)  # 将图像从 BGR 转换为 RGB 格式
        except:
            raise ValueError(
                f"无法读取 {joints_data['imgPath']}")  # 如果图像读取失败，则抛出错误

        # 获取关节和可见性信息
        joints = joints_data['joints']
        joints_vis = joints_data['joints_visibility']

        # 获取关节中心和缩放因子
        c = joints_data['center']
        s = joints_data['scale']
        # 获取得分，默认值为1
        score = joints_data['score'] if 'score' in joints_data else 1
        r = 0  # 初始化旋转角度

        # 应用数据增强，适用于训练阶段
        if self.is_train:
            # 获取缩放因子和旋转因子
            sf = self.scale_prob  # 缩放因子
            rf = self.rotation_factor  # 旋转因子

            # 应用随机缩放
            if self.scale_prob:
                s = s * np.clip(random.random() * sf + 1, 1 - sf,
                                1 + sf)  # 在 [1 - sf, 1 + sf] 范围内的随机缩放因子

            # 应用随机旋转
            if self.rotate_prob and random.random() < self.rotate_prob:
                r = np.clip(random.random() * rf, -rf * 2,
                            rf * 2)  # 在 [-2 * rf, 2 * rf] 范围内的随机旋转因子
            else:
                r = 0  # 如果没有旋转，则保持角度为0

            # 应用随机翻转
            if self.flip_prob and random.random() < self.flip_prob:
                image = image[:, ::-1, :]  # 水平翻转图像
                joints, joints_vis = fliplr_joints(joints, joints_vis,
                                                   image.shape[1],
                                                   self.flip_pairs)  # 翻转关节和可见性
                c[0] = image.shape[1] - c[0] - 1  # 更新中心点的x坐标

        # 对关节和图像应用仿射变换
        trans = get_affine_transform(c, s, self.pixel_std, r,
                                     self.image_size)  # 获取仿射变换矩阵
        image = cv2.warpAffine(
            image,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),  # 变换后的图像大小
            flags=cv2.INTER_LINEAR  # 使用线性插值法
        )

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.:  # 检查关节的可见性
                joints[i, 0:2] = affine_transform(joints[i, 0:2],
                                                  trans)  # 应用仿射变换到关节位置

        # 将图像转换为张量并进行归一化处理
        if self.transform is not None:  # 如果定义了转换函数
            image = self.transform(image)

        target, target_weight = self._generate_target(joints,
                                                      joints_vis)  # 生成目标和权重

        # 更新元数据
        joints_data['joints'] = joints  # 更新关节信息
        joints_data['joints_visibility'] = joints_vis  # 更新关节可见性
        joints_data['center'] = c  # 更新中心点
        joints_data['scale'] = s  # 更新缩放因子
        joints_data['rotation'] = r  # 更新旋转角度
        joints_data['score'] = score  # 更新得分

        return image, target.astype(np.float32), target_weight.astype(
            np.float32), joints_data  # 返回处理后的图像、目标、权重和关节数据

        # 私有方法
    def _box2cs(self, box):
        x, y, w, h = box[:4]  # 解包边界框的坐标和尺寸
        return self._xywh2cs(x, y, w, h)  # 转换为中心和尺度

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2, ), dtype=np.float32)  # 初始化中心点
        center[0] = x + w * 0.5  # 计算中心 x 坐标
        center[1] = y + h * 0.5  # 计算中心 y 坐标

        # 根据宽高比调整宽度和高度
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        # 计算尺度
        scale = np.array([w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
                         dtype=np.float32)

        if center[0] != -1:  # 如果中心 x 坐标有效
            scale = scale * 1.25  # 增加尺度

        return center, scale  # 返回中心和尺度

    def _half_body_transform(self, joints, joints_vis):
        upper_joints = []  # 上半身关节
        lower_joints = []  # 下半身关节

        # 根据可见性将关节分类为上半身和下半身
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        # 随机选择上半身或下半身关节
        if random.random() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints if len(
                lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:  # 如果选择的关节少于两个，返回 None
            return None, None

        selected_joints = np.array(selected_joints,
                                   dtype=np.float32)  # 转换为 NumPy 数组
        center = selected_joints.mean(axis=0)[:2]  # 计算中心

        left_top = np.amin(selected_joints, axis=0)  # 计算左上角
        right_bottom = np.amax(selected_joints, axis=0)  # 计算右下角

        w = right_bottom[0] - left_top[0]  # 计算宽度
        h = right_bottom[1] - left_top[1]  # 计算高度

        # 根据宽高比调整宽度和高度
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array([w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
                         dtype=np.float32)

        scale = scale * 1.5  # 增加尺度

        return center, scale  # 返回中心和尺度

    def _generate_target(self, joints, joints_vis):
        """
        :param joints:  [num_joints, 2] 关节坐标
        :param joints_vis: [num_joints, 2] 关节可见性
        :return: target, target_weight(1: visible, 0: invisible) 目标热图和目标权重
        """
        # 初始化目标权重为全1
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]  # 根据可见性更新目标权重

        if self.heatmap_type == 'gaussian':
            # 初始化目标热图
            target = np.zeros(
                (self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32)

            tmp_size = self.heatmap_sigma * 3  # 高斯模糊的临时大小

            # 逐个关节生成热图
            for joint_id in range(self.num_joints):
                # 计算特征图的步幅
                feat_stride = np.asarray(self.image_size) / np.asarray(
                    self.heatmap_size)
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)  # x坐标
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)  # y坐标

                # 确保高斯的任何部分都在边界内
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]  # 左上角
                br = [int(mu_x + tmp_size + 1),
                      int(mu_y + tmp_size + 1)]  # 右下角

                # 检查是否越界
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # 如果越界，更新目标权重并继续
                    target_weight[joint_id] = 0
                    continue

                # 生成高斯分布
                size = 2 * tmp_size + 1  # 高斯核大小
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2  # 高斯中心
                # 高斯没有归一化，我们希望中心值为1
                g = np.exp(-((x - x0)**2 + (y - y0)**2) /
                           (2 * self.heatmap_sigma**2))

                # 可用的高斯范围
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # 图像范围
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]  # 当前关节的权重
                if v > 0.5:  # 如果关节可见
                    # 将生成的高斯分布赋值到目标热图
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        else:
            raise NotImplementedError  # 如果热图类型未实现，则引发错误

        # 如果使用不同的关节权重
        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight,
                                        self.joints_weight)  # 更新目标权重

        return target, target_weight  # 返回目标热图和权重

    def _write_coco_keypoint_results(self, keypoints, res_file):
        """
        将 COCO 格式的关键点结果写入文件
        :param keypoints: 关键点数据
        :param res_file: 结果文件路径
        """
        # 准备数据包
        data_pack = [{
            'cat_id': 1,  # 1 == 'person'
            'cls': 'person',  # 类别
            'ann_type': 'keypoints',  # 注释类型
            'keypoints': keypoints  # 关键点
        }]

        # 获取 COCO 格式的单类关键点结果
        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])

        # 将结果写入文件
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

        # 尝试加载结果文件以检查格式是否正确
        try:
            json.load(open(res_file))
        except Exception:
            # 如果格式不正确，修复文件
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'  # 修正为有效的 JSON 格式
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        """
        处理单类别的 COCO 格式关键点结果
        :param data_pack: 包含关键点和类别信息的数据包
        :return: 类别结果列表
        """
        cat_id = data_pack['cat_id']  # 类别 ID
        keypoints = data_pack['keypoints']  # 关键点数据
        cat_results = []  # 初始化类别结果列表

        # 遍历每个图像的关键点
        for img_kpts in keypoints:
            if len(img_kpts) == 0:  # 如果没有关键点，跳过
                continue

            # 提取关键点数据
            _key_points = np.array(
                [img_kpts[k]['keypoints'] for k in range(len(img_kpts))],
                dtype=np.float32)
            key_points = np.zeros((_key_points.shape[0], self.num_joints * 3),
                                  dtype=np.float32)

            # 将关键点数据整理为所需格式
            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]  # x 坐标
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]  # y 坐标
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # 关键点分数

            # 准备结果
            result = [
                {
                    'image_id': img_kpts[k]['image'],  # 图像 ID
                    'category_id': cat_id,  # 类别 ID
                    'keypoints': list(key_points[k]),  # 关键点列表
                    'score': img_kpts[k]['score'].astype(np.float32),  # 关键点分数
                    'center': list(img_kpts[k]['center']),  # 中心点
                    'scale': list(img_kpts[k]['scale'])  # 缩放
                } for k in range(len(img_kpts))
            ]
            cat_results.extend(result)  # 添加到类别结果列表中

        return cat_results  # 返回类别结果列表
