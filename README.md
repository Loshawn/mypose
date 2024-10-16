# 2D Human Pose Estimation

2D人体姿态估计项目

## 目录
- [项目简介](#项目简介)
- [更新日志](#更新日志)
- [环境要求](#环境要求)
- [安装](#安装)
- [使用方法](#使用方法)
- [训练模型](#训练模型)
  - [数据集构建](#数据集构建)
  - [开始训练](#开始训练)
- [模型评估](#模型评估)


## 项目简介
待完善

## 更新日志
- 2024.10.16  
    - 添加学习率衰减和学习率预热
    - 实现模型推理

## 环境要求

参考项目环境：

- Python 3.9
- PyTorch 2.4.0

## 安装

克隆本项目

```bash
git clone https://github.com/Loshawn/mypose.git
cd mypose
```

## 使用方法

通过以下命令，您可以轻松对图像进行图像关键点预测（`inference.sh`）：

```bash
python src/inference.py --image_path test/000000000113.jpg --model ViTPose --model_size s --pretrain model.pth
```

- --image_path 输入图像路径。
- --model 训练模型名称 `[ViTPose]`
- --model_size 模型规模 `[s, b, l, h]`
- --pretrain 权重文件路径。

处理后的图片会自动保存到 `test/` 文件夹。

## 训练模型

### 数据集构建
项目使用 `coco-wholebody` 数据集进行训练。  
您可以下载 [coco](https://cocodataset.org/#download) 数据集，以及用于 [训练](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/js20_connect_hku_hk/EfE4vxMce2NNiEfJUySLTmwBS5Ay2rbp5-7sHxN6BoldFw)/[验证](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/js20_connect_hku_hk/EQuxJ51ZSXVPv6EeGnLT65YBvkaVQLAMRYW6pnk6sobfPA?e=jjV2u4) 的标注，并按照如下格式构建数据集。

```
mypose/
├── dataset/               # 数据存放位置
│   └── cocowholebody/
│       ├── annotations/
│       │   ├── coco_wholebody_train_v1.0.json
│       │   └── coco_wholebody_val_v1.0.json
│       ├── train2017/
│       └── val2017/
├── config/
│   └── ...
└── src/
    └── ...
```


### 开始训练
如果您想重新训练 ViTPose 模型，使用以下命令 `train.sh`：  

- 单卡训练 (`train.sh`)
```bash
export CUDA_VISIBLE_DEVICES=0 
python src/train.py --save ViTPose --model ViTPose --model_size s
```
- 多卡训练 (`train_dist.sh`)
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 
torchrun --nproc_per_node=4 src/train.py --save ViTPose --model ViTPose --model_size s --distributed 
```

参数解释：
- --save 保存文件夹名
- --model 训练模型名称 `[ViTPose]`
- --model_size 模型规模 `[s, b, l, h]`
- --resume 断点恢复，中断的训练任务 (如 `exp/ViTPose/log_2024-10-15-19-18`)


## 模型评估
尚未实现