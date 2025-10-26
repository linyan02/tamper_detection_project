# 图像篡改检测学习研究项目

## 项目简介
短周期（2周）学习研究项目，基于PSCC-Net合成数据集，实现图像篡改检测与基础定位功能，支持ResNet18与PSCC-Net模型对比。

## 环境配置
1. 克隆项目：`git clone git@github.com:linyan02/tamper_detection_project.git`
2. 安装依赖：`pip install -r requirements.txt`
3. 准备资源：将数据集与预训练权重放入`resources/`目录（参考`config/paths.py`路径）

## 快速启动
### 1. 训练模型
```bash
# 训练ResNet18（基础模型）
python train.py --model resnet18 --epochs 15

# 训练PSCC-Net（进阶模型）
python train.py --model pscc_net --epochs 20