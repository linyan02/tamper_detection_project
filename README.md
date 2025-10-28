# 图像篡改检测学习研究项目

## 项目简介
短周期（2周）学习研究项目，基于PSCC-Net合成数据集，实现图像篡改检测与基础定位功能，支持ResNet18与PSCC-Net模型对比。

## 系统要求
- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)
- Git

## 快速开始

### 1. 克隆项目代码

使用 Git 克隆项目到本地：

```bash
# 使用 SSH (推荐)
git clone git@github.com:T-Curiosity/tamper_detection_project.git

# 或使用 HTTPS
git clone https://github.com/T-Curiosity/tamper_detection_project.git

# 进入项目目录
cd tamper_detection_project
```

### 2. 环境安装

项目提供两种安装方式，可根据需要选择其一：

#### 方式一：使用 uv 安装（推荐，更快速）

[uv](https://github.com/astral-sh/uv) 是一个极快的 Python 包管理器和项目管理工具。

**步骤 1: 安装 uv**

Windows (PowerShell):
```powershell
# 使用 PowerShell 安装
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Linux/macOS:
```bash
# 使用 curl 安装
curl -LsSf https://astral.sh/uv/install.sh | sh
```

或使用 pip 安装:
```bash
pip install uv
```

**步骤 2: 使用 uv 创建虚拟环境并安装依赖**

```bash
# 根据你的硬件配置，调整 pyproject.toml 中的 PyTorch 安装源：
# - GPU (CUDA 12.6): 保持默认配置
# - GPU (其他 CUDA 版本): 将所有 "126" 替换为对应版本号（如 "128"）
# - CPU: 删除 [tool.uv.sources] 和 [[tool.uv.index]] 部分
# 详细说明请查看 pyproject.toml 文件中的注释

# 安装依赖（会自动创建虚拟环境 .venv）
uv sync
```

#### 方式二：使用 pip 安装（传统方式）

**步骤 1: 创建虚拟环境（推荐）**

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate
```

**步骤 2: 安装项目依赖**

```bash
# 安装所有依赖包
pip install -r requirements.txt

# 如果使用 GPU，确保安装正确的 PyTorch 版本
# 访问 https://pytorch.org 获取适合您系统的安装命令
```

### 3. 准备数据和预训练权重

将数据集与预训练权重放入 `resources/` 目录：

```
resources/
├── dataset/
│   ├── train/
│   │   ├── Pristine/    # 原始图像
│   │   └── Tampered/    # 篡改图像
│   ├── val/
│   │   ├── Pristine/
│   │   └── Tampered/
│   └── masks/
│       ├── train/
│       │   └── Tampered/  # 训练集篡改掩码
│       └── val/
│           └── Tampered/  # 验证集篡改掩码
└── pretrained_weights/
    └── resnet18.pth      # ResNet18 预训练权重
```

**注意**: 详细的路径配置请参考 `config/paths.py` 文件。

### 4. 训练模型

```bash
# 训练ResNet18（基础模型）
python train.py --model resnet18 --epochs 15

# 训练PSCC-Net（进阶模型）
python train.py --model pscc_net --epochs 5

# 训练参数说明：
# --model: 选择模型类型 (resnet18 或 pscc_net)
# --epochs: 训练轮数
```

训练过程中会自动：
- 保存最佳模型到 `outputs/models/`
- 记录训练日志到 `outputs/logs/`
- 在验证集上评估模型性能

### 5. 启动 Web 界面

项目提供了基于 Gradio 的 Web 交互界面：

```bash
uv run run.py
```

启动后，在浏览器中打开显示的本地地址（通常是 `http://127.0.0.1:7860`），即可：
- 上传图像进行篡改检测
- 可视化检测结果
- 对比不同模型的性能

## 项目结构说明

```
tamper_detection_project/
├── config/              # 配置文件
│   ├── params.py       # 训练参数配置
│   └── paths.py        # 路径配置
├── data/               # 数据处理模块
│   ├── dataset.py      # 数据集定义
│   └── preprocess.py   # 数据预处理
├── models/             # 模型定义
│   ├── resnet18.py     # ResNet18 模型
│   └── pscc_net.py     # PSCC-Net 模型
├── train/              # 训练脚本
│   ├── train_resnet.py # ResNet18 训练
│   └── train_pscc.py   # PSCC-Net 训练
├── inference/          # 推理模块
│   ├── model_loader.py # 模型加载
│   └── predictor.py    # 预测器
├── frontend/           # Web 前端
│   └── app.py          # Gradio 应用
├── utils/              # 工具函数
│   ├── logger.py       # 日志工具
│   └── metrics.py      # 评估指标
├── outputs/            # 输出目录
│   ├── models/         # 训练好的模型
│   ├── logs/           # 训练日志
│   └── visualizations/ # 可视化结果
├── resources/          # 资源文件
│   ├── dataset/        # 数据集
│   └── pretrained_weights/  # 预训练权重
├── tests/              # 测试文件
├── main.py             # 主程序入口
├── train.py            # 训练入口
├── run.py              # 推理入口
└── requirements.txt    # 依赖列表
```

## 常见问题

### 1. 虚拟环境激活失败

**Windows PowerShell 报错：无法加载文件，因为在此系统上禁止运行脚本**

解决方案：
```powershell
# 以管理员身份运行 PowerShell，执行：
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. CUDA 相关错误

如果遇到 CUDA 不可用或版本不匹配的问题：

```bash
# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回 False，请安装对应 CUDA 版本的 PyTorch
# 访问 https://pytorch.org 获取正确的安装命令
```

### 3. 依赖安装失败

如果某些包安装失败，可以尝试：

```bash
# 使用国内镜像源（加速下载）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用 uv（通常更快）
uv pip install -r requirements.txt
```

## 技术栈

- **深度学习框架**: PyTorch
- **图像处理**: OpenCV, PIL
- **Web 界面**: Gradio
- **数据处理**: NumPy, Pandas
- **可视化**: Matplotlib
- **日志**: Python logging

## 参考资料

- [PSCC-Net 论文](相关论文链接)
- [ResNet 论文](https://arxiv.org/abs/1512.03385)
- [项目文档](doc/)

## 许可证

本项目仅用于学习研究目的。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请通过 GitHub Issues 联系。
