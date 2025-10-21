import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 资源路径
RESOURCES_DIR = PROJECT_ROOT / "resources"
DATASET_DIR = RESOURCES_DIR / "dataset"
TRAIN_DATA_DIR = DATASET_DIR / "train"
VAL_DATA_DIR = DATASET_DIR / "val"
PRETRAINED_DIR = RESOURCES_DIR / "pretrained_weights"

# 输出路径
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_SAVE_DIR = OUTPUTS_DIR / "models"
LOG_DIR = OUTPUTS_DIR / "logs"
VIS_DIR = OUTPUTS_DIR / "visualizations"

# 创建目录
for dir_path in [MODEL_SAVE_DIR, LOG_DIR, VIS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)