import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# ==================== 配置参数 ====================
# 原始数据集根目录（替换为你的PSCC-Net原始数据集路径）
RAW_DATA_ROOT = Path("pscc_net_dataset")
# 目标数据集保存路径（项目的resources/dataset）
TARGET_DATA_ROOT = Path("dataset")
# 总筛选图片数量（1000张）
TOTAL_IMAGES = 1000
# 真实图（Pristine）占比（建议40%，与篡改图比例均衡）
PRISTINE_RATIO = 0.4
# 训练集占比（80%，验证集20%）
TRAIN_RATIO = 0.8
# 支持的图像格式
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

def get_image_paths(raw_dir, is_pristine=True):
    """获取原始目录下所有图像路径（支持递归查找）"""
    image_paths = []
    # 真实图路径（假设原始数据集Pristine目录结构：raw_dir/Pristine/xxx.jpg）
    if is_pristine:
        search_dir = raw_dir / "authentic/train2014resize_256/"
        if not search_dir.exists():
            raise FileNotFoundError(f"真实图目录不存在：{search_dir}")
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(search_dir.glob(f"*{ext}"))
    # 篡改图路径（合并所有篡改类型：CopyMove/Splicing/Removal等）
    else:
        tamper_types = ["copymove/fake/", "splice/fake/", "removal/fake/"]  # 按实际篡改类型修改
        for tamper_type in tamper_types:
            search_dir = raw_dir / tamper_type
            if not search_dir.exists():
                print(f"警告：篡改类型目录不存在，跳过：{search_dir}")
                continue
            for ext in IMAGE_EXTENSIONS:
                image_paths.extend(search_dir.glob(f"*{ext}"))
    return list(set(image_paths))  # 去重


def get_mask_path(image_path):
    """根据图像路径获取对应掩码路径（假设掩码与图像同名，存于mask子目录）"""
    # 假设掩码路径规则：原始图像路径 → 同级mask目录下同名文件
    # 例：raw_dir/CopyMove/img1.jpg → raw_dir/CopyMove/mask/img1.png
    mask_dir = image_path.parent.parent / "mask"
    mask_path = mask_dir / f"{image_path.stem}{image_path.suffix}"
    return mask_path if mask_path.exists() else None


def copy_with_mask(image_path, target_image_dir, copy_mask=True):
    """复制图像到目标目录，若为篡改图则同步复制对应掩码"""
    # 复制图像
    target_image_path = target_image_dir / image_path.name
    shutil.copy2(image_path, target_image_path)  # 保留元数据

    # 复制掩码（仅篡改图需要）
    if not copy_mask:
        return

    mask_path = get_mask_path(image_path)
    if mask_path:
        # 目标掩码目录结构：TARGET_DATA_ROOT/masks/[train/val]/Tampered/xxx.png
        target_mask_dir = TARGET_DATA_ROOT / "masks" / target_image_dir.parent.name / target_image_dir.name
        target_mask_dir.mkdir(parents=True, exist_ok=True)
        target_mask_path = target_mask_dir / mask_path.name
        shutil.copy2(mask_path, target_mask_path)
    else:
        print(f"警告：未找到{image_path.name}的掩码文件，跳过掩码复制")


def split_dataset():
    """执行数据集分割主逻辑"""
    # 1. 计算各类别数量
    num_pristine = int(TOTAL_IMAGES * PRISTINE_RATIO)  # 真实图总数
    num_tampered = TOTAL_IMAGES - num_pristine  # 篡改图总数
    num_train_pristine = int(num_pristine * TRAIN_RATIO)  # 训练集真实图
    num_val_pristine = num_pristine - num_train_pristine  # 验证集真实图
    num_train_tampered = int(num_tampered * TRAIN_RATIO)  # 训练集篡改图
    num_val_tampered = num_tampered - num_train_tampered  # 验证集篡改图

    print(f"===== 数据集分割配置 =====")
    print(f"总图像数：{TOTAL_IMAGES}（真实图：{num_pristine}，篡改图：{num_tampered}）")
    print(f"训练集：{int(TOTAL_IMAGES * TRAIN_RATIO)}（真实图：{num_train_pristine}，篡改图：{num_train_tampered}）")
    print(f"验证集：{int(TOTAL_IMAGES * (1 - TRAIN_RATIO))}（真实图：{num_val_pristine}，篡改图：{num_val_tampered}）")
    print(f"目标路径：{TARGET_DATA_ROOT.resolve()}")

    # 2. 创建目标目录结构
    dirs = {
        "train_pristine": TARGET_DATA_ROOT / "train" / "Pristine",
        "val_pristine": TARGET_DATA_ROOT / "val" / "Pristine",
        "train_tampered": TARGET_DATA_ROOT / "train" / "Tampered",
        "val_tampered": TARGET_DATA_ROOT / "val" / "Tampered"
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # 3. 筛选并复制真实图（Pristine）
    print("\n===== 处理真实图 =====")
    pristine_paths = get_image_paths(RAW_DATA_ROOT, is_pristine=True)
    if len(pristine_paths) < num_pristine:
        raise ValueError(f"真实图数量不足：现有{len(pristine_paths)}张，需{num_pristine}张")

    # 随机选择训练集和验证集真实图
    random.shuffle(pristine_paths)
    train_pristine = pristine_paths[:num_train_pristine]
    val_pristine = pristine_paths[num_train_pristine:num_train_pristine + num_val_pristine]

    # 复制到目标目录（真实图无掩码，copy_mask=False）
    for img_path in tqdm(train_pristine, desc="复制训练集真实图"):
        copy_with_mask(img_path, dirs["train_pristine"], copy_mask=False)
    for img_path in tqdm(val_pristine, desc="复制验证集真实图"):
        copy_with_mask(img_path, dirs["val_pristine"], copy_mask=False)

    # 4. 筛选并复制篡改图（Tampered）
    print("\n===== 处理篡改图 =====")
    tampered_paths = get_image_paths(RAW_DATA_ROOT, is_pristine=False)
    if len(tampered_paths) < num_tampered:
        raise ValueError(f"篡改图数量不足：现有{len(tampered_paths)}张，需{num_tampered}张")

    # 随机选择训练集和验证集篡改图
    random.shuffle(tampered_paths)
    train_tampered = tampered_paths[:num_train_tampered]
    val_tampered = tampered_paths[num_train_tampered:num_train_tampered + num_val_tampered]

    # 复制到目标目录（篡改图需同步复制掩码，copy_mask=True）
    for img_path in tqdm(train_tampered, desc="复制训练集篡改图（含掩码）"):
        copy_with_mask(img_path, dirs["train_tampered"], copy_mask=True)
    for img_path in tqdm(val_tampered, desc="复制验证集篡改图（含掩码）"):
        copy_with_mask(img_path, dirs["val_tampered"], copy_mask=True)

    print("\n===== 分割完成 =====")
    print(f"训练集真实图：{len(train_pristine)}张 → {dirs['train_pristine']}")
    print(f"验证集真实图：{len(val_pristine)}张 → {dirs['val_pristine']}")
    print(f"训练集篡改图：{len(train_tampered)}张 → {dirs['train_tampered']}")
    print(f"验证集篡改图：{len(val_tampered)}张 → {dirs['val_tampered']}")
    print(f"掩码文件路径：{TARGET_DATA_ROOT / 'masks'}")


if __name__ == "__main__":
    # 确保随机种子固定，结果可复现
    random.seed(42)
    split_dataset()