import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import cv2
import numpy as np
from tqdm import tqdm
from config.paths import MODEL_SAVE_DIR, LOG_DIR, TRAIN_DATA_DIR, VAL_DATA_DIR
from config.params import *
from data.dataset import TamperDataset  # 复用数据集类（需扩展掩码加载）
from models.pscc_net import PSCCNet
from utils.logger import get_logger
from utils.metrics import iou_score, accuracy_score_new

# 初始化日志
logger = get_logger(LOG_DIR / "pscc_train.log")


# 扩展数据集类以支持掩码加载（PSCC-Net需要篡改区域掩码用于定位训练）
class PSCCDataset(TamperDataset):
    def __init__(self, is_train=True, transform=None):
        super().__init__(is_train=is_train, transform=transform)
        # 假设掩码与图像同名，存放在同级masks目录（如train/masks/Tampered/xxx.png）
        self.mask_dir = self.data_dir.parent / "masks" / ("train" if is_train else "val")

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # 加载图像（复用父类逻辑）
        image = super()._load_image(img_path)  # 需在TamperDataset中实现_load_image方法
        # 加载对应掩码（仅篡改图像有掩码，真实图像掩码为全0）
        mask = self._load_mask(img_path, label)
        # 应用预处理（图像和掩码需同步增强）
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image, torch.tensor(label, dtype=torch.float32), mask.float().unsqueeze(0)  # 掩码增加通道维度

    def _load_mask(self, img_path, label):
        """加载篡改区域掩码（0=背景，1=篡改区域）"""
        if label == 0:  # 真实图像，掩码全为0
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        # 篡改图像：稳健构建掩码路径（兼容 Windows/Unix 路径）
        from pathlib import Path
        import re

        p = Path(img_path)
        stem = p.stem  # 文件名不含扩展

        # 尝试不同的掩码文件夹名（可能为 Tampered 或 tampered）
        mask_subdirs = [self.mask_dir / "Tampered", self.mask_dir / "tampered"]

        # 首先尝试常见命名：<stem>.png, <stem>_mask.png
        candidates = []
        for d in mask_subdirs:
            candidates.append(d / f"{stem}.png")
            candidates.append(d / f"{stem}_mask.png")

        mask = None
        for cand in candidates:
            if cand.exists():
                mask = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    break

        # 如果没有直接匹配，尝试基于数字 id 的匹配（从文件名提取数字序列）
        if mask is None:
            digits = re.findall(r"\d+", stem)
            if digits:
                # 构建所有现有掩码 stems索引以便快速查找
                all_mask_stems = []
                mask_paths = []
                for d in mask_subdirs:
                    if d.exists():
                        for mf in d.iterdir():
                            if mf.is_file() and mf.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                                all_mask_stems.append(mf.stem)
                                mask_paths.append(mf)

                # 查找包含数字串的掩码名
                found = False
                for num in digits:
                    for ms, mp in zip(all_mask_stems, mask_paths):
                        if num in ms:
                            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                            if mask is not None:
                                found = True
                                break
                    if found:
                        break

        # 最后尝试模糊匹配（掩码名包含图像名或反之）
        if mask is None:
            for d in mask_subdirs:
                if d.exists():
                    for mf in d.iterdir():
                        if mf.is_file() and mf.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                            mstem = mf.stem
                            if mstem in stem or stem in mstem:
                                mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
                                if mask is not None:
                                    break
                    if mask is not None:
                        break

        if mask is None:
            logger.warning(f"掩码不存在或无法读取，使用空掩码。Checked candidates: {candidates}")
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        # 若读取成功，再进行 resize 和二值化
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
        return (mask > 127).astype(np.uint8)  # 二值化（0或1）


# 定义PSCC-Net专用损失函数（Dice损失+BCELoss，适合分割任务）
class DiceBCELoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Dice损失（衡量掩码重叠度）
        intersection = (pred * target).sum()
        dice = 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        # BCE损失（衡量像素级分类）
        bce = torch.nn.BCELoss()(pred, target)
        return dice + bce


def evaluate(model, val_loader, device):
    """评估模型在验证集上的性能（IoU和分类准确率）"""
    model.eval()
    total_iou = 0.0
    total_tampered = 0
    total_acc = 0.0
    with torch.no_grad():
        for images, labels, masks in val_loader:
            images, labels, masks = images.to(device), labels.to(device), masks.to(device)
            outputs = model(images)  # logits
            probs = torch.sigmoid(outputs)  # 转为概率
            pred_mask = (probs > THRESHOLD).float()

            # per-sample IoU，仅对 label==1 的图片计算并累加
            for i in range(images.size(0)):
                if labels[i].item() == 1:
                    mt = masks[i].cpu().numpy()
                    mp = pred_mask[i].cpu().numpy()
                    iou = iou_score(mt, mp)
                    total_iou += iou
                    total_tampered += 1

            # 计算分类准确率（根据掩码是否存在判断是否为篡改图像）
            pred_labels = (probs.view(probs.size(0), -1).sum(dim=1) > 0).float()
            total_acc += accuracy_score_new(labels.cpu().numpy(), pred_labels.cpu().numpy()) * images.size(0)

    mean_iou = (total_iou / total_tampered) if total_tampered > 0 else 0.0
    return {
        "iou": mean_iou,
        "acc": total_acc / len(val_loader.dataset)
    }

class DiceBCELoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits, target):
        # pred_logits: 模型原始输出（logits）
        pred_probs = torch.sigmoid(pred_logits)
        # Dice loss 使用概率
        intersection = (pred_probs * target).sum()
        dice = 1 - (2. * intersection + self.smooth) / (pred_probs.sum() + target.sum() + self.smooth)
        # 对数值稳定的 BCE 使用 BCEWithLogitsLoss（直接接受 logits）
        bce = torch.nn.BCEWithLogitsLoss()(pred_logits, target)
        return dice + bce
    
def train_pscc(epochs=None, quick_mode=False):
    """训练PSCC-Net模型（支持快速模式，适配短周期项目）"""
    epochs = epochs or (10 if quick_mode else PSCC_EPOCHS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"开始训练PSCC-Net（设备：{device}，轮次：{epochs}）")

    # 加载数据集（快速模式使用部分数据加速训练）
    train_dataset = PSCCDataset(is_train=True)
    val_dataset = PSCCDataset(is_train=False)

    if quick_mode:
        # 仅使用50%训练数据和30%验证数据
        train_idx = np.random.choice(len(train_dataset), len(train_dataset) // 2, replace=False)
        val_idx = np.random.choice(len(val_dataset), len(val_dataset) // 3, replace=False)
        train_dataset = Subset(train_dataset, train_idx)
        val_dataset = Subset(val_dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4 if quick_mode else BATCH_SIZE,
        shuffle=True,
        num_workers=1 if quick_mode else 2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=1
    )

    # 初始化模型、损失函数、优化器
    model = PSCCNet().to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=2e-4 if quick_mode else LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # 训练循环
    best_iou = 0.0  # 以IoU作为最优模型指标
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, labels, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # 输出：(batch_size, 1, 512, 512)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        # 每轮训练后验证
        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {train_loss / len(train_dataset):.4f}, "
            f"Val IoU: {val_metrics['iou']:.4f}, "
            f"Val Acc: {val_metrics['acc']:.4f}"
        )

        # 保存最优模型（按IoU）
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_path = MODEL_SAVE_DIR / "pscc_net_best.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"✅ 最优模型已保存至：{save_path}(IoU:{best_iou:.4f})")

        scheduler.step()

    logger.info(f"训练完成! 最佳验证IoU: {best_iou:.4f}")