import os
import cv2
import torch
from torch.utils.data import Dataset
from config.paths import TRAIN_DATA_DIR, VAL_DATA_DIR
from config.params import IMAGE_SIZE
from data.preprocess import get_train_transform, get_val_transform

class TamperDataset(Dataset):
    def __init__(self, is_train=True, transform=None):
        self.is_train = is_train
        self.data_dir = TRAIN_DATA_DIR if is_train else VAL_DATA_DIR
        self.transform = transform or (get_train_transform() if is_train else get_val_transform())
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        # 加载真实图像
        pristine_dir = self.data_dir / "Pristine"
        for fname in os.listdir(pristine_dir):
            if fname.lower().endswith(('.jpg', '.png')):
                samples.append((str(pristine_dir / fname), 0))
        # 加载篡改图像
        tamper_dir = self.data_dir / "Tampered"
        for fname in os.listdir(tamper_dir):
            if fname.lower().endswith(('.jpg', '.png')):
                samples.append((str(tamper_dir / fname), 1))
        return samples

    def _load_image(self, img_path):
        """加载图像并返回 RGB 的 numpy array，供 transform 使用。

        返回格式：H x W x C (uint8 或 float)，与原有代码兼容（transform 期望 numpy image）。
        """
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")

        # 如果是灰度图，转换为 RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 统一缩放为配置的 IMAGE_SIZE，保证与掩码尺寸一致，避免 albumentations 的 shapes 检查报错
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        return image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # 使用统一的加载函数，子类可以直接调用 super()._load_image
        image = self._load_image(img_path)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, torch.tensor(label, dtype=torch.float32)