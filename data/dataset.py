import os
import cv2
import torch
from torch.utils.data import Dataset
from config.paths import TRAIN_DATA_DIR, VAL_DATA_DIR
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, torch.tensor(label, dtype=torch.float32)