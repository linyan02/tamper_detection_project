import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from config.paths import MODEL_SAVE_DIR, LOG_DIR
from config.params import *
from data.dataset import TamperDataset
from models.resnet18 import ResNet18Tamper
from utils.logger import get_logger
from utils.metrics import accuracy_score

logger = get_logger(LOG_DIR / "resnet_train.log")


def train_resnet(epochs=None):
    epochs = epochs or RESNET_EPOCHS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_dataset = TamperDataset(is_train=True)
    val_dataset = TamperDataset(is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 模型初始化
    model = ResNet18Tamper().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # 训练循环
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_preds.extend((outputs > THRESHOLD).float().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # 验证
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_preds.extend((outputs > THRESHOLD).float().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # 计算指标
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        logger.info(
            f"Epoch {epoch + 1} - "
            f"Train Loss: {train_loss / len(train_dataset):.4f}, "
            f"Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss / len(val_dataset):.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_DIR / "resnet18_best.pth")

        scheduler.step()

    logger.info(f"Training complete. Best Val Acc: {best_acc:.4f}")