import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18Tamper(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = resnet18(pretrained=pretrained)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        return self.sigmoid(x)

if __name__ == "__main__":
    model = ResNet18Tamper()
    x = torch.randn(2, 3, 512, 512)
    print(model(x).shape)  # 应输出(2, 1)