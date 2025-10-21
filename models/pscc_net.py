import torch
import torch.nn as nn

class PSCCNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # 简化版PSCC-Net结构（复用核心特征提取逻辑）
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, out_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.sigmoid(x)

if __name__ == "__main__":
    model = PSCCNet()
    x = torch.randn(2, 3, 512, 512)
    print(model(x).shape)  # 应输出(2, 1, 512, 512)