import torch.nn as nn

class SimpleSegmentationModel(nn.Module):
    def __init__(self, num_classes=21):
        super(SimpleSegmentationModel, self).__init__()
        # 骨干网络（简化版）
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 56x56
        )
        # 分类头
        self.head = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 上采样到112x112
            nn.ReLU(),
            nn.ConvTranspose2d(128, num_classes, kernel_size=4, stride=2, padding=1)  # 输出224x224
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return self.backbone(x)['out']
