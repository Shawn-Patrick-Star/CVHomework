import torch
import torch.nn as nn
import torch.nn.functional as F
class DepthwiseSeparableConv(nn.Module):
    """修正深度可分离卷积实现"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels  # 关键：groups必须等于输入通道数
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class LightSegNet(nn.Module):
    def __init__(self, num_classes=21, in_channels=3):
        super().__init__()
        
        # 下采样路径
        self.down1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            DepthwiseSeparableConv(32, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 224 → 112
        )
        
        self.down2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            DepthwiseSeparableConv(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 112 → 56
        )

        # 中间瓶颈层
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            DepthwiseSeparableConv(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 上采样路径
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, 
                             padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                             padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # 修正特征融合层通道数
        self.conv_fuse1 = DepthwiseSeparableConv(64+32, 64)  # 输入96→64
        self.conv_fuse2 = DepthwiseSeparableConv(32+32, 32)  # 输入64→32
        
        # 最终输出层
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # 下采样
        x1 = self.down1(x)  # 输入3 → 输出32@112x112
        x2 = self.down2(x1) # 输入32 → 输出64@56x56
        
        # 瓶颈层
        x = self.bottleneck(x2)  # 输入64 → 输出128@56x56
        
        # 第一次上采样与融合
        x = self.up1(x)         # 输入128 → 输出64@112x112
        x = torch.cat([x, x1], dim=1)  # 64+32=96通道
        x = self.conv_fuse1(x)  # 输入96 → 输出64@112x112
        
        # 第二次上采样与融合
        x = self.up2(x)         # 输入64 → 输出32@224x224
        x = torch.cat([x, x[:, :32]], dim=1)  # 使用前半部分特征 (32+32=64)
        x = self.conv_fuse2(x)  # 输入64 → 输出32@224x224
        
        return self.final_conv(x)  # 输入32 → 输出21@224x224

if __name__ == "__main__":
    model = LightSegNet(num_classes=21)
    test_input = torch.randn(2, 3, 224, 224)  # batch_size=2
    output = model(test_input)
    print(output.shape)  # 应该输出 torch.Size([2, 21, 224, 224])