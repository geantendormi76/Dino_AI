# src/brain/cnn_encoder.py
import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    """
    一个简单的CNN编码器，用于将网格化的状态表示转换为一维特征向量。
    """
    def __init__(self, in_channels: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 这里的 in_features 需要根据 GRID_SHAPE 和 CNN 结构计算得出
        # 示例计算: 初始 25x80 -> 13x40 -> 7x20 -> 4x10
        # 所以 in_features = 128 * 4 * 10 = 5120
        # 建议在第一次运行时打印一下flatten后的形状来确定这个值
        self.linear = nn.Linear(5120, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入的网格状态，形状为 (B, C, H, W)。
        Returns:
            torch.Tensor: 输出的一维特征向量，形状为 (B, output_dim)。
        """
        x = self.network(x)
        return self.linear(x)