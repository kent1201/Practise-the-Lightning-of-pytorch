import torch
import torch.nn as nn
import torch.nn.functional as F

# 定義一個殘差塊 (Residual Block)
class ResidualBlock(nn.Module):
    """
    Residual block for ResNet architectures, designed for ResNet18/34.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride size for downsampling. Default is 1.

    Returns:
        torch.Tensor: Output tensor after passing through the residual block.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # 兩層卷積層，每層都有 BatchNorm 和 ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果輸入與輸出維度不匹配，我們需要進行調整 (downsample)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 將輸入 (x) 和通過卷積層後的輸出相加
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class BottleneckBlock(nn.Module):
    """
    Bottleneck block used in deeper ResNet models like ResNet50/101/152.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride size for downsampling. Default is 1.

    Returns:
        torch.Tensor: Output tensor after passing through the bottleneck block.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        width = out_channels // 4  # Bottleneck層的通道縮減

        # 第一層1x1卷積，用於通道縮減
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        # 第二層3x3卷積，主要特徵提取
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        # 第三層1x1卷積，用於通道擴展
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Shortcut: 若輸入和輸出形狀不同，進行下採樣
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 定義 ResNet 主體架構
class ResNet(nn.Module):
    """
    Main ResNet architecture supporting various depths.

    Args:
        block (nn.Module): Type of block to use (ResidualBlock or BottleneckBlock).
        num_blocks (list): Number of blocks in each layer.
        num_classes (int, optional): Number of output classes for classification. Default is 10.

    Attributes:
        conv1 (nn.Conv2d): Initial convolution layer.
        bn1 (nn.BatchNorm2d): Batch normalization layer for initial conv layer.
        layer1, layer2, layer3, layer4 (nn.Sequential): Residual layers.
        avgpool (nn.AdaptiveAvgPool2d): Global average pooling layer.
        fc (nn.Linear): Final fully connected layer.
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 初始卷積層
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 堆疊多層殘差塊
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 全局平均池化和全連接層
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Creates a layer by stacking residual or bottleneck blocks.

        Args:
            block (nn.Module): Block type to be used in the layer.
            out_channels (int): Number of output channels for the layer.
            num_blocks (int): Number of blocks in the layer.
            stride (int): Stride size for the first block.

        Returns:
            nn.Sequential: A sequential container with the stacked blocks.
        """
        strides = [stride] + [1]*(num_blocks - 1)  # 第一個塊的步長可能不同
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward_features(self, x):
        """
        Extracts features from the input using all layers except the final pooling and FC layer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output feature tensor after passing through all layers.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        features = self.layer4(out)
        return features

    def forward(self, x):
        """
        Defines the forward pass of the entire ResNet model, including classification.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Logits for each class after classification.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# 定義 ResNet-18
def ResNet18(args):
    """
    Builds a ResNet18 model with 2 residual blocks per layer.

    Args:
        args (Namespace): Arguments containing the number of classes for the model.

    Returns:
        ResNet: A ResNet18 model instance.
    """
    return ResNet(ResidualBlock, [2, 2, 2, 2], args.num_classes)

def ResNet50(args):
    """
    Builds a ResNet50 model with bottleneck blocks.

    Args:
        args (Namespace): Arguments containing the number of classes for the model.

    Returns:
        ResNet: A ResNet50 model instance.
    """
    return ResNet(BottleneckBlock, [3, 4, 6, 3], args.num_classes)

def ResNet101(args):
    """
    Builds a ResNet101 model with 23 bottleneck blocks.

    Args:
        args (Namespace): Arguments containing the number of classes for the model.

    Returns:
        ResNet: A ResNet101 model instance.
    """
    return ResNet(BottleneckBlock, [3, 4, 23, 3], args.num_classes)

def ResNet152(args):
    """
    Builds a ResNet152 model with 36 bottleneck blocks.

    Args:
        args (Namespace): Arguments containing the number of classes for the model.

    Returns:
        ResNet: A ResNet152 model instance.
    """
    return ResNet(BottleneckBlock, [3, 8, 36, 3], args.num_classes)

