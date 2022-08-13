import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.dropout = nn.Dropout2d(p=0.6)
        self.activation = nn.ReLU()


    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.bn(x)
        x = self.dropout(x)

        x = self.activation(x)

        return x


class TinyConvNet(nn.Module):
    def __init__(self, in_channels, n_classes) -> None:
        super().__init__()
        # input image is (1, 64, 64)

        self.conv1 = ConvBlock(in_channels, out_channels=8, kernel_size=3, stride=1)
        self.conv2 = ConvBlock(in_channels=8, out_channels=16, kernel_size=3, stride=1)
        self.conv3 = ConvBlock(in_channels=16, out_channels=48, kernel_size=3, stride=1)

        self.linear = nn.Linear(48 * 5 * 5, n_classes)


    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(batch_size, -1)

        return self.linear(x)
