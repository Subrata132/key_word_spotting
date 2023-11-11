import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import ConvBlock, FullyConnectedBlock


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential(
            ConvBlock(input_channel=1, output_channel=32),
            ConvBlock(input_channel=32, output_channel=64),
            ConvBlock(input_channel=64, output_channel=128),
            ConvBlock(input_channel=128, output_channel=128),
            nn.Flatten()
        )

        self.class_classifier = nn.Sequential(
            FullyConnectedBlock(in_features=10752, out_features=512),
            nn.Dropout(p=0.5),
            FullyConnectedBlock(in_features=512, out_features=64),
            nn.Dropout(p=0.5),
            FullyConnectedBlock(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.class_classifier(x)
        return x


class CNNModelTiny(nn.Module):
    def __init__(self):
        super(CNNModelTiny, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.bn6 = nn.BatchNorm1d(num_features=64)
        self.d2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool4(x)
        x = F.relu(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.d2(x)
        x = self.fc3(x)
        return x


def test():
    sample_input = torch.rand((2, 1, 65, 65))
    model = CNNModelTiny()
    y = model(sample_input)
    print(y.shape)


if __name__ == '__main__':
    test()