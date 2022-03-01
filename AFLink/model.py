"""
@Author: Du Yunhao
@Filename: model.py
@Contact: dyh_bupt@163.com
@Time: 2021/12/28 14:13
@Discription: model
"""
import torch
from torch import nn

class TemporalBlock(nn.Module):
    def __init__(self, cin, cout):
        super(TemporalBlock, self).__init__()
        self.conv = nn.Conv2d(cin, cout, (7, 1), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bnf = nn.BatchNorm1d(cout)
        self.bnx = nn.BatchNorm1d(cout)
        self.bny = nn.BatchNorm1d(cout)

    def bn(self, x):
        x[:, :, :, 0] = self.bnf(x[:, :, :, 0])
        x[:, :, :, 1] = self.bnx(x[:, :, :, 1])
        x[:, :, :, 2] = self.bny(x[:, :, :, 2])
        return x

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FusionBlock(nn.Module):
    def __init__(self, cin, cout):
        super(FusionBlock, self).__init__()
        self.conv = nn.Conv2d(cin, cout, (1, 3), bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Classifier(nn.Module):
    def __init__(self, cin):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(cin*2, cin//2)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(cin//2, 2)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class PostLinker(nn.Module):
    def __init__(self):
        super(PostLinker, self).__init__()
        self.TemporalModule_1 = nn.Sequential(
            TemporalBlock(1, 32),
            TemporalBlock(32, 64),
            TemporalBlock(64, 128),
            TemporalBlock(128, 256)
        )
        self.TemporalModule_2 = nn.Sequential(
            TemporalBlock(1, 32),
            TemporalBlock(32, 64),
            TemporalBlock(64, 128),
            TemporalBlock(128, 256)
        )
        self.FusionBlock_1 = FusionBlock(256, 256)
        self.FusionBlock_2 = FusionBlock(256, 256)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = Classifier(256)

    def forward(self, x1, x2):
        x1 = x1[:, :, :, :3]
        x2 = x2[:, :, :, :3]
        x1 = self.TemporalModule_1(x1)  # [B,1,30,3] -> [B,256,6,3]
        x2 = self.TemporalModule_2(x2)
        x1 = self.FusionBlock_1(x1)
        x2 = self.FusionBlock_2(x2)
        x1 = self.pooling(x1).squeeze(-1).squeeze(-1)
        x2 = self.pooling(x2).squeeze(-1).squeeze(-1)
        y = self.classifier(x1, x2)
        if not self.training:
            y = torch.softmax(y, dim=1)
        return y


if __name__ == '__main__':
    x1 = torch.ones((2, 1, 30, 3))
    x2 = torch.ones((2, 1, 30, 3))
    m = PostLinker()
    y = m(x1, x2)
    print(y)