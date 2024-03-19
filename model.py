from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
import torch.nn.functional as F


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 0, 0.01)
            m.bias.data.zero_()


class Sum(nn.Module):
    def forward(self, inps):
        return sum(inps)


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.C1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.C2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.C3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.C4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.C5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        C1 = self.C1(x)  # 1, 64, 640, 640
        C2 = self.C2(C1)  # 1, 128, 320, 320
        C3 = self.C3(C2)  # 1, 256, 160, 160
        C4 = self.C4(C3)  # 1, 512, 80, 80
        C5 = self.C5(C4)  # 1, 512, 40, 40

        out = OrderedDict(C2=C2, C3=C3, C4=C4, C5=C5)

        return out


class PanopticFPN(nn.Module):
    def __init__(self):
        super(PanopticFPN, self).__init__()

        # 1, 256, 320, 320
        self.c2 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=(1,1), stride=(1,1)),
                nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1,1), padding=(1,1)),
                nn.GroupNorm(32, 128, eps=1e-05, affine=True),
                nn.ReLU(inplace=True)
        )

        # 1, 256, 160, 160
        self.c3 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
                nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.GroupNorm(32, 128, eps=1e-05, affine=True),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2)
            )
        # 1，512，80，80
        self.c4 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.GroupNorm(32, 256, eps=1e-05, affine=True),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.GroupNorm(32, 128, eps=1e-05, affine=True),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2))
        self.c5 = nn.Sequential(
                    nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.GroupNorm(32, 256, eps=1e-05, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.GroupNorm(32, 256, eps=1e-05, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.GroupNorm(32, 128, eps=1e-05, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2))
        self.sum = Sum()
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        c2 = self.c2(x['C2'])
        c3 = self.c3(x['C3'])
        c4 = self.c4(x['C4'])
        c5 = self.c5(x['C5'])
        result = self.sum([c2, c3, c4, c5])

        result = self.fusion(result)

        return result


class BackboneWithFPN(nn.Module):
    def __init__(self):
        super(BackboneWithFPN, self).__init__()
        self.backbone = VGG16()
        self.fpn = PanopticFPN()

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        return x


class head(nn.Module):
    def __init__(self):
        super(head, self).__init__()
        self.semantic = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1)),
        )

        self.density_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.density_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.result = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1),
                                    nn.BatchNorm2d(1),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        result = dict()
        density1 = self.density_1(x)
        density2 = self.density_2(x)
        density = torch.cat([density1, density2], 1)
        result["density"] = self.result(density)
        result["mask"] = self.semantic(x)
        return result


class FlowerNet(nn.Module):
    def __init__(self):
        super(FlowerNet, self).__init__()
        self.backboneWithFPN = BackboneWithFPN()
        self.head = head()

    def forward(self, x):
        x = self.backboneWithFPN(x)
        x = self.head(x)
        return x

