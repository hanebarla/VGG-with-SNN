from os import name
import numpy as np
import torch
import torch.nn as nn


class Vgg16_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_class = 10
        self.blockparams1 = ((3, 64), (64, 128))
        self.blockparams2 = ((128, 256), (256, 512), (512, 512))

        block_concat = []
        for b in self.blockparams:
            block_concat.extend([
                nn.Conv2d(b[0], b[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(b[1]),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(b[1]),
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(b[1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
        """
        for b in self.blockparams1:
            block_concat.extend([
                nn.Conv2d(b[0], b[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(b[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(b[1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
        for b in self.blockparams2:
            block_concat.extend([
                nn.Conv2d(b[0], b[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(b[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(b[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(b[1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
        """

        self.block = nn.Sequential(*block_concat)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 32),
            nn.ReLU(True),
            nn.Linear(32, self.num_class),
        )

    def forward(self, x):
        x = self.block(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Vgg16(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_class = 10
        self.blockparams1 = ((3, 64), (64, 128))
        self.blockparams2 = ((128, 256), (256, 512), (512, 512))

        block_concat = []
        for b in self.blockparams1:
            block_concat.extend([
                nn.Conv2d(b[0], b[1], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
        for b in self.blockparams2:
            block_concat.extend([
                nn.Conv2d(b[0], b[1], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])

        self.block = nn.Sequential(*block_concat)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 32),
            nn.ReLU(True),
            nn.Linear(32, self.num_class),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.block(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SpikingLinear(nn.Module):
    def __init__(self, N_in, N_out, initW=None, Vth=0.5, Vres=0.0):
        super().__init__()

        self.n = None
        self.Vth = Vth

        self.linear = nn.Linear(N_in, N_out)
        self.conv2d.bias = nn.Parameter(torch.zeros(N_out))
        if initW is not None:
            self.linear.weight = initW
        else:
            self.W = nn.Parameter(torch.Tensor(
                np.zeros(N_out, N_in)
            ))

    def forward(self, x):
        self.n += self.linear(x)
        spike = torch.where(self.n > self.Vth, 1.0, -1.0)
        self.n[self.n > self.Vth] -= self.Vth

        return spike


class SpikingConv2d(nn.Module):
    def __init__(self, N_in_ch, N_out_ch, kernel_size=3, padding=1, initW=None, Vth=0.5, Vres=0.0):
        super().__init__()

        self.n = None
        self.Vth = Vth

        self.conv2d = nn.Conv2d(N_in_ch, N_out_ch, kernel_size=kernel_size, padding=padding)
        self.conv2d.bias = nn.Parameter(torch.zeros(N_out_ch))
        if initW is not None:
            self.conv2d.weight = initW
        else:
            self.conv2d.weight = nn.Parameter(torch.zeros((N_out_ch, N_in_ch, kernel_size, kernel_size)))
    
    def forward(self, x):
        self.n += self.conv2d(x)
        spike = torch.where(self.n > self.Vth, 1.0, -1.0)
        self.n[self.n > self.Vth] -= self.Vth

        return spike


class SpikingVGG16(nn.Module):
    def __init__(self, chckpoint):
        super().__init__()
        self.num_class = 10
        self.blockparams1 = ((3, 64), (64, 128))
        self.blockparams2 = ((128, 256), (256, 512), (512, 512))

        block_concat = []
        for b in self.blockparams1:
            block_concat.extend([
                SpikingConv2d(b[0], b[1], kernel_size=3, padding=1),
                SpikingConv2d(b[1], b[1], kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
        for b in self.blockparams2:
            block_concat.extend([
                SpikingConv2d(b[0], b[1], kernel_size=3, padding=1),
                SpikingConv2d(b[1], b[1], kernel_size=3, padding=1),
                SpikingConv2d(b[1], b[1], kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])

        self.block = nn.Sequential(*block_concat)

        self.classifir_concat = nn.Sequential(
            SpikingLinear(512, 512),
            SpikingLinear(512, 32),
            SpikingLinear(32, self.num_class),
        )

    def forward(self, x):
        self.neurons.append()
        for b in self.block_concat:
            x = b(x)
            self.neurons


if __name__ == "__main__":
    stdict = torch.load("/home/thabara/Documents/VGG-with-SNN/0624/model_best.pth.tar")['state_dict']
    model = Vgg16()
    print(stdict.keys())