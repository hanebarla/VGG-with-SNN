import numpy as np
import torch
import torch.nn as nn


class Vgg16(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_class = 10
        self.blockparams = ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512))

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

class SpikieFC(nn.Module):
    def __init__(self, N_in, N_out, initW=None):
        super().__init__()
        if initW is not None:
            self.W = initW
        else:
            self.W = nn.Parameter(torch.Tensor(
                0.1 * np.random.rand(N_out, N_in)
            ))

    def forward(self, x):
        return torch.matmul(x, self.W)
