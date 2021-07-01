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
        """
        for b in self.blockparams:
            block_concat.extend([
                nn.Conv2d(b[0], b[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(b[1]),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(b[1]),
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(b[1]),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
        """
        for b in self.blockparams1:
            block_concat.extend([
                nn.Conv2d(b[0], b[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(b[1]),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(b[1]),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
        for b in self.blockparams2:
            block_concat.extend([
                nn.Conv2d(b[0], b[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(b[1]),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(b[1]),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(b[1]),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])

        self.block = nn.Sequential(*block_concat)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 32),
            nn.LeakyReLU(inplace=True),
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
    def __init__(self, N_in, N_out, initW=None, initB=None, device=None, Vth=0.5, Vres=0.0, alpha=0.01):
        super().__init__()
        self.device = device

        self.n = None
        self.Vth = Vth
        self.Vres = Vres
        self.alpha = alpha

        self.linear = nn.Linear(N_in, N_out)
        if initW is not None:
            self.linear.weight = nn.Parameter(initW)
        else:
            self.linear.weight = nn.Parameter(torch.Tensor(
                np.zeros(N_out, N_in)
            ))

        if initB is not None:
            self.linear.bias = nn.Parameter(initB)
        else:
            self.linear.weight = nn.Parameter(
                torch.zeros(N_out)
            )

    def set_neurons(self, x):
        n_tmp = self.linear(x)
        self.n = torch.zeros_like(n_tmp)

        return self.n

    def reset_batch_neurons(self, bsize):
        self.n = torch.zeros(bsize, self.n.size()[1])
        self.n = self.n.to(self.device)

    def forward(self, x):
        self.n += self.linear(x)
        spike = torch.where(self.n > self.Vth, 1.0, self.n)
        spike = torch.where(spike < self.n_Vth, -1.0, 0.0)
        self.n[self.n > self.Vth] -= self.Vth
        self.n[self.n < self.n_Vth] -= self.n_Vth

        return spike


class SpikingConv2d(nn.Module):
    def __init__(self, N_in_ch, N_out_ch, kernel_size=3, padding=1, initW=None, initB=None, device=None, Vth=0.5, Vres=0.0, alpha=0.01):
        super().__init__()
        self.device = device

        self.n = None
        self.Vth = Vth
        self.Vres = Vres
        self.n_Vth = - Vth / alpha

        self.conv2d = nn.Conv2d(N_in_ch, N_out_ch, kernel_size=kernel_size, padding=padding)
        self.conv2d.bias = nn.Parameter(torch.zeros(N_out_ch))
        if initW is not None:
            self.conv2d.weight = nn.Parameter(initW)
        else:
            self.conv2d.weight = nn.Parameter(torch.zeros((N_out_ch, N_in_ch, kernel_size, kernel_size)))

    def set_neurons(self, x):
        n_tmp = self.conv2d(x)
        self.n = torch.zeros_like(n_tmp)

        return self.n

    def reset_batch_neurons(self, bsize):
        self.n = torch.zeros(bsize, self.n.size()[1], self.n.size()[2], self.n.size()[3])
        self.n = self.n.to(self.device)
    
    def forward(self, x):
        self.n += self.conv2d(x)
        spike = torch.where(self.n > self.Vth, 1.0, self.n)
        spike = torch.where(spike < self.n_Vth, -1.0, 0.0)
        self.n[self.n > self.Vth] -= self.Vth
        self.n[self.n < self.n_Vth] -= self.n_Vth

        return spike


class SpikingVGG16(nn.Module):
    def __init__(self, stdict, block1, block2, classifier, set_x, device):
        super().__init__()
        self.num_class = 10
        self.blockparams1 = block1
        self.blockparams2 = block2
        self.classifier = classifier

        block_concat = []
        for b in self.blockparams1:
            block_concat.extend([
                SpikingConv2d(b[0], b[1], kernel_size=3, padding=1, initW=stdict[b[2]], initB=stdict[b[3]], device=device),
                SpikingConv2d(b[1], b[1], kernel_size=3, padding=1, initW=stdict[b[4]], initB=stdict[b[5]], device=device),
                nn.AvgPool2d(kernel_size=2, stride=2)  # Use Avepool2d instead of nn.MaxPool2d(kernel_size=2, stride=2) in SNN
            ])
        for b in self.blockparams2:
            block_concat.extend([
                SpikingConv2d(b[0], b[1], kernel_size=3, padding=1, initW=stdict[b[2]], initB=stdict[b[3]], device=device),
                SpikingConv2d(b[1], b[1], kernel_size=3, padding=1, initW=stdict[b[4]], initB=stdict[b[5]], device=device),
                SpikingConv2d(b[1], b[1], kernel_size=3, padding=1, initW=stdict[b[6]], initB=stdict[b[7]], device=device),
                nn.AvgPool2d(kernel_size=2, stride=2)  # Use Avepool2d instead of nn.MaxPool2d(kernel_size=2, stride=2) in SNN
            ])

        self.block = nn.Sequential(*block_concat)

        self.classifier_concat = nn.Sequential(
            SpikingLinear(512, 512, initW=stdict[self.classifier[0]], initB=stdict[self.classifier[1]], device=device),
            SpikingLinear(512, 32, initW=stdict[self.classifier[2]], initB=stdict[self.classifier[3]], device=device),
            SpikingLinear(32, self.num_class, initW=stdict[self.classifier[4]], initB=stdict[self.classifier[5]], device=device),
        )

        self._set_neurons(set_x)

    def _set_neurons(self, x):
        before_linear = False
        for m in self.modules():
            if isinstance(m, nn.AvgPool2d):
                x = m(x)
            elif isinstance(m, SpikingConv2d):
                x = m.set_neurons(x)
            elif isinstance(m, SpikingLinear):
                if not before_linear:
                    x = x.view(x.size(0), -1)
                    before_linear = True
                x = m.set_neurons(x)

    def reset(self, bsize):
        for m in self.modules():
            if isinstance(m, SpikingConv2d):
                m.reset_batch_neurons(bsize)
            elif isinstance(m, SpikingLinear):
                m.reset_batch_neurons(bsize)

    def forward(self, x):
        x = self.block(x)
        x = x.view(x.size(0), -1)
        output = self.classifier_concat(x)

        return output


if __name__ == "__main__":
    stdict = torch.load("/home/thabara/Documents/VGG-with-SNN/0624/model_best.pth.tar", map_location=torch.device('cpu'))['state_dict']
    # model = SpikingVGG16(stdict)
    block1 = (
        (3, 64, 'block.0.weight', 'block.2.weight'),
        (64, 128, 'block.5.weight', 'block.7.weight')
    )
    block2 = (
        (128, 256, 'block.10.weight', 'block.12.weight', 'block.14.weight'),
        (256, 512, 'block.17.weight', 'block.19.weight', 'block.21.weight'),
        (512, 512, 'block.24.weight', 'block.26.weight', 'block.28.weight')
    )
    classifier = ('classifier.0.weight', 'classifier.2.weight', 'classifier.4.weight')
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(stdict.keys())
    inp = torch.ones(1, 3, 32, 32)
    model = SpikingVGG16(stdict, block1, block2, classifier, inp, device)
