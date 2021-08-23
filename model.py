from os import isatty
import numpy as np
import torch
import torch.nn as nn


class Vgg16_BN(nn.Module):
    def __init__(self):
        """
        Create VGG16 with BatchNormalization, activate is LeakyRelu
        """
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
    """
        Create VGG16 without BatchNormalization, activate is LeakyRelu
        """
    def __init__(self):
        super().__init__()
        self.num_class = 10
        self.blockparams1 = ((3, 64), (64, 128))
        self.blockparams2 = ((128, 256), (256, 512), (512, 512))

        block_concat = []
        for b in self.blockparams1:
            block_concat.extend([
                nn.Conv2d(b[0], b[1], kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
        for b in self.blockparams2:
            block_concat.extend([
                nn.Conv2d(b[0], b[1], kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])

        self.block = nn.Sequential(*block_concat)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(True),
            nn.Linear(512, 32),
            nn.LeakyReLU(True),
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
    """
    Spiking Linear(FC) layer

    Parameters
    ----------
    N_in : int
        input channel size
    N_out : int
        output channel size
    initW : 
        Pretrained layer's weight
    init B :
        Pretrained layer's Bias
    device : String
        cpu or gpu
    Vth : float
        Spike threshold
    Vres : float
        reset voltage
    alpha : float
        slope of LeakyRelu in negative area
    """
    def __init__(self, N_in, N_out, initW=None, initB=None, device=None, Vth=0.5, Vres=0.0, alpha=0.01, scale=1.0):
        super().__init__()
        self.device = device

        self.n = None
        self.Amax = None
        self.Vth = Vth
        self.Vres = Vres
        self.n_Vth = - Vth / alpha
        self.scale = scale
        self.peak = 1.0
        
        self.firecout = 0
        self.firemax = 0

        self.lambda_before = 0.0
        self.lambda_after = 0.0

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
            self.linear.bias = nn.Parameter(
                torch.zeros(N_out)
            )

    def set_neurons(self, x):
        n_tmp = self.linear(x)
        self.n = torch.zeros_like(n_tmp)

        return self.n

    def reset_batch_neurons(self, bsize):
        self.n = torch.zeros(bsize, self.n.size()[1])
        self.n = self.n.to(self.device)

        self.firecout = 0
        self.firemax = 0

    def get_lambda(self, x):
        lambda_tmp = torch.max(x)
        if lambda_tmp > self.lambda_before:
            self.lambda_before = lambda_tmp

        n_tmp = self.linear(x)
        lambda_tmp = torch.max(n_tmp)
        if lambda_tmp > self.lambda_after:
            self.lambda_after = lambda_tmp

        return n_tmp

    def maxactivation_normalize(self):
        self.linear.weight.data = self.scale * self.linear.weight.data / self.lambda_after * self.lambda_before
        self.linear.bias.data = self.scale * self.linear.bias.data / self.lambda_after

    def forward(self, x):
        self.n += self.linear(x)

        spike = torch.zeros_like(self.n)
        spike[self.n > self.Vth] = self.peak
        spike[self.n < self.n_Vth] = -self.peak
        spike.to(self.device)

        peak_count = torch.sum(torch.abs(spike)).item()
        peak_max_count = torch.sum(torch.ones_like(spike)).item()
        self.firecout += peak_count
        self.firemax += peak_max_count

        self.n[self.n > self.Vth] -= self.Vth
        self.n[self.n < self.n_Vth] -= self.n_Vth

        return spike


class SpikingConv2d(nn.Module):
    """
    Spiking Conv2d layer

    Parameters
    ----------
    N_in : int
        input channel size
    N_out : int
        output channel size
    initW : 
        Pretrained layer's weight
    init B :
        Pretrained layer's Bias
    device : String
        cpu or gpu
    Vth : float
        Spike threshold
    Vres : float
        reset voltage
    alpha : float
        slope of LeakyRelu in negative area
    """
    def __init__(self, N_in_ch, N_out_ch, kernel_size=3, padding=1, initW=None, initB=None, device=None, Vth=0.5, Vres=0.0, alpha=0.01, scale=1.0, out_h=1, out_w=1):
        super().__init__()
        self.device = device

        self.n = None
        self.Vth = Vth
        self.Vres = Vres
        self.n_Vth = - Vth / alpha
        self.scale = scale
        self.peak = 1.0

        self.firecout = 0
        self.firemax = 0

        self.lambda_before = torch.zeros(N_in_ch).to(device)
        self.lambda_after = torch.zeros(N_out_ch).to(device)

        self.conv2d = nn.Conv2d(N_in_ch, N_out_ch, kernel_size=kernel_size, padding=padding)
        self.conv2d.bias = nn.Parameter(torch.zeros(N_out_ch))
        if initW is not None:
            self.conv2d.weight = nn.Parameter(initW)
        else:
            self.conv2d.weight = nn.Parameter(torch.zeros((N_out_ch, N_in_ch, kernel_size, kernel_size)))

        if initB is not None:
            self.conv2d.bias = nn.Parameter(initB)
        else:
            self.conv2d.bias = nn.Parameter(
                torch.zeros(N_out_ch, out_h, out_w)
            )

    def set_neurons(self, x):
        n_tmp = self.conv2d(x)
        self.n = torch.zeros_like(n_tmp)

        return self.n

    def get_lambda(self, x):
        """
        Calculate each channel's max
        """
        x_tmp = torch.clone(x)
        x_tmp = x_tmp.view(x_tmp.size(0), x_tmp.size(1), -1)
        x_batch_ch_max, _ = torch.max(x_tmp, 2)
        x_ch_max, _  = torch.max(x_batch_ch_max, 0)
        self.lambda_before = torch.where(self.lambda_before < x_ch_max, x_ch_max, self.lambda_before)

        n_tmp = self.conv2d(x)
        n_tmp_reshaped = n_tmp.view(n_tmp.size(0), n_tmp.size(1), -1) # size: (batch_size, channel_size, w*h)
        batch_ch_max, _ = torch.max(n_tmp_reshaped, 2) # size: (batch_size, channel_size)
        ch_max, _ = torch.max(batch_ch_max, 0) # size(channel_size)
        self.lambda_after = torch.where(self.lambda_after < ch_max, ch_max, self.lambda_after)

        return n_tmp

    def channel_wise_normalize(self):
        out_ch = self.lambda_after.size(0)
        inp_ch = self.lambda_before.size(0)

        for i in range(out_ch):
            self.conv2d.bias.data[i] = self.scale * self.conv2d.bias.data[i] / self.lambda_after[i]
            for j in range(inp_ch):
                self.conv2d.weight.data[i, j, :, :] = self.scale * self.conv2d.weight.data[i, j, :, :] / self.lambda_after[i] * self.lambda_before[j]

    def reset_batch_neurons(self, bsize):
        self.n = torch.zeros(bsize, self.n.size()[1], self.n.size()[2], self.n.size()[3])
        self.n = self.n.to(self.device)

        self.firecout = 0
        self.firemax = 0
    
    def forward(self, x):
        self.n += self.conv2d(x)

        spike = torch.zeros_like(self.n)
        spike[self.n > self.Vth] = self.peak
        spike[self.n < self.n_Vth] = -self.peak
        spike.to(self.device)

        peak_count = torch.sum(torch.abs(spike)).item()
        peak_max_count = torch.sum(torch.ones_like(spike)).item()
        self.firecout += peak_count
        self.firemax += peak_max_count

        self.n[self.n > self.Vth] -= self.Vth
        self.n[self.n < self.n_Vth] -= self.n_Vth

        return spike


class SpikingAvgPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None, device=None, Vth=0.5, Vres=0.0) -> None:
        super().__init__()
        self.AvgPool2d = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override
        )

        self.device = device
        self.Vth = Vth
        self.Vres = Vres
        self.n = None
        self.peak = float(1.0)

        self.firecout = 0
        self.firemax = 0

    def set_neurons(self, x):
        n_tmp = self.AvgPool2d(x)
        self.n = torch.zeros_like(n_tmp)
        return n_tmp

    # Spiking AveratePool2d is not need to get lambda, but other layer needs.
    def get_lambda(self, x):
        return self.AvgPool2d(x)

    def reset_batch_neurons(self, bsize):
        self.n = torch.zeros(bsize, self.n.size()[1], self.n.size()[2], self.n.size()[3])
        self.n = self.n.to(self.device)

        self.firecout = 0
        self.firemax = 0

    def forward(self, x):
        self.n += self.AvgPool2d(x)

        spike = torch.zeros_like(self.n)
        spike[self.n > self.Vth] = self.peak
        spike[self.n < -self.Vth] = -self.peak # avepool2dの負方向のスパイクの閾値はとりあえず-Vthに
        spike.to(self.device)

        peak_count = torch.sum(torch.abs(spike)).item()
        peak_max_count = torch.sum(torch.ones_like(spike)).item()
        self.firecout += peak_count
        self.firemax += peak_max_count

        self.n[self.n > self.Vth] -= self.Vth
        self.n[self.n < -self.Vth] += self.Vth

        return spike


class SpikingVGG16(nn.Module):
    def __init__(self, stdict, block1, block2, classifier, set_x, device, Vth=0.5, Vres=0.0, scale=1.0):
        super().__init__()
        self.num_class = 10
        self.blockparams1 = block1
        self.blockparams2 = block2
        self.classifier = classifier

        block_concat = []
        for b in self.blockparams1:
            block_concat.extend([
                SpikingConv2d(b[0], b[1], kernel_size=3, padding=1, initW=stdict[b[2]], initB=stdict[b[3]], device=device, Vth=Vth, Vres=Vres, scale=scale),
                SpikingConv2d(b[1], b[1], kernel_size=3, padding=1, initW=stdict[b[4]], initB=stdict[b[5]], device=device, Vth=Vth, Vres=Vres, scale=scale),
                SpikingAvgPool2d(kernel_size=2, stride=2, device=device, Vth=Vth, Vres=Vres)  # Use Avepool2d instead of nn.MaxPool2d(kernel_size=2, stride=2) in SNN
            ])
        for b in self.blockparams2:
            block_concat.extend([
                SpikingConv2d(b[0], b[1], kernel_size=3, padding=1, initW=stdict[b[2]], initB=stdict[b[3]], device=device, Vth=Vth, Vres=Vres, scale=scale),
                SpikingConv2d(b[1], b[1], kernel_size=3, padding=1, initW=stdict[b[4]], initB=stdict[b[5]], device=device, Vth=Vth, Vres=Vres, scale=scale),
                SpikingConv2d(b[1], b[1], kernel_size=3, padding=1, initW=stdict[b[6]], initB=stdict[b[7]], device=device, Vth=Vth, Vres=Vres, scale=scale),
                SpikingAvgPool2d(kernel_size=2, stride=2, device=device, Vth=Vth, Vres=Vres)  # Use Avepool2d instead of nn.MaxPool2d(kernel_size=2, stride=2) in SNN
            ])

        self.block = nn.Sequential(*block_concat)

        self.classifier_concat = nn.Sequential(
            SpikingLinear(512, 512, initW=stdict[self.classifier[0]], initB=stdict[self.classifier[1]], device=device, Vth=Vth, Vres=Vres, scale=scale),
            SpikingLinear(512, 32, initW=stdict[self.classifier[2]], initB=stdict[self.classifier[3]], device=device, Vth=Vth, Vres=Vres, scale=scale),
            SpikingLinear(32, self.num_class, initW=stdict[self.classifier[4]], initB=stdict[self.classifier[5]], device=device, Vth=Vth, Vres=Vres, scale=scale),
        )

        self._set_neurons(set_x) # set membrem voltage

    def calculate_lambda(self, x):
        before_linear = False
        for m in self.modules():
            if isinstance(m, SpikingAvgPool2d):
                x = m.get_lambda(x)
            elif isinstance(m, SpikingConv2d):
                x = m.get_lambda(x)
            elif isinstance(m, SpikingLinear):
                if not before_linear:
                    x = x.view(x.size(0), -1)
                    before_linear = True
                x = m.get_lambda(x)

    def channel_wised_normlization(self):
        for m in self.modules():
            if isinstance(m, SpikingConv2d):
                m.channel_wise_normalize()
            elif isinstance(m, SpikingLinear):
                m.maxactivation_normalize()

    def _set_neurons(self, x):
        before_linear = False
        for m in self.modules():
            if isinstance(m, SpikingAvgPool2d):
                x = m.set_neurons(x)
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
            elif isinstance(m, SpikingAvgPool2d):
                m.reset_batch_neurons(bsize)

    def FireCount(self):
        firecnt = 0
        firemax = 0
        for m in self.modules():
            if isinstance(m, SpikingAvgPool2d):
                firecnt += m.firecout
                firemax += m.firemax
            elif isinstance(m, SpikingConv2d):
                firecnt += m.firecout
                firemax += m.firemax
            elif isinstance(m, SpikingLinear):
                firecnt += m.firecout
                firemax += m.firemax
        print("Fire Count: {}, Fire Max {}, Fire Rate {}".format(firecnt, firemax, firecnt / firemax))

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
