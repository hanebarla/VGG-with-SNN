import copy
import numpy as np
import torch
import torch.nn as nn


class Vgg16_BN(nn.Module):
    def __init__(self, activate="leaky"):
        """
        Create VGG16 with BatchNormalization, activate is LeakyRelu or ReLU
        """
        super().__init__()
        self.num_class = 10
        self.blockparams1 = ((3, 64), (64, 128))
        self.blockparams2 = ((128, 256), (256, 512), (512, 512))

        activate_factory = {
            "leaky": nn.LeakyReLU(inplace=True),
            "relu": nn.ReLU(inplace=True)
        }
        activate_func = activate_factory[activate]

        block_concat = []
        for b in self.blockparams1:
            block_concat.extend([
                nn.Conv2d(b[0], b[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(b[1]),
                activate_func,
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(b[1]),
                activate_func,
                nn.AvgPool2d(kernel_size=2, stride=2)
            ])
        for b in self.blockparams2:
            block_concat.extend([
                nn.Conv2d(b[0], b[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(b[1]),
                activate_func,
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(b[1]),
                activate_func,
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(b[1]),
                activate_func,
                nn.AvgPool2d(kernel_size=2, stride=2)
            ])

        self.block = nn.Sequential(*block_concat)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            activate_func,
            nn.Linear(512, 32),
            activate_func,
            nn.Linear(32, self.num_class),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.block(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def layer_debug(self, x, layer_num=1, act_mode="leaky"):
        activate_fact = {
            "leaky": nn.LeakyReLU(True),
            "relu": nn.ReLU(True)
        }
        activate = activate_fact[act_mode]
        before_linear = False
        layer_cnt = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                x = m(x)
                x = self.block[layer_cnt+1](x) # batch norm layer
                x = activate(x)
                layer_cnt += 1
            elif isinstance(m, nn.Linear):
                if not before_linear:
                    x = x.view(x.size(0), -1)
                    before_linear = True
                x = m(x)
                x = activate(x)
                layer_cnt += 1
            elif isinstance(m, nn.AvgPool2d):
                x = m(x)
                layer_cnt += 1

            if layer_cnt == layer_num:
                break
        
        return x


class Vgg16(nn.Module):
    """
        Create VGG16 without BatchNormalization, activate is LeakyRelu
        """
    def __init__(self, activate="leaky"):
        super().__init__()
        self.num_class = 10
        self.blockparams1 = ((3, 64), (64, 128))
        self.blockparams2 = ((128, 256), (256, 512), (512, 512))
        
        activate_factory = {
            "leaky": nn.LeakyReLU(inplace=True),
            "relu": nn.ReLU(inplace=True)
        }
        activate_func = activate_factory[activate]

        block_concat = []
        for b in self.blockparams1:
            block_concat.extend([
                nn.Conv2d(b[0], b[1], kernel_size=3, padding=1),
                activate_func,
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                activate_func,
                nn.AvgPool2d(kernel_size=2, stride=2)
            ])
        for b in self.blockparams2:
            block_concat.extend([
                nn.Conv2d(b[0], b[1], kernel_size=3, padding=1),
                activate_func,
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                activate_func,
                nn.Conv2d(b[1], b[1], kernel_size=3, padding=1),
                activate_func,
                nn.AvgPool2d(kernel_size=2, stride=2)
            ])

        self.block = nn.Sequential(*block_concat)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            activate_func,
            nn.Linear(512, 32),
            activate_func,
            nn.Linear(32, self.num_class),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                

    def forward(self, x):
        x = self.block(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def layer_debug(self, x, layer_num=1, act_mode="leaky"):
        activate_fact = {
            "leaky": nn.LeakyReLU(True),
            "relu": nn.ReLU(True)
        }
        activate = activate_fact[act_mode]
        before_linear = False
        layer_cnt = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                x = m(x)
                x = activate(x)
                layer_cnt += 1
            elif isinstance(m, nn.Linear):
                if not before_linear:
                    x = x.view(x.size(0), -1)
                    before_linear = True
                x = m(x)
                x = activate(x)
                layer_cnt += 1
            elif isinstance(m, nn.AvgPool2d):
                x = m(x)
                layer_cnt += 1

            if layer_cnt == layer_num:
                break
        
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
    def __init__(self, N_in, N_out, initW=None, initB=None, device=None, Vth=0.5, Vres=0.0, alpha=0.01, scale=1.0, percentile=0.999):
        super().__init__()
        self.device = device

        self.n = None
        self.Amax = None
        self.alpha = alpha
        self.Vth = Vth
        self.Vres = Vres
        self.n_Vth = - Vth / alpha
        self.scale = scale
        self.percentile = percentile
        self.peak = 1.0
        
        self.firecout = None

        self.lambda_before = 0
        self.lambda_after = 0

        if alpha == 0:
            self.activate = nn.ReLU()
        else:
            self.activate = nn.LeakyReLU()

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

        self.firecout = torch.zeros_like(self.n).to(self.device)

    def ann_forward(self, x):
        x = self.linear(x)
        return self.activate(x)

    def get_lambda(self, x):
        x_tmp = x.detach()
        x_tmp = torch.quantile(x_tmp, self.percentile)
        self.lambda_before = x_tmp

        n_tmp = self.linear(x)
        n_tmp = self.activate(n_tmp)
        n_tmp_detach = n_tmp.detach()
        n_tmp_detach = torch.quantile(n_tmp_detach, self.percentile)

        self.lambda_after = n_tmp_detach

        return n_tmp

    def maxactivation_normalize(self):
        self.lambda_after = abs(self.lambda_after) + 1e-5
        self.linear.weight.data = (self.linear.weight.data / self.lambda_after) * abs(self.lambda_before)
        self.linear.bias.data = self.linear.bias.data / self.lambda_after

    def forward(self, x):
        self.n += self.linear(x)

        spike = torch.zeros_like(self.n)
        spike[self.n > self.Vth] = self.peak
        spike[self.n < self.n_Vth] = -self.peak
        spike.to(self.device)

        self.firecout += spike * self.scale

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
    def __init__(self, N_in_ch, N_out_ch, kernel_size=3, padding=1, initW=None, initB=None, device=None, Vth=0.5, Vres=0.0, alpha=0.01, scale=1.0, out_h=1, out_w=1, bn=None, percentile=0.999, input_minus=False):
        super().__init__()
        self.device = device

        self.n = None
        self.alpha = alpha
        self.Vth = Vth
        self.Vres = Vres
        self.n_Vth = - Vth / alpha
        self.scale = scale
        self.percentile = percentile
        self.peak = 1.0

        self.firecout = None

        self.input_minus = input_minus
        self.lambda_before = None
        self.lambda_after = None

        if alpha <= 1e-5:
            self.activate = nn.ReLU()
        else:
            self.activate = nn.LeakyReLU()

        self.conv2d = nn.Conv2d(N_in_ch, N_out_ch, kernel_size=kernel_size, padding=padding)
        self.conv2d.bias = nn.Parameter(torch.zeros(N_out_ch))

        if bn is not None:
            gamma, beta, mean, var, _ = bn
            sigma = torch.sqrt(var + 1e-5)
            gamma_sigma = gamma / sigma

        if initW is not None:
            if bn is not None:
                initW = initW.permute(1, 2, 3, 0)
                initW = initW * gamma_sigma
                initW = initW.permute(3, 0, 1, 2)

            self.conv2d.weight = nn.Parameter(initW)
        else:
            self.conv2d.weight = nn.Parameter(torch.zeros((N_out_ch, N_in_ch, kernel_size, kernel_size)))

        if initB is not None:
            if bn is not None:
                initB = gamma_sigma * (initB - mean) + beta

            self.conv2d.bias = nn.Parameter(initB)
        else:
            self.conv2d.bias = nn.Parameter(
                torch.zeros(N_out_ch, out_h, out_w)
            )


    def set_neurons(self, x):
        n_tmp = self.conv2d(x)
        self.n = torch.zeros_like(n_tmp)

        return self.n

    def reset_batch_neurons(self, bsize):
        self.n = torch.zeros(bsize, self.n.size()[1], self.n.size()[2], self.n.size()[3])
        self.n = self.n.to(self.device)

        self.firecout = torch.zeros_like(self.n).to(self.device)

    def ann_forward(self, x):
        x = self.conv2d(x)
        return self.activate(x)

    def get_lambda(self, x):
        """
        Calculate each channel's max
        """
        x_tmp = x.detach()
        x_tmp = x_tmp.permute(1, 0, 2, 3)
        x_tmp = x_tmp.contiguous().view(x_tmp.size(0), -1)
        x_tmp = torch.quantile(x_tmp, self.percentile, dim=1)
        # x_tmp = x_tmp.contiguous().view(-1, x_tmp.size(1))
        self.lambda_before = x_tmp.detach()

        n_tmp = self.conv2d(x)
        n_tmp = self.activate(n_tmp)

        n_tmp_reshaped = n_tmp.detach()
        n_tmp_reshaped = n_tmp_reshaped.permute(1, 0, 2, 3) # size: (batch_size, channel_size, w, h) -> (channel_size, batch_size, w, h)
        n_tmp_reshaped = n_tmp_reshaped.contiguous().view(n_tmp_reshaped.size(0), -1) # size: (channel_size, batch_size, w*h) -> (channel_size, batch_size*w*h)
        n_tmp_reshaped = torch.quantile(n_tmp_reshaped, self.percentile, dim=1) # size: (channel_size, batch_size*w*h) -> (channel_size)
        # n_tmp_reshaped = n_tmp_reshaped.contiguous().view(-1, n_tmp_reshaped.size(1))
        self.lambda_after = n_tmp_reshaped.detach()

        return n_tmp

    def channel_wise_normalize(self):
        out_ch = self.lambda_after.size(0)
        inp_ch = self.lambda_before.size(0)
        self.lambda_after = torch.abs(self.lambda_after) + 1e-5

        if self.input_minus:
            self.lambda_before = torch.ones_like(self.lambda_before).to(self.device)

        # テンソル計算に直す
        for i in range(out_ch):
            self.conv2d.bias.data[i] = self.conv2d.bias.data[i] / self.lambda_after[i]
            for j in range(inp_ch):
                self.conv2d.weight.data[i, j, :, :] = (self.conv2d.weight.data[i, j, :, :] / self.lambda_after[i]) * abs(self.lambda_before[j])
    
    def forward(self, x):
        self.n += self.conv2d(x)

        spike = torch.zeros_like(self.n)
        spike[self.n > self.Vth] = self.peak
        spike[self.n < self.n_Vth] = -self.peak
        spike.to(self.device)

        self.firecout += spike * self.scale

        self.n[self.n > self.Vth] -= self.Vth
        self.n[self.n < self.n_Vth] -= self.n_Vth

        return spike


class SpikingRandomPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0, device=None, Vth=0.5, Vres=0.0) -> None:
        super().__init__()
        self.AvgPool2d = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.kernal_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device

    def set_neurons(self, x):
        n_tmp = self.AvgPool2d(x)
        return n_tmp

    def get_lambda(self, x):
        return self.AvgPool2d(x)

    def ann_forward(self, x):
        return self.AvgPool2d(x)

    def forward(self, x):
        # 2, 3 is map index(0 is batch, 1 is channel)
        ba, ch, h, w = x.size()
        h_out = int((h + 2*self.padding - self.kernal_size) / 2 + 1)
        w_out = int((w + 2*self.padding - self.kernal_size) / 2 + 1)
        x_patch = x.unfold(2, self.kernal_size, self.stride).unfold(3, self.kernal_size, self.stride)

        rand1 = torch.randint(self.kernal_size, (ba, ch, h_out, w_out, self.kernal_size, 1)).to(self.device)
        rand2 = torch.randint(self.kernal_size, (ba, ch, h_out, w_out, 1, 1)).to(self.device)

        ga1 = torch.gather(x_patch, -1, rand1)
        ga2 = torch.gather(ga1, -2, rand2)

        outputs = ga2.view(ba, ch, h_out, w_out)

        return outputs


class SpikingVGG16(nn.Module):
    def __init__(self, stdict, block1, block2, bn1, bn2, classifier, set_x, device, Vth=1.0, Vres=0.0, activate="leaky", scale=1.0, percentile=0.999):
        super().__init__()
        self.num_class = 10
        self.blockparams1 = block1
        self.blockparams2 = block2
        self.classifier = classifier
        self.outscale = 1.0 * Vth

        if activate == "relu":
            alpha = 1e-9
        elif activate == "leaky":
            alpha = 1e-2
        else:
            raise ValueError

        block_concat = []
        layer = 0
        input_minus = False
        for b, bnk in zip(self.blockparams1, bn1):
            bns = []
            for bn_keys in bnk:
                if bn_keys is not None:
                    bns.append([stdict[k] for k in bn_keys])
                else:
                    bns.append(None)
            if layer == 0:
                input_minus = True
            block_concat.extend([
                SpikingConv2d(b[0], b[1], kernel_size=3, padding=1, initW=stdict[b[2]], initB=stdict[b[3]], device=device, Vth=Vth, Vres=Vres, alpha=alpha, scale=scale, bn=bns[0], percentile=percentile, input_minus=input_minus),
                SpikingConv2d(b[1], b[1], kernel_size=3, padding=1, initW=stdict[b[4]], initB=stdict[b[5]], device=device, Vth=Vth, Vres=Vres, alpha=alpha, scale=scale, bn=bns[1], percentile=percentile),
                SpikingRandomPool2d(kernel_size=2, stride=2, device=device)
            ])
            layer += 1
            input_minus = False
        for b, bnk in zip(self.blockparams2, bn2):
            bns = []
            for bn_keys in bnk:
                if bn_keys is not None:
                    bns.append([stdict[k] for k in bn_keys])
                else:
                    bns.append(None)

            block_concat.extend([
                SpikingConv2d(b[0], b[1], kernel_size=3, padding=1, initW=stdict[b[2]], initB=stdict[b[3]], device=device, Vth=Vth, Vres=Vres, alpha=alpha, scale=scale, bn=bns[0], percentile=percentile),
                SpikingConv2d(b[1], b[1], kernel_size=3, padding=1, initW=stdict[b[4]], initB=stdict[b[5]], device=device, Vth=Vth, Vres=Vres, alpha=alpha, scale=scale, bn=bns[1], percentile=percentile),
                SpikingConv2d(b[1], b[1], kernel_size=3, padding=1, initW=stdict[b[6]], initB=stdict[b[7]], device=device, Vth=Vth, Vres=Vres, alpha=alpha, scale=scale, bn=bns[2], percentile=percentile),
                SpikingRandomPool2d(kernel_size=2, stride=2, device=device)
            ])

        self.block = nn.Sequential(*block_concat)

        self.classifier_concat = nn.Sequential(
            SpikingLinear(512, 512, initW=stdict[self.classifier[0]], initB=stdict[self.classifier[1]], device=device, Vth=Vth, Vres=Vres, alpha=alpha, scale=scale, percentile=percentile),
            SpikingLinear(512, 32, initW=stdict[self.classifier[2]], initB=stdict[self.classifier[3]], device=device, Vth=Vth, Vres=Vres, alpha=alpha, scale=scale, percentile=percentile),
            SpikingLinear(32, self.num_class, initW=stdict[self.classifier[4]], initB=stdict[self.classifier[5]], device=device, Vth=Vth, Vres=Vres, alpha=alpha, scale=scale, percentile=percentile),
        )

        self._set_neurons(set_x) # set membrem voltage

    # kwargsを使ってほかのものも
    def _manual_forward(self, x, mode="set_neurons", layer_num=None):
        layer_cnt = 0
        apply_instance_2d = (SpikingConv2d, SpikingRandomPool2d)
        before_linear = False
        for m in self.modules():
            if isinstance(m, apply_instance_2d):
                x = getattr(m, mode)(x)
                layer_cnt += 1
            elif isinstance(m, SpikingLinear):
                if not before_linear:
                    x = x.view(x.size(0), -1)
                    before_linear = True
                x = getattr(m, mode)(x)
                layer_cnt += 1

            if layer_cnt == layer_num:
                break

        return x

    def ann_forward(self, x, layer_num=None):
        return self._manual_forward(x, mode="ann_forward", layer_num=layer_num)

    def calculate_lambda(self, x):
        out = self._manual_forward(x, mode="get_lambda")
        return out

    def channel_wised_normlization(self):
        for m in self.modules():
            if isinstance(m, SpikingConv2d):
                m.channel_wise_normalize()
            elif isinstance(m, SpikingLinear):
                m.maxactivation_normalize()

    def _set_neurons(self, x):
        _ = self._manual_forward(x, mode="set_neurons")

    def reset(self, bsize):
        apply_instance = (SpikingConv2d, SpikingLinear)
        for m in self.modules():
            if isinstance(m, apply_instance):
                m.reset_batch_neurons(bsize)

    def layer_debug(self, layer_num=1):
        fire_map = None
        layer_cnt = 0
        apply_instance = (SpikingConv2d, SpikingLinear)

        for m in self.modules():
            if isinstance(m, SpikingRandomPool2d):
                layer_cnt += 1
            elif isinstance(m, apply_instance):
                fire_map = m.firecout
                layer_cnt += 1

            if layer_num == layer_cnt:
                break

        return fire_map

    def FireCount(self, timestep=100):
        apply_instance = (SpikingConv2d, SpikingLinear)
        firerate_max = []
        firerate_min = []
        firerate_mean = []
        for m in self.modules():
            if isinstance(m, apply_instance):
                firerate_max.append(torch.max(m.firecout/timestep).item())
                firerate_min.append(torch.min(m.firecout/timestep).item())
                firerate_mean.append(torch.mean(m.firecout/timestep).item())

        firemax = max(firerate_max)
        firemin = min(firerate_min)
        firemean = sum(firerate_mean)/len(firerate_mean)
        print("Time Step {}, Fire Max Rate: {}, Fire Min Rate: {}, Fire Mean Rate {}".format(timestep, firemax, firemin, firemean))

    def forward(self, x):
        x = self.block(x)
        x = x.view(x.size(0), -1)
        output = self.classifier_concat(x) * self.outscale # when Vth is changed, we should scale this

        return output

    def changeVth(self, Vth):
        apply_instance = (SpikingConv2d, SpikingLinear)
        for m in self.modules():
            if isinstance(m, apply_instance):
                m.scale = 1.0 * Vth
                m.Vth = Vth
                m.n_Vth = -Vth / m.alpha
                self.outscale = 1.0 * Vth


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
