import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from model import SpikingConv2d, SpikingVGG16, Vgg16, Vgg16_BN
from utils import SpikeEncodeDatasets, date2foldername, saveCSVrow, saveCSVrows
from ThresholdModel import AdaptiveModel


parser = argparse.ArgumentParser(description='PyTorch Spiking Test')
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--scale', default=1.0, type=float)
parser.add_argument('--Vth', default=1.0, type=float)
parser.add_argument('--Vres', default=0.0, type=float)
parser.add_argument('--activate', default="leaky")
parser.add_argument('--debug_layer', default=1, type=int)
parser.add_argument('--timelength', default=1000, type=int)
parser.add_argument('--bn', default=0, type=int)
parser.add_argument('--load_weight', default=None, help="ANN trained model file path")
parser.add_argument('--load_normalized_weight', default=None)
parser.add_argument('--savefolder', default="SNN_Test_Results/")
parser.add_argument('--changeStep', default=200, type=int)
parser.add_argument('--logging', default=1, type=int, help="if we want to save csv data per image")  # bool


# Calculate lambda(max activations), and channel-wise Normalize
def CW_Normalize(args, model, trainset, device):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize)
    dataForNormalize = next(iter(trainloader))
    with torch.no_grad():
        inputs, labels = dataForNormalize
        inputs = inputs.to(device)
        outputs = model.calculate_lambda(inputs)

    print("Normalizing Model ...")
    model.channel_wised_normlization()
    print("=> Model Normalize Success")

    return model


def spike_test(args, trainset, testset,  spikeset, device):
    softmax = nn.Softmax(dim=1)
    VthFunc = AdaptiveModel(chagestep=args.changeStep)
    # logging set up
    if args.logging == 1:
        savecsv = os.path.join(args.savefolder, "Vth-Adaptive_result_per_image.csv")
        accTimestepCSV = os.path.join(args.savefolder, "Vth-Adaptive_batchAcc_per_time.csv")
        vthTimestepCSV = os.path.join(args.savefolder, "Vth-Adaptive_Vth_per_time.csv")
        fireTimestepCSV = os.path.join(args.savefolder, "Vth-Adaptive_Firecount_per_time.csv")
        saveCSVrow(["Vth","index","spike_argmax","label","acc"], savecsv)

    # Load Mmodels
    if args.load_weight is not None:
        stdict = torch.load(args.load_weight, map_location=torch.device('cpu'))['state_dict']
    else:
        stdict = None
    # model = SpikingVGG16(stdict)

    # 学習済みモデルをパースするスクリプトを作成する
    # if you use without batch norm2d model
    if args.bn == 0:
        block1 = (
            (3, 64, 'block.0.weight', 'block.0.bias', 'block.2.weight', 'block.2.bias'),
            (64, 128, 'block.5.weight', 'block.5.bias', 'block.7.weight', 'block.7.bias')
        )
        bn1 = (
            (None, None),
            (None, None)
        )
        block2 = (
            (128, 256, 'block.10.weight', 'block.10.bias', 'block.12.weight', 'block.12.bias', 'block.14.weight', 'block.14.bias'),
            (256, 512, 'block.17.weight', 'block.17.bias', 'block.19.weight', 'block.19.bias', 'block.21.weight', 'block.21.bias'),
            (512, 512, 'block.24.weight', 'block.24.bias', 'block.26.weight', 'block.26.bias', 'block.28.weight', 'block.28.bias')
        )
        bn2 = (
            (None, None, None),
            (None, None, None),
            (None, None, None)
        )
    else:
        # if you use batch norm2d model
        block1 = (
            (3, 64, 'block.0.weight', 'block.0.bias', 'block.3.weight', 'block.3.bias'),
            (64, 128, 'block.7.weight', 'block.7.bias', 'block.10.weight', 'block.10.bias')
        )
        bn1 = (
            (
                ('block.1.weight', 'block.1.bias', 'block.1.running_mean', 'block.1.running_var', 'block.1.num_batches_tracked'),
                ('block.4.weight', 'block.4.bias', 'block.4.running_mean', 'block.4.running_var', 'block.4.num_batches_tracked')
            ),
            (
                ('block.8.weight', 'block.8.bias', 'block.8.running_mean', 'block.8.running_var', 'block.8.num_batches_tracked'),
                ('block.11.weight', 'block.11.bias', 'block.11.running_mean', 'block.11.running_var', 'block.11.num_batches_tracked')
            )
        )
        block2 = (
            (128, 256, 'block.14.weight', 'block.14.bias', 'block.17.weight', 'block.17.bias', 'block.20.weight', 'block.20.bias'),
            (256, 512, 'block.24.weight', 'block.24.bias', 'block.27.weight', 'block.27.bias', 'block.30.weight', 'block.30.bias'),
            (512, 512, 'block.34.weight', 'block.34.bias', 'block.37.weight', 'block.37.bias', 'block.40.weight', 'block.40.bias')
        )
        bn2 = (
            (
                ('block.15.weight', 'block.15.bias', 'block.15.running_mean', 'block.15.running_var', 'block.15.num_batches_tracked'),
                ('block.18.weight', 'block.18.bias', 'block.18.running_mean', 'block.18.running_var', 'block.18.num_batches_tracked'),
                ('block.21.weight', 'block.21.bias', 'block.21.running_mean', 'block.21.running_var', 'block.21.num_batches_tracked'),
            ),
            (
                ('block.25.weight', 'block.25.bias', 'block.25.running_mean', 'block.25.running_var', 'block.25.num_batches_tracked'),
                ('block.28.weight', 'block.28.bias', 'block.28.running_mean', 'block.28.running_var', 'block.28.num_batches_tracked'),
                ('block.31.weight', 'block.31.bias', 'block.31.running_mean', 'block.31.running_var', 'block.31.num_batches_tracked'),
            ),
            (
                ('block.35.weight', 'block.35.bias', 'block.35.running_mean', 'block.35.running_var', 'block.35.num_batches_tracked'),
                ('block.38.weight', 'block.38.bias', 'block.38.running_mean', 'block.38.running_var', 'block.38.num_batches_tracked'),
                ('block.41.weight', 'block.41.bias', 'block.41.running_mean', 'block.41.running_var', 'block.41.num_batches_tracked'),
            ),
        )
    classifier = ('classifier.0.weight', 'classifier.0.bias', 'classifier.2.weight', 'classifier.2.bias', 'classifier.4.weight', 'classifier.4.bias')
    print("=> Load Weight Success")
    # print(stdict['block.0.bias'].size())

    # ANN_model to Debug
    if args.bn == 1:
        ANN_model = Vgg16_BN(activate=args.activate)
    else:
        ANN_model = Vgg16(activate=args.activate)
    ANN_model.load_state_dict(stdict)
    ANN_model.to(device)

    # SNN Initialize
    inp = torch.ones(1, 3, 32, 32)  # To get neauron size, so simulate as a scalemodel
    model = SpikingVGG16(stdict, block1, block2, bn1, bn2, classifier, inp, device, Vth=args.Vth, Vres=args.Vres, activate=args.activate)
    if args.load_normalized_weight is None:
        model.to(device)
        model = CW_Normalize(args, model, trainset, device)
        NormalizeSaveName = os.path.join(os.path.dirname(args.load_weight), "normalized.pth.tar")
        torch.save(model.state_dict(), NormalizeSaveName)
    else:
        model.load_state_dict(torch.load(args.load_normalized_weight, map_location=torch.device('cpu')))
        model.to(device)
        print("=> Already Normalized")

    # Data load
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False)
    spikeloader = torch.utils.data.DataLoader(spikeset, batch_size=args.batchsize, shuffle=False)
    print("Data nums: ANN-{}, SNN-{}".format(len(testloader), len(spikeloader)))
    dlen = len(spikeset)
    acc = 0

    for i, data in enumerate(zip(testloader, spikeloader)):
        test_data = data[0]
        sp = data[1]
        Acc_step_per_batch = []

        outspike = torch.zeros(sp[0].size()[0], args.timelength, 10)
        labels = sp[1]
        model.reset(sp[0].size()[0])  # batch size to argument
        print("{} start. Voltage reseted".format(i))
        Vth = args.Vth

        vth_timestep = [Vth]
        # time step proccess
        for j in range(args.timelength):
            spike = sp[0][:, j, :, :, :]
            spike = spike.to(device)
            with torch.no_grad():
                out = model(spike)
                outspike[:, j, :] = out
                spikecount = torch.sum(outspike, axis=1)
                firerate = softmax(spikecount)
                firerate_softmax = softmax(firerate)
                entropy = torch.sum(-1 * firerate_softmax * torch.log2(firerate_softmax), dim=1)
                VthFunc.entropylog(entropy)

            # log acc per step
            if args.logging == 1:
                _, spikecount_argmax = torch.max(spikecount, dim=1)
                acc_tensor = torch.zeros_like(labels)
                acc_tensor[spikecount_argmax==labels] = 1
                Acc_step_per_batch.append(acc_tensor.sum().item())
                model_count, overfire, whole_fire_cnt = model.SaveFireCount(fireTimestepCSV, j)
            VthFunc.firelog([model_count, overfire])

            if (j + 1) % args.changeStep == 0:
                Vth = VthFunc(j+1, Vth)
                model.changeVth(Vth)
            vth_timestep.append(Vth)

        model.FireCount(timestep=args.timelength)

        spikecount = torch.sum(outspike, axis=1)
        _, spikecount_argmax = torch.max(spikecount, dim=1)
        acc_tensor = torch.zeros_like(labels)
        acc_tensor[spikecount_argmax==labels] = 1

        # Save accurate per image
        if args.logging == 1:
            bsize = spikecount_argmax.size(0)
            vth_list = torch.full((bsize, 1), args.Vth)
            indecies = torch.arange(bsize).reshape(-1, 1) + (args.batchsize * i)
            spikecount_argmax_info = spikecount_argmax.view(-1, 1)
            labels_info = labels.view(-1, 1)
            acc_tensor_info = acc_tensor.view(-1, 1)
            saveinfo = np.concatenate([vth_list, indecies, spikecount_argmax_info, labels_info, acc_tensor_info], -1)
            saveCSVrows(saveinfo.tolist(), savecsv)
            saveCSVrow(Acc_step_per_batch, accTimestepCSV)
            saveCSVrow(vth_timestep, vthTimestepCSV)

        acc += acc_tensor.sum().item()

    acc /= dlen
    print("Acc: {}".format(acc))



def main():
    args = parser.parse_args()

    traintransform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5 ,0.5), (0.5, 0.5, 0.5))]
    )
    testtransform = transforms.ToTensor()
    trainset = CIFAR10(root='./data', train=True, download=False, transform=traintransform)
    spiketestset = CIFAR10(root='./data', train=False, download=False, transform=testtransform)
    testset = CIFAR10(root='./data', train=False, download=False, transform=traintransform)
    Spikes = SpikeEncodeDatasets(spiketestset, timelength=args.timelength)
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    if args.logging == 1:
        args.savefolder = os.path.join(args.savefolder, date2foldername())
        os.makedirs(args.savefolder, exist_ok=True)

    spike_test(args, trainset, testset, Spikes, device)


if __name__ == "__main__":
    main()
