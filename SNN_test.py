import argparse
import os
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from model import SpikingConv2d, SpikingVGG16
from utils import SpikeEncodeDatasets, date2foldername


parser = argparse.ArgumentParser(description='PyTorch CANNet2s')
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--scale', default=1.0, type=float)
parser.add_argument('--vth', default=0.9, type=float)
parser.add_argument('--vres', default=0.0, type=float)
parser.add_argument('--timelength', default=100, type=int)
parser.add_argument('--load_weight', default="0822/model_best.pth.tar")
parser.add_argument('--load_normalized_weight', default=None)
parser.add_argument('--savefolder', default="SNN_Test_Results/")


def CW_Normalize(args, model, trainset, device):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize)

    for data in trainloader:
        with torch.no_grad():
            inputs, _ = data
            inputs = inputs.to(device)
            model.calculate_lambda(inputs)
    
    model.channel_wised_normlization()
    print("=> Model Normalize Success")

    return model


def spike_test(args, trainset, spikeset, device):
    # Load Mmodels
    if args.load_normalized_weight is None:
        stdict = torch.load(args.load_weight, map_location=torch.device('cpu'))['state_dict']
    else:
        stdict = torch.load(args.load_normalized_weight, map_location=torch.device('cpu'))
    # model = SpikingVGG16(stdict)
    block1 = (
        (3, 64, 'block.0.weight', 'block.0.bias', 'block.2.weight', 'block.2.bias'),
        (64, 128, 'block.5.weight', 'block.5.bias', 'block.7.weight', 'block.7.bias')
    )
    block2 = (
        (128, 256, 'block.10.weight', 'block.10.bias', 'block.12.weight', 'block.12.bias', 'block.14.weight', 'block.14.bias'),
        (256, 512, 'block.17.weight', 'block.17.bias', 'block.19.weight', 'block.19.bias', 'block.21.weight', 'block.21.bias'),
        (512, 512, 'block.24.weight', 'block.24.bias', 'block.26.weight', 'block.26.bias', 'block.28.weight', 'block.28.bias')
    )
    classifier = ('classifier.0.weight', 'classifier.0.bias', 'classifier.2.weight', 'classifier.2.bias', 'classifier.4.weight', 'classifier.4.bias')
    print("=> Load Weight Success")
    # print(stdict['block.0.bias'].size())

    inp = torch.ones(1, 3, 32, 32)  # To get neauron size, so simulate as a scalemodel
    model = SpikingVGG16(stdict, block1, block2, classifier, inp, device)
    model.to(device)


    if args.load_normalized_weight is None:
        model = CW_Normalize(args, model, trainset, device)
        torch.save(model.state_dict(), "normalized.pth.tar")
    else:
        print("=> Already Normalized")


    spikeloader = torch.utils.data.DataLoader(spikeset, batch_size=args.batchsize)

    dlen = len(spikeset)
    acc = 0
    for i, sp in enumerate(spikeloader):
        outspike = torch.zeros(sp[0].size()[0], args.timelength, 10)
        labels = sp[1]
        model.reset(sp[0].size()[0])  # batch size to argument
        print("{} start. Voltage reseted".format(i))

        # time step proccess
        for i in range(args.timelength):
            spike = sp[0][:, i, :, :, :]
            spike = spike.to(device)
            with torch.no_grad():
                out = model(spike)
            outspike[:, i, :] = out
            # break

        model.FireCount()
        break

        spikecount = torch.sum(outspike, axis=1)
        spikecount_argmax = torch.max(spikecount, dim=1)
        acc_tensor = torch.zeros_like(labels)
        acc_tensor[spikecount_argmax==labels] = 1

        acc += acc_tensor.sum().item()

    acc /= dlen
    print("Acc: {}".format(acc))



def main():
    args = parser.parse_args()
    args.savefolder = os.path.join(args.savefolder, date2foldername())
    print(args.savefolder)

    traintransform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5 ,0.5), (0.5, 0.5, 0.5))]
    )
    testtransform = transforms.ToTensor()
    trainset = CIFAR10(root='./data', train=True, download=False, transform=traintransform)
    testset = CIFAR10(root='./data', train=False, download=False, transform=testtransform)
    Spikes = SpikeEncodeDatasets(testset, timelength=args.timelength)
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    spike_test(args, trainset, Spikes, device)



if __name__ == "__main__":
    main()
    """
    batchsize = 32
    transform = transforms.ToTensor()
    trainset = CIFAR10(root='./data', train=True, download=False, transform=transform)
    testset = CIFAR10(root='./data', train=False, download=False, transform=transform)
    Spikes = SpikeEncodeDatasets(testset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize)
    spikeloader = torch.utils.data.DataLoader(Spikes, batch_size=batchsize)
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # Load Mmodels
    stdict = torch.load("/home/thabara/Documents/VGG-with-SNN/0624/model_best.pth.tar", map_location=torch.device('cpu'))['state_dict']
    # model = SpikingVGG16(stdict)
    block1 = (
        (3, 64, 'block.0.weight', 'block.0.bias', 'block.2.weight', 'block.2.bias'),
        (64, 128, 'block.5.weight', 'block.5.bias', 'block.7.weight', 'block.7.bias')
    )
    block2 = (
        (128, 256, 'block.10.weight', 'block.10.bias', 'block.12.weight', 'block.12.bias', 'block.14.weight', 'block.14.bias'),
        (256, 512, 'block.17.weight', 'block.17.bias', 'block.19.weight', 'block.19.bias', 'block.21.weight', 'block.21.bias'),
        (512, 512, 'block.24.weight', 'block.24.bias', 'block.26.weight', 'block.26.bias', 'block.28.weight', 'block.28.bias')
    )
    classifier = ('classifier.0.weight', 'classifier.0.bias', 'classifier.2.weight', 'classifier.2.bias', 'classifier.4.weight', 'classifier.4.bias')

    print(stdict['block.0.bias'].size())

    # raise ValueError

    inp = torch.ones(1, 3, 32, 32)  # To get neauron size, so simulate as a scalemodel
    model = SpikingVGG16(stdict, block1, block2, classifier, inp, device)
    model.to(device)

    # Calculate lambda
    for data in trainloader:
        inputs, _ = data
        inputs = inputs.to(device)

        model.calculate_lambda(inputs)

    model.channel_wised_normlization()

    for sp in spikeloader:
        outspike = torch.zeros(batchsize, 100, 10)
        model.reset(sp[0].size()[0])  # batch size to argument
        
        # 100 time step
        for i in range(100):
            spike = sp[0][:, i, :, :, :]
            spike = spike.to(device)
            with torch.no_grad():
                out = model(spike)
            outspike[:, i, :] = out

        spikecount = torch.sum(outspike, axis=1)
        print(spikecount.size())

        break
    """
        