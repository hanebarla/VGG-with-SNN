import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from model import SpikingConv2d, SpikingVGG16, Vgg16
from utils import SpikeEncodeDatasets, date2foldername, saveCSVrow, saveCSVrows


parser = argparse.ArgumentParser(description='PyTorch Spiking Test')
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--scale', default=1.0, type=float)
parser.add_argument('--Vth', default=1.0, type=float)
parser.add_argument('--Vres', default=0.0, type=float)
parser.add_argument('--activate', default="leaky")
parser.add_argument('--debug_layer', default=1, type=int)
parser.add_argument('--timelength', default=100, type=int)
parser.add_argument('--load_weight', default="0826/model_best.pth.tar")
parser.add_argument('--load_normalized_weight', default=None)
parser.add_argument('--savefolder', default="SNN_Test_Results/")


# Calculate lambda(max activations), and channel-wise Normalize
def CW_Normalize(args, model, trainset, device):
    dleng = len(trainset)
    acc = 0
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize)

    for data in trainloader:
        with torch.no_grad():
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model.calculate_lambda(inputs)

            output_argmax = torch.argmax(outputs, dim=1)

        acc_tensor = torch.zeros_like(labels)
        acc_tensor[output_argmax==labels] = 1
        acc += acc_tensor.sum().item()

    acc /= dleng
    print("ANN Train Acc: {}".format(acc)) # Check the accuracy in Trainset with ANN
    model.channel_wised_normlization()
    print("=> Model Normalize Success")

    return model


def LayerViewer(layer_num, ann_layer, snn_layer, batch=0):
    savefolder = "Layer_Debug/layer-{}_batch-{}".format(layer_num, batch)
    os.makedirs(savefolder, exist_ok=True)

    num_ann_layer = ann_layer[batch, ...].to('cpu').detach().numpy().copy()
    num_snn_layer = snn_layer[batch, ...].to('cpu').detach().numpy().copy()

    ch_size = num_ann_layer.shape[0]
    
    fig = plt.figure(figsize=(24, 8))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    for i in range(ch_size):
        view_config = "Layer-{} Channel-{}".format(layer_num, i)
        ax1.set_title("ANN " + view_config)
        ax2.set_title("SNN " + view_config)
        ax3.set_title("Difference " + view_config)
        ch_ann_layer = num_ann_layer[i, :, :]
        ch_snn_layer = num_snn_layer[i, :, :]

        num_ann_min = np.min(ch_ann_layer)
        print("ANN Layer channel-{}: max-{}, min-{}".format(i, np.max(ch_ann_layer), num_ann_min))
        ch_ann_layer += min(num_ann_min, 0)
        num_ann_max = np.max(ch_ann_layer)
        ch_ann_layer /= num_ann_max
        ax1.imshow(ch_ann_layer, cmap="jet")

        num_snn_min = np.min(ch_snn_layer)
        print("SNN Layer channel-{}: max-{}, min-{}".format(i, np.max(ch_snn_layer), num_snn_min))
        ch_snn_layer += min(num_snn_min, 0)
        num_snn_max = np.max(ch_snn_layer)
        ch_snn_layer /= num_snn_max
        ax2.imshow(ch_snn_layer, cmap="jet")

        diff = np.abs(ch_ann_layer - ch_snn_layer)
        ax3.imshow(diff, cmap="jet")

        fig.savefig(os.path.join(savefolder, "channel-{}.png".format(i)))
    
    print("Layer-{} View Completed".format(layer_num))


def spike_test(args, trainset, testset,  spikeset, device):
    savecsv = os.path.join(args.savefolder, "Vth-{}_result_per_image.csv".format(args.Vth))
    saveCSVrow(["Vth","index","spike_argmax","label","acc"], savecsv)

    # Load Mmodels
    if args.load_weight is not None:
        stdict = torch.load(args.load_weight, map_location=torch.device('cpu'))['state_dict']
    else:
        stdict = None
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

    # ANN_model to Debug
    ANN_model = Vgg16(activate=args.activate)
    ANN_model.load_state_dict(stdict)
    ANN_model.to(device)

    # SNN Initialize
    inp = torch.ones(1, 3, 32, 32)  # To get neauron size, so simulate as a scalemodel
    model = SpikingVGG16(stdict, block1, block2, classifier, inp, device, Vth=args.Vth, Vres=args.Vres)
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

        # ANN Test
        """
        ann_input, ann_label = test_data
        with torch.no_grad():
            ann_input = ann_input.to(device)
            ann_debug_layer = ANN_model.layer_debug(ann_input, layer_num=args.debug_layer, act_mode=args.activate)
            # print(debug_layer.size())
        """

        outspike = torch.zeros(sp[0].size()[0], args.timelength, 10)
        labels = sp[1]
        model.reset(sp[0].size()[0])  # batch size to argument
        print("{} start. Voltage reseted".format(i))

        # time step proccess
        for j in range(args.timelength):
            spike = sp[0][:, j, :, :, :]
            spike = spike.to(device)
            with torch.no_grad():
                out = model(spike)
            outspike[:, j, :] = out
            # break

        model.FireCount(timestep=args.timelength)
        # snn_debug_layer = model.layer_debug(layer_num=args.debug_layer)
        # print(snn_debug_layer.size())
        # LayerViewer(0, test_data[0], torch.sum(sp[0], 1))
        # LayerViewer(args.debug_layer, ann_debug_layer, snn_debug_layer)

        spikecount = torch.sum(outspike, axis=1)
        _, spikecount_argmax = torch.max(spikecount, dim=1)
        # print(spikecount_argmax.size())
        # print(torch.cat((torch.unsqueeze(labels, 1), torch.unsqueeze(spikecount_argmax, 1)), 1))
        acc_tensor = torch.zeros_like(labels)
        acc_tensor[spikecount_argmax==labels] = 1

        # Save accurate per image
        bsize = spikecount_argmax.size(0)
        vth_list = torch.full((bsize, 1), args.Vth)
        indecies = torch.arange(bsize).reshape(-1, 1) + (args.batchsize * i)
        spikecount_argmax_info = spikecount_argmax.view(-1, 1)
        labels_info = labels.view(-1, 1)
        acc_tensor_info = acc_tensor.view(-1, 1)
        saveinfo = np.concatenate([vth_list, indecies, spikecount_argmax_info, labels_info, acc_tensor_info], -1)
        saveCSVrows(saveinfo.tolist(), savecsv)

        acc += acc_tensor.sum().item()

    acc /= dlen
    print("Acc: {}".format(acc))



def main():
    VthCond = [(i+5)/10 for i in range(11)]
    args = parser.parse_args()
    args.savefolder = os.path.join(args.savefolder, date2foldername())
    os.makedirs(args.savefolder, exist_ok=True)

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

    for vth in VthCond:
        print("========== Vth: {} ==========".format(vth))
        args.Vth = vth
        spike_test(args, trainset, testset, Spikes, device)



if __name__ == "__main__":
    main()
