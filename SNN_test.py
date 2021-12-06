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


parser = argparse.ArgumentParser(description='PyTorch Spiking Test')
parser.add_argument('--batchsize', default=128, type=int, help="specify bachsize")
parser.add_argument('--scale', default=1.0, type=float, help="To change weight for fire corresponding to Vth")
parser.add_argument('--Vth', default=1.0, type=float, help="spike threshold")
parser.add_argument('--Vres', default=0.0, type=float, help="membren voltage when reset")
parser.add_argument('--activate', default="leaky", help="Specify activate function")
parser.add_argument('--debug_layer', default=0, type=int, help="Specify to view debug layer")
parser.add_argument('--timelength', default=100, type=int, help="Specify time length")
parser.add_argument('--bn', default=0, type=int, help="Activate Batch Norm layer")
parser.add_argument('--percentile', default=0.999, type=float, help="Spcify normalize percentile")
parser.add_argument('--load_weight', default=None, help="ANN trained model file path")
parser.add_argument('--load_normalized_weight', default=None, help="SNN trained model normalized from ANN model")
parser.add_argument('--savefolder', default="SNN_Test_Results/", help="Experiment or Debug layer Save path")
parser.add_argument('--change_Vth', default=0, type=int, help="Explement condition one Vth or some Vth's")  # bool
parser.add_argument('--logging', default=0, type=int, help="if we want to save csv data per image")  # bool


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
        ax2.set_title("SNN(ANN forward) " + view_config)
        ax3.set_title("Difference " + view_config)
        ch_ann_layer = num_ann_layer[i, :, :]
        ch_snn_layer = num_snn_layer[i, :, :]

        num_ann_min = np.min(ch_ann_layer)
        print("ANN Layer channel-{}: max-{}, min-{}".format(i, np.max(ch_ann_layer), num_ann_min))
        ch_ann_layer += min(num_ann_min, 0)
        num_ann_max = np.max(ch_ann_layer)
        ch_ann_layer /= num_ann_max + 1e-5
        ax1.imshow(ch_ann_layer, cmap="jet")

        num_snn_min = np.min(ch_snn_layer)
        print("SNN(ANN forward) Layer channel-{}: max-{}, min-{}".format(i, np.max(ch_snn_layer), num_snn_min))
        ch_snn_layer += min(num_snn_min, 0)
        num_snn_max = np.max(ch_snn_layer)
        ch_snn_layer /= num_snn_max + 1e-5
        ax2.imshow(ch_snn_layer, cmap="jet")

        diff = np.abs(ch_ann_layer - ch_snn_layer)
        ax3.imshow(diff, cmap="jet")

        fig.savefig(os.path.join(savefolder, "channel-{}.png".format(i)))
    
    print("Layer-{} View Completed".format(layer_num))


def spike_test(args, trainset, testset,  spikeset, device):
    # logging set up
    if args.logging == 1:
        savecsv = os.path.join(args.savefolder, "Vth-{}_result_per_image.csv".format(args.Vth))
        saveCSVrow(["Vth","index","spike_argmax","label","acc"], savecsv)

    # Data load
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False)
    spikeloader = torch.utils.data.DataLoader(spikeset, batch_size=args.batchsize, shuffle=False)
    print("Data nums: ANN-{}, SNN-{}".format(len(testloader), len(spikeloader)))
    dlen = len(spikeset)

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
    ANN_model.eval()

    # ANN test with spike dataset
    AnnWithSpike_acc = 0
    for sp in spikeloader:
        outspike = torch.zeros(sp[0].size()[0], args.timelength, 10)
        labels = sp[1]

        # time step proccess
        for j in range(args.timelength):
            spike = sp[0][:, j, :, :, :]
            spike = spike.to(device)
            with torch.no_grad():
                out = ANN_model(spike)
            outspike[:, j, :] = out
            # break
        spikecount = torch.sum(outspike, axis=1)
        _, spikecount_argmax = torch.max(spikecount, dim=1)
        acc_tensor = torch.zeros_like(labels)
        acc_tensor[spikecount_argmax==labels] = 1
        AnnWithSpike_acc += acc_tensor.sum().item()
    print("Ann with Spiking data Acc:{}".format(AnnWithSpike_acc / dlen))

    # SNN Initialize
    inp = torch.ones(1, 3, 32, 32)  # To get neauron size, so simulate as a scalemodel
    model = SpikingVGG16(stdict, block1, block2, bn1, bn2, classifier, inp, device, Vth=args.Vth, Vres=args.Vres, activate=args.activate, percentile=args.percentile)

    # snn model acc check
    model.to(device)
    ann2snn_ann_acc = 0
    for i, data in enumerate(testloader):
        ann_input, ann_label = data
        with torch.no_grad():
            ann_input = ann_input.to(device)
            ann_label = ann_label.to(device)
            ann_out = model.ann_forward(ann_input)
            ann_out_argmax = torch.argmax(ann_out, dim=1)
        acc_tensor = torch.zeros_like(ann_label)
        acc_tensor[ann_out_argmax==ann_label] = 1
        ann2snn_ann_acc += acc_tensor.sum().item()
    print("ANN to SNN's ann acc: {}".format(ann2snn_ann_acc / dlen))
    model.to('cpu')

    # Normalize parameters of ann model to the snn model's
    if args.load_normalized_weight is None:
        model.to(device)
        model = CW_Normalize(args, model, trainset, device)
        NormalizeSaveName = os.path.join(os.path.dirname(args.load_weight), "normalized.pth.tar")
        torch.save(model.state_dict(), NormalizeSaveName)
    else:
        model.load_state_dict(torch.load(args.load_normalized_weight, map_location=torch.device('cpu')))
        model.to(device)
        print("=> Already Normalized")
    model.eval()

    # snn model acc check
    # model.to(device)
    ann2snn_ann_acc = 0
    for i, data in enumerate(testloader):
        ann_input, ann_label = data
        with torch.no_grad():
            ann_input = ann_input.to(device)
            ann_label = ann_label.to(device)
            ann_out = model.ann_forward(ann_input)
            ann_out_argmax = torch.argmax(ann_out, dim=1)
        acc_tensor = torch.zeros_like(ann_label)
        acc_tensor[ann_out_argmax==ann_label] = 1
        ann2snn_ann_acc += acc_tensor.sum().item()
    print("ANN to SNN's ann acc(normalized): {}".format(ann2snn_ann_acc / dlen))
    # model.to('cpu')

    acc = 0
    # Finally, remodeled snn's acc
    for i, data in enumerate(zip(testloader, spikeloader)):
        test_data = data[0]
        sp = data[1]

        # ANN Test
        ann_input, ann_label = test_data
        with torch.no_grad():
            ann_input = ann_input.to(device)
            ann_debug_layer = ANN_model.layer_debug(ann_input, layer_num=args.debug_layer, act_mode=args.activate)
            # print(debug_layer.size())
            # normalized_model_map = model.ann_forward(ann_input, layer_num=args.debug_layer)

        """
        view_batch = 0
        LayerViewer(0, test_data[0], torch.sum(sp[0], 1), batch=view_batch)
        LayerViewer(args.debug_layer, ann_debug_layer, normalized_model_map, batch=view_batch)

        print(model.block[0].lambda_before)
        print(model.block[0].lambda_after)
        """
        
        # break
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
        if args.debug_layer != 0:
            view_batch = 0
            snn_debug_layer = model.layer_debug(layer_num=args.debug_layer)
            print(snn_debug_layer.size())
            LayerViewer(0, test_data[0], torch.sum(sp[0], 1), batch=view_batch)
            LayerViewer(args.debug_layer, ann_debug_layer, snn_debug_layer, batch=view_batch)

        spikecount = torch.sum(outspike, axis=1)
        _, spikecount_argmax = torch.max(spikecount, dim=1)
        # print("View batch: Pred {}, Label {}".format(spikecount_argmax[view_batch], labels[view_batch]))
        # print(spikecount_argmax.size())
        # print(torch.cat((torch.unsqueeze(labels, 1), torch.unsqueeze(spikecount_argmax, 1)), 1))
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

        acc += acc_tensor.sum().item()

        if args.debug_layer != 0:
            break

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

    if args.change_Vth == 0:
        spike_test(args, trainset, testset, Spikes, device)
    elif args.change_Vth == 1:
        args.savefolder = os.path.join(args.savefolder, date2foldername())
        os.makedirs(args.savefolder, exist_ok=True)
        VthCond = [(i+5)/10 for i in range(11)]
        for vth in VthCond:
            print("========== Vth: {} ==========".format(vth))
            args.Vth = vth
            spike_test(args, trainset, testset, Spikes, device)
    



if __name__ == "__main__":
    main()
