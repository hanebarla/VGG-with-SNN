import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from model import SpikingConv2d, SpikingVGG16
from utils import SpikeEncodeDatasets


if __name__ == "__main__":
    batchsize = 32
    transform = transforms.ToTensor()
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    Spikes = SpikeEncodeDatasets(testset)
    spikeloader = torch.utils.data.DataLoader(Spikes, batch_size=batchsize)
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu") 

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

    # print(stdict['classifier.4.weight'].size())

    # raise ValueError

    inp = torch.ones(1, 3, 32, 32)  # To get neauron size, so simulate as a scalemodel
    model = SpikingVGG16(stdict, block1, block2, classifier, inp, device)
    model.to(device)

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

        