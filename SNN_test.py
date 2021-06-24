import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from model import SpikingVGG16
from utils import SpikeEncodeDatasets


if __name__ == "__main__":
    batchsize = 32
    transform = transforms.ToTensor()
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    Spikes = SpikeEncodeDatasets(testset)
    spikeloader = torch.utils.data.DataLoader(Spikes, batch_size=batchsize)

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
    inp = torch.ones(1, 3, 32, 32)
    model = SpikingVGG16(stdict, block1, block2, classifier, inp)
    model.cuda()

    with torch.no_grad():
        for sp in spikeloader:
            outspike = torch.zeros_like(batchsize, 100, 10)
            for i in range(100):
                spike = sp[0][:, i, :, :, :]
                out = model(spike)
                outspike[:, i, :, :] = out

            spikecount = torch.sum(outspike, axis=1)
            print(spikecount.size())

            break

        
