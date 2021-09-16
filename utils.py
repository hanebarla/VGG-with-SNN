import shutil
import datetime
import torch
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar', bestname='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


def date2foldername():
    dt_now = datetime.datetime.now()
    return dt_now.isoformat()

class SpikeEncodeDatasets():
    def __init__(self, dsets, timelength=100) -> None:
        self.dsets = dsets
        self.leng = len(dsets)
        self.timelength = timelength

    def __len__(self):
        return self.leng

    def __getitem__(self, idx):
        inputs, label = self.dsets[idx]
        spikes = torch.zeros(self.timelength, inputs.size()[0], inputs.size()[1], inputs.size()[2])

        for i in range(self.timelength):
            p_sp = torch.bernoulli(inputs)
            m_sp = -1 * torch.bernoulli(1.0 - inputs)
            spikes[i,:,:,:] = p_sp + m_sp


        # spikes[spikes==0] = -1

        return spikes, label


def printSave(message, filename):
    with open(filename, mode='a') as f:
        f.write(message + "\n")

    print(message)


if __name__ == "__main__":
    transform = transforms.ToTensor()
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    Spikes = SpikeEncodeDatasets(testset)
    print(type(testset))
    print(Spikes[0][0].size())

    stdict = torch.load("/home/thabara/Documents/VGG-with-SNN/normalized.pth.tar", map_location=torch.device('cpu'))
    print(stdict.keys())
