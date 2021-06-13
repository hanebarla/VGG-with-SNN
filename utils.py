import shutil
import torch
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar', bestname='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


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
            spikes[i,:,:,:] = torch.bernoulli(inputs)

        spikes[spikes==0] = -1

        return spikes, label


if __name__ == "__main__":
    transform = transforms.ToTensor()
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    Spikes = SpikeEncodeDatasets(testset)
    print(Spikes[0][0].size())
