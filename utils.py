import shutil
import torch
import torchvision


def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar', bestname='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


class SpikeDatasets():
    def __init__(self, dsets) -> None:
        self.dsets = dsets
        self.leng = len(dsets)
