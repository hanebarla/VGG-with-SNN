import torch
import torchvision


class SpikeDatasets():
    def __init__(self, dsets) -> None:
        self.dsets = dsets
        self.leng = len(dsets)
