import torch
import torch.nn as nn
from model import Vgg16


if __name__ == "__main__":
    model = Vgg16()
    checkpoint = torch.load("Ann_train_exp/checkpoint.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
