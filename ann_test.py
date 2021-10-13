import argparse
import os

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch
import torch.nn as nn
import torch.optim as optim
from model import Vgg16_BN, Vgg16
from utils import save_checkpoint


parser = argparse.ArgumentParser(description='PyTorch SNN')
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--bn', default=0, type=int)
parser.add_argument('--load_weight', default="0822/model_best.pth.tar")


def test(testset, model, criterion, args, device):
    model.eval()
    dlengs = len(testset)
    valloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)

    test_loss = 0.0
    test_acc = 0.0
    for i, data in enumerate(valloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            output_argmax = torch.argmax(outputs, dim=1)

        acc_tensor = torch.zeros_like(labels)
        acc_tensor[output_argmax==labels] = 1

        test_loss += loss.item()
        test_acc += acc_tensor.sum().item()
    
    test_loss /= dlengs
    test_acc /= dlengs
    print("#Test-acc: {}".format(test_acc))
    print('#Test-loss: {}'.format(test_loss))

    return test_acc


def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    best_pred = 0

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    if args.bn == 1:
        model = Vgg16_BN()
    else:
        model = Vgg16()
    print(model.block[0])
    model.load_state_dict(torch.load(args.load_weight)["state_dict"])
    if torch.cuda.device_count() > 1:
        print("You can use {} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    pred = test(testset, model, criterion, args, device)

if __name__ == "__main__":
    main()