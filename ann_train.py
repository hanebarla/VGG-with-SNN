import argparse
import os

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch
import torch.nn as nn
import torch.optim as optim
from model import Vgg16_BN
from utils import save_checkpoint


parser = argparse.ArgumentParser(description='PyTorch CANNet2s')
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--decay', default=1e-3, type=float)
parser.add_argument('--savefolder', default="/home/thabara/Documents/VGG-with-SNN/Ann_train_exp")


def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    best_pred = 0

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    model = Vgg16_BN()
    if torch.cuda.device_count() > 1:
        print("You can use {} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.decay)

    for e in range(args.epochs):
        print("=== [{} / {}] epochs ===".format(e+1, args.epochs))

        train(trainset, model, criterion, optimizer, args, device)

        pred = val(testset, model, criterion, args, device)

        is_best = pred > best_pred
        best_pred = max(pred, best_pred)

        if torch.cuda.device_count() > 1:
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'val': pred
            }, is_best,
                filename=os.path.join(args.savefolder, 'checkpoint.pth.tar'),
                bestname=os.path.join(args.savefolder, 'model_best.pth.tar'))
        else:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'val': pred
            }, is_best,
                filename=os.path.join(args.savefolder, 'checkpoint.pth.tar'),
                bestname=os.path.join(args.savefolder, 'model_best.pth.tar'))

        print("Model Saved \n")


def train(trainset, model, criterion, optimizer, args, device):
    dlengs = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i % 50) == 0:
            print('[{} / {}] loss: {}'.format(min(args.batchsize*(i+1), 50000), dlengs, loss.item()))


def val(testset, model, criterion, args, device):
    dlengs = len(testset)
    valloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)

    val_loss = 0.0
    val_acc = 0.0
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

        val_loss += loss.item()
        val_acc += acc_tensor.sum().item()
    
    val_loss /= dlengs
    val_acc /= dlengs
    print("#Val-acc: {}".format(val_acc))
    print('#Val-loss: {}'.format(val_loss))

    return val_acc


if __name__ == "__main__":
    main()
