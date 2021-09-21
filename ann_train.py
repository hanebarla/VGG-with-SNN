import argparse
import os

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch
import torch.nn as nn
import torch.optim as optim
from model import Vgg16_BN, Vgg16
from utils import date2foldername, printSave, save_checkpoint


parser = argparse.ArgumentParser(description='PyTorch SNN')
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--optim', default="adam", choices=["sgd", "adam", "amsgrad"])
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--decay', default=1e-3, type=float)
parser.add_argument('--split_train_num', default=0.8, type=float)
parser.add_argument('--activate', default="leaky", choices=["leaky", "relu"])
parser.add_argument('--bn', default=0, type=int)
parser.add_argument('--DA', default=0, type=int)
parser.add_argument('--savefolder', default="/home/thabara/Documents/VGG-with-SNN/Ann_train_exp")


def main():
    args = parser.parse_args()
    savefolder = os.path.join(args.savefolder,
                              "optim-{}_lr-{}_decay-{}_split-{}_activate-{}_bn-{}_DA-{}".format(args.optim, args.lr, args.decay, args.split_train_num, args.activate, args.bn, args.DA)
                              )
    os.makedirs(savefolder, exist_ok=True)
    logfile = os.path.join(savefolder, "log.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    best_pred = 0

    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233, 0.24348505, 0.26158768))]
    )
    """

    if args.DA:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((28, 28)),
                transforms.Pad((2, 2, 2, 2), padding_mode="edge"),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
    # transform = transforms.ToTensor()
    trainset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    if args.bn == 1:
        model = Vgg16_BN(activate=args.activate)
    else:
        model = Vgg16(activate=args.activate)
    # for m in model.modules():
    #     print(m)
    # raise ValueError
    if torch.cuda.device_count() > 1:
        printSave("You can use {} GPUs!".format(torch.cuda.device_count()), filename=logfile)
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optim_fact = {
        "sgd": optim.SGD(model.parameters(), args.lr, weight_decay=args.decay),
        "adam": optim.Adam(model.parameters(), args.lr, weight_decay=args.decay),
        "amsgrad": optim.Adam(model.parameters(), args.lr, weight_decay=args.decay, amsgrad=True)
    }
    optimizer = optim_fact[args.optim]

    for e in range(args.epochs):
        printSave("=== [{} / {}] epochs ===".format(e+1, args.epochs), filename=logfile)

        train(trainset, model, criterion, optimizer, args, device, logfile)

        pred = val(testset, model, criterion, args, device, logfile)

        is_best = pred > best_pred
        best_pred = max(pred, best_pred)

        if torch.cuda.device_count() > 1:
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'val': pred
            }, is_best,
                filename=os.path.join(savefolder, 'checkpoint.pth.tar'),
                bestname=os.path.join(savefolder, 'model_best.pth.tar'))
        else:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'val': pred
            }, is_best,
                filename=os.path.join(savefolder, 'checkpoint.pth.tar'),
                bestname=os.path.join(savefolder, 'model_best.pth.tar'))

        printSave("Model Saved \n", filename=logfile)


def train(trainset, model, criterion, optimizer, args, device, logfilename):
    model.train()
    dlengs = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    
    train_loss = 0.0
    train_acc = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        output_argmax = torch.argmax(outputs, dim=1)
        acc_tensor = torch.zeros_like(labels)
        acc_tensor[output_argmax==labels] = 1

        train_loss += loss.item()
        train_acc += acc_tensor.sum().item()


        if (i % 100) == 0:
            printSave('[{} / {}] loss: {}'.format(min(args.batchsize*(i+1), dlengs), dlengs, loss.item()), filename=logfilename)

    train_loss /= dlengs
    train_acc /= dlengs

    printSave("#Train-acc: {}".format(train_acc), filename=logfilename)
    printSave('#Train-loss: {}'.format(train_loss), filename=logfilename)


def val(testset, model, criterion, args, device, logfilename):
    model.eval()
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
    printSave("#Val-acc: {}".format(val_acc), filename=logfilename)
    printSave('#Val-loss: {}'.format(val_loss), filename=logfilename)

    return val_acc


if __name__ == "__main__":
    main()
