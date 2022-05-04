from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms

def get_cifar10_ANN_train_dataset(root):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((28, 28)),
            transforms.Pad((2, 2, 2, 2), padding_mode="edge"),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    trainset = CIFAR10(root=root, train=True, download=False, transform=train_transform)

    return trainset

def get_cifar10_ANN_test_dataset(root):
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    testset = CIFAR10(root=root, train=False, download=False, transform=test_transform)

    return testset
    