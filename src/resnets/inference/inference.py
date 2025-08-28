import torch
import torchvision
import torchvision.transforms as transforms
from resnets.blocks.myResNet import ResNet18


def load():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 64

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    resnet18 = ResNet18()

    batchX, _ = next(iter(trainloader))
    print(batchX.shape)
    print(resnet18(batchX))
