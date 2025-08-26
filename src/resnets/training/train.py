# """Training."""
from mlflow import artifacts
import torch
from torch import nn
from resnets.training.runNEpochs import runNEpochs
from torch.utils.data import random_split
from resnets.training.TrainingSettings import (
    SplittingRatios,
    GlobalParameters,
    TrainingParameters,
    TrainingDatasets,
)

from torchinfo import summary
import mlflow.pytorch as mlpt
import mlflow
from resnets.training.runOneEpoch import computeLoss
from torch.utils.data import DataLoader


import torchvision
import torchvision.transforms as transforms
from resnets.blocks.myResNet import ResNetMini


def load():
    pass


def train():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 64

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    globalParameters = GlobalParameters(randomSeed=42)
    splittingRatios = SplittingRatios(
        trainingRatio=0.8, validationRatio=0.2, testingRatio=0
    )
    trainingDataset, validationDataset = random_split(
        trainset,
        [splittingRatios.trainingRatio, splittingRatios.validationRatio],
        torch.Generator().manual_seed(globalParameters.randomSeed),
    )
    testingDataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    trainingDatasets = TrainingDatasets(
        trainingDataset=trainingDataset,
        validationDataset=validationDataset,
        testingDataset=testingDataset,
    )

    # classes = (
    #     "plane",
    #     "car",
    #     "bird",
    #     "cat",
    #     "deer",
    #     "dog",
    #     "frog",
    #     "horse",
    #     "ship",
    #     "truck",
    # )

    resnetMini = ResNetMini()
    loss_fn = nn.CrossEntropyLoss()
    globalParameters = GlobalParameters(randomSeed=42)

    trainingParameters = TrainingParameters(
        epochs=10,
        learningRate=5e-3,
        betas=(0.9, 0.8),
        optimizer="Adam",
        batchSize=64,
        shuffle=True,
        loss=loss_fn.__class__.__name__,
        splittingRatios=splittingRatios,
        trainingSize=len(trainingDataset),
        validationSize=len(validationDataset),
        testingSize=len(testingDataset),
    )

    optimizer = torch.optim.Adam(
        resnetMini.parameters(),
        lr=trainingParameters.learningRate,
        betas=trainingParameters.betas,
    )
    with mlflow.start_run():
        # Log model summary.
        with open("resnet_summary.txt", "w") as f:
            f.write(str(summary(resnetMini)))
            mlflow.log_artifact("resnet_summary.txt")

        runNEpochs(
            trainingDatasets=trainingDatasets,
            trainingParameters=trainingParameters,
            model=resnetMini,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        testingDataloader = DataLoader(
            testingDataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        X = next(iter(testingDataloader))
        testingLoss = computeLoss(
            dataloader=testingDataloader, model=resnetMini, loss_fn=loss_fn
        )
        mlflow.log_metric(key="Testing Loss", value=testingLoss)
        mlpt.log_model(
            resnetMini,
            name="ResNetMini",
            input_example=X[0].numpy(),
        )
