# """Training."""
import wandb
import torch
from torch import nn
from resnets.training.runNEpochs import runNEpochs
from torch.utils.data import random_split
from resnets.training.TrainingSettings import (
    AdamParameters,
    OptimiserParameters,
    SplittingRatios,
    GlobalParameters,
    TrainingParameters,
    TrainingDatasets,
    WandbParameters,
    ExperimentParameters,
)

from torchinfo import summary
from resnets.training.runOneEpoch import computeLoss
from torch.utils.data import DataLoader


import torchvision
import torchvision.transforms as transforms
from resnets.blocks.myResNet import (
    ResNetMedium,
    ResNetMini,
    ResNetMiniDeep,
    LMResNetMiniDeep,
)


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

    modelChoice = ResNetMedium()
    entity = "adrien-jamelot-dev-personal-project"
    project = "resnets-dynamics"
    wandbParameters = WandbParameters(entity=entity, project=project)
    loss_fn = nn.CrossEntropyLoss()
    globalParameters = GlobalParameters(randomSeed=42)
    optimiserParameters = AdamParameters(learningRate=5e-3, betas=(0.9, 0.8))
    trainingParameters = TrainingParameters(
        epochs=5,
        optimiserParameters=optimiserParameters,
        batchSize=64,
        shuffle=True,
        loss=loss_fn.__class__.__name__,
        splittingRatios=splittingRatios,
        trainingSize=len(trainingDataset),
        validationSize=len(validationDataset),
        testingSize=len(testingDataset),
    )
    experimentParameters = ExperimentParameters(
        globalParameters=globalParameters,
        trainingParameters=trainingParameters,
        wandbParameters=wandbParameters,
    )

    with wandb.init(config=experimentParameters.model_dump()) as run:
        optimizer = (
            experimentParameters.trainingParameters.optimiserParameters.createOptimiser(
                modelChoice.parameters()
            )
        )
        runNEpochs(
            trainingDatasets=trainingDatasets,
            trainingParameters=trainingParameters,
            model=modelChoice,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        testingDataloader = DataLoader(
            testingDataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        testingLoss = computeLoss(
            dataloader=testingDataloader, model=modelChoice, loss_fn=loss_fn
        )
        run.log({"Testing Loss": testingLoss})
