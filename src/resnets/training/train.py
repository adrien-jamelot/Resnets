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
from metaflow.flowspec import FlowSpec
from metaflow.decorators import step


class TrainingFlow(FlowSpec):
    @step
    def start(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
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
        self.splittingRatios = splittingRatios
        self.batch_size = batch_size
        self.trainingDatasets = trainingDatasets
        self.next(self.choose_settings)

    @step
    def choose_settings(self):
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
            splittingRatios=self.splittingRatios,
            trainingSize=len(self.trainingDatasets.trainingDataset),
            validationSize=len(self.trainingDatasets.validationDataset),
            testingSize=len(self.trainingDatasets.testingDataset),
        )
        self.experimentParameters = ExperimentParameters(
            globalParameters=globalParameters,
            trainingParameters=trainingParameters,
            wandbParameters=wandbParameters,
        )
        self.loss_fn = loss_fn
        self.modelChoice = modelChoice
        self.next(self.run_experiment)

    @step
    def run_experiment(self):
        with wandb.init(config=self.experimentParameters.model_dump()) as run:
            optimizer = self.experimentParameters.trainingParameters.optimiserParameters.createOptimiser(
                self.modelChoice.parameters()
            )
            runNEpochs(
                trainingDatasets=self.trainingDatasets,
                trainingParameters=self.trainingParameters,
                model=self.modelChoice,
                loss_fn=self.loss_fn,
                optimizer=optimizer,
            )
            testingDataloader = DataLoader(
                self.trainingDatasets.testingDataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
            )
            testingLoss = computeLoss(
                dataloader=testingDataloader,
                model=self.modelChoice,
                loss_fn=self.loss_fn,
            )
            run.log({"Testing Loss": testingLoss})
        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass


def train():
    TrainingFlow()
