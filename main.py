# import click  #


# @click.command()
# @click.option(
#     "--mode",
#     default="inference",
#     type=click.Choice(["inference", "training"]),
# )
# def main(mode):
#     print(mode)
#     print("hello")
#     if mode == "inference":
#         print("hi!")
#         train()

# """Training."""
from ray.tune.schedulers import ASHAScheduler
import wandb
import torch
from torch import nn
from resnets.training.runNEpochs import runNEpochs
from torch.utils.data import random_split
from resnets.training.TrainingSettings import (
    AdamParameters,
    OptimiserWrapper,
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
        optimiserParameters = OptimiserWrapper(
            optimiser=AdamParameters(learningRate=5e-3, betas=(0.9, 0.8))
        )
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
        optimiserParameters2 = OptimiserWrapper(
            optimiser=AdamParameters(learningRate=1e-2, betas=(0.9, 0.8))
        )
        trainingParameters2 = trainingParameters.model_copy(
            update={"optimiserParameters": optimiserParameters2}
        )
        experimentParameters = ExperimentParameters(
            globalParameters=globalParameters,
            trainingParameters=trainingParameters,
            wandbParameters=wandbParameters,
        )
        experimentParameters2 = experimentParameters.model_copy(
            update={"trainingParameters": trainingParameters2}
        )
        self.experimentsParameters = [
            experimentParameters,
            experimentParameters2,
        ]
        self.trainingParameters = trainingParameters
        self.loss_fn = loss_fn
        self.modelChoice = modelChoice
        self.next(self.run_experiment, foreach="experimentsParameters")

    @step
    def run_experiment(self):
        self.experimentParameters = self.input
        with wandb.init(config=self.experimentParameters.model_dump()) as run:
            optimizer = self.experimentParameters.trainingParameters.optimiserParameters.optimiser.createOptimiser(
                self.modelChoice.parameters()
            )
            runNEpochs(
                trainingDatasets=self.trainingDatasets,
                trainingParameters=self.experimentParameters.trainingParameters,
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
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass


if __name__ == "__main__":
    TrainingFlow()
