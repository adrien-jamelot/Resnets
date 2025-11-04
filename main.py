# """Training."""
import os
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import optuna
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

import torchvision
import torchvision.transforms as transforms
from resnets.blocks.myResNet import ResNetMedium
from metaflow.flowspec import FlowSpec
from metaflow.decorators import step
from metaflow.multicore_utils import parallel_map


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
        self.next(self.run_hop)

    @step
    def run_hop(self):
        modelChoice = ResNetMedium()
        entity = "adrien-jamelot-dev-personal-projects"
        project = "resnets-dynamics"
        wandbParameters = WandbParameters(entity=entity, project=project)
        loss_fn = nn.CrossEntropyLoss()
        globalParameters = GlobalParameters(randomSeed=42)

        from optuna.integration import WeightsAndBiasesCallback

        wandbc = WeightsAndBiasesCallback(
            wandb_kwargs=wandbParameters.model_dump(), as_multirun=True
        )

        @wandbc.track_in_wandb()
        def objective(trial):
            print(f"Running trial {trial.number=} in process {os.getpid()}")
            learning_rate = trial.suggest_float(
                name="learning_rate", low=1e-4, high=1e-3
            )
            beta1 = trial.suggest_float(name="beta_1", low=0.6, high=1.0)
            beta2 = trial.suggest_float(name="beta_2", low=beta1, high=1.0)

            optimiserParameters = OptimiserWrapper(
                optimiser=AdamParameters(
                    learningRate=learning_rate, betas=(beta1, beta2)
                )
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
            experimentParameters = ExperimentParameters(
                globalParameters=globalParameters,
                trainingParameters=trainingParameters,
                wandbParameters=wandbParameters,
            )
            wandb.log(experimentParameters.model_dump())
            optimizer = experimentParameters.trainingParameters.optimiserParameters.optimiser.createOptimiser(
                modelChoice.parameters()
            )
            validationLoss = runNEpochs(
                trainingDatasets=self.trainingDatasets,
                trainingParameters=experimentParameters.trainingParameters,
                model=modelChoice,
                loss_fn=loss_fn,
                optimizer=optimizer,
            )
            return validationLoss

        def run_optimization():
            study = optuna.create_study(
                study_name="journal_storage_multiprocess",
                storage=JournalStorage(JournalFileBackend(file_path="./journal.log")),
                load_if_exists=True,  # Useful for multi-process or multi-node optimization.
            )
            study.optimize(objective, n_trials=3, callbacks=[wandbc])

        parallel_map(lambda _: run_optimization(), range(18), max_parallel=6)

        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass


if __name__ == "__main__":
    TrainingFlow()
