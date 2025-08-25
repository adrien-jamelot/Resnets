import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from resnets.training.TrainingSettings import TrainingDatasets, TrainingParameters
from resnets.training.runOneEpoch import runOneEpoch
from torch.utils.data import DataLoader


def runNEpochs(
    trainingDatasets: TrainingDatasets,
    trainingParameters: TrainingParameters,
    model: nn.Module,
    loss_fn: _Loss,
    optimizer: torch.optim.Optimizer,
):
    trainingDataloader = DataLoader(
        trainingDatasets.trainingDataset,
        batch_size=trainingParameters.batchSize,
        shuffle=True,
    )
    validationDataloader = DataLoader(
        trainingDatasets.validationDataset,
        batch_size=trainingParameters.batchSize,
        shuffle=True,
    )

    for t in range(trainingParameters.epochs):
        print(f"Epoch {t}\n-------------------------------")
        runOneEpoch(
            trainingDataloader,
            validationDataloader,
            model,
            loss_fn,
            optimizer,
            t,
        )
