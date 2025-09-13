import torch
from torch import Tensor
from torch.types import Number
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
import torch.nn.functional as F
import math
import wandb
import metaflow
from torchmetrics.classification import MulticlassAccuracy


def runOneEpoch(
    trainingDataloader: DataLoader,
    validationDataloader: DataLoader,
    model: nn.Module,
    loss_fn: _Loss,
    optimizer: Optimizer,
    epoch: int,
):
    print(f"Running epoch {epoch}")
    datasetSize = len(trainingDataloader.dataset)
    batchSize = trainingDataloader.batch_size
    nbBatches = len(trainingDataloader)
    model.train()
    logFrequency = math.floor((datasetSize / batchSize) // 100)
    print(f"logFrequency: {logFrequency}")
    for batchIndex, (X, y) in enumerate(trainingDataloader):
        pred, loss = trainingStep(X, y, model, loss_fn, optimizer)
        if batchIndex % logFrequency == 0:
            with torch.no_grad():
                accuracy = MulticlassAccuracy(num_classes=10)
                lossValue = loss.item()
                printTrainingStatus(
                    lossValue,
                    batchIndex,
                    datasetSize,
                    batchSize,
                )
                wandb.log(
                    {"Training Loss": lossValue},
                    step=(nbBatches // logFrequency + 1) * logFrequency * epoch
                    + batchIndex,
                )
                wandb.log(
                    {"Training accuracy": accuracy(model(X), y)},
                    step=(nbBatches // logFrequency + 1) * logFrequency * epoch
                    + batchIndex,
                )

    trainingLoss = computeLoss(
        dataloader=validationDataloader, model=model, loss_fn=loss_fn
    )
    validationLoss = computeLoss(
        dataloader=validationDataloader, model=model, loss_fn=loss_fn
    )
    wandb.log({"Validation Loss": validationLoss}, step=epoch)
    wandb.log({"Total Training Loss": trainingLoss}, step=epoch)
    return trainingLoss


def trainingStep(
    X: Tensor, y: Tensor, model: nn.Module, loss_fn: _Loss, optimizer: Optimizer
) -> tuple[Tensor, Tensor]:
    # Compute prediction and loss
    pred = model(X)
    target = F.one_hot(y.long(), 10).float()
    loss = loss_fn(pred, target)
    # Backpropagation
    loss.backward()
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    return pred, loss


@torch.no_grad()
def computeLoss(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: _Loss,
) -> float:
    model.eval()
    validationLoss = 0
    for X, y in dataloader:
        target = F.one_hot(y.long(), 10).float()
        validationLoss += loss_fn(model(X), target)
    return validationLoss / len(dataloader)


def printTrainingStatus(
    lossValue: Number,
    batch: int,
    datasetSize: int,
    batchSize: int,
):
    currentExample = batch * batchSize + 1
    print(f"loss: {lossValue:>7f}  [{currentExample:>5d}/{datasetSize:>5d}]")


@torch.no_grad()
def computeMetrics(
    dataloader: DataLoader,
    model: nn.Module,
) -> float:
    model.eval()
    acc = 0
    for X, y in dataloader:
        accuracy = MulticlassAccuracy(num_classes=10)
        acc += accuracy(model(X), y)
    return acc / len(dataloader)
