from pydantic import BaseModel, model_validator, model_serializer
from torch.nn.parameter import Parameter
from torch.optim.optimizer import ParamsT
from torch.utils.data import Dataset
import torch
from typing import Iterator


class WandbParameters(BaseModel):
    entity: str
    project: str


class SplittingRatios(BaseModel):
    trainingRatio: float
    validationRatio: float
    testingRatio: float

    @model_validator(mode="after")
    def assertSumOfRatiosIs1(self):
        sumRatios = self.trainingRatio + self.validationRatio + self.testingRatio
        if self.trainingRatio + self.validationRatio + self.testingRatio != 1:
            raise ValueError(
                f"The sum of the ratios must equal 1.0 but it is f{sumRatios}"
            )
        return self


class GlobalParameters(BaseModel):
    randomSeed: int


class ModelParameters(BaseModel):
    pass


class OptimiserParameters(BaseModel):
    def createOptimiser(self, modelParameters: ParamsT) -> torch.optim.Optimizer:
        raise NotImplementedError(
            "createOptimiser method is missing for class " + self.__class__.__name__
        )


class AdamParameters(OptimiserParameters):
    learningRate: float
    betas: tuple[float, float]

    def createOptimiser(self, modelParameters):
        return torch.optim.Adam(
            params=modelParameters, lr=self.learningRate, betas=self.betas
        )


class OptimiserWrapper(BaseModel):
    optimiser: OptimiserParameters

    @model_serializer
    def serialise_optimiser(self):
        return self.optimiser.model_dump()


class TrainingParameters(BaseModel):
    epochs: int
    optimiserParameters: OptimiserWrapper
    batchSize: int
    shuffle: bool
    loss: str
    splittingRatios: SplittingRatios
    trainingSize: int
    validationSize: int
    testingSize: int


class TrainingDatasets(BaseModel):
    trainingDataset: Dataset
    validationDataset: Dataset
    testingDataset: Dataset

    class Config:
        arbitrary_types_allowed = True


class ExperimentParameters(BaseModel):
    trainingParameters: TrainingParameters
    globalParameters: GlobalParameters
    wandbParameters: WandbParameters
