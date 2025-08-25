from pydantic import BaseModel, model_validator
from torch.utils.data import Dataset


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


class TrainingParameters(BaseModel):
    epochs: int
    learningRate: float
    betas: tuple[float, float]
    optimizer: str
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
