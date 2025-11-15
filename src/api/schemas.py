from typing import List
from pydantic import BaseModel


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class IrisBatchRequest(BaseModel):
    samples: List[IrisFeatures]


class IrisPrediction(BaseModel):
    class_id: int
    class_name: str
    probability: float


class IrisBatchPrediction(BaseModel):
    predictions: List[IrisPrediction]
