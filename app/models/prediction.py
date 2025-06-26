from pydantic import BaseModel
from typing import List, Optional


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictionInput(BaseModel):
    features: List[float]
    model_version: Optional[str] = "v1"


class PredictionOutput(BaseModel):
    prediction: int
    probabilities: List[float]
    model_version: str
    timestamp: str
    status: str = "success"