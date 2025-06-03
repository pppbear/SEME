from pydantic import BaseModel
from typing import List

class PredictResponse(BaseModel):
    predictions: List[float] 