from pydantic import BaseModel
from typing import List

class DataRow(BaseModel):
    longitude: float
    latitude: float
    value: float

class DataResponse(BaseModel):
    data: List[DataRow] 