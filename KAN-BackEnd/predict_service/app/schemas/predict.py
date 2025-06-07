from pydantic import BaseModel
from typing import List
from fastapi import status

class PredictResponse(BaseModel):
    """预测响应模型"""
    code: int = status.HTTP_200_OK
    message: str = "success"
    data: List[float] 

