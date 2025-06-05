from pydantic import BaseModel
from typing import List

class PredictResponse(BaseModel):
    """预测响应模型"""
    code: int = 200
    message: str = "success"
    data: List[float] 

