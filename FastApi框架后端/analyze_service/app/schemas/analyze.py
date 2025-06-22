from pydantic import BaseModel
from typing import List
from fastapi import status

class KeyFeature(BaseModel):
    """关键特征模型"""
    feature_name: str
    feature_value: float

class DependentKeyFeatureAnalyzeResponse(BaseModel):
    """因变量关键特征预测响应模型"""
    code: int = status.HTTP_200_OK
    message: str = "success"
    data: List[KeyFeature]

