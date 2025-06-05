from pydantic import BaseModel
from typing import List

class KeyFeature(BaseModel):
    """关键特征模型"""
    feature_name: str
    feature_value: float

class DependentKeyFeatureAnalyzeResponse(BaseModel):
    """因变量关键特征预测响应模型"""
    code: int = 200
    message: str = "success"
    data: List[KeyFeature]

