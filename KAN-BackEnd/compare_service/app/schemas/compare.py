from typing import List
from pydantic import BaseModel
from fastapi import status

class CompareResult(BaseModel):
    """单个因变量的预测结果和指标模型"""
    dependent_name: str
    true_values: List[float]
    mlp_predictions: List[float]
    rf_predictions: List[float]
    kan_predictions: List[float]
    mse_mlp: float
    r2_mlp: float
    mse_rf: float
    r2_rf: float
    mse_kan: float
    r2_kan: float

class CompareResponse(BaseModel):
    """预测比较服务的响应模型"""
    code: int = status.HTTP_200_OK
    message: str = "success"
    data: List[CompareResult]
