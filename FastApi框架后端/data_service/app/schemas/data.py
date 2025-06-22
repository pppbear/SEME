from pydantic import BaseModel
from typing import List
from fastapi import status

class DataRow(BaseModel):
    """每行数据对应的三种特征值模型"""
    longitude: float
    latitude: float
    value: float

class DataResponse(BaseModel):
    """数据响应模型"""
    code: int = status.HTTP_200_OK
    message: str = "success"
    data: List[DataRow] 