from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import json

from data_service.app.schemas.data import DataResponse, DataRow
from data_service.app.service.data import load_data_by_target
import os

data_router = APIRouter()

@data_router.get("/get_data", response_model=DataResponse)
async def get_data(
    data: str = Query(..., description="目标因变量（nighttime_、lst_day_c、lst_night_c 三选一）")
):
    """
    接收目标因变量，返回栅格数据。
    """
    if data not in ['nighttime_', 'lst_day_c', 'lst_night_c']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="目标因变量不合法"
        )
    try:
        # 加载数据
        data = load_data_by_target(data)

        #返回数据
        return DataResponse(data=[DataRow(**item) for item in data])
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )