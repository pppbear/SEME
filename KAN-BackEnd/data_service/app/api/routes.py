from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import json

from app.schemas.data import DataResponse, DataRow
from app.crud.data import load_data_by_target
import os

data_router = APIRouter()

@data_router.get("/get_data", response_model=DataResponse)
async def get_data(
    target: str = Query(..., description="目标因变量（nighttime_、lst_day_c、lst_night_ 三选一）")
):
    """
    接收目标因变量，返回栅格数据。
    """
    if target not in ['nighttime_', 'lst_day_c', 'lst_night_']:
        raise HTTPException(status_code=400, detail="目标因变量不合法")
    try:
        data = load_data_by_target(target)
        return DataResponse(data=[DataRow(**item) for item in data])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))