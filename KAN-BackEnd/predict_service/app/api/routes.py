from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List
import json
import os
from app.schemas.predict import PredictResponse
from app.crud.predict import predict_from_excel

predict_router = APIRouter()

@predict_router.post("/dependent_predict", response_model=PredictResponse)
async def dependent_predict(
    file: UploadFile = File(...),
    dependent_name: str = Form(..., description="目标因变量（nighttime_、lst_day_c、lst_night_c 三选一）")
):
    """
    接收Excel文件和因变量名，进行数据预测。
    """
    if dependent_name not in ['nighttime_', 'lst_day_c', 'lst_night_c']:
        raise HTTPException(status_code=400, detail="目标因变量不合法")
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名为空")
    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', file.filename)
    with open(filepath, "wb") as f:
        f.write(await file.read())
    try:
        predictions = predict_from_excel(filepath, dependent_name)
        os.remove(filepath)
        return PredictResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))