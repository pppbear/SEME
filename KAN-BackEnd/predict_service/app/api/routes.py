from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query, status
from fastapi.responses import JSONResponse
from typing import List
import json
import os
from predict_service.app.schemas.predict import PredictResponse
from predict_service.app.service.predict import predict_from_excel
from common_utils.file import read_excel_file

predict_router = APIRouter()

# 预测接口
@predict_router.post("/dependent_predict", response_model=PredictResponse)
async def dependent_predict(
    file: UploadFile = File(...),
    dependent_name: str = Form(..., description="目标因变量（nighttime_、lst_day_c、lst_night_c 三选一）")
):
    """
    接收Excel文件和因变量名，进行模型预测和比较。
    """
    if dependent_name not in ['nighttime_', 'lst_day_c', 'lst_night_c']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="目标因变量不合法"
        )
    try:
        # 读取Excel文件
        df = await read_excel_file(file)

        # 进行预测和分析
        results = predict_from_excel(df, dependent_name)
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="预测失败或未找到指定的因变量列"
            )

        return PredictResponse(data=results)
        
    except HTTPException as he:
        # 直接重新抛出HTTPException
        raise he
    except Exception as e:
        print(f"处理请求时发生错误: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"内部服务器错误: {str(e)}"
        ) 
    
