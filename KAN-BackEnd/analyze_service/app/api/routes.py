from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query, status
from fastapi.responses import JSONResponse
from typing import List
import json
import os
from analyze_service.app.schemas.analyze import KeyFeature, DependentKeyFeatureAnalyzeResponse
from analyze_service.app.service.analyze import analyze_key_features_from_df
from common_utils.file import read_excel_file

analyze_router = APIRouter()


@analyze_router.post("/dependent_feature_analyze", response_model=DependentKeyFeatureAnalyzeResponse)
async def dependent_feature_analyze(
    file: UploadFile = File(...),
    dependent_name: str = Form(..., description="目标因变量（nighttime_、lst_day_c、lst_night_c 三选一）")
):
    """
    接收Excel文件和因变量名，进行数据分析。
    """
    if dependent_name not in ['nighttime_', 'lst_day_c', 'lst_night_c']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="目标因变量不合法"
        )
    try:
        # 读取Excel文件
        df = await read_excel_file(file)

        # 进行特征重要性分析
        key_features = analyze_key_features_from_df(df, dependent_name)
        if not key_features:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="分析失败或未找到指定的因变量列"
            )

        return DependentKeyFeatureAnalyzeResponse(data=key_features)
        
    except HTTPException as he:
        # 直接重新抛出HTTPException
        raise he
    except Exception as e:
        print(f"处理请求时发生错误: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"内部服务器错误: {str(e)}"
        ) 
