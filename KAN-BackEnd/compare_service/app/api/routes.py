from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import json

from app.schemas.compare import CompareResponse, CompareResult
from app.crud.compare import analyze_and_predict
from app.utils.file import read_excel_file

compare_router = APIRouter()

@compare_router.post("/compare", response_model=CompareResponse)
async def compare_endpoint(
    file: UploadFile = File(...),
    dependent_names_json: str = Form(...)
):
    """
    接收Excel文件和因变量名，进行模型预测和比较。
    """
    try:
        # 读取Excel文件
        df = await read_excel_file(file)
            
        # 解析因变量名
        try:
            dependent_names: List[str] = json.loads(dependent_names_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="dependent_names_json格式错误")

        # 进行预测和分析
        results = analyze_and_predict(df, dependent_names)
        
        if not results:
            raise HTTPException(status_code=500, detail="预测失败或未找到指定的因变量列")

        return CompareResponse(data=results)
        
    except HTTPException as he:
        # 直接重新抛出HTTPException
        raise he
    except Exception as e:
        print(f"处理请求时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}") 