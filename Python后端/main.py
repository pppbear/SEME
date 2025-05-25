from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from services.predictor import predict_from_excel
from schemas.predict import PredictResponse
from schemas.analyze import AnalyzeResponse, VariableImportance
from services.analyzer import analyze_variables_with_kan

app = FastAPI(
    title="上海市宜居性分析后端API",
    description="这是我的后端API文档说明。",
    version="1.0.0"
)

# 允许跨域（如有需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="需要上传的 Excel 文件"),
    target: str = Form(..., description="目标因变量（nighttime_、lst_day_c、lst_night_ 三选一）")
):
    if target not in ['nighttime_', 'lst_day_c', 'lst_night_']:
        raise HTTPException(status_code=400, detail="目标因变量不合法")
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名为空")
    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', file.filename)
    with open(filepath, "wb") as f:
        f.write(await file.read())
    try:
        predictions = predict_from_excel(filepath, target)
        os.remove(filepath)
        return PredictResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(..., description="需要上传的 Excel 文件"),
    target: str = Form(..., description="目标因变量名称")
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名为空")
    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', file.filename)
    with open(filepath, "wb") as f:
        f.write(await file.read())
    try:
        importances = analyze_variables_with_kan(filepath, target)
        os.remove(filepath)
        return AnalyzeResponse(importances=[VariableImportance(**imp) for imp in importances])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 