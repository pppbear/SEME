# 添加项目根目录到Python路径
import sys
import os
# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os.path as op

from analyze_service.app.api.routes import analyze_router
from analyze_service.app.core.config import settings

# 调试信息
print("===== 关键特征值分析服务启动 =====")
print(f"JWT密钥: {settings.SECRET_KEY[:10]}...")
print(f"CORS配置: {settings.BACKEND_CORS_ORIGINS}")

# 获取项目根目录
ROOT_DIR = op.abspath(op.join(op.dirname(__file__), ".."))

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="上海市宜居性系统关键特征值分析服务",
    version="0.1.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含所有API路由
app.include_router(analyze_router, prefix=settings.API_V1_STR)
