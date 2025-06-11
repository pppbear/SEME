# 添加项目根目录到Python路径
import sys
import os
# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os.path as op

from auth_service.app.api.routes import auth_router
from auth_service.app.core.config import settings

from auth_service.app.db.base import Base
from auth_service.app.db.session import engine

# 调试信息
print("===== 认证服务启动 =====")
print(f"数据库连接: {settings.SQLITE_DATABASE_URI}")
print(f"JWT密钥: {settings.SECRET_KEY[:10]}...")
print(f"CORS配置: {settings.BACKEND_CORS_ORIGINS}")

# 获取项目根目录
ROOT_DIR = op.abspath(op.join(op.dirname(__file__), ".."))
db_file = op.join(ROOT_DIR, "common_db", "common_db.db")
    
print(f"数据库文件路径: {db_file}")
if op.exists(db_file):
    print(f"数据库文件存在，大小: {op.getsize(db_file) / 1024:.2f} KB")
else:
    print(f"警告: 数据库文件不存在，将尝试创建")

# 创建数据库表
Base.metadata.create_all(bind=engine)
print("数据库表已更新")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="上海市宜居性系统认证服务",
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
app.include_router(auth_router, prefix=settings.API_V1_STR)
