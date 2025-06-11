# 添加项目根目录到Python路径
import sys
import os
# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_gateway.app.api.routes import api_router # 主路由文件
from api_gateway.app.core.config import settings # 系统配置

# 调试信息
print("===== API网关启动 =====")
print(f"认证服务URL: {settings.AUTH_SERVICE_URL}")
print(f"模型对比服务URL: {settings.COMPARE_SERVICE_URL}")
print(f"栅格数据服务URL: {settings.DATA_SERVICE_URL}")
print(f"CORS配置: {settings.BACKEND_CORS_ORIGINS}")

# ------------------------- FastAPI应用初始化 -------------------------
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="上海市宜居性系统API网关",
    version="0.1.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)
"""
FastAPI实例化关键参数说明：
- title/docs标签页显示的标题
- description/API文档的详细描述
- version/API版本（遵循语义化版本）
- openapi_url/自定义OpenAPI schema路径（默认在/docs可见）
"""

# ------------------------- CORS跨域配置 -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS, # 允许的源列表（从配置读取）
    allow_credentials=True, # 允许携带Cookie
    allow_methods=["*"], # 允许所有HTTP方法
    allow_headers=["*"], # 允许所有请求头
)
"""
CORS策略说明：
1. allow_origins: 生产环境应严格指定前端地址（如["https://your-domain.com"]）
2. allow_credentials: 为True时前端才能接收Set-Cookie
"""

# ------------------------- 路由注册 -------------------------
app.include_router(api_router, prefix=settings.API_V1_STR)
print(f"API路由注册完成，前缀: {settings.API_V1_STR}")
print(f"认证转发路由: {settings.API_V1_STR}/auth/*")
print(f"模型对比转发路由: {settings.API_V1_STR}/compare/*")
print(f"栅格数据转发路由: {settings.API_V1_STR}/data/*")
print(f"栅格数据转发路由: {settings.API_V1_STR}/analyze/*")
"""
路由挂载说明：
- api_router: 来自routes.py的所有路由
- prefix: 为所有路由添加前缀（如/api/v1/auth/login）
效果：
  原路由 /health => 实际访问路径 /api/v1/health
"""

