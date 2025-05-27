from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import httpx
from app.core.config import settings

# 创建API路由实例，用于组织所有接口
api_router = APIRouter()

# ------------------------- 基础接口 -------------------------
@api_router.get("/health")
async def health_check():
    """健康检查接口
    功能：用于服务探活或负载均衡检测
    """
    return {"status": "ok"}


# ------------------------- 根接口 -------------------------
@api_router.get("/")
async def root():
    """根接口
    功能：提供API基本信息
    """
    return {
        "message": "上海市宜居性系统API网关",
        "version": "0.1.0",
        "docs": f"{settings.API_V1_STR}/docs"
    }


# ------------------------- 微服务代理接口 -------------------------
# 认证服务代理
@api_router.api_route("/auth/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def auth_proxy(path: str, request: Request):
    """认证服务代理
    功能：将/auth开头的请求转发到认证微服务
    实现逻辑：
        1. 动态拼接目标URL（从配置获取AUTH_SERVICE_URL）
        2. 使用httpx异步转发原始请求
        3. 将响应原样返回给客户端
    参数：
        - path: URL路径的通配符
        - request: 原始请求对象（自动注入）
    """
    async with httpx.AsyncClient() as client:
        # 拼接目标地址，例如：http://auth_service:8001/api/v1/login
        url = f"{settings.AUTH_SERVICE_URL}/api/v1/auth/{path}"

        # 转换FastAPI Request为httpx请求
        headers = dict(request.headers)

        # 获取请求体
        body = await request.body() if request.method in ["POST", "PUT"] else None
        
        # 转发请求（保留方法、头、参数、体）
        response = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
            params=dict(request.query_params)
        )

        # 将响应包装为FastAPI的JSONResponse
        return JSONResponse(
            content=response.json(),
            status_code=response.status_code
        )


#TODO: 上海市宜居性系统服务代理
