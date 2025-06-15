from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
import httpx
from api_gateway.app.core.config import settings

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
        # 拼接目标地址
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
            params=dict(request.query_params),
            timeout=60
        )

        # 健壮处理响应内容
        try:
            content = response.json()
            return JSONResponse(content=content, status_code=response.status_code)
        except Exception:
            return Response(
                content=response.text,
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "text/plain")
            )

# 模型对比代理
@api_router.api_route("/compare/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def compare_proxy(path: str, request: Request):
    """模型对比服务代理
    功能：将/compare开头的请求转发到模型对比微服务
    实现逻辑：
        1. 动态拼接目标URL（从配置获取COMPARE_SERVICE_URL）
        2. 使用httpx异步转发原始请求
        3. 将响应原样返回给客户端
    参数：
        - path: URL路径的通配符
        - request: 原始请求对象（自动注入）
    """
    async with httpx.AsyncClient() as client:
        # 拼接目标地址，例如：http://compare_service:8002/api/v1/compare
        url = f"{settings.COMPARE_SERVICE_URL}/api/v1/compare/{path}"

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
            params=dict(request.query_params),
            timeout=60
        )

        # 健壮处理响应内容
        try:
            content = response.json()
            return JSONResponse(content=content, status_code=response.status_code)
        except Exception:
            return Response(
                content=response.text,
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "text/plain")
            )

# 数据服务代理
@api_router.api_route("/data/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def data_proxy(path: str, request: Request):
    """数据服务代理
    功能：将/data开头的请求转发到数据微服务
    实现逻辑：
        1. 动态拼接目标URL（从配置获取DATA_SERVICE_URL）
        2. 使用httpx异步转发原始请求
        3. 将响应原样返回给客户端
    参数：
        - path: URL路径的通配符
        - request: 原始请求对象（自动注入）
    """
    async with httpx.AsyncClient() as client:
        # 拼接目标地址，例如：http://data_service:8003/api/v1/data
        url = f"{settings.DATA_SERVICE_URL}/api/v1/data/{path}"

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
            params=dict(request.query_params),
            timeout=60
        )

        # 健壮处理响应内容
        try:
            content = response.json()
            return JSONResponse(content=content, status_code=response.status_code)
        except Exception:
            return Response(
                content=response.text,
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "text/plain")
            )

# 数据预测服务代理
@api_router.api_route("/predict/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def predict_proxy(path: str, request: Request):
    """数据预测服务代理
    功能：将/predict开头的请求转发到数据预测微服务
    实现逻辑：
        1. 动态拼接目标URL（从配置获取PREDICT_SERVICE_URL）
        2. 使用httpx异步转发原始请求
        3. 将响应原样返回给客户端
    参数：
        - path: URL路径的通配符
        - request: 原始请求对象（自动注入）
    """
    async with httpx.AsyncClient() as client:
        # 拼接目标地址，例如：http://predict_service:8004/api/v1/predict
        url = f"{settings.PREDICT_SERVICE_URL}/api/v1/predict/{path}"

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
            params=dict(request.query_params),
            timeout=60
        )

        # 健壮处理响应内容
        try:
            content = response.json()
            return JSONResponse(content=content, status_code=response.status_code)
        except Exception:
            return Response(
                content=response.text,
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "text/plain")
            )

# 特征关键值分析服务代理
@api_router.api_route("/analyze/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def analyze_proxy(path: str, request: Request):
    """特征关键值分析服务代理
    功能：将/analyze开头的请求转发到特征关键值分析服务
    实现逻辑：
        1. 动态拼接目标URL（从配置获取ANALYZE_SERVICE_URL）
        2. 使用httpx异步转发原始请求
        3. 将响应原样返回给客户端
    参数：
        - path: URL路径的通配符
        - request: 原始请求对象（自动注入）
    """
    async with httpx.AsyncClient() as client:
        # 拼接目标地址，例如：http://analyze_service:8005/api/v1/analyze
        url = f"{settings.ANALYZE_SERVICE_URL}/api/v1/analyze/{path}"

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
            params=dict(request.query_params),
            timeout=90
        )

        # 健壮处理响应内容
        try:
            content = response.json()
            return JSONResponse(content=content, status_code=response.status_code)
        except Exception:
            return Response(
                content=response.text,
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "text/plain")
            )

