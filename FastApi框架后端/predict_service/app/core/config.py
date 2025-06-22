import secrets
from typing import Any, Dict, List, Optional, Union
from pydantic import AnyHttpUrl, BaseSettings, validator
import os


class Settings(BaseSettings):
    """
    系统核心配置类（继承自 Pydantic 的 BaseSettings）
    特性：
      - 自动从环境变量或 .env 文件加载配置
      - 支持类型验证和默认值
      - 通过类方法实现自定义验证逻辑
    """

    # >>>>>>>>>>>>>>> API 基础配置 <<<<<<<<<<<<<<<<
    API_V1_STR: str = "/api/v1/predict"
    # API 接口前缀，用于路由分组和文档生成

    # 简化密钥，确保两个服务使用完全相同的字符串
    SECRET_KEY: str = "simple-shared-secret-key-123456"
    # 安全密钥（静态密钥，用于多服务共享）
    # 用途：
    #   - JWT 令牌签名
    #   - 敏感数据加密
    # 生产环境必须通过 .env 文件覆盖此值

    PROJECT_NAME: str = "上海市宜居性系统数据预测服务"
    # 服务名称（用于日志标识和 API 文档展示）
    

    # >>>>>>>>>>>>>>> 令牌有效期配置 <<<<<<<<<<<<<<<<
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8 # 8天
    # 访问令牌有效期（单位：分钟）


    # >>>>>>>>>>>>>>> 跨域配置 <<<<<<<<<<<<<<<<
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = ["http://localhost:5173"]
    # 允许跨域请求的来源列表
    # 类型：AnyHttpUrl 确保必须是合法的 HTTP/HTTPS URL
    # TODO:部署后需改为前端域名

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """
        CORS 来源预处理验证器
        功能：将字符串形式的配置自动转换为列表
        """
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    

    # >>>>>>>>>>>>>>> JWT 配置 <<<<<<<<<<<<<<<<
    JWT_ALGORITHM: str = "HS256"
    # JWT 签名算法（HS256 表示 HMAC-SHA256）
    # TODO:生产环境建议使用 RS256（非对称加密）

    # 模型路径
    MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "..", "service", "models") # 模型目录路径
    KAN_MODEL_DIR: str = os.path.join(MODEL_DIR, "kan") # KAN模型目录


# 全局配置实例（其他模块通过 from app.core.config import settings 使用）
settings = Settings()