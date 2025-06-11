import secrets
from typing import Any, Dict, List, Optional, Union
from pydantic import AnyHttpUrl, BaseSettings, validator, EmailStr
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
    API_V1_STR: str = "/api/v1/auth"
    # API 接口前缀，用于路由分组和文档生成

    # 简化密钥，确保两个服务使用完全相同的字符串
    SECRET_KEY: str = "simple-shared-secret-key-123456"
    # 安全密钥（静态密钥，用于多服务共享）
    # 用途：
    #   - JWT 令牌签名
    #   - 敏感数据加密
    # 生产环境必须通过 .env 文件覆盖此值

    PROJECT_NAME: str = "上海市宜居性系统认证服务"
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
    

    # >>>>>>>>>>>>>>> 数据库配置 <<<<<<<<<<<<<<<<
    SQLITE_DATABASE_URI: str = "sqlite:///../../common_db/common_db.db"
    # SQLite 数据库连接字符串
    # 格式说明：sqlite:///<相对路径>
    # 相对于 auth_service/app/core/ 的位置指向 common_db/common_db.db
    

    # >>>>>>>>>>>>>>> 密码哈希配置 <<<<<<<<<<<<<<<<
    PASSWORD_HASHING_ALGORITHM: str = "bcrypt"
    # 密码哈希算法（支持 bcrypt, argon2 等）
    # 需与 security.py 中的 CryptContext 配置一致
    

    # >>>>>>>>>>>>>>> JWT 配置 <<<<<<<<<<<<<<<<
    JWT_ALGORITHM: str = "HS256"
    # JWT 签名算法（HS256 表示 HMAC-SHA256）
    # TODO:生产环境建议使用 RS256（非对称加密）
    

    # >>>>>>>>>>>>>>> 邮件配置 <<<<<<<<<<<<<<<<
    EMAIL_SENDER: EmailStr = "2784892686@qq.com"  # 发件人邮箱
    SMTP_SERVER: str = "smtp.qq.com"  # SMTP服务器地址
    SMTP_PORT: int = 465  # SMTP服务器端口
    SMTP_USE_TLS: bool = False  # 465端口不使用TLS
    SMTP_USE_SSL: bool = True  # 使用SSL连接
    SMTP_USERNAME: str = "2784892686@qq.com"  # SMTP登录用户名
    SMTP_PASSWORD: str = "lftkwqcvmxaedhbg"  # SMTP登录密码

    # >>>>>>>>>>>>>>> Redis 配置 <<<<<<<<<<<<<<<<
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", None)


# 全局配置实例（其他模块通过 from app.core.config import settings 使用）
settings = Settings()