import secrets
from typing import Any, Dict, List, Optional, Union
from pydantic import AnyHttpUrl, BaseSettings, validator


class Settings(BaseSettings):
    """
    系统核心配置类（基于Pydantic的BaseSettings）
    特点：
      1. 自动从环境变量或.env文件加载配置
      2. 支持类型验证和默认值
      3. 可通过类方法添加自定义验证逻辑
    """

    # >>>>>>>>>>>>>>> 基础配置 <<<<<<<<<<<<<<<<
    API_V1_STR: str = "/api/v1"
    # API接口前缀，用于Swagger文档路由等

    SECRET_KEY: str = secrets.token_urlsafe(32)
    # 安全密钥（默认生成随机值）
    # 用途：
    #   - JWT令牌签名
    #   - CSRF保护
    #   - 加密敏感数据
    # 生产环境应通过环境变量覆盖（避免使用默认值）
    
    PROJECT_NAME: str = "上海市宜居性系统"
    # 项目名称（用于文档展示和日志标识）

    # >>>>>>>>>>>>>>> JWT配置 <<<<<<<<<<<<<<<<
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    # Token过期时间（分钟）
    # 计算公式：60分钟 * 24小时 * 8天 = 11520分钟


    # >>>>>>>>>>>>>>> CORS跨域配置 <<<<<<<<<<<<<<<<
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = ["http://localhost:5173"]
    # 允许的跨域请求来源列表
    # 类型：AnyHttpUrl确保必须是合法URL格式


    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """
        CORS来源预处理验证器
        功能：将字符串形式的配置自动转换为列表
        示例：
          输入 "http://a.com,http://b.com" => ["http://a.com", "http://b.com"]
          输入 ["http://a.com"] => 保持不变
        """
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)


    # >>>>>>>>>>>>>>> 微服务通信配置 <<<<<<<<<<<<<<<<
    AUTH_SERVICE_URL: str = "http://localhost:8001"
    # 认证服务地址

    COMPARE_SERVICE_URL: str = "http://localhost:8002"
    # 模型对比服务地址

    DATA_SERVICE_URL: str = "http://localhost:8003"
    # 数据服务地址

    PREDICT_SERVICE_URL: str = "http://localhost:8004"
    # 数据预测服务地址

    ANALYZE_SERVICE_URL: str = "http://localhost:8005"
    # 特征关键值分析服务地址


# 全局配置实例
settings = Settings() 
