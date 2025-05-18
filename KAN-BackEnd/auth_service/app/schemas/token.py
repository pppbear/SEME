from typing import Optional
from pydantic import BaseModel


class Token(BaseModel):
    """
    JWT令牌响应模型
    用于API返回给客户端的访问令牌信息
    """
    access_token: str # 访问令牌（JWT字符串）
    token_type: str # 令牌类型


class TokenPayload(BaseModel):
    """
    JWT令牌负载模型
    定义JWT令牌解码后的负载内容
    """
    sub: Optional[int] = None # 主题标识
    exp: Optional[int] = None # 过期时间