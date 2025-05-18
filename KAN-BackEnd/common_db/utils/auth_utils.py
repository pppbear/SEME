"""
共享认证工具模块
在不同服务间共享JWT验证逻辑
"""
from datetime import datetime, timedelta
from typing import Optional, Any, Dict, Union
from jose import jwt, JWTError
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auth_utils")

# 共享密钥配置
# 优先从环境变量读取，如果不存在则使用默认值
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "simple-shared-secret-key-123456")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 8))  # 默认8天

def create_access_token(
    subject: Union[str, Any], 
    expires_delta: Optional[timedelta] = None,
    extra_claims: Optional[Dict[str, Any]] = None
) -> str:
    """
    创建JWT访问令牌
    
    参数:
        subject: 令牌主题(通常是用户ID)
        expires_delta: 过期时间增量
        extra_claims: 额外的声明数据
    
    返回:
        JWT令牌字符串
    """
    try:
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        # 基本声明
        to_encode = {"exp": expire, "sub": str(subject)}
        
        # 添加额外声明
        if extra_claims:
            to_encode.update(extra_claims)
        
        # 编码并返回令牌
        token = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        logger.debug(f"创建令牌: subject={subject}, 过期时间={expire}")
        return token
    except Exception as e:
        logger.error(f"创建令牌失败: {str(e)}")
        raise

def decode_access_token(token: str, verify_signature: bool = True) -> Dict[str, Any]:
    """
    解码并验证JWT访问令牌
    
    参数:
        token: JWT令牌字符串
        verify_signature: 是否验证签名
    
    返回:
        解码后的令牌数据
    
    异常:
        JWTError: 令牌格式错误或验证失败
    """
    try:
        options = {"verify_signature": verify_signature}
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM], options=options)
        logger.debug(f"解码令牌成功: payload={payload}")
        return payload
    except JWTError as e:
        logger.warning(f"解码令牌失败: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"解码令牌时发生未知错误: {str(e)}")
        raise

def get_token_user_id(token: str) -> Optional[int]:
    """
    从令牌中获取用户ID
    
    参数:
        token: JWT令牌字符串
    
    返回:
        用户ID(整数)或None(解码失败)
    """
    try:
        payload = decode_access_token(token)
        user_id = int(payload.get("sub"))
        return user_id
    except (JWTError, ValueError) as e:
        logger.warning(f"从令牌获取用户ID失败: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"获取用户ID时发生未知错误: {str(e)}")
        return None 