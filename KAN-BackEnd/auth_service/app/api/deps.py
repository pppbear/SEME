from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from pydantic import ValidationError
from sqlalchemy.orm import Session

from auth_service.app.core.config import settings
from auth_service.app.db.session import SessionLocal # 数据库会话工厂
from common_db.models.user import User # 数据库用户模型
from auth_service.app.schemas.token import TokenPayload # JWT负载模型
from auth_service.app.crud.user import get_user # 用户查询方法

# 定义OAuth2密码模式（令牌获取URL指向登录接口）
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/login")


def get_db() -> Generator:
    """依赖项：获取数据库会话
    生命周期管理：
      - 每个请求独立会话
      - 请求结束后自动关闭
    """
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


def get_current_user(
    db: Session = Depends(get_db), 
    token: str = Depends(oauth2_scheme) # 自动从请求头提取Bearer Token
) -> User:
    """依赖项：通过JWT令牌获取当前用户
    安全流程：
      1. 解码并验证JWT令牌
      2. 查询对应用户
      3. 检查用户状态
    """
    try:
        # 解码JWT（使用配置中的密钥和算法）
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        # 将payload转换为Pydantic模型（自动验证字段）
        token_data = TokenPayload(**payload)
        if token_data.sub is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无法验证凭据",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except (jwt.JWTError, ValidationError): # 捕获所有JWT相关错误
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无法验证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 从数据库查询用户
    user = get_user(db, id=token_data.sub)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户未激活"
        )
    return user


def get_current_active_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    """依赖项：验证当前用户是否为超级管理员
    权限控制：
      - 普通用户访问此接口会返回403错误
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="权限不足"
        )
    return current_user 