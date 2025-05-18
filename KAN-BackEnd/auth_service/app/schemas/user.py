from typing import Optional
from pydantic import BaseModel, EmailStr
from datetime import datetime


# 共享属性
class UserBase(BaseModel):
    """
    用户模型的基类
    包含用户实体的通用属性（所有操作共享）
    """
    email: Optional[EmailStr] = None # 邮箱
    username: Optional[str] = None # 用户名
    is_active: Optional[bool] = True # 账户激活状态


# 创建时需要额外的属性
class UserCreate(UserBase):
    """
    用户创建模型
    继承自UserBase，添加注册时必须提供的字段
    """
    email: EmailStr # 强制要求邮箱
    username: str # 强制要求用户名
    password: str # 明文密码
    verification_code: str # 邮箱验证码


# 请求发送注册验证码
class EmailVerification(BaseModel):
    """
    邮箱验证码请求模型
    用于发送验证码到指定邮箱
    """
    email: EmailStr # 邮箱地址


# 重置密码请求
class PasswordReset(BaseModel):
    """
    密码重置请求模型
    用于重置用户密码
    """
    email: EmailStr # 邮箱地址
    new_password: str # 新密码
    verification_code: str # 邮箱验证码


# 更新时的属性
class UserUpdate(UserBase):
    """
    用户更新模型
    允许用户更新部分信息（密码可选）
    """
    password: Optional[str] = None # 新密码


# 登录请求
class UserLogin(BaseModel):
    """
    用户登录请求模型
    支持用户名或邮箱登录
    """
    username: str # 可以是用户名或邮箱
    password: str # 密码


# 数据库中存储的完整用户信息
class UserInDB(UserBase):
    """
    数据库用户模型
    表示数据库中存储的完整用户信息（包含敏感字段）
    """
    id: int # 用户唯一ID
    hashed_password: str # 哈希后的密码（非明文存储）
    is_superuser: bool = False # 超级管理员标识
    created_at: datetime # 账户创建时间
    updated_at: datetime # 最后更新时间

    class Config:
        orm_mode = True # 启用ORM兼容模式（允许从SQLAlchemy模型转换）


# API返回的用户信息（不包含密码等敏感信息）
class User(UserBase):
    """
    API用户响应模型
    返回给前端的用户信息（排除敏感字段）
    """
    id: int
    is_superuser: bool = False
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True 