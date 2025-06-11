from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta, datetime
from sqlalchemy.orm import Session

from auth_service.app.api import deps # 依赖注入模块
from auth_service.app.core import security # 安全相关工具
from auth_service.app.core.config import settings # 配置管理
from auth_service.app.schemas.token import Token # Token响应模型
from auth_service.app.schemas.user import User, UserCreate, UserUpdate, EmailVerification, UserLogin, PasswordReset # 用户模型
from auth_service.app.schemas.auth import AuthResponse
from common_db.models.user import User as UserModel # 数据库用户模型
from auth_service.app.crud.user import (
    create_user_with_verification,
    get_user_by_email,
    authenticate_user,
    reset_password,
    get_user_by_username
)
from auth_service.app.utils.email import send_verification_code

# 创建API路由实例
auth_router = APIRouter()


@auth_router.post("/login", 
                 summary="用户登录",
                 description="通过用户名或邮箱和密码获取访问令牌",
                 response_model=dict)
async def login_for_access_token(
    form_data: UserLogin, # 使用自定义登录模型
    db: Session = Depends(deps.get_db)
):
    """用户登录获取令牌
    流程：
      1. 验证用户名/邮箱和密码
      2. 生成JWT访问令牌
      3. 返回令牌给客户端
    """
    # 调用CRUD层的认证函数
    print(f"尝试验证用户: {form_data.username}")
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="用户名/邮箱或密码不正确"
        )
    
    print(f"用户验证成功: {user.username}, ID: {user.id}")
    # 生成令牌过期时间（从配置读取）
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # 生成JWT令牌，包含额外用户信息
    extra_claims = {
        "username": user.username,
        "email": user.email
    }
    
    # 使用共享库生成访问令牌
    from common_db.utils.auth_utils import create_access_token
    access_token = create_access_token(
        subject=user.id, 
        expires_delta=access_token_expires,
        extra_claims=extra_claims
    )
    
    # 确定用户角色
    role = "admin" if user.is_active and getattr(user, 'is_superuser', False) else "user"
    
    # 计算令牌过期时间
    expire = (datetime.utcnow() + access_token_expires).timestamp()
    
    # 返回标准格式响应
    print(f"生成令牌成功，过期时间: {expire}")

    return AuthResponse(
        code=status.HTTP_200_OK,
        message="success",
        data={
            "token": access_token,
            "expire": expire,
            "role": role,
            "username": user.username
        }
    )


@auth_router.post("/oauth-login", 
                 summary="OAuth2表单登录",
                 description="支持OAuth2标准表单登录")
async def oauth_login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(deps.get_db)
):
    """OAuth2兼容的登录端点（用于第三方客户端）"""
    # 调用CRUD层的认证函数
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        # 使用统一的错误响应格式
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="用户名/邮箱或密码不正确"
        )
        
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # 使用共享库生成访问令牌
    from common_db.utils.auth_utils import create_access_token
    access_token = create_access_token(
        subject=user.id, 
        expires_delta=access_token_expires
    )
    
    # 确定用户角色
    role = "admin" if user.is_superuser else "user"
    
    # 计算令牌过期时间
    expire = (datetime.utcnow() + access_token_expires).timestamp()
    
    # 返回标准格式响应
    return AuthResponse(
        code=status.HTTP_200_OK,
        message="success",
        data={
            "token": access_token,
            "expire": expire,
            "role": role,
            "username": user.username
        }
    )
    

@auth_router.post("/register/send-code",
                 summary="发送注册验证码",
                 description="向指定邮箱发送注册验证码")
async def send_register_code(
    email_data: EmailVerification,
    db: Session = Depends(deps.get_db)
):
    """发送注册验证码
    流程：
      1. 检查邮箱是否已注册
      2. 生成验证码并发送
    """
    # 检查邮箱是否已注册
    user = get_user_by_email(db, email=email_data.email)
    if user:
        # 使用统一的错误响应格式
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="该邮箱已被注册"
        )
        
    # 发送验证码
    success = send_verification_code(email_data.email, "register")
    if not success:
        # 使用统一的错误响应格式
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="验证码发送失败，请稍后重试"
        )
        
    return AuthResponse(
        code=status.HTTP_200_OK,
        message="验证码已发送至邮箱，有效期为10分钟"
    )
    

@auth_router.post("/register", 
                 summary="用户注册",
                 description="通过邮箱验证码注册新用户")
async def register_user(
    user_in: UserCreate,
    db: Session = Depends(deps.get_db)
):
    """用户注册（带验证码）
    流程：
      1. 检查邮箱是否已注册
      2. 检查用户名是否已存在
      3. 验证邮箱验证码
      4. 创建新用户
    """
    # 检查邮箱是否已注册
    existing_user = get_user_by_email(db, email=user_in.email)
    if existing_user:
        # 使用统一的错误响应格式
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, 
            detail="该邮箱已被注册"
        )
        
    # 检查用户名是否已存在
    existing_username = get_user_by_username(db, username=user_in.username)
    if existing_username:
        # 使用统一的错误响应格式
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, 
            detail="该用户名已被使用"
        )
        
    # 创建用户（包含验证码验证）
    user = create_user_with_verification(db, user_in)
    if not user:
        # 使用统一的错误响应格式
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="验证码无效或已过期，请重新获取"
        )
        
    # 转换用户对象为API响应格式
    user_data = {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat() if hasattr(user, 'created_at') and user.created_at else None
    }
    
    return AuthResponse(
        code=status.HTTP_200_OK,
        message="success",
        data={
            "data": user_data
        }
    )
    

@auth_router.post("/reset-password/send-code",
                summary="发送密码重置验证码",
                description="向指定邮箱发送密码重置验证码")
async def send_reset_password_code(
    email_data: EmailVerification,
    db: Session = Depends(deps.get_db)
):
    """发送密码重置验证码
    流程：
      1. 检查邮箱是否存在
      2. 生成验证码并发送
    """
    # 检查邮箱是否存在
    user = get_user_by_email(db, email=email_data.email)
    if not user:
        # 使用统一的错误响应格式
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="该邮箱未注册"
        )
        
    # 发送验证码
    success = send_verification_code(email_data.email, "reset_password")
    if not success:
        # 使用统一的错误响应格式
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="验证码发送失败，请稍后重试"
        )
        
    return AuthResponse(
        code=status.HTTP_200_OK,
        message="验证码已发送至邮箱，有效期为10分钟"
    )
    

@auth_router.post("/reset-password",
                summary="重置密码",
                description="通过邮箱验证码重置密码")
async def reset_user_password(
    reset_data: PasswordReset,
    db: Session = Depends(deps.get_db)
):
    """重置用户密码
    流程：
      1. 验证邮箱验证码
      2. 更新用户密码
    """
    # 调用CRUD层的密码重置函数
    user = reset_password(db, reset_data)
    if not user:
        # 使用统一的错误响应格式
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="密码重置失败，请确认信息正确后重试"
        )
        
    return AuthResponse(
        code=status.HTTP_200_OK,
        message="密码重置成功，请使用新密码登录"
    )