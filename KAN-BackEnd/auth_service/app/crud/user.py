from typing import Optional, Union
from sqlalchemy.orm import Session

from auth_service.app.core.security import get_password_hash, verify_password
from common_db.models.user import User
from auth_service.app.schemas.user import UserCreate, UserUpdate, PasswordReset
from auth_service.app.utils.email import verify_code


def get_user(db: Session, id: int) -> Optional[User]:
    """根据用户ID查询用户
    参数：
      - db: 数据库会话
      - id: 用户唯一标识
    返回：
      - User 对象（如果存在）或 None
    """
    return db.query(User).filter(User.id == id).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """根据邮箱查询用户
    用途：
      - 注册时检查邮箱是否重复
      - 通过邮箱找回密码
    """
    return db.query(User).filter(User.email == email).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """根据用户名查询用户
    用途：
      - 用户登录时的身份验证
    """
    return db.query(User).filter(User.username == username).first()


def create_user_with_verification(db: Session, user_in: UserCreate) -> Optional[User]:
    """创建新用户（带验证码）
    流程：
      1. 验证验证码是否正确
      2. 将明文密码哈希化
      3. 构建用户对象并提交到数据库
    """
    # 验证邮箱验证码
    is_valid = verify_code(user_in.email, user_in.verification_code, "register")
    if not is_valid:
        return None  # 验证码无效，返回None
    
    # 创建用户
    db_user = User(
        email=user_in.email,
        username=user_in.username,
        hashed_password=get_password_hash(user_in.password),
        is_active=True,
        is_superuser=False,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def create_user(db: Session, user_in: UserCreate) -> User:
    """创建新用户（无验证码，仅内部使用）
    流程：
      1. 将明文密码哈希化
      2. 构建用户对象
      3. 提交到数据库
    安全注意：
      - 永远不存储明文密码
    """
    db_user = User(
        email=user_in.email,
        username=user_in.username,
        hashed_password=get_password_hash(user_in.password),
        is_active=True,
        is_superuser=False,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def reset_password(db: Session, reset_data: PasswordReset) -> Optional[User]:
    """重置用户密码
    流程：
      1. 根据邮箱查找用户
      2. 验证重置验证码
      3. 更新密码
    """
    # 查找用户
    user = get_user_by_email(db, reset_data.email)
    if not user:
        return None
    
    # 验证重置验证码
    is_valid = verify_code(reset_data.email, reset_data.verification_code, "reset_password")
    if not is_valid:
        return None
    
    # 更新密码
    user.hashed_password = get_password_hash(reset_data.new_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return user


def authenticate_user(db: Session, username_or_email: str, password: str) -> Optional[User]:
    """用户认证（支持用户名或邮箱登录）
    流程：
      1. 尝试通过用户名或邮箱查询用户
      2. 验证密码哈希是否匹配
    """
    # 先尝试作为用户名查询
    user = get_user_by_username(db, username=username_or_email)
    
    # 如果找不到，再尝试作为邮箱查询
    if not user:
        user = get_user_by_email(db, email=username_or_email)
    
    # 如果还是找不到用户或密码不匹配，返回None
    if not user or not verify_password(password, user.hashed_password):
        return None
    
    return user 