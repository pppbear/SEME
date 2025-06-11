from datetime import datetime, timedelta
from typing import Any, Union, Optional
from jose import jwt
from passlib.context import CryptContext
from auth_service.app.core.config import settings

# 导入共享认证库
import sys
import os
# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from common_db.utils.auth_utils import create_access_token, decode_access_token

# 密码哈希上下文配置（与 config.py 中的算法配置联动）
pwd_context = CryptContext(schemes=[settings.PASSWORD_HASHING_ALGORITHM], # 使用的哈希算法
                           deprecated="auto" # 自动标记旧算法为弃用
                           )


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证明文密码与哈希值是否匹配
    参数：
      - plain_password: 用户输入的明文密码
      - hashed_password: 数据库存储的哈希值
    返回：验证结果（True/False）
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """生成密码哈希值
    用途：用户注册或修改密码时调用
    示例：get_password_hash("123456") -> "$2b$12$xxxx..."
    """
    return pwd_context.hash(password) 