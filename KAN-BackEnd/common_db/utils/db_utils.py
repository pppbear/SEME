"""
共享数据库工具模块
提供统一的数据库连接和操作方法
"""
import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.orm import declarative_base

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("db_utils")

# 数据库配置
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "common_db.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# 创建数据库引擎
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}  # 仅用于SQLite
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建基类
Base = declarative_base()

def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """初始化数据库"""
    logger.info("创建数据库表...")
    Base.metadata.create_all(bind=engine)
    logger.info("数据库表创建完成")

def get_user_by_id(db: Session, user_id: int):
    """根据ID获取用户"""
    try:
        # 动态导入User模型，避免循环导入
        from common_db.models.user import User
        return db.query(User).filter(User.id == user_id).first()
    except Exception as e:
        logger.error(f"获取用户失败: {str(e)}")
        return None 