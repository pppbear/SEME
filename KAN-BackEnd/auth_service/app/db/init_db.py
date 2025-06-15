import os
from sqlalchemy.orm import Session
from sqlalchemy import text

from auth_service.app.crud.user import create_user, get_user_by_email
from auth_service.app.schemas.user import UserCreate
from common_db.utils.db_utils import Base, engine, SessionLocal
from auth_service.app.core.config import settings
from common_db.models.user import User  # 确保模型被注册


def init_db():
    """
    初始化数据库函数
    功能：
      1. 创建所有数据表
      2. 插入初始管理员用户（如果不存在）
    """
    # 确保数据表存在
    Base.metadata.create_all(bind=engine)
    print("✅ 数据表结构已创建/更新")

    # 使用新会话操作（确保事务隔离）
    new_db = SessionLocal()
    try:
        admin_email = "2784892686@qq.com"
        admin = get_user_by_email(new_db, email=admin_email)
        
        if not admin:
            user_in = UserCreate(
                email=admin_email,
                username="admin",
                password="Powderblue437",
                is_active=True,
            )
            admin_user = create_user(new_db, user_in)
            admin_user.is_superuser = True
            new_db.add(admin_user)
            new_db.commit()
            print("🆗 管理员用户创建成功")
        else:
            print("⏩ 管理员用户已存在")
    finally:
        new_db.close()

if __name__ == "__main__":
    # 打印数据库实际路径
    db_url = str(engine.url)
    if db_url.startswith("sqlite:///"):
        db_file = db_url.replace("sqlite:///", "")
        db_file = os.path.abspath(os.path.join(os.path.dirname(__file__), db_file))
        print(f"数据库文件实际路径: {db_file}")
    else:
        print(f"数据库连接: {db_url}")
    init_db()