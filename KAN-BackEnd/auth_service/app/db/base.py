from sqlalchemy.ext.declarative import declarative_base
from common_db.utils.db_utils import Base

# 创建共享的Base类
Base = declarative_base() 