from common_db.utils.db_utils import engine, SessionLocal, Base
from common_db.models.user import User

# 注意：直接使用common_db模块提供的数据库引擎和会话
# 这样可以确保auth_service和其他服务使用同一个数据库连接

# 如果需要针对auth_service的特殊配置，可以在这里添加
# 但要避免重新创建engine和SessionLocal
