import pytest
from sqlalchemy.orm import Session
from common_db.utils import db_utils
from common_db.models.user import User

# 通过conftest.py中的db_session和test_user fixture自动完成数据库初始化和测试用户创建

def test_init_db(db_session):
    # 只要不抛异常即可
    pass

def test_get_user_by_id(db_session, test_user):
    # 查询
    found = db_utils.get_user_by_id(db_session, test_user.id)
    assert found is not None
    assert found.email == test_user.email
    # 查询不存在用户
    not_found = db_utils.get_user_by_id(db_session, -1)
    assert not_found is None 