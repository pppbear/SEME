import pytest
from sqlalchemy.orm import Session
from common_db.utils import db_utils
from common_db.models.user import User
import uuid

@pytest.fixture(scope="function")
def db_session():
    db_utils.init_db()
    db: Session = next(db_utils.get_db())
    yield db
    db.close()

@pytest.fixture(scope="function")
def test_user(db_session):
    # 生成唯一用户名和邮箱
    unique = str(uuid.uuid4())
    user = User(email=f'fixture_user_{unique}@example.com', username=f'fixtureuser_{unique}', hashed_password='fixturepwd')
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user