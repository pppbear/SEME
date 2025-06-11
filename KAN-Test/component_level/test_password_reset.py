import pytest
import uuid
from auth_service.app.crud import user as user_crud
from auth_service.app.schemas.user import PasswordReset, UserCreate
from auth_service.app.utils import email as email_utils

def test_reset_password_success(db_session, monkeypatch):
    # 创建测试用户
    unique = str(uuid.uuid4())
    email = f'reset_{unique}@test.com'
    username = f'resetuser_{unique}'
    user_in = UserCreate(email=email, username=username, password='oldpwd', verification_code='123456')
    test_user = user_crud.create_user(db_session, user_in)
    # mock 验证码校验（mock user_crud作用域下的verify_code）
    monkeypatch.setattr(user_crud, 'verify_code', lambda e, c, t: True)
    reset_data = PasswordReset(email=email, new_password='newpwd', verification_code='123456')
    old_hashed_password = test_user.hashed_password  # 先保存旧密码哈希
    updated_user = user_crud.reset_password(db_session, reset_data)
    assert updated_user is not None
    assert updated_user.email == email
    assert updated_user.hashed_password != old_hashed_password

def test_reset_password_invalid_code(db_session, monkeypatch):
    unique = str(uuid.uuid4())
    email = f'reset2_{unique}@test.com'
    username = f'resetuser2_{unique}'
    user_in = UserCreate(email=email, username=username, password='oldpwd', verification_code='123456')
    user_crud.create_user(db_session, user_in)
    monkeypatch.setattr(user_crud, 'verify_code', lambda e, c, t: False)
    reset_data = PasswordReset(email=email, new_password='newpwd', verification_code='wrong')
    updated_user = user_crud.reset_password(db_session, reset_data)
    assert updated_user is None

def test_reset_password_user_not_found(db_session, monkeypatch):
    monkeypatch.setattr(user_crud, 'verify_code', lambda e, c, t: True)
    reset_data = PasswordReset(email='notfound@test.com', new_password='newpwd', verification_code='123456')
    updated_user = user_crud.reset_password(db_session, reset_data)
    assert updated_user is None 