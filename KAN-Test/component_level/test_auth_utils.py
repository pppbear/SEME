import pytest
from datetime import timedelta
from common_db.utils import auth_utils

def test_create_and_decode_access_token():
    user_id = 123
    token = auth_utils.create_access_token(subject=user_id, expires_delta=timedelta(minutes=5))
    assert isinstance(token, str)
    payload = auth_utils.decode_access_token(token)
    assert payload['sub'] == str(user_id)
    assert 'exp' in payload

def test_get_token_user_id():
    user_id = 456
    token = auth_utils.create_access_token(subject=user_id)
    uid = auth_utils.get_token_user_id(token)
    assert uid == user_id

def test_decode_access_token_invalid():
    with pytest.raises(Exception):
        auth_utils.decode_access_token('invalid.token.string') 