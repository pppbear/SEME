import pytest
from auth_service.app.utils import email

def test_generate_verification_code():
    code = email.generate_verification_code()
    assert isinstance(code, str)
    assert len(code) == 6
    assert code.isdigit()

def test_save_and_verify_code(monkeypatch):
    # mock redis_client
    class DummyRedis:
        def __init__(self):
            self.store = {}
        def setex(self, key, time, value):
            self.store[key] = value
        def get(self, key):
            return self.store.get(key)
        def delete(self, key):
            self.store.pop(key, None)
    dummy_redis = DummyRedis()
    monkeypatch.setattr(email, 'redis_client', dummy_redis)
    email.save_verification_code('a@test.com', '123456', 'register')
    assert dummy_redis.get('verify:register:a@test.com') == '123456'
    # 正确校验
    assert email.verify_code('a@test.com', '123456', 'register') is True
    # 校验后已删除
    assert dummy_redis.get('verify:register:a@test.com') is None
    # 错误校验
    email.save_verification_code('a@test.com', '654321', 'register')
    assert email.verify_code('a@test.com', 'wrong', 'register') is False

def test_send_email(monkeypatch):
    # mock smtplib
    monkeypatch.setattr(email, 'settings', type('S', (), {
        'EMAIL_SENDER': 'from@test.com',
        'SMTP_SERVER': 'smtp.test.com',
        'SMTP_PORT': 465,
        'SMTP_USERNAME': 'user',
        'SMTP_PASSWORD': 'pwd'
    })())
    class DummyServer:
        def login(self, u, p): pass
        def send_message(self, msg): pass
        def quit(self): pass
    monkeypatch.setattr(email.smtplib, 'SMTP_SSL', lambda *a, **k: DummyServer())
    assert email.send_email('to@test.com', 'subj', 'body') is True

def test_send_verification_code(monkeypatch):
    monkeypatch.setattr(email, 'send_email', lambda *a, **k: True)
    monkeypatch.setattr(email, 'save_verification_code', lambda *a, **k: None)
    assert email.send_verification_code('to@test.com', 'register') is True
    assert email.send_verification_code('to@test.com', 'reset_password') is True
    assert email.send_verification_code('to@test.com', 'other') is False 