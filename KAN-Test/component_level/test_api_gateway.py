import pytest
from api_gateway.app.core import config

def test_config_settings():
    # 检查配置项存在
    assert hasattr(config, 'settings')
    s = config.settings
    assert hasattr(s, 'API_V1_STR')
    assert hasattr(s, 'PROJECT_NAME') 