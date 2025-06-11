import pytest
import requests

# 可根据实际端口调整
PREDICT_URL = 'http://localhost:8000/predict'
DATA_URL = 'http://localhost:8001/data'
COMPARE_URL = 'http://localhost:8002/compare'
ANALYZE_URL = 'http://localhost:8003/analyze'
AUTH_URL = 'http://localhost:8004/auth'
GATEWAY_URL = 'http://localhost:8005/api/v1'

@pytest.mark.integration
def test_predict_service():
    # 假设predict服务有/predict POST接口
    data = {"input": [[0.5, 0.2, 1, 4, 2, 1]]}
    resp = requests.post(PREDICT_URL, json=data)
    assert resp.status_code == 200
    assert 'result' in resp.json() or isinstance(resp.json(), list)

@pytest.mark.integration
def test_data_service():
    # 假设data服务有/data GET接口
    resp = requests.get(DATA_URL)
    assert resp.status_code == 200
    assert isinstance(resp.json(), dict) or isinstance(resp.json(), list)

@pytest.mark.integration
def test_compare_service():
    # 假设compare服务有/compare POST接口
    data = {"a": [1,2,3], "b": [1,2,3]}
    resp = requests.post(COMPARE_URL, json=data)
    assert resp.status_code == 200
    assert 'result' in resp.json() or resp.json() is not None

@pytest.mark.integration
def test_analyze_service():
    # 假设analyze服务有/analyze POST接口
    data = {"input": [1,2,3]}
    resp = requests.post(ANALYZE_URL, json=data)
    assert resp.status_code == 200
    assert 'result' in resp.json() or resp.json() is not None

@pytest.mark.integration
def test_auth_service():
    # 假设auth服务有/auth/login POST接口
    data = {"username": "testuser", "password": "testpass"}
    resp = requests.post(f"{AUTH_URL}/login", json=data)
    # 允许401等，主要测试服务联通
    assert resp.status_code in (200, 401, 403)

@pytest.mark.integration
def test_gateway_service():
    # 假设api_gateway有聚合接口
    resp = requests.get(f"{GATEWAY_URL}/status")
    assert resp.status_code == 200
    assert 'status' in resp.json() or resp.json() is not None 