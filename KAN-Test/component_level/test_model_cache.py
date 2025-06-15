import os
import pickle
import joblib
import torch
import numpy as np
import pytest
from compare_service.app.service import model_cache as compare_model_cache
from predict_service.app.service import model_cache as predict_model_cache

class DummyKAN(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    def load_state_dict(self, state_dict):
        return None

def test_compare_model_cache(tmp_path):
    # 1. 测试get_scaler
    scaler_path = tmp_path / 'scaler.joblib'
    joblib.dump({'mean': 1}, scaler_path)
    scaler = compare_model_cache.get_scaler(str(scaler_path))
    assert scaler == {'mean': 1}
    # 缓存命中
    scaler2 = compare_model_cache.get_scaler(str(scaler_path))
    assert scaler2 is scaler

    # 2. 测试get_rf_model
    rf_path = tmp_path / 'rf.joblib'
    joblib.dump({'rf': 2}, rf_path)
    rf = compare_model_cache.get_rf_model(str(rf_path))
    assert rf == {'rf': 2}
    rf2 = compare_model_cache.get_rf_model(str(rf_path))
    assert rf2 is rf

    # 3. 测试get_mlp_model
    mlp_path = tmp_path / 'mlp.pth'
    model = compare_model_cache.MLPRegressor(2, 1, [(4, 'relu')])
    torch.save(model.state_dict(), mlp_path)
    mlp = compare_model_cache.get_mlp_model(str(mlp_path), 2, 1, [(4, 'relu')])
    assert isinstance(mlp, compare_model_cache.MLPRegressor)
    mlp2 = compare_model_cache.get_mlp_model(str(mlp_path), 2, 1, [(4, 'relu')])
    assert mlp2 is mlp

    # 4. 测试get_kan_scaler
    kan_scaler_path = tmp_path / 'kan_scaler.pkl'
    with open(kan_scaler_path, 'wb') as f:
        pickle.dump({'k': 3}, f)
    kan_scaler = compare_model_cache.get_kan_scaler(str(kan_scaler_path))
    assert kan_scaler == {'k': 3}
    kan_scaler2 = compare_model_cache.get_kan_scaler(str(kan_scaler_path))
    assert kan_scaler2 is kan_scaler

    # 5. 测试get_kan_model_and_scaler
    kan_model_path = tmp_path / 'kan_model.pth'
    torch.save({'model_state_dict': DummyKAN().state_dict(), 'model_params': {'width': [2, 2, 1]}}, kan_model_path)
    result = compare_model_cache.get_kan_model_and_scaler(
        str(kan_model_path), str(kan_scaler_path), str(kan_scaler_path),
        DummyKAN, {'width': [2, 2, 1]}, str(tmp_path)
    )
    assert isinstance(result[0], DummyKAN)
    # 缓存命中
    result2 = compare_model_cache.get_kan_model_and_scaler(
        str(kan_model_path), str(kan_scaler_path), str(kan_scaler_path),
        DummyKAN, {'width': [2, 2, 1]}, str(tmp_path)
    )
    assert result2 is result

    # 6. 异常分支
    with pytest.raises(FileNotFoundError):
        compare_model_cache.get_scaler(str(tmp_path / 'not_exist.joblib'))
    with pytest.raises(FileNotFoundError):
        compare_model_cache.get_rf_model(str(tmp_path / 'not_exist_rf.joblib'))
    with pytest.raises(FileNotFoundError):
        compare_model_cache.get_kan_scaler(str(tmp_path / 'not_exist_kan_scaler.pkl'))

def test_predict_model_cache(tmp_path):
    # 1. 测试get_kan_scaler
    kan_scaler_path = tmp_path / 'kan_scaler.pkl'
    with open(kan_scaler_path, 'wb') as f:
        pickle.dump({'k': 5}, f)
    kan_scaler = predict_model_cache.get_kan_scaler(str(kan_scaler_path))
    assert kan_scaler == {'k': 5}
    kan_scaler2 = predict_model_cache.get_kan_scaler(str(kan_scaler_path))
    assert kan_scaler2 is kan_scaler

    # 2. 测试get_kan_model_and_scaler
    kan_model_path = tmp_path / 'kan_model.pth'
    torch.save({'model_state_dict': DummyKAN().state_dict(), 'model_params': {'width': [2, 2, 1]}}, kan_model_path)
    result = predict_model_cache.get_kan_model_and_scaler(
        str(kan_model_path), str(kan_scaler_path), str(kan_scaler_path),
        DummyKAN, {'width': [2, 2, 1]}, str(tmp_path)
    )
    assert isinstance(result[0], DummyKAN)
    # 缓存命中
    result2 = predict_model_cache.get_kan_model_and_scaler(
        str(kan_model_path), str(kan_scaler_path), str(kan_scaler_path),
        DummyKAN, {'width': [2, 2, 1]}, str(tmp_path)
    )
    assert result2 is result

    # 3. 异常分支
    with pytest.raises(FileNotFoundError):
        predict_model_cache.get_kan_scaler(str(tmp_path / 'not_exist_kan_scaler.pkl')) 