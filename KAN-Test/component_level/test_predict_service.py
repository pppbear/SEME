import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from predict_service.app.service import predict, model_cache


def test_model_cache_get_kan_scaler(tmp_path):
    # 创建一个假的scaler文件
    import pickle
    scaler_path = tmp_path / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump({'a': 1}, f)
    # 第一次加载
    scaler = model_cache.get_kan_scaler(str(scaler_path))
    assert scaler == {'a': 1}
    # 第二次应命中缓存
    scaler2 = model_cache.get_kan_scaler(str(scaler_path))
    assert scaler2 is scaler

def test_predict_from_excel(monkeypatch):
    # 构造DataFrame和mock依赖
    df = pd.DataFrame({
        'NDVI_MEAN': [0.5, 0.6],
        '不透水面比例': [0.2, 0.3],
        'POI购物': [1, 2],
        'POI生活': [4, 5],
        '容积率': [2, 3],
        '建筑密度': [1, 2],
    })
    # mock preprocess_for_model
    monkeypatch.setattr(predict, 'preprocess_for_model', lambda df, cols: df)
    # mock KAN模型和scaler
    class DummyModel:
        def eval(self): return self
        def __call__(self, x):
            import torch
            return torch.tensor([[1.0],[2.0]])
    dummy_scaler = MagicMock()
    dummy_scaler.transform.side_effect = lambda x: x
    dummy_scaler.inverse_transform.side_effect = lambda x: x
    monkeypatch.setattr(predict, 'get_kan_model_and_scaler', lambda *a, **k: (DummyModel(), dummy_scaler, dummy_scaler))
    # mock settings
    monkeypatch.setattr(predict.settings, 'KAN_MODEL_DIR', '.')
    # mock torch.load
    monkeypatch.setattr('torch.load', lambda *a, **k: {'model_params': {}})
    # mock open/features文件
    m = mock_open(read_data="NDVI_MEAN\n不透水面比例\n")
    monkeypatch.setattr('builtins.open', m)
    # mock os.path.exists
    monkeypatch.setattr('os.path.exists', lambda x: True)
    # 执行
    result = predict.predict_from_excel(df, 'lst_day_c')
    assert isinstance(result, list)
    assert result == [1.0, 2.0] 