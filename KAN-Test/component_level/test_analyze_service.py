import pytest
import pandas as pd
from analyze_service.app.service.analyze import analyze_key_features_from_df
from analyze_service.app.schemas.analyze import KeyFeature
import os
import types
import torch

class DummyModel:
    def __init__(self):
        self.feature_score = None
    def train(self): pass
    def eval(self): pass
    def __call__(self, x): return x
    def attribute(self): pass

# 主流程测试
@pytest.mark.parametrize('target', ['y'])
def test_analyze_key_features_main(monkeypatch, target):
    df = pd.DataFrame({
        'x1': [1,2,3,4,5],
        'x2': [2,3,4,5,6],
        'y': [1,0,1,0,1]
    })
    # mock KAN，直接mock analyze模块内的KAN
    import analyze_service.app.service.analyze as analyze_mod
    # monkeypatch MultKAN类本身
    monkeypatch.setattr('common_utils.kan.MultKAN', lambda *a, **kw: DummyModel())
    # mock torch
    mock_torch = types.SimpleNamespace(
        tensor=lambda x: torch.tensor(x, dtype=torch.float32),
        no_grad=lambda : None,
        FloatTensor=lambda x: torch.tensor(x, dtype=torch.float32),
        optim=types.SimpleNamespace(
            Adam=lambda params, lr: types.SimpleNamespace(
                zero_grad=lambda: None,
                step=lambda: None
            )
        ),
        device=lambda x: None,
        load=lambda *a, **k: {},
    )
    monkeypatch.setattr('analyze_service.app.service.analyze.torch', mock_torch)
    # mock os.path.exists 和 os.path.join，避免FileNotFoundError
    monkeypatch.setattr(os.path, 'exists', lambda path: True)
    monkeypatch.setattr('os.path.join', lambda *a: 'mocked_path')
    # mock settings
    monkeypatch.setattr(analyze_mod, 'settings', types.SimpleNamespace(MODEL_DIR=''))
    # mock scaler
    import sklearn.preprocessing
    monkeypatch.setattr(sklearn.preprocessing.StandardScaler, 'fit', lambda self, x: self)
    monkeypatch.setattr(sklearn.preprocessing.StandardScaler, 'transform', lambda self, x: x)
    # mock KeyFeature
    monkeypatch.setattr(analyze_mod, 'KeyFeature', KeyFeature)
    result = analyze_key_features_from_df(df, target)
    assert isinstance(result, list)
    assert all(isinstance(r, KeyFeature) for r in result)

# 异常分支测试
@pytest.mark.parametrize('df,target,err', [
    (pd.DataFrame({'y': [1, 1]}), 'y', ValueError),
    (pd.DataFrame({'x1': [1, 1], 'x2': [2, 2], 'y': [1, 1]}), 'y', ValueError),
    (pd.DataFrame({'x1': [1, 1], 'x2': [2, 2], 'y': [1, 2]}), 'y', ValueError),
])
def test_analyze_key_features_exceptions(df, target, err):
    import analyze_service.app.service.analyze as analyze_mod
    with pytest.raises(err):
        analyze_mod.analyze_key_features_from_df(df, target) 