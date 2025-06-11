import warnings
warnings.filterwarnings("ignore", message="警告：以下特征列在数据中未找到", category=UserWarning)
import pytest
import pandas as pd
from compare_service.app.service.compare import analyze_and_predict
from compare_service.app.schemas.compare import CompareResult
from compare_service.app.service import model_cache, compare
import numpy as np
import types
from common_utils.kan import MultKAN

class DummyScaler:
    def __init__(self):
        self.mean_ = np.array([0.])
    def transform(self, x): return x
    def inverse_transform(self, x): return x

class DummyTensor(np.ndarray):
    def numpy(self):
        return self
    def clone(self):
        return self.copy()
    def cpu(self):
        return self

class DummyModel:
    def __call__(self, x):
        arr = np.ones((len(x), 1)).view(DummyTensor)
        return arr
    def eval(self): pass
    def predict(self, x):
        arr = np.ones((len(x), 1)).view(DummyTensor)
        return arr
    def load_state_dict(self, state_dict, *a, **k):
        return None

@pytest.mark.parametrize('target', ['y'])
def test_analyze_and_predict_main(monkeypatch, target):
    df = pd.DataFrame({
        'x1': [1,2,3],
        'x2': [2,3,4],
        'y': [1,2,3]
    })
    # mock依赖
    monkeypatch.setattr('compare_service.app.service.compare.get_scaler', lambda path: DummyScaler())
    monkeypatch.setattr('compare_service.app.service.compare.get_mlp_model', lambda *a, **k: DummyModel())
    monkeypatch.setattr('compare_service.app.service.compare.get_rf_model', lambda *a, **k: DummyModel())
    monkeypatch.setattr('compare_service.app.service.compare.get_kan_model_and_scaler', lambda *a, **k: (DummyModel(), DummyScaler(), DummyScaler()))
    import os
    monkeypatch.setattr(os.path, 'exists', lambda path: True)
    monkeypatch.setattr('os.path.join', lambda *a: 'mocked_path')
    import compare_service.app.service.compare as compare_mod
    monkeypatch.setattr(compare_mod, 'settings', type('S', (), {'MLP_MODEL_DIR':'','RF_MODEL_DIR':'','KAN_MODEL_DIR':''})())
    # mock open，返回一个模拟文件对象
    class DummyFile:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def __iter__(self): return iter(['x1\n','x2\n','y\n'])
        def close(self): pass
        def tell(self): return 0
        def seek(self, offset, whence=0): return 0
        def read(self, n=-1): return b''
        def readline(self, n=-1): return b''
        def write(self, s): return len(s)
    monkeypatch.setattr('builtins.open', lambda *a, **k: DummyFile())
    # mock preprocess
    monkeypatch.setattr('compare_service.app.service.compare.preprocess_for_model', lambda df, cols: df)
    def dummy_float_tensor(x):
        arr = np.array(x).view(DummyTensor)
        return arr
    mock_torch = types.SimpleNamespace(
        tensor=dummy_float_tensor,
        no_grad=lambda : None,
        FloatTensor=dummy_float_tensor,
        optim=types.SimpleNamespace(
            Adam=lambda params, lr: types.SimpleNamespace(
                zero_grad=lambda: None,
                step=lambda: None
            )
        ),
        device=lambda x: None,
        load=lambda *a, **k: {},
    )
    monkeypatch.setattr('compare_service.app.service.model_cache.torch', mock_torch)
    monkeypatch.setattr('compare_service.app.service.model_cache.torch.load', lambda *a, **k: {'model_params': {'width': [2, 2, 1]}, 'model_state_dict': {}, 'other': 1})
    monkeypatch.setattr('torch.load', lambda *a, **k: {'model_params': {'width': [2, 2, 1]}, 'model_state_dict': {}, 'other': 1})
    monkeypatch.setattr(MultKAN, 'load_state_dict', lambda self, state_dict, *a, **k: None)
    monkeypatch.setattr('compare_service.app.service.model_cache.get_kan_scaler', lambda path: DummyScaler())
    result = analyze_and_predict(df, target)
    assert isinstance(result, list)
    assert isinstance(result[0], CompareResult)

@pytest.mark.parametrize('df,target,err', [
    (pd.DataFrame({'x1': [1,2], 'x2': [3,4], 'y': [1,2]}), 'y', FileNotFoundError),
    (pd.DataFrame({'x1': [1,2], 'x2': [3,4], 'y': [1,2]}), 'y', ValueError),
])
def test_analyze_and_predict_exceptions(monkeypatch, df, target, err):
    import compare_service.app.service.compare as compare_mod
    # 针对FileNotFoundError
    if err is FileNotFoundError:
        monkeypatch.setattr('os.path.exists', lambda path: False)
    else:
        # 针对ValueError，mock open和exists
        monkeypatch.setattr('os.path.exists', lambda path: True)
        class DummyFile:
            def __enter__(self): return self
            def __exit__(self, exc_type, exc_val, exc_tb): pass
            def __iter__(self): return iter(['x1\n','x2\n','y\n'])
            def close(self): pass
            def tell(self): return 0
            def seek(self, offset, whence=0): return 0
            def read(self, n=-1): return b''
            def readline(self, n=-1): return b''
            def write(self, s): return len(s)
        monkeypatch.setattr('builtins.open', lambda *a, **k: DummyFile())
        monkeypatch.setattr('compare_service.app.service.compare.preprocess_for_model', lambda df, cols: df)
        monkeypatch.setattr('compare_service.app.service.compare.get_scaler', lambda path: DummyScaler())
        monkeypatch.setattr('torch.load', lambda *a, **k: {'model_params': {'width': [2, 2, 1]}, 'model_state_dict': {}, 'other': 1})
        monkeypatch.setattr('compare_service.app.service.compare.get_mlp_model', lambda *a, **k: DummyModel())
        monkeypatch.setattr('compare_service.app.service.compare.get_rf_model', lambda *a, **k: DummyModel())
        monkeypatch.setattr('compare_service.app.service.model_cache.torch.load', lambda *a, **k: {'model_params': {'width': [2, 2, 1]}, 'model_state_dict': {}, 'other': 1})
        monkeypatch.setattr('compare_service.app.service.compare.get_kan_model_and_scaler', lambda *a, **k: (_ for _ in ()).throw(ValueError("mocked error")))
        monkeypatch.setattr('compare_service.app.service.model_cache.get_kan_scaler', lambda path: DummyScaler())
    with pytest.raises(err):
        compare_mod.analyze_and_predict(df, target)

def test_model_cache(tmp_path):
    import pickle
    scaler_path = tmp_path / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump({'b': 2}, f)
    scaler = model_cache.get_kan_scaler(str(scaler_path))
    assert scaler == {'b': 2}
    scaler2 = model_cache.get_kan_scaler(str(scaler_path))
    assert scaler2 is scaler

def test_compare_main():
    # 假设 compare.py 有 main_compare 函数
    if hasattr(compare, 'main_compare'):
        result = compare.main_compare([1,2,3], [1,2,3])
        assert result is not None