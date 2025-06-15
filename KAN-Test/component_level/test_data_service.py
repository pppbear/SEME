import pytest
import pandas as pd
from data_service.app.service import data
from data_service.app.service.data import dms_str_to_float, load_data_by_target

def test_validate_data():
    # 假设有 validate_data 函数
    if hasattr(data, 'validate_data'):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        result = data.validate_data(df)
        assert result is True or result is None

def test_transform_data():
    # 假设有 transform_data 函数
    if hasattr(data, 'transform_data'):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = data.transform_data(df)
        assert isinstance(df2, pd.DataFrame)
        assert df2.shape == df.shape 

def test_dms_str_to_float():
    assert dms_str_to_float("30° 15' 50.5\" N") == pytest.approx(30 + 15/60 + 50.5/3600)
    assert dms_str_to_float("120° 10' 20.0\" E") == pytest.approx(120 + 10/60 + 20/3600)
    assert dms_str_to_float("30° 15' 50.5\" S") == pytest.approx(-(30 + 15/60 + 50.5/3600))
    assert dms_str_to_float("120° 10' 20.0\" W") == pytest.approx(-(120 + 10/60 + 20/3600))
    with pytest.raises(ValueError):
        dms_str_to_float("无效格式")

def test_load_data_by_target(tmp_path, monkeypatch):
    df = pd.DataFrame({
        "Longitude": ["120° 10' 20.0\" E"],
        "Latitude": ["30° 15' 50.5\" N"],
        "target": [123.45]
    })
    file_path = tmp_path / "test.xlsx"
    df.to_excel(file_path, index=False)
    from data_service.app.core import config
    monkeypatch.setattr(config.settings, "DATA_PATH", str(file_path))
    monkeypatch.setattr('pandas.read_excel', lambda *a, **k: df)
    data = load_data_by_target("target")
    assert data[0]["longitude"] == pytest.approx(120 + 10/60 + 20/3600)
    assert data[0]["latitude"] == pytest.approx(30 + 15/60 + 50.5/3600)
    assert data[0]["value"] == 123.45
    # 异常分支：目标因变量不存在
    monkeypatch.setattr('pandas.read_excel', lambda *a, **k: pd.DataFrame({"Longitude": ["120° 10' 20.0\" E"]}))
    with pytest.raises(ValueError, match="目标因变量不存在于Excel中"):
        load_data_by_target("not_exist") 