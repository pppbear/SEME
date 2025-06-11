import pytest
import pandas as pd
from common_utils.preprocess import preprocess_for_model, preprocess_for_analyze

def test_preprocess_for_model_basic():
    df = pd.DataFrame({
        'NDVI_MEAN': [0.5, 0, 0.3],
        '不透水面比例': [0.2, 0.1, 0],
        'POI购物': [1, 2, 3],
        'POI生活': [4, 5, 6],
        '容积率': [2, 3, 4],
        '建筑密度': [1, 2, 2],
    })
    feature_cols = ['NDVI_MEAN', '不透水面比例', 'POI购物', 'POI生活', 'POI总数', '平均建筑高度']
    df2 = preprocess_for_model(df, feature_cols)
    # 检查过滤
    assert (df2['NDVI_MEAN'] != 0).all()
    assert (df2['不透水面比例'] != 0).all()
    # 检查自动补齐POI总数
    assert 'POI总数' in df2.columns
    assert (df2['POI总数'] == df2['POI购物'] + df2['POI生活']).all()
    # 检查自动生成平均建筑高度
    assert '平均建筑高度' in df2.columns
    assert (df2['平均建筑高度'] == df2['容积率'] / df2['建筑密度']).all()

def test_preprocess_for_analyze_basic():
    df = pd.DataFrame({
        'NDVI_MEAN': [0.5, 0, 0.3],
        '不透水面比例': [0.2, 0.1, 0],
        'POI购物': [1, 2, 3],
        'POI生活': [4, 5, 6],
        '容积率': [2, 3, 4],
        '建筑密度': [1, 2, 2],
    })
    df2 = preprocess_for_analyze(df)
    # 检查过滤
    assert (df2['NDVI_MEAN'] != 0).all()
    assert (df2['不透水面比例'] != 0).all()
    # 检查自动补齐POI总数
    assert 'POI总数' in df2.columns
    assert (df2['POI总数'] == df2['POI购物'] + df2['POI生活']).all()
    # 检查自动生成平均建筑高度
    assert '平均建筑高度' in df2.columns
    assert (df2['平均建筑高度'] == df2['容积率'] / df2['建筑密度']).all() 