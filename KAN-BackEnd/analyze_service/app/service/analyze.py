import pandas as pd
from typing import List
from analyze_service.app.schemas.analyze import KeyFeature
from common_utils.preprocess import preprocess_for_analyze
import logging
import torch
import numpy as np
from analyze_service.app.core.config import settings
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predictor_utils")

# 分析关键特征
def analyze_key_features_from_df(
    df: pd.DataFrame,
    target: str,
    n_epochs: int = 100,
    hidden_dim: int = 8,
    grid: int = 5,
    k: int = 3,
    lr: float = 0.01
) -> List[KeyFeature]:
    """
    直接接收DataFrame和目标变量名，返回关键特征列表（特征重要性）。
    """
    try:
        # 1. 数据预处理
        df = preprocess_for_analyze(df)

        # 2. 检查目标变量
        if target not in df.columns:
            raise ValueError(f'数据中未找到目标变量: {target}')

        # 3. 自动筛选特征列：去除目标变量和非数值型、常数列
        exclude_targets = ['lst_day_c', 'lst_night_c', 'nighttime_', 'uhi_day_c', 'uhi_night_c']
        feature_cols = [col for col in df.columns if col not in exclude_targets and col != target]
        feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
        feature_cols = [col for col in feature_cols if df[col].nunique() > 1]
        if not feature_cols:
            raise ValueError('无有效特征可用于分析！')

        # 4. 只保留特征和目标变量
        X = df[feature_cols].values
        y = df[target].values.reshape(-1, 1)

        # 标准化
        from sklearn.preprocessing import StandardScaler
        scaler_X = StandardScaler().fit(X)
        scaler_y = StandardScaler().fit(y)
        X_std = scaler_X.transform(X)
        y_std = scaler_y.transform(y)

        # 训练KAN模型
        from common_utils.kan.MultKAN import KAN
        input_dim = X_std.shape[1]
        output_dim = 1
        model = KAN(width=[input_dim, hidden_dim, output_dim], grid=grid, k=k, symbolic_enabled=True, ckpt_path=settings.MODEL_DIR)
        model.train()
        X_tensor = torch.FloatTensor(X_std)
        y_tensor = torch.FloatTensor(y_std)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = ((pred - y_tensor) ** 2).mean()
            loss.backward()
            optimizer.step()
        # 归因分析
        model.eval()
        model(X_tensor)
        model.attribute()
        if hasattr(model, 'feature_score') and model.feature_score is not None:
            feature_importance = model.feature_score.detach().cpu().numpy()
            if np.isnan(feature_importance).any():
                logger.warning('特征重要性中存在NaN值，将被替换为0')
                feature_importance = np.nan_to_num(feature_importance, 0.0)
            # 排序并组装为 KeyFeature 列表
            importance = list(zip(feature_cols, feature_importance))
            importance.sort(key=lambda x: x[1], reverse=True)
            return [KeyFeature(feature_name=feat, feature_value=float(val)) for feat, val in importance]
        else:
            logger.warning('无法获取特征重要性分数，使用均匀分布')
            uniform_importance = np.ones(len(feature_cols)) / len(feature_cols)
            importance = list(zip(feature_cols, uniform_importance))
            return [KeyFeature(feature_name=feat, feature_value=float(val)) for feat, val in importance]
    except Exception as e:
        logger.error(f"特征重要性分析出错: {e}")
        raise

