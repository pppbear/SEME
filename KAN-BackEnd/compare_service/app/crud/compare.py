import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import os
import logging
import torch
import torch.nn as nn
import joblib
from typing import List
from ..core.config import settings
from ..schemas.compare import CompareResult

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predictor_utils")

# MLP模型定义
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.model(x)

def analyze_and_predict(
    df: pd.DataFrame,
    target_dependent_var: str
) -> List[CompareResult]:
    """
    预测流程与 predict_with_model.py 完全一致，支持MLP和RF，输出CompareResult。
    """
    try:
        # 1. 特征选择，排除所有因变量和不需要的列
        pred_col_names = ["nighttime_", "lst_day_c", "lst_night_"]
        feature_cols = [col for col in df.columns if col not in ['FID', 'Id', 'Longitude', 'Latitude'] + pred_col_names]
        X = df[feature_cols].values

        # 2. 加载scaler
        mlp_model_path = os.path.join(settings.MLP_MODEL_DIR, "mlp_model.pth")
        rf_model_path = os.path.join(settings.RF_MODEL_DIR, "rf_model.joblib")
        scaler_x_path = os.path.join(settings.MLP_MODEL_DIR, "scaler_x.save")
        scaler_y_path = os.path.join(settings.MLP_MODEL_DIR, "scaler_y.save")
        scaler_X = joblib.load(scaler_x_path)
        scaler_Y = joblib.load(scaler_y_path)

        # 3. 标准化特征
        X_scaled = scaler_X.transform(X)

        # 4. 加载模型
        # 获取输出维度
        output_dim = scaler_Y.mean_.shape[0] if hasattr(scaler_Y, 'mean_') else 3
        # MLP
        mlp_model = MLPRegressor(X_scaled.shape[1], output_dim)
        mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=torch.device('cpu')))
        mlp_model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            y_pred_mlp = mlp_model(X_tensor).numpy()
        # RF
        rf_model = joblib.load(rf_model_path)
        y_pred_rf = rf_model.predict(X_scaled)

        # 5. 反标准化
        y_pred_mlp_inv = scaler_Y.inverse_transform(y_pred_mlp)
        y_pred_rf_inv = scaler_Y.inverse_transform(y_pred_rf)

        # 6. 只返回目标因变量
        # 兼容大小写
        target_idx = [c.lower() for c in pred_col_names].index(target_dependent_var.lower())
        y_true = df[target_dependent_var].values
        y_pred_mlp_i = y_pred_mlp_inv[:, target_idx]
        y_pred_rf_i = y_pred_rf_inv[:, target_idx]

        # 7. 计算指标
        mse_mlp = mean_squared_error(y_true, y_pred_mlp_i)
        r2_mlp = r2_score(y_true, y_pred_mlp_i)
        mse_rf = mean_squared_error(y_true, y_pred_rf_i)
        r2_rf = r2_score(y_true, y_pred_rf_i)

        # 8. 日志输出
        logger.info(f"\n==== {target_dependent_var} 预测结果统计信息 ====")
        logger.info(f"MLP: MSE={mse_mlp:.4f}, R2={r2_mlp:.4f}")
        logger.info(f"RF: MSE={mse_rf:.4f}, R2={r2_rf:.4f}")
        logger.info(f"真实值范围: [{y_true.min():.2f}, {y_true.max():.2f}], 均值: {y_true.mean():.2f}")
        logger.info(f"MLP预测值范围: [{y_pred_mlp_i.min():.2f}, {y_pred_mlp_i.max():.2f}], 均值: {y_pred_mlp_i.mean():.2f}")
        logger.info(f"RF预测值范围: [{y_pred_rf_i.min():.2f}, {y_pred_rf_i.max():.2f}], 均值: {y_pred_rf_i.mean():.2f}")

        # 9. 返回CompareResult
        result = CompareResult(
            dependent_name=target_dependent_var,
            true_values=y_true.tolist(),
            mlp_predictions=y_pred_mlp_i.tolist(),
            rf_predictions=y_pred_rf_i.tolist(),
            mse_mlp=float(mse_mlp),
            r2_mlp=float(r2_mlp),
            mse_rf=float(mse_rf),
            r2_rf=float(r2_rf)
        )
        return [result]
    except Exception as e:
        logger.error(f"预测过程中出错: {e}")
        raise 