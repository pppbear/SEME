import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import os
import logging
import torch
import torch.nn as nn
import joblib
from typing import List
from compare_service.app.core.config import settings
from compare_service.app.schemas.compare import CompareResult
from common_utils.kan.MultKAN import KAN
import pickle
import sys
from compare_service.app.service.model_cache import MLPRegressor, get_mlp_model, get_scaler, get_rf_model, get_kan_model_and_scaler
from common_utils.preprocess import preprocess_for_model
from concurrent.futures import ThreadPoolExecutor
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predictor_utils")

def analyze_and_predict(
    df: pd.DataFrame,
    target_dependent_var: str
) -> List[CompareResult]:
    """
    - 每个因变量独立特征文件、独立模型和scaler
    - 特征顺序严格对齐，缺失特征警告但不报错
    - 只预测和返回目标因变量
    """
    import warnings
    try:
        # 1. 读取特征文件
        features_dir = os.path.join(os.path.dirname(__file__), "independent")
        features_file = os.path.join(features_dir, f"{target_dependent_var}_features.txt")
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"未找到特征列文件: {features_file}")
        with open(features_file, 'r', encoding='utf-8') as f:
            feature_cols = [line.strip() for line in f if line.strip()]

        # 2. 数据预处理
        df = preprocess_for_model(df, feature_cols)

        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            warnings.warn(f"警告：以下特征列在数据中未找到，将被忽略: {missing_cols}")
        feature_cols_valid = [col for col in feature_cols if col in df.columns]
        if not feature_cols_valid:
            raise ValueError("错误：无有效特征列可用于预测！")
        X = df[feature_cols_valid].values

        # 3. 拼接模型和scaler路径
        mlp_model_path = os.path.join(settings.MLP_MODEL_DIR, f"mlp_model_{target_dependent_var}.pth")
        rf_model_path = os.path.join(settings.RF_MODEL_DIR, f"rf_model_{target_dependent_var}.joblib")
        scaler_x_path = os.path.join(settings.MLP_MODEL_DIR, f"scaler_x_{target_dependent_var}.save")
        scaler_y_path = os.path.join(settings.MLP_MODEL_DIR, f"scaler_y_{target_dependent_var}.save")
        scaler_x_rf_path = os.path.join(settings.RF_MODEL_DIR, f"scaler_x_{target_dependent_var}.save")
        scaler_y_rf_path = os.path.join(settings.RF_MODEL_DIR, f"scaler_y_{target_dependent_var}.save")

        # 4. 加载scaler
        scaler_X = get_scaler(scaler_x_path) if os.path.exists(scaler_x_path) else get_scaler(scaler_x_rf_path)
        scaler_Y = get_scaler(scaler_y_path) if os.path.exists(scaler_y_path) else get_scaler(scaler_y_rf_path)
        X_scaled = scaler_X.transform(X)

        # 5. 真实值
        if target_dependent_var not in df.columns:
            raise ValueError(f"数据中未找到目标因变量: {target_dependent_var}")
        y_true = df[target_dependent_var].values

        # 6.1. MLP预测函数
        def predict_mlp():
            output_dim = scaler_Y.mean_.shape[0] if hasattr(scaler_Y, 'mean_') else 1
            hidden_layers = [(1024, 'relu'), (1024, 'relu')]
            mlp_model = get_mlp_model(mlp_model_path, X_scaled.shape[1], output_dim, hidden_layers)
            with torch.no_grad():
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                y_pred_mlp = mlp_model(X_tensor).numpy()
            if y_pred_mlp.ndim == 1:
                y_pred_mlp = y_pred_mlp.reshape(-1, 1)
            y_pred_mlp_inv = scaler_Y.inverse_transform(y_pred_mlp)
            if y_pred_mlp_inv.shape[1] == 1:
                y_pred_mlp_i = y_pred_mlp_inv.ravel()
            else:
                y_pred_mlp_i = y_pred_mlp_inv[:, 0]
            mse_mlp = mean_squared_error(y_true, y_pred_mlp_i)
            r2_mlp = r2_score(y_true, y_pred_mlp_i)
            return y_pred_mlp_i, mse_mlp, r2_mlp

        # 6.2. RF预测函数
        def predict_rf():
            rf_model = get_rf_model(rf_model_path)
            y_pred_rf = rf_model.predict(X_scaled)
            if y_pred_rf.ndim == 1:
                y_pred_rf = y_pred_rf.reshape(-1, 1)
            y_pred_rf_inv = scaler_Y.inverse_transform(y_pred_rf)
            if y_pred_rf_inv.shape[1] == 1:
                y_pred_rf_i = y_pred_rf_inv.ravel()
            else:
                y_pred_rf_i = y_pred_rf_inv[:, 0]
            mse_rf = mean_squared_error(y_true, y_pred_rf_i)
            r2_rf = r2_score(y_true, y_pred_rf_i)
            return y_pred_rf_i, mse_rf, r2_rf

        # 6.3. KAN预测函数
        def predict_kan():
            kan_model_dir = settings.KAN_MODEL_DIR
            kan_model_path = os.path.join(kan_model_dir, f"{target_dependent_var}_model.pth")
            kan_scaler_x_path = os.path.join(kan_model_dir, f"{target_dependent_var}_scaler.pkl")
            kan_scaler_y_path = os.path.join(kan_model_dir, f"{target_dependent_var}_y_scaler.pkl")
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils/kan')))
            checkpoint = torch.load(kan_model_path, map_location=torch.device('cpu'))
            model_params = checkpoint['model_params']
            model_kan, kan_scaler_x, kan_scaler_y = get_kan_model_and_scaler(
                kan_model_path, kan_scaler_x_path, kan_scaler_y_path,
                KAN, model_params, settings.KAN_MODEL_DIR + "/model"
            )
            X_kan = kan_scaler_x.transform(X)
            X_kan_tensor = torch.FloatTensor(X_kan)
            with torch.no_grad():
                y_pred_kan = model_kan(X_kan_tensor)
            y_pred_kan = y_pred_kan.cpu().numpy().flatten()
            y_pred_kan_inv = kan_scaler_y.inverse_transform(y_pred_kan.reshape(-1, 1)).ravel()
            mse_kan = mean_squared_error(y_true, y_pred_kan_inv)
            r2_kan = r2_score(y_true, y_pred_kan_inv)
            return y_pred_kan_inv, mse_kan, r2_kan

        # 6. 并行执行三模型预测
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_mlp = executor.submit(predict_mlp)
            future_rf = executor.submit(predict_rf)
            future_kan = executor.submit(predict_kan)
            y_pred_mlp_i, mse_mlp, r2_mlp = future_mlp.result()
            y_pred_rf_i, mse_rf, r2_rf = future_rf.result()
            y_pred_kan_inv, mse_kan, r2_kan = future_kan.result()

        # 7. 日志输出
        logger.info(f"\n==== {target_dependent_var} 预测结果统计信息 ====")
        logger.info(f"MLP: MSE={mse_mlp:.4f}, R2={r2_mlp:.4f}")
        logger.info(f"RF: MSE={mse_rf:.4f}, R2={r2_rf:.4f}")
        logger.info(f"KAN: MSE={mse_kan:.4f}, R2={r2_kan:.4f}")
        logger.info(f"真实值范围: [{y_true.min():.2f}, {y_true.max():.2f}], 均值: {y_true.mean():.2f}")
        logger.info(f"MLP预测值范围: [{y_pred_mlp_i.min():.2f}, {y_pred_mlp_i.max():.2f}], 均值: {y_pred_mlp_i.mean():.2f}")
        logger.info(f"RF预测值范围: [{y_pred_rf_i.min():.2f}, {y_pred_rf_i.max():.2f}], 均值: {y_pred_rf_i.mean():.2f}")
        logger.info(f"KAN预测值范围: [{y_pred_kan_inv.min():.2f}, {y_pred_kan_inv.max():.2f}], 均值: {y_pred_kan_inv.mean():.2f}")

        # 8. 返回CompareResult
        result = CompareResult(
            dependent_name=target_dependent_var,
            true_values=y_true.tolist(),
            mlp_predictions=y_pred_mlp_i.tolist(),
            rf_predictions=y_pred_rf_i.tolist(),
            kan_predictions=y_pred_kan_inv.tolist(),
            mse_mlp=float(mse_mlp),
            r2_mlp=float(r2_mlp),
            mse_rf=float(mse_rf),
            r2_rf=float(r2_rf),
            mse_kan=float(mse_kan),
            r2_kan=float(r2_kan)
        )
        return [result]
    except Exception as e:
        logger.error(f"预测过程中出错: {e}")
        raise 
