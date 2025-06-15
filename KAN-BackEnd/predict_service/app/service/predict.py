import pandas as pd
import os
import logging
from typing import List
from predict_service.app.schemas.predict import PredictResponse
from common_utils.preprocess import preprocess_for_model
from predict_service.app.core.config import settings
from common_utils.kan.MultKAN import KAN
from predict_service.app.service.model_cache import get_kan_model_and_scaler
import torch
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predictor_utils")

def predict_from_excel(
    df: pd.DataFrame,
    target_dependent_var: str
) -> PredictResponse:
    """
    只用KAN模型进行预测，返回预测值列表
    :param df: Excel文件路径
    :param target: 目标因变量（nighttime_、lst_day_c、lst_night_c）
    :return: PredictResponse
    """
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
        feature_cols_valid = [col for col in feature_cols if col in df.columns]
        if not feature_cols_valid:
            raise ValueError("错误：无有效特征列可用于预测！")
        X = df[feature_cols_valid].values

        # 3. KAN模型预测
        kan_model_dir = settings.KAN_MODEL_DIR
        kan_model_path = os.path.join(kan_model_dir, f"{target_dependent_var}_model.pth")
        kan_scaler_x_path = os.path.join(kan_model_dir, f"{target_dependent_var}_scaler.pkl")
        kan_scaler_y_path = os.path.join(kan_model_dir, f"{target_dependent_var}_y_scaler.pkl")
        # 加载模型参数
        import sys
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

        logger.info(f"KAN预测完成，预测值范围: [{y_pred_kan_inv.min():.2f}, {y_pred_kan_inv.max():.2f}], 均值: {y_pred_kan_inv.mean():.2f}")
        
        predictions=y_pred_kan_inv.tolist()
        
        return predictions
    except Exception as e:
        logger.error(f"KAN预测过程中出错: {e}")
        raise 
