import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import pandas as pd
import torch
import pickle
from sklearn.metrics import mean_squared_error, r2_score

# KAN模型依赖
from kan.kan.MultKAN import KAN

def load_features(features_path):
    with open(features_path, 'r', encoding='utf-8') as f:
        features = [line.strip() for line in f if line.strip()]
    return features

def predict_kan(input_path, target_var, output_path=None):
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    # 1. 特征文件
    features_path = os.path.join(models_dir, f'{target_var}_features.txt')
    if not os.path.exists(features_path):
        raise FileNotFoundError(f'未找到特征列文件: {features_path}')
    feature_cols = load_features(features_path)

    # 2. 读取数据
    if input_path.endswith('.xlsx'):
        df = pd.read_excel(input_path)
    elif input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        raise ValueError('仅支持Excel或CSV文件')

    # 过滤NDVI_MEAN和不透水面比例为0的行
    for col in ['NDVI_MEAN', '不透水面比例']:
        if col in df.columns:
            before = len(df)
            df = df[df[col] != 0]
            print(f"已过滤{col}为0的行，剩余{len(df)}/{before}")

    # 自动补齐POI总数
    if 'POI总数' in feature_cols and 'POI总数' not in df.columns:
        poi_cols = [col for col in df.columns if col.startswith('POI')]
        if poi_cols:
            df['POI总数'] = df[poi_cols].sum(axis=1)
            print(f"已自动补齐POI总数列，使用{len(poi_cols)}个POI相关列加和。")
        else:
            print("警告：特征文件要求POI总数，但数据中没有任何POI相关列，POI总数将为NaN。")

    # 自动生成平均建筑高度
    if '平均建筑高度' in feature_cols and '平均建筑高度' not in df.columns:
        if '容积率' in df.columns and '建筑密度' in df.columns:
            df['平均建筑高度'] = df.apply(
                lambda row: row['容积率'] / row['建筑密度'] if row['建筑密度'] != 0 and not pd.isnull(row['建筑密度']) else 0,
                axis=1
            )
            print("已自动生成平均建筑高度列。")

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f'警告：以下特征列在数据中未找到，将被忽略: {missing_cols}')
    feature_cols_valid = [col for col in feature_cols if col in df.columns]
    if not feature_cols_valid:
        raise ValueError('无有效特征列可用于预测！')
    X = df[feature_cols_valid].values
    print(df[feature_cols].head())
    print(df[feature_cols_valid].head())
    # 3. 加载KAN模型和scaler
    kan_model_path = os.path.join(models_dir, f'{target_var}_model.pth')
    kan_scaler_x_path = os.path.join(models_dir, f'{target_var}_scaler.pkl')
    kan_scaler_y_path = os.path.join(models_dir, f'{target_var}_y_scaler.pkl')
    if not os.path.exists(kan_model_path):
        raise FileNotFoundError(f'未找到KAN模型: {kan_model_path}')
    if not os.path.exists(kan_scaler_x_path) or not os.path.exists(kan_scaler_y_path):
        raise FileNotFoundError('未找到scaler文件')
    # 加载scaler
    with open(kan_scaler_x_path, 'rb') as f:
        kan_scaler_x = pickle.load(f)
    with open(kan_scaler_y_path, 'rb') as f:
        kan_scaler_y = pickle.load(f)
    X_kan = kan_scaler_x.transform(X)
    X_kan_tensor = torch.FloatTensor(X_kan)
    # 加载模型参数
    checkpoint = torch.load(kan_model_path, map_location=torch.device('cpu'))
    model_kan = KAN(
        width=checkpoint['model_params']['width'],
        grid=checkpoint['model_params']['grid'],
        k=checkpoint['model_params']['k'],
        symbolic_enabled=True,
        ckpt_path=models_dir
    )
    model_kan.load_state_dict(checkpoint['model_state_dict'])
    model_kan.eval()
    # 预测
    with torch.no_grad():
        y_pred_kan = model_kan(X_kan_tensor)
    y_pred_kan = y_pred_kan.cpu().numpy().flatten()
    y_pred_kan_inv = kan_scaler_y.inverse_transform(y_pred_kan.reshape(-1, 1)).ravel()
    # 保存结果
    result_df = pd.DataFrame({target_var: y_pred_kan_inv})
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + f'_pred_{target_var}.xlsx'
    result_df.to_excel(output_path, index=False)
    print(f'预测完成，结果已保存到: {output_path}')

    # 如果有真实值，计算MSE和R2
    if target_var in df.columns:
        y_true = df[target_var].values
        mse = mean_squared_error(y_true, y_pred_kan_inv)
        r2 = r2_score(y_true, y_pred_kan_inv)
        print(f'MSE: {mse:.4f}')
        print(f'R2: {r2:.4f}')
    else:
        print('输入数据中未包含真实值，无法计算MSE和R2。')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KAN模型预测脚本')
    parser.add_argument('--input', type=str, required=True, help='输入Excel或CSV文件路径')
    parser.add_argument('--target', type=str, required=True, help='目标因变量名')
    parser.add_argument('--output', type=str, default=None, help='输出Excel文件路径')
    args = parser.parse_args()
    predict_kan(args.input, args.target, args.output)
