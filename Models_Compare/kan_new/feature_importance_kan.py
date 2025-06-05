import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
from kan.kan.MultKAN import KAN

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def preprocess_for_model(df, feature_cols):
    """
    通用数据预处理：
    1. 过滤NDVI_MEAN和不透水面比例为0的行
    2. 自动补齐POI总数
    3. 自动生成平均建筑高度
    """
    for col in ['NDVI_MEAN', '不透水面比例']:
        if col in df.columns:
            before = len(df)
            df = df[df[col] != 0]
            print(f"已过滤{col}为0的行，剩余{len(df)}/{before}")
    if 'POI总数' in feature_cols and 'POI总数' not in df.columns:
        poi_cols = [col for col in df.columns if col.startswith('POI')]
        if poi_cols:
            df['POI总数'] = df[poi_cols].sum(axis=1)
            print(f"已自动补齐POI总数列，使用{len(poi_cols)}个POI相关列加和。")
        else:
            print("警告：特征文件要求POI总数，但数据中没有任何POI相关列，POI总数将为NaN。")
    if '平均建筑高度' in feature_cols and '平均建筑高度' not in df.columns:
        if '容积率' in df.columns and '建筑密度' in df.columns:
            df['平均建筑高度'] = df.apply(
                lambda row: row['容积率'] / row['建筑密度'] if row['建筑密度'] != 0 and not pd.isnull(row['建筑密度']) else 0,
                axis=1
            )
            print("已自动生成平均建筑高度列。")
    return df 

def calculate_feature_importance(models_dir, target_var, data_path, save_dir='feature_importance_results'):
    """
    加载KAN模型、scaler、特征文件，计算特征重要性并保存排序和可视化。
    models_dir: 模型和特征文件目录
    target_var: 目标变量名（如 lst_night_c）
    data_path: 用于归因分析的数据文件（如shanghai_nozero.xlsx）
    save_dir: 结果保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    # 路径
    model_path = os.path.join(models_dir, f'{target_var}_model.pth')
    scaler_path = os.path.join(models_dir, f'{target_var}_scaler.pkl')
    features_path = os.path.join(models_dir, f'{target_var}_features.txt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'未找到模型文件: {model_path}')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f'未找到scaler文件: {scaler_path}')
    if not os.path.exists(features_path):
        raise FileNotFoundError(f'未找到特征文件: {features_path}')
    # 加载特征
    with open(features_path, 'r', encoding='utf-8') as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    # 加载scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    # 加载数据
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError('仅支持Excel或CSV文件')
    # 统一预处理，保证特征补齐和顺序一致
    df = preprocess_for_model(df, feature_cols)
    # 只保留特征列
    X = df[feature_cols].values
    X_std = scaler.transform(X)
    # 加载模型
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = KAN(
        width=checkpoint['model_params']['width'],
        grid=checkpoint['model_params']['grid'],
        k=checkpoint['model_params']['k'],
        symbolic_enabled=True,
        ckpt_path=models_dir
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # 计算特征重要性
    print('计算特征重要性...')
    x_tensor = torch.FloatTensor(X_std)
    model(x_tensor)  # 先做一次前向传播，激活和cache_data会被保存
    model.attribute()  # 再归因
    if hasattr(model, 'feature_score') and model.feature_score is not None:
        feature_importance = model.feature_score.detach().cpu().numpy()
        if np.isnan(feature_importance).any():
            print('警告：特征重要性中存在NaN值，将被替换为0')
            feature_importance = np.nan_to_num(feature_importance, 0.0)
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': feature_importance
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        print('\n前10个最重要的特征:')
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"{i+1}. {row['feature']}: {row['importance']:.4f}")
        # 可视化
        plt.figure(figsize=(10, 6))
        top_n = min(10, len(importance_df))
        top_features = importance_df.head(top_n)
        plt.barh(top_features['feature'][::-1], top_features['importance'][::-1])
        plt.xlabel('重要性')
        plt.title(f'{target_var} 特征重要性（前10）')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/feature_importance_{target_var}.png', dpi=200)
        plt.close()
        # 保存数据
        importance_df.to_csv(f'{save_dir}/feature_importance_{target_var}.csv', index=False)
        print(f'特征重要性已保存到: {save_dir}')
        return importance_df
    else:
        print('无法获取特征重要性分数，使用均匀分布')
        uniform_importance = np.ones(len(feature_cols)) / len(feature_cols)
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': uniform_importance
        })
        return importance_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='KAN模型特征重要性分析')
    parser.add_argument('--models_dir', type=str, default='models', help='模型和特征文件目录')
    parser.add_argument('--target', type=str, required=True, help='目标因变量名')
    parser.add_argument('--data_path', type=str, required=True, help='用于归因分析的数据文件（如shanghai_nozero.xlsx）')
    parser.add_argument('--save_dir', type=str, default='feature_importance_results', help='结果保存目录')
    args = parser.parse_args()
    calculate_feature_importance(args.models_dir, args.target, args.data_path, args.save_dir)
