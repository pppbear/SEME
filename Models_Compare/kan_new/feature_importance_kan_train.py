import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
from kan.kan.MultKAN import KAN

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def preprocess_for_model(df, target_var):
    # 自动去除无用列
    exclude_cols = ['FID', 'Shape *', 'FID_', 'Id', 'Shape_Leng', 'Shape_Le_1', 'Shape_Area', 'Longitude', 'Latitude']
    df = df.drop(columns=exclude_cols, errors='ignore')
    # 过滤NDVI_MEAN和不透水面比例为0的行
    for col in ['NDVI_MEAN', '不透水面比例']:
        if col in df.columns:
            before = len(df)
            df = df[df[col] != 0]
            print(f"已过滤{col}为0的行，剩余{len(df)}/{before}")
    # 自动补齐POI总数
    if 'POI总数' not in df.columns:
        poi_cols = [col for col in df.columns if col.startswith('POI')]
        if poi_cols:
            df['POI总数'] = df[poi_cols].sum(axis=1)
            print(f"已自动补齐POI总数列，使用{len(poi_cols)}个POI相关列加和。")
        else:
            print("警告：数据中没有任何POI相关列，POI总数将为NaN。")
    # 自动生成平均建筑高度
    if '平均建筑高度' not in df.columns:
        if '容积率' in df.columns and '建筑密度' in df.columns:
            df['平均建筑高度'] = df.apply(
                lambda row: row['容积率'] / row['建筑密度'] if row['建筑密度'] != 0 and not pd.isnull(row['建筑密度']) else 0,
                axis=1
            )
            print("已自动生成平均建筑高度列。")
    return df

def train_and_analyze_feature_importance(data_path, target_var, save_dir='feature_importance_results', n_epochs=200, hidden_dim=8, grid=5, k=3, lr=0.01):
    os.makedirs(save_dir, exist_ok=True)
    # 加载数据
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError('仅支持Excel或CSV文件')
    # 预处理
    df = preprocess_for_model(df, target_var)
    # 检查目标变量
    if target_var not in df.columns:
        raise ValueError(f'数据中未找到目标变量: {target_var}')
    # 自动筛选特征列：去除目标变量和非数值型、常数列
    exclude_targets = ['lst_day_c', 'lst_night_c', 'nighttime_', 'uhi_day_c', 'uhi_night_c']
    feature_cols = [col for col in df.columns if col not in exclude_targets and col != target_var]
    # 只保留数值型特征
    feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    # 移除常数特征
    feature_cols = [col for col in feature_cols if df[col].nunique() > 1]
    print(f"最终用于训练的特征数: {len(feature_cols)}")
    print(f"特征列表: {feature_cols}")
    # 只保留特征和目标变量
    X = df[feature_cols].values
    y = df[target_var].values.reshape(-1, 1)
    # 标准化
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)
    X_std = scaler_X.transform(X)
    y_std = scaler_y.transform(y)
    # 训练KAN模型
    input_dim = X_std.shape[1]
    output_dim = 1
    model = KAN(width=[input_dim, hidden_dim, output_dim], grid=grid, k=k, symbolic_enabled=True)
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
        if (epoch+1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
    # 归因分析
    model.eval()
    model(X_tensor)
    model.attribute()
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
    parser = argparse.ArgumentParser(description='KAN模型现训特征重要性分析（自动去除无用列）')
    parser.add_argument('--data_path', type=str, required=True, help='输入数据文件（如shanghai_nozero.xlsx）')
    parser.add_argument('--target', type=str, required=True, help='目标因变量名')
    parser.add_argument('--save_dir', type=str, default='feature_importance_results', help='结果保存目录')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--hidden_dim', type=int, default=8, help='隐藏层神经元数')
    parser.add_argument('--grid', type=int, default=5, help='KAN网格数')
    parser.add_argument('--k', type=int, default=3, help='样条阶数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    args = parser.parse_args()
    train_and_analyze_feature_importance(
        args.data_path, args.target, args.save_dir,
        n_epochs=args.epochs, hidden_dim=args.hidden_dim, grid=args.grid, k=args.k, lr=args.lr
    ) 