import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from mlp_utils import MLPRegressor as MLP
from mlp_utils import train_and_validate_mlp_model, validate_mlp_model, evaluate_mlp_model, generate_mlp_params
from preprocess import preprocess_for_model

# ========== 配置 =============
data_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'shanghai_nozero.xlsx')
independent_dir = os.path.join(os.path.dirname(__file__), 'independent')
model_dir = os.path.join(os.path.dirname(__file__), 'models')
result_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(model_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

targets = {
    'nighttime_': 'nighttime__features.txt',
    'lst_day_c': 'lst_day_c_features.txt',
    'lst_night_c': 'lst_night_c_features.txt',
}

# ========== 读取大表 =============
df = pd.read_excel(data_file)
print(f"数据文件 {data_file} 加载完成，形状: {df.shape}")
print(f"独立变量文件路径: {independent_dir}")
print(f"模型保存路径: {model_dir}")
print(f"结果保存路径: {result_dir}")

# ========== 超参数配置 =============
learning_rates = [0.001, 0.005]
patiences = [5, 10]
hidden_layers_configs = [
    [(1024, 'relu')],
    [(1024, 'relu'), (1024, 'relu')],
]
loss_functions = [nn.MSELoss()]
param_combinations = generate_mlp_params(learning_rates, patiences, hidden_layers_configs, loss_functions)

n_splits = 5  # 外层折数
batch_size = 64
max_epochs = 400

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for target, feat_file in targets.items():
    print(f"\n{'='*60}\n处理因变量: {target}")
    feat_path = os.path.join(independent_dir, feat_file)
    with open(feat_path, encoding='utf-8') as f:
        feature_names = [line.strip() for line in f if line.strip()]
    print(f"自变量数量: {len(feature_names)}，自变量名: {feature_names}")
    df = preprocess_for_model(df, feature_names)
    missing = [col for col in feature_names + [target] if col not in df.columns]
    if missing:
        raise ValueError(f"数据中缺少列: {missing}")
    X = df[feature_names].values
    y = df[[target]].values
    outer_kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(outer_kf.split(X)):
        print(f"\n--- 外层第{fold+1}折 ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train)
        X_train_std = scaler_X.transform(X_train)
        X_test_std = scaler_X.transform(X_test)
        y_train_std = scaler_y.transform(y_train)
        y_test_std = scaler_y.transform(y_test)
        # 内层交叉验证选超参数
        inner_kf = KFold(n_splits=3, shuffle=True, random_state=fold)
        best_params = None
        best_val_loss = float('inf')
        for params in param_combinations:
            inner_losses = []
            for inner_train_idx, inner_val_idx in inner_kf.split(X_train_std):
                X_inner_train, X_inner_val = X_train_std[inner_train_idx], X_train_std[inner_val_idx]
                y_inner_train, y_inner_val = y_train_std[inner_train_idx], y_train_std[inner_val_idx]
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                    torch.tensor(X_inner_train, dtype=torch.float32),
                    torch.tensor(y_inner_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
                val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                    torch.tensor(X_inner_val, dtype=torch.float32),
                    torch.tensor(y_inner_val, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
                input_dim = X_inner_train.shape[1]
                output_dim = 1
                model = MLP(input_dim, output_dim, params['hidden_layers']).to(device)
                criterion = params['loss_function']
                train_and_validate_mlp_model(train_loader, val_loader, model, criterion, patience=params['patience'], learning_rate=params['learning_rate'], max_epochs=max_epochs, device=device)
                val_loss = validate_mlp_model(model, val_loader, device)
                inner_losses.append(val_loss)
            avg_inner_loss = np.mean(inner_losses)
            if avg_inner_loss < best_val_loss:
                best_val_loss = avg_inner_loss
                best_params = params
        print(f"最佳超参数: {best_params}")
        # 用最佳超参数在本折训练集上训练
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.tensor(X_train_std, dtype=torch.float32),
            torch.tensor(y_train_std, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.tensor(X_test_std, dtype=torch.float32),
            torch.tensor(y_test_std, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
        input_dim = X_train_std.shape[1]
        output_dim = 1
        model = MLP(input_dim, output_dim, best_params['hidden_layers']).to(device)
        criterion = best_params['loss_function']
        train_and_validate_mlp_model(train_loader, test_loader, model, criterion, patience=best_params['patience'], learning_rate=best_params['learning_rate'], max_epochs=max_epochs, device=device)
        # 保存模型和scaler
        torch.save(model.state_dict(), os.path.join(model_dir, f"mlp_model_{target}_fold{fold+1}.pth"))
        joblib.dump(scaler_X, os.path.join(model_dir, f"scaler_x_{target}_fold{fold+1}.save"))
        joblib.dump(scaler_y, os.path.join(model_dir, f"scaler_y_{target}_fold{fold+1}.save"))
        # 评估
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test_std, dtype=torch.float32).to(device)).cpu().numpy()
        preds_inv = scaler_y.inverse_transform(preds)
        y_test_inv = scaler_y.inverse_transform(y_test_std)
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test_inv, preds_inv)
        r2 = r2_score(y_test_inv, preds_inv)
        print(f"Fold {fold+1} MSE: {mse:.4f}, R2: {r2:.4f}")
        fold_metrics.append({'mse': mse, 'r2': r2})
    # 输出平均指标
    avg_mse = np.mean([m['mse'] for m in fold_metrics])
    avg_r2 = np.mean([m['r2'] for m in fold_metrics])
    print(f"\n{target} 平均MSE: {avg_mse:.4f}, 平均R2: {avg_r2:.4f}")
    with open(os.path.join(result_dir, f"{target}_nestedcv_metrics.txt"), 'w', encoding='utf-8') as f:
        f.write(f"平均MSE: {avg_mse:.4f}\n平均R2: {avg_r2:.4f}\n")

