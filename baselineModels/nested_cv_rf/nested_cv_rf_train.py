import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
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
n_estimators_list = [50, 100, 150]
max_depth_list = [None, 10, 20]
param_combinations = [
    {'n_estimators': n, 'max_depth': d}
    for n in n_estimators_list for d in max_depth_list
]

n_splits = 5  # 外层折数
random_seed = 42

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
    outer_kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(outer_kf.split(X)):
        print(f"\n--- 外层第{fold+1}折 ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train)
        X_train_std = scaler_X.transform(X_train)
        X_test_std = scaler_X.transform(X_test)
        y_train_std = scaler_y.transform(y_train).ravel()
        y_test_std = scaler_y.transform(y_test).ravel()
        # 内层交叉验证选超参数
        inner_kf = KFold(n_splits=3, shuffle=True, random_state=fold)
        best_params = None
        best_val_loss = float('inf')
        for params in param_combinations:
            inner_losses = []
            for inner_train_idx, inner_val_idx in inner_kf.split(X_train_std):
                X_inner_train, X_inner_val = X_train_std[inner_train_idx], X_train_std[inner_val_idx]
                y_inner_train, y_inner_val = y_train_std[inner_train_idx], y_train_std[inner_val_idx]
                rf_model = RandomForestRegressor(**params, random_state=random_seed)
                rf_model.fit(X_inner_train, y_inner_train)
                val_preds = rf_model.predict(X_inner_val)
                val_loss = mean_squared_error(y_inner_val, val_preds)
                inner_losses.append(val_loss)
            avg_inner_loss = np.mean(inner_losses)
            if avg_inner_loss < best_val_loss:
                best_val_loss = avg_inner_loss
                best_params = params
        print(f"最佳超参数: {best_params}")
        # 用最佳超参数在本折训练集上训练
        rf_model = RandomForestRegressor(**best_params, random_state=random_seed)
        rf_model.fit(X_train_std, y_train_std)
        # 保存模型和scaler
        joblib.dump(rf_model, os.path.join(model_dir, f"rf_model_{target}_fold{fold+1}.joblib"))
        joblib.dump(scaler_X, os.path.join(model_dir, f"scaler_x_{target}_fold{fold+1}.save"))
        joblib.dump(scaler_y, os.path.join(model_dir, f"scaler_y_{target}_fold{fold+1}.save"))
        # 评估
        preds = rf_model.predict(X_test_std)
        preds_inv = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
        y_test_inv = scaler_y.inverse_transform(y_test_std.reshape(-1, 1)).ravel()
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


