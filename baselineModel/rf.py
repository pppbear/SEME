from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset import load_dataset

file_name = "shanghai.xlsx"

# 加载数据（已标准化）
X_train, X_test, y_train, y_test, scaler_y = load_dataset(file_name)

# 转换为 DataFrame
y_train = pd.DataFrame(y_train, columns=["nighttime_", "lst_day_c", "lst_night_"])
y_test = pd.DataFrame(y_test, columns=["nighttime_", "lst_day_c", "lst_night_"])

# 训练随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 模型预测
y_pred = rf_model.predict(X_test)

# 评估模型性能
for i, col in enumerate(y_train.columns):
    mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"目标变量: {col}")
    print(f"MSE: {mse:.4f}")
    print(f"R² : {r2:.4f}")

# 反标准化
y_test_inverse = scaler_y.inverse_transform(y_test)
y_pred_inverse = scaler_y.inverse_transform(y_pred)

# 可视化预测效果（前100个样本）
# 为了绘制 y=x 参考线，先找出最大值
max_val = max(np.max(y_test_inverse), np.max(y_pred_inverse))
min_val = min(np.min(y_test_inverse), np.min(y_pred_inverse))

# 画图
plt.figure(figsize=(10, 6))
plt.scatter(y_test_inverse, y_pred_inverse, alpha=0.6, label="Predictions vs Actual", edgecolors='k')
plt.plot([min_val, max_val], [min_val, max_val], 'orange', linestyle='--', label='Ideal Fit')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual Values (Test Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
