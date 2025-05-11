from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from dataset import load_dataset
import joblib

# 定义输入数据文件名（使用完整路径）
file_name = None  # 使用dataset.py中的默认路径

# 调用dataset.py中的函数加载并预处理数据
X_train, X_test, y_train, y_test, scaler_y = load_dataset(file_name)

# 将NumPy数组转换为DataFrame，方便后续处理和结果分析
y_train = pd.DataFrame(y_train, columns=["nighttime_", "lst_day_c", "lst_night_"])
y_test = pd.DataFrame(y_test, columns=["nighttime_", "lst_day_c", "lst_night_"])

# 初始化并训练随机森林回归模型
# n_estimators=100: 使用100棵决策树
# random_state=42: 设定随机种子，确保结果可复现
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 使用训练好的模型对测试集进行预测
y_pred = rf_model.predict(X_test)

# 保存模型（可选）
model_save_path = os.path.join(os.path.dirname(__file__), "models", "rf_model.joblib")
joblib.dump(rf_model, model_save_path)
print(f"模型已保存到: {model_save_path}")

# 评估模型性能：对每个目标变量计算MSE和R²
for i, col in enumerate(y_train.columns):
    mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"目标变量: {col}")
    print(f"MSE: {mse:.4f}")
    print(f"R² : {r2:.4f}")

# 将标准化的结果反转回原始尺度，便于解释
y_test_inverse = scaler_y.inverse_transform(y_test)
y_pred_inverse = scaler_y.inverse_transform(y_pred)

# 找出数据的最大值和最小值，用于绘制理想拟合线
max_val = max(np.max(y_test_inverse), np.max(y_pred_inverse))
min_val = min(np.min(y_test_inverse), np.min(y_pred_inverse))

# 绘制预测值与实际值的散点图，用于可视化评估模型性能
plt.figure(figsize=(10, 6))
plt.scatter(y_test_inverse, y_pred_inverse, alpha=0.6, label="Predictions vs Actual", edgecolors='k')
plt.plot([min_val, max_val], [min_val, max_val], 'orange', linestyle='--', label='Ideal Fit')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual Values (Test Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存图表
results_save_path = os.path.join(os.path.dirname(__file__), "results", "rf_predictions.png")
plt.savefig(results_save_path)
print(f"结果图表已保存到: {results_save_path}")
plt.show()
