import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from dataset import load_dataset
from mlp import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 加载数据
file_name = None  # 使用dataset.py中的默认路径
X_train, X_test, y_train, y_test, scaler_y = load_dataset(file_name)

# 创建目标变量名称列表
target_names = ["夜间光", "白天地表温度", "夜间地表温度"]

# 将y_train和y_test转换为DataFrame（方便随机森林模型使用）
y_train_df = pd.DataFrame(y_train, columns=["nighttime_", "lst_day_c", "lst_night_"])
y_test_df = pd.DataFrame(y_test, columns=["nighttime_", "lst_day_c", "lst_night_"])

# ============= 训练和评估MLP模型 =============
# 将数据转换为张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 初始化MLP模型
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
mlp_model = MLPRegressor(input_dim, output_dim)

# 加载预训练模型（如果有）或训练新模型
mlp_model_path = os.path.join(os.path.dirname(__file__), "models", "mlp_model.pth")
if os.path.exists(mlp_model_path):
    print("加载预训练MLP模型...")
    mlp_model.load_state_dict(torch.load(mlp_model_path))
else:
    print("训练新的MLP模型...")
    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)
    
    # 训练模型
    epochs = 400
    for epoch in range(epochs):
        mlp_model.train()
        optimizer.zero_grad()
        outputs = mlp_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    # 保存训练好的模型
    torch.save(mlp_model.state_dict(), mlp_model_path)

# 用MLP模型进行预测
mlp_model.eval()
with torch.no_grad():
    mlp_preds = mlp_model(X_test_tensor).numpy()

# ============= 训练和评估随机森林模型 =============
# 初始化随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练随机森林模型
rf_model.fit(X_train, y_train_df)

# 用随机森林模型进行预测
rf_preds = rf_model.predict(X_test)

# ============= 计算两个模型的性能指标 =============
print("\n两个模型的性能比较:")
print("=" * 50)
print(f"{'目标变量':<15}{'MLP-MSE':<12}{'RF-MSE':<12}{'MLP-R²':<12}{'RF-R²':<12}")
print("-" * 50)

mlp_mse_values = []
rf_mse_values = []
mlp_r2_values = []
rf_r2_values = []

for i in range(output_dim):
    # 计算MLP的MSE和R²
    mlp_mse = mean_squared_error(y_test[:, i], mlp_preds[:, i])
    mlp_r2 = r2_score(y_test[:, i], mlp_preds[:, i])
    
    # 计算随机森林的MSE和R²
    rf_mse = mean_squared_error(y_test_df.iloc[:, i], rf_preds[:, i])
    rf_r2 = r2_score(y_test_df.iloc[:, i], rf_preds[:, i])
    
    # 保存结果
    mlp_mse_values.append(mlp_mse)
    rf_mse_values.append(rf_mse)
    mlp_r2_values.append(mlp_r2)
    rf_r2_values.append(rf_r2)
    
    print(f"{target_names[i]:<15}{mlp_mse:<12.4f}{rf_mse:<12.4f}{mlp_r2:<12.4f}{rf_r2:<12.4f}")

print("=" * 50)
print(f"{'平均':<15}{np.mean(mlp_mse_values):<12.4f}{np.mean(rf_mse_values):<12.4f}"
      f"{np.mean(mlp_r2_values):<12.4f}{np.mean(rf_r2_values):<12.4f}")

# ============= 可视化比较 =============
# 1. 反标准化预测结果
y_test_inverse = scaler_y.inverse_transform(y_test)
mlp_preds_inverse = scaler_y.inverse_transform(mlp_preds)
rf_preds_inverse = scaler_y.inverse_transform(rf_preds)

# 2. 绘制散点图比较
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('MLP与随机森林(RF)模型预测效果对比', fontsize=16)

for i in range(3):
    ax = axes[i]
    
    # 找出最大值和最小值
    min_val = min(np.min(y_test_inverse[:, i]), 
                  np.min(mlp_preds_inverse[:, i]), 
                  np.min(rf_preds_inverse[:, i]))
    max_val = max(np.max(y_test_inverse[:, i]), 
                  np.max(mlp_preds_inverse[:, i]), 
                  np.max(rf_preds_inverse[:, i]))
    
    # 绘制MLP和RF的散点图
    ax.scatter(y_test_inverse[:, i], mlp_preds_inverse[:, i], 
               alpha=0.6, label='MLP模型', color='blue', marker='o')
    ax.scatter(y_test_inverse[:, i], rf_preds_inverse[:, i], 
               alpha=0.6, label='RF模型', color='red', marker='x')
    
    # 绘制理想拟合线
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='理想拟合')
    
    # 添加标签和标题
    ax.set_xlabel('实际值')
    ax.set_ylabel('预测值')
    ax.set_title(f'目标变量: {target_names[i]}')
    ax.legend()
    ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，留出总标题的空间
# 保存模型对比图
model_comparison_path = os.path.join(os.path.dirname(__file__), "results", "model_comparison.png")
plt.savefig(model_comparison_path, dpi=300)
plt.show()

# 3. 绘制性能指标对比条形图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MSE对比图（越低越好）
axes[0].bar(
    np.arange(len(target_names)) - 0.2, 
    mlp_mse_values, 
    width=0.4, 
    label='MLP', 
    color='blue'
)
axes[0].bar(
    np.arange(len(target_names)) + 0.2, 
    rf_mse_values, 
    width=0.4, 
    label='随机森林', 
    color='red'
)
axes[0].set_xticks(np.arange(len(target_names)))
axes[0].set_xticklabels(target_names)
axes[0].set_ylabel('均方误差 (MSE)')
axes[0].set_title('MSE对比 (越低越好)')
axes[0].legend()
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# R²对比图（越高越好）
axes[1].bar(
    np.arange(len(target_names)) - 0.2, 
    mlp_r2_values, 
    width=0.4, 
    label='MLP', 
    color='blue'
)
axes[1].bar(
    np.arange(len(target_names)) + 0.2, 
    rf_r2_values, 
    width=0.4, 
    label='随机森林', 
    color='red'
)
axes[1].set_xticks(np.arange(len(target_names)))
axes[1].set_xticklabels(target_names)
axes[1].set_ylabel('决定系数 (R²)')
axes[1].set_title('R²对比 (越高越好)')
axes[1].legend()
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.suptitle('MLP与随机森林模型性能指标对比', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
# 保存指标对比图
metrics_comparison_path = os.path.join(os.path.dirname(__file__), "results", "metrics_comparison.png")
plt.savefig(metrics_comparison_path, dpi=300)
plt.show()

print("\n比较结果已保存为图片：")
print(f"1. {model_comparison_path} - 预测值散点图对比")
print(f"2. {metrics_comparison_path} - 性能指标对比") 