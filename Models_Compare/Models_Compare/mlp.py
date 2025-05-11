from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from dataset import load_dataset
from torch.utils.data import TensorDataset, DataLoader

# 定义输入数据文件名（使用完整路径）
file_name = None  # 使用dataset.py中的默认路径

# 调用dataset.py中的函数加载并预处理数据
X_train, X_test, y_train, y_test, scaler_y = load_dataset(file_name)

# 将NumPy数组转换为PyTorch张量，并创建数据加载器
# 转换为 TensorDataset 再转换为 DataLoader
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)

# 创建批处理数据加载器，训练时随机打乱数据
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义多层感知器(MLP)模型类
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPRegressor, self).__init__()
        # 构建三层神经网络：输入层->64节点隐藏层->32节点隐藏层->输出层
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),   # 输入层到第一隐藏层
            nn.ReLU(),                  # ReLU激活函数
            nn.Linear(64, 32),          # 第一隐藏层到第二隐藏层
            nn.ReLU(),                  # ReLU激活函数
            nn.Linear(32, output_dim)   # 第二隐藏层到输出层
        )

    def forward(self, x):
        # 前向传播函数
        return self.model(x)

# 从第一个批次中获取输入维度和输出维度，然后初始化模型
for X_batch, y_batch in train_loader:
    input_dim = X_batch.shape[1]
    output_dim = y_batch.shape[1]
    break  # 只取第一个 batch 就够了
model = MLPRegressor(input_dim, output_dim)

# 定义优化器和损失函数
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器，学习率=0.001

# 训练模型
epochs = 400  # 训练轮数
for epoch in range(epochs):
    model.train()  # 设置为训练模式
    total_loss = 0.0

    # 遍历每个批次进行训练
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()  # 清除梯度
        outputs = model(X_batch)  # 前向传播
        loss = criterion(outputs, y_batch)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 保存模型
model_save_path = os.path.join(os.path.dirname(__file__), "models", "mlp_model.pth")
torch.save(model.state_dict(), model_save_path)
print(f"模型已保存到: {model_save_path}")

# 在测试集上评估模型性能
model.eval()  # 设置为评估模式
preds = []
trues = []

# 遍历测试集批次收集预测结果
with torch.no_grad():  # 不计算梯度
    for X_batch, y_batch in test_loader:
        output = model(X_batch)
        preds.append(output)
        trues.append(y_batch)

# 合并所有批次的结果
pred_tensor = torch.cat(preds)
true_tensor = torch.cat(trues)

# 计算每个输出维度的均方误差(MSE)
mse_values = []
for i in range(true_tensor.shape[1]):  # 遍历每个输出维度
    mse = mean_squared_error(true_tensor[:, i].numpy(), pred_tensor[:, i].numpy())
    mse_values.append(mse)
    print(f'Mean Squared Error for output {i+1}: {mse:.4f}')

# 计算决定系数(R²)，衡量模型解释目标变量变异性的能力
r2_values = []
for i in range(true_tensor.shape[1]):  # 遍历每个输出维度
    r2 = r2_score(true_tensor[:, i].numpy(), pred_tensor[:, i].numpy())
    r2_values.append(r2)
    print(f'R² for output {i+1}: {r2:.4f}')

# 可视化实际值和预测值的散点图
plt.figure(figsize=(10, 6))
plt.scatter(true_tensor.numpy(), pred_tensor.numpy(), c='blue', label='Predictions vs Actuals')
plt.plot([true_tensor.min(), true_tensor.max()], [true_tensor.min(), true_tensor.max()], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.title('Model Predictions vs Actual Values')

# 保存图表
results_save_path = os.path.join(os.path.dirname(__file__), "results", "mlp_predictions.png")
plt.savefig(results_save_path)
print(f"结果图表已保存到: {results_save_path}")
plt.show()
