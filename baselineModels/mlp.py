from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from dataset import load_dataset
from torch.utils.data import TensorDataset, DataLoader

file_name = "shanghai.xlsx"

# 加载数据
X_train, X_test, y_train, y_test, scaler_y = load_dataset(file_name)

# 转换为 TensorDataset 再转换为 DataLoader
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# 初始化模型
for X_batch, y_batch in train_loader:
    input_dim = X_batch.shape[1]
    output_dim = y_batch.shape[1]
    break  # 只取第一个 batch 就够了
model = MLPRegressor(input_dim, output_dim)

# 优化器和损失函数
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
epochs = 400
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 完成训练后，在测试集上评估一次
model.eval()  # 设置为评估模式
preds = []
trues = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        output = model(X_batch)
        preds.append(output)
        trues.append(y_batch)

pred_tensor = torch.cat(preds)
true_tensor = torch.cat(trues)

# 计算每个输出维度的 MSE
mse_values = []
for i in range(true_tensor.shape[1]):  # 遍历每个输出维度
    mse = mean_squared_error(true_tensor[:, i].numpy(), pred_tensor[:, i].numpy())
    mse_values.append(mse)
    print(f'Mean Squared Error for output {i+1}: {mse:.4f}')

# 计算 R²
r2_values = []
for i in range(true_tensor.shape[1]):  # 遍历每个输出维度
    r2 = r2_score(true_tensor[:, i].numpy(), pred_tensor[:, i].numpy())
    r2_values.append(r2)
    print(f'R² for output {i+1}: {r2:.4f}')

# 可视化实际值和预测值
plt.figure(figsize=(10, 6))
plt.scatter(true_tensor.numpy(), pred_tensor.numpy(), c='blue', label='Predictions vs Actuals')
plt.plot([true_tensor.min(), true_tensor.max()], [true_tensor.min(), true_tensor.max()], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.title('Model Predictions vs Actual Values')
plt.show()
