import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[(1024, 'relu')]):
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim, act in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            if act == 'relu':
                layers.append(nn.ReLU())
            elif act == 'tanh':
                layers.append(nn.Tanh())
            # 可扩展更多激活函数
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def create_dataloader(X, y, indices, batch_size=64, shuffle=True):
    subset = Subset(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)), indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

def train_and_validate_mlp_model(train_loader, val_loader, model, criterion, patience=5, learning_rate=0.001, max_epochs=500, device="cpu"):
    early_stopping = EarlyStopping(patience=patience)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    last_train_loss = None
    last_val_loss = None
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_train_loss = running_loss / len(train_loader)
        last_train_loss = average_train_loss
        average_val_loss = validate_mlp_model(model, val_loader, device)
        last_val_loss = average_val_loss
        early_stopping(average_val_loss)
        if early_stopping.early_stop:
            break
    return last_train_loss, last_val_loss

def validate_mlp_model(model, val_loader, device="cpu"):
    model.eval()
    val_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels.view(-1, 1))
            val_loss += loss.item()
    return val_loss / len(val_loader)

def evaluate_mlp_model(model, test_loader, device="cpu"):
    model.eval()
    test_true_values = []
    test_predicted_values = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_features)
            predicted = outputs.squeeze()
            test_true_values.extend(batch_labels.view(-1).tolist())
            test_predicted_values.extend(predicted.view(-1).tolist())
    mse = mean_squared_error(test_true_values, test_predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_true_values, test_predicted_values)
    r2 = r2_score(test_true_values, test_predicted_values)
    ci95 = 1.96 * np.std(np.array(test_predicted_values) - np.array(test_true_values)) / np.sqrt(len(test_true_values))
    pearson_corr, _ = pearsonr(test_true_values, test_predicted_values)
    spearman_corr, _ = spearmanr(test_true_values, test_predicted_values)
    within_2 = np.mean(np.abs(np.array(test_true_values) - np.array(test_predicted_values)) < 2) * 100
    within_5 = np.mean(np.abs(np.array(test_true_values) - np.array(test_predicted_values)) < 5) * 100
    within_10 = np.mean(np.abs(np.array(test_true_values) - np.array(test_predicted_values)) < 10) * 100
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'CI95': ci95,
        'Pearson': pearson_corr,
        'Spearman': spearman_corr,
        'within_2': within_2,
        'within_5': within_5,
        'within_10': within_10
    }
    return metrics

def generate_mlp_params(learning_rates, patiences, hidden_layers_configs, loss_functions):
    from itertools import product
    param_combinations = [
        {
            'learning_rate': lr,
            'patience': p,
            'hidden_layers': hl,
            'loss_function': lf
        }
        for lr, p, hl, lf in product(learning_rates, patiences, hidden_layers_configs, loss_functions)
    ]
    return param_combinations
