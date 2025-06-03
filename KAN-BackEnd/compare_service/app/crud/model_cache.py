import torch
import joblib
import torch.nn as nn

# MLP模型定义
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
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

mlp_model_cache = {}
scaler_cache = {}

def get_mlp_model(model_path, input_dim, output_dim, hidden_layers):
    key = (model_path, input_dim, output_dim, tuple(hidden_layers))
    if key not in mlp_model_cache:
        print(f"[模型缓存] 加载新MLP模型: {model_path}, input_dim={input_dim}, output_dim={output_dim}, hidden_layers={hidden_layers}")
        model = MLPRegressor(input_dim, output_dim, hidden_layers)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        mlp_model_cache[key] = model
    else:
        print(f"[模型缓存] 命中缓存: {model_path}, input_dim={input_dim}, output_dim={output_dim}, hidden_layers={hidden_layers}")
    return mlp_model_cache[key]

def get_scaler(scaler_path):
    if scaler_path not in scaler_cache:
        print(f"[Scaler缓存] 加载新scaler: {scaler_path}")
        scaler_cache[scaler_path] = joblib.load(scaler_path)
    else:
        print(f"[Scaler缓存] 命中缓存: {scaler_path}")
    return scaler_cache[scaler_path] 


#TODO: rf模型缓存
#TODO: kan模型缓存
#TODO: 模型缓存清理
