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

# scaler缓存
scaler_cache = {}
def get_scaler(scaler_path):
    if scaler_path not in scaler_cache:
        print(f"[Scaler缓存] 加载新scaler: {scaler_path}")
        scaler_cache[scaler_path] = joblib.load(scaler_path)
    else:
        print(f"[Scaler缓存] 命中缓存: {scaler_path}")
    return scaler_cache[scaler_path]

# MLP模型缓存
mlp_model_cache = {}
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

# RF模型缓存
rf_model_cache = {}
def get_rf_model(model_path):
    if model_path not in rf_model_cache:
        print(f"[RF模型缓存] 加载新RF模型: {model_path}")
        rf_model_cache[model_path] = joblib.load(model_path)
    else:
        print(f"[RF模型缓存] 命中缓存: {model_path}")
    return rf_model_cache[model_path]

# KAN scaler缓存
kan_scaler_cache = {}
def get_kan_scaler(scaler_path):
    if scaler_path not in kan_scaler_cache:
        print(f"[KAN Scaler缓存] 加载新scaler: {scaler_path}")
        import pickle
        with open(scaler_path, 'rb') as f:
            kan_scaler_cache[scaler_path] = pickle.load(f)
    else:
        print(f"[KAN Scaler缓存] 命中缓存: {scaler_path}")
    return kan_scaler_cache[scaler_path]

# KAN模型和scaler缓存
kan_model_cache = {}
def get_kan_model_and_scaler(model_path, scaler_x_path, scaler_y_path, KAN_class, model_params, ckpt_path):
    key = (model_path, scaler_x_path, scaler_y_path)
    if key not in kan_model_cache:
        print(f"[KAN模型缓存] 加载新KAN模型: {model_path}")
        import torch
        # 加载模型
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model_kan = KAN_class(**model_params, symbolic_enabled=True, ckpt_path=ckpt_path)
        model_kan.load_state_dict(checkpoint['model_state_dict'])
        model_kan.eval()
        # 加载scaler（用缓存）
        kan_scaler_x = get_kan_scaler(scaler_x_path)
        kan_scaler_y = get_kan_scaler(scaler_y_path)
        kan_model_cache[key] = (model_kan, kan_scaler_x, kan_scaler_y)
    else:
        print(f"[KAN模型缓存] 命中缓存: {model_path}")
    return kan_model_cache[key]
