import torch

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

