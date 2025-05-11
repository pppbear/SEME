import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from kan.MultKAN import KAN  # 修正导入
import gc  # 导入垃圾回收模块

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置设备（GPU或CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 设置CUDA优化
if torch.cuda.is_available():
    # 设置cudnn基准模式以提高计算速度
    torch.backends.cudnn.benchmark = True
    # 显示CUDA设备信息
    current_device = torch.cuda.current_device()
    print(f"当前GPU设备: {torch.cuda.get_device_name(current_device)}")
    print(f"GPU显存总量: {torch.cuda.get_device_properties(current_device).total_memory / 1024**3:.2f} GB")
    print(f"当前已分配显存: {torch.cuda.memory_allocated(current_device) / 1024**3:.2f} GB")

# MLP模型定义
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# 加载数据集
def load_data():
    file_path = 'data.xlsx'  # 使用项目根目录下的数据文件
    print(f"正在加载数据: {file_path}")
    
    df = pd.read_excel(file_path)
    print(f"数据加载完成，原始形状: {df.shape}")
    
    # 检查并删除无用列
    columns_to_drop = ["FID", "Shape *", "FID_", "Id", "Shape_Leng", "SUM_1", "LENGTH", "LENGTH_1", "Land02", "Land50234", "Land505", "sidewalk_M", "building_M", "vegetation",
                      "sky_MEAN", "POI餐饮", "POI风景", "POI公司", "POI购物", "POI科教", "POI医疗", "POI政府", "railway_m", "Subway_m", "car_road_m", "high_grade", "Shape_Le_1",
                      "Shape_Area", "Longitude", "Latitude"]
    
    # 只删除实际存在的列
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True)
        print(f"删除了 {len(columns_to_drop)} 个无用列")
    
    # 检查数据中的目标变量
    available_targets = []
    target_candidates = ["nighttime_", "uhi_night_", "lst_day_c", "lst_night_"]
    
    for target in target_candidates:
        matching_cols = [col for col in df.columns if target in col]
        if matching_cols:
            available_targets.extend(matching_cols)
    
    if not available_targets:
        raise ValueError(f"未找到任何目标变量列。请确保数据中包含以下列之一: {target_candidates}")
    
    print(f"找到的目标变量列: {available_targets}")
    
    # 分离目标变量
    target_cols = available_targets
    y = df[target_cols].values
    X = df.drop(columns=target_cols).values
    
    print(f"特征矩阵形状: {X.shape}, 目标矩阵形状: {y.shape}")
    
    # 返回特征和目标变量
    return X, y, target_cols

# 训练MLP模型
def train_mlp(X_train, y_train, X_val, y_val, params):
    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)
    
    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val_scaled)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    
    # 初始化模型
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = MLPRegressor(input_dim, params['hidden_dim'], output_dim)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # 训练模型
    model.train()
    for epoch in range(params['epochs']):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor).numpy()
        val_preds_inverse = scaler_y.inverse_transform(val_preds)
        y_val_inverse = scaler_y.inverse_transform(y_val_scaled)
    
    # 计算每个目标变量的MSE和R²
    mse_values = []
    r2_values = []
    for i in range(output_dim):
        mse = mean_squared_error(y_val_inverse[:, i], val_preds_inverse[:, i])
        r2 = r2_score(y_val_inverse[:, i], val_preds_inverse[:, i])
        mse_values.append(mse)
        r2_values.append(r2)
    
    # 返回平均指标和模型
    avg_mse = np.mean(mse_values)
    avg_r2 = np.mean(r2_values)
    
    return model, scaler_X, scaler_y, avg_mse, avg_r2, mse_values, r2_values

# 训练随机森林模型
def train_rf(X_train, y_train, X_val, y_val, params):
    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)
    
    # 训练随机森林模型
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train_scaled)
    
    # 预测验证集
    val_preds = model.predict(X_val_scaled)
    val_preds_inverse = scaler_y.inverse_transform(val_preds)
    y_val_inverse = scaler_y.inverse_transform(y_val_scaled)
    
    # 计算每个目标变量的MSE和R²
    mse_values = []
    r2_values = []
    for i in range(y_train.shape[1]):
        mse = mean_squared_error(y_val_inverse[:, i], val_preds_inverse[:, i])
        r2 = r2_score(y_val_inverse[:, i], val_preds_inverse[:, i])
        mse_values.append(mse)
        r2_values.append(r2)
    
    # 返回平均指标和模型
    avg_mse = np.mean(mse_values)
    avg_r2 = np.mean(r2_values)
    
    return model, scaler_X, scaler_y, avg_mse, avg_r2, mse_values, r2_values

# 训练KAN模型
def train_kan(X_train, y_train, X_val, y_val, params):
    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)
    
    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val_scaled).to(device)
    
    # 准备KAN模型的数据集字典
    dataset = {
        'train_input': X_train_tensor,
        'train_label': y_train_tensor,
        'test_input': X_val_tensor,
        'test_label': y_val_tensor
    }
    
    # 初始化KAN模型
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    # 清理GPU缓存以释放内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # 使用更优化的参数初始化KAN模型
    model = KAN(
        width=[input_dim, params['hidden_dim'], output_dim],
        grid=params['grid'],
        k=params['k'],
        symbolic_enabled=True,
        optimizer_params={"lr": params['learning_rate']}  # 预设优化器参数
    ).to(device)
    
    # 使用混合精度训练来提高速度
    try:
        from torch.cuda.amp import autocast, GradScaler
        use_amp = torch.cuda.is_available()
        scaler = GradScaler() if use_amp else None
    except ImportError:
        use_amp = False
        scaler = None
    
    # 训练模型
    if use_amp:
        print(f"使用混合精度训练以加速...")
    
    # 修改KAN训练参数以提高性能
    history = model.fit(
        dataset=dataset,
        opt=params['optimizer'],
        steps=params['steps'],
        lamb_l1=params['lamb_l1'],
        lamb_entropy=params['lamb_entropy'],
        lr=params['learning_rate'],
        batch_size=params.get('batch_size', 256),  # 增加批量大小利用GPU并行性
        plateau_patience=15,  # 增加早停耐心值
        plateau_factor=0.7,   # 更积极的学习率降低
        verbose=True          # 显示进度
    )
    
    # 评估模型
    with torch.no_grad():
        val_preds = model(X_val_tensor).cpu().numpy()
        val_preds_inverse = scaler_y.inverse_transform(val_preds)
        y_val_inverse = scaler_y.inverse_transform(y_val_scaled.cpu().numpy())
    
    # 计算每个目标变量的MSE和R²
    mse_values = []
    r2_values = []
    for i in range(output_dim):
        mse = mean_squared_error(y_val_inverse[:, i], val_preds_inverse[:, i])
        r2 = r2_score(y_val_inverse[:, i], val_preds_inverse[:, i])
        mse_values.append(mse)
        r2_values.append(r2)
    
    # 返回平均指标和模型
    avg_mse = np.mean(mse_values)
    avg_r2 = np.mean(r2_values)
    
    return model, scaler_X, scaler_y, avg_mse, avg_r2, mse_values, r2_values

# 执行十折嵌套交叉验证
def nested_cross_validation(X, y, target_cols, model_type='mlp'):
    n_samples = X.shape[0]
    
    # 定义外部折
    outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
    # 定义内部折
    inner_cv = KFold(n_splits=9, shuffle=True, random_state=42)
    
    # 存储每个外部折的结果
    outer_results = {
        'mse': [],
        'r2': [],
        'mse_by_target': [[] for _ in range(len(target_cols))],
        'r2_by_target': [[] for _ in range(len(target_cols))]
    }
    
    # 定义参数网格
    if model_type == 'mlp':
        param_grid = {
            'hidden_dim': [32, 64, 128],
            'learning_rate': [0.001, 0.01],
            'batch_size': [32, 64],
            'epochs': [200, 400]
        }
    elif model_type == 'rf':  # 随机森林
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20]
        }
    else:  # KAN模型
        param_grid = {
            'hidden_dim': [8, 12, 16],
            'grid': [4, 5],
            'k': [2, 3],
            'optimizer': ['Adam'],
            'steps': [400, 600],
            'lamb_l1': [0.4, 0.5],
            'lamb_entropy': [0.4, 0.5],
            'learning_rate': [0.0005, 0.001],
            'batch_size': [128, 256]  # 添加批量大小参数
        }
    
    # 生成所有参数组合
    def generate_param_combinations(param_grid):
        keys = list(param_grid.keys())
        combinations = []
        
        def backtrack(index, current_params):
            if index == len(keys):
                combinations.append(current_params.copy())
                return
            
            key = keys[index]
            for value in param_grid[key]:
                current_params[key] = value
                backtrack(index + 1, current_params)
        
        backtrack(0, {})
        return combinations
    
    param_combinations = generate_param_combinations(param_grid)
    print(f"共生成 {len(param_combinations)} 种参数组合进行网格搜索")
    
    # 外部循环
    for outer_fold, (train_val_idx, test_idx) in enumerate(outer_cv.split(X)):
        print(f"\n===== 外部折 {outer_fold + 1}/10 =====")
        
        # 清理GPU缓存以释放内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"当前GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
        
        # 存储内部折的结果
        inner_results = []
        
        # 内部循环：网格搜索
        for param_set in param_combinations:
            inner_mse = []
            inner_r2 = []
            
            for inner_fold, (train_idx, val_idx) in enumerate(inner_cv.split(X_train_val)):
                X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
                y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
                
                # 训练和评估模型
                if model_type == 'mlp':
                    _, _, _, avg_mse, avg_r2, _, _ = train_mlp(
                        X_train, y_train, X_val, y_val, param_set
                    )
                elif model_type == 'rf':
                    _, _, _, avg_mse, avg_r2, _, _ = train_rf(
                        X_train, y_train, X_val, y_val, param_set
                    )
                else:  # KAN模型
                    _, _, _, avg_mse, avg_r2, _, _ = train_kan(
                        X_train, y_train, X_val, y_val, param_set
                    )
                
                inner_mse.append(avg_mse)
                inner_r2.append(avg_r2)
            
            # 计算当前参数组合的平均性能
            avg_inner_mse = np.mean(inner_mse)
            avg_inner_r2 = np.mean(inner_r2)
            
            inner_results.append({
                'params': param_set,
                'avg_mse': avg_inner_mse,
                'avg_r2': avg_inner_r2
            })
            
            print(f"参数: {param_set}, 内部平均MSE: {avg_inner_mse:.4f}, 内部平均R²: {avg_inner_r2:.4f}")
        
        # 选择最佳参数（根据R²值）
        best_result = max(inner_results, key=lambda x: x['avg_r2'])
        best_params = best_result['params']
        print(f"\n选择最佳参数: {best_params}\n")
        
        # 使用最佳参数在整个训练+验证数据上训练模型
        if model_type == 'mlp':
            model, scaler_X, scaler_y, _, _, final_mse_values, final_r2_values = train_mlp(
                X_train_val, y_train_val, X_test, y_test, best_params
            )
        elif model_type == 'rf':
            model, scaler_X, scaler_y, _, _, final_mse_values, final_r2_values = train_rf(
                X_train_val, y_train_val, X_test, y_test, best_params
            )
        else:  # KAN模型
            # 在训练最终模型前清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
            model, scaler_X, scaler_y, _, _, final_mse_values, final_r2_values = train_kan(
                X_train_val, y_train_val, X_test, y_test, best_params
            )
            
            # KAN特有：尝试提取符号化方程
            if outer_fold == 0:  # 仅对第一个外部折执行此操作
                try:
                    os.makedirs('results/kan', exist_ok=True)
                    
                    # 准备完整数据集
                    X_full_tensor = torch.FloatTensor(scaler_X.transform(X)).to(device)
                    y_full_tensor = torch.FloatTensor(scaler_y.transform(y)).to(device)
                    
                    dataset_full = {
                        'train_input': X_full_tensor,
                        'train_label': y_full_tensor,
                        'test_input': X_full_tensor,
                        'test_label': y_full_tensor
                    }
                    
                    # 使用KAN的符号化特征方程提取功能
                    try:
                        # 定义符号库
                        print("\n使用KAN的符号化特征方程提取功能...")
                        lib = ['x', 'x^2', 'x^3', 'exp', 'log', 'sqrt', 'sin', 'cos', 'abs']
                        
                        # 应用自动符号化提取
                        model.auto_symbolic(lib=lib)
                        
                        # 获取符号化公式
                        print("提取符号化方程...")
                        symbolic_result = model.symbolic_formula()
                        
                        # 将结果保存到文件
                        with open(f'results/kan/symbolic_equations.txt', 'w', encoding='utf-8') as f:
                            f.write(f"KAN符号化方程 (外部折 {outer_fold + 1}):\n")
                            f.write(f"最佳参数: {best_params}\n\n")
                            f.write(f"符号化结果:\n{symbolic_result}")
                            
                            # 计算特征重要性
                            if hasattr(model, 'attribute'):
                                try:
                                    model.attribute()
                                    if hasattr(model, 'feature_score') and model.feature_score is not None:
                                        feature_importance = model.feature_score.detach().cpu().numpy()
                                        feature_names = [f"X{i}" for i in range(X.shape[1])]
                                        
                                        # 创建特征重要性DataFrame
                                        importance_df = pd.DataFrame({
                                            'feature': feature_names,
                                            'importance': feature_importance
                                        })
                                        
                                        # 按重要性排序
                                        importance_df = importance_df.sort_values('importance', ascending=False)
                                        
                                        f.write("\n\n特征重要性排序:\n")
                                        for i, (_, row) in enumerate(importance_df.iterrows()):
                                            f.write(f"{i+1}. {row['feature']}: {row['importance']:.6f}\n")
                                except Exception as e:
                                    f.write(f"\n\n计算特征重要性时出错: {str(e)}")
                    except Exception as e:
                        print(f"提取符号化方程时出错: {str(e)}")
                        
                except Exception as e:
                    print(f"尝试提取KAN公式时出错: {str(e)}")
        
        # 保存外部测试结果
        outer_results['mse'].append(np.mean(final_mse_values))
        outer_results['r2'].append(np.mean(final_r2_values))
        
        # 保存每个目标变量的指标
        for i, (mse, r2) in enumerate(zip(final_mse_values, final_r2_values)):
            outer_results['mse_by_target'][i].append(mse)
            outer_results['r2_by_target'][i].append(r2)
        
        print(f"外部折 {outer_fold + 1} 测试性能:")
        for i, col in enumerate(target_cols):
            print(f"{col}: MSE = {final_mse_values[i]:.4f}, R² = {final_r2_values[i]:.4f}")
    
    # 计算并打印最终性能
    final_results = {}
    final_results['avg_mse'] = np.mean(outer_results['mse'])
    final_results['avg_r2'] = np.mean(outer_results['r2'])
    final_results['std_mse'] = np.std(outer_results['mse'])
    final_results['std_r2'] = np.std(outer_results['r2'])
    
    final_results['target_mse'] = {}
    final_results['target_r2'] = {}
    final_results['target_mse_std'] = {}
    final_results['target_r2_std'] = {}
    
    print("\n最终模型性能（十折平均）:")
    print("=" * 50)
    print(f"总体: MSE = {final_results['avg_mse']:.4f} ± {final_results['std_mse']:.4f}, "
          f"R² = {final_results['avg_r2']:.4f} ± {final_results['std_r2']:.4f}")
    print("-" * 50)
    
    for i, col in enumerate(target_cols):
        mse_values = outer_results['mse_by_target'][i]
        r2_values = outer_results['r2_by_target'][i]
        
        final_results['target_mse'][col] = np.mean(mse_values)
        final_results['target_r2'][col] = np.mean(r2_values)
        final_results['target_mse_std'][col] = np.std(mse_values)
        final_results['target_r2_std'][col] = np.std(r2_values)
        
        print(f"{col}: MSE = {final_results['target_mse'][col]:.4f} ± {final_results['target_mse_std'][col]:.4f}, "
              f"R² = {final_results['target_r2'][col]:.4f} ± {final_results['target_r2_std'][col]:.4f}")
    
    return final_results, outer_results

# 可视化结果函数
def visualize_results(mlp_results, rf_results, kan_results, target_cols):
    # 转换目标变量名称为中文
    target_names = ["夜间光", "白天地表温度", "夜间地表温度"]
    
    # 创建性能指标对比图表
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 收集数据
    mlp_mse = [mlp_results['target_mse'][col] for col in target_cols]
    mlp_r2 = [mlp_results['target_r2'][col] for col in target_cols]
    rf_mse = [rf_results['target_mse'][col] for col in target_cols]
    rf_r2 = [rf_results['target_r2'][col] for col in target_cols]
    kan_mse = [kan_results['target_mse'][col] for col in target_cols]
    kan_r2 = [kan_results['target_r2'][col] for col in target_cols]
    
    # 定义条形宽度和位置
    width = 0.25
    x = np.arange(len(target_names))
    
    # MSE对比图
    axes[0].bar(x - width, mlp_mse, width=width, label='MLP', color='blue')
    axes[0].bar(x, rf_mse, width=width, label='随机森林', color='red')
    axes[0].bar(x + width, kan_mse, width=width, label='KAN', color='green')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(target_names)
    axes[0].set_ylabel('均方误差 (MSE)')
    axes[0].set_title('MSE对比 (越低越好)')
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # R²对比图
    axes[1].bar(x - width, mlp_r2, width=width, label='MLP', color='blue')
    axes[1].bar(x, rf_r2, width=width, label='随机森林', color='red')
    axes[1].bar(x + width, kan_r2, width=width, label='KAN', color='green')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(target_names)
    axes[1].set_ylabel('决定系数 (R²)')
    axes[1].set_title('R²对比 (越高越好)')
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle('多模型十折交叉验证性能对比', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存结果图
    os.makedirs('results', exist_ok=True)
    results_path = os.path.join(os.path.dirname(__file__), "results", "nested_cv_comparison.png")
    plt.savefig(results_path, dpi=300)
    print(f"交叉验证结果图已保存到: {results_path}")
    plt.show()

# 主函数
if __name__ == "__main__":
    # 限制内部OpenMP线程数，避免CPU争用
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    # 加载数据
    X, y, target_cols = load_data()
    
    # 选择要评估的模型
    model_type = input("请选择要评估的模型类型 (mlp/rf/kan/all): ").strip().lower()
    
    if model_type == 'all':
        print("\n===== 开始MLP模型的十折嵌套交叉验证 =====")
        mlp_final_results, mlp_outer_results = nested_cross_validation(X, y, target_cols, 'mlp')
        
        print("\n===== 开始随机森林模型的十折嵌套交叉验证 =====")
        rf_final_results, rf_outer_results = nested_cross_validation(X, y, target_cols, 'rf')
        
        print("\n===== 开始KAN模型的十折嵌套交叉验证 =====")
        kan_final_results, kan_outer_results = nested_cross_validation(X, y, target_cols, 'kan')
        
        # 可视化比较结果
        visualize_results(mlp_final_results, rf_final_results, kan_final_results, target_cols)
    elif model_type in ['mlp', 'rf', 'kan']:
        print(f"\n===== 开始{model_type.upper()}模型的十折嵌套交叉验证 =====")
        final_results, _ = nested_cross_validation(X, y, target_cols, model_type)
    else:
        print("无效的模型类型，请选择 'mlp', 'rf', 'kan' 或 'all'。") 