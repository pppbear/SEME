import numpy as np
import matplotlib.pyplot as plt
import torch
from kan.MultKAN import KAN
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import warnings
import pickle
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 检查CUDA是否可用并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建必要的目录
os.makedirs('simplified_results', exist_ok=True)
os.makedirs('simplified_results/figures', exist_ok=True)
os.makedirs('simplified_results/models', exist_ok=True)  # 创建模型保存目录

def load_data(file_path):
    """加载Excel数据并进行清洗"""
    df = pd.read_excel(file_path)
    print(f"原始数据量: {len(df)}")
    
    # 检查原始数据中的NaN值
    print("\n原始数据中的NaN值统计:")
    for col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"{col}: {nan_count} 个NaN值")
    
    # 重命名列
    if 'lst_night_' in df.columns:
        df = df.rename(columns={'lst_night_': 'lst_night_c'})
    if 'uhi_night_' in df.columns:
        df = df.rename(columns={'uhi_night_': 'uhi_night_c'})
    
    # 只保留有用的列
    exclude_cols = ['FID', 'Shape *', 'FID_', 'Id', 'Shape_Leng', 'Shape_Le_1', 'Shape_Area','Longitude','Latitude']
    df = df.drop(columns=exclude_cols, errors='ignore')
    
    # 过滤掉目标变量为0的值
    target_variables = ['lst_day_c', 'lst_night_c', 'nighttime_']
    
    # 创建一个掩码，标记所有目标变量都不为0的行
    valid_mask = pd.Series(True, index=df.index)
    for target in target_variables:
        if target in df.columns:
            # 检查目标变量中的NaN值
            nan_count = df[target].isna().sum()
            if nan_count > 0:
                print(f"\n警告：目标变量 {target} 中存在 {nan_count} 个NaN值")
            valid_mask = valid_mask & (df[target] != 0)
    
    # 应用掩码过滤数据
    df = df[valid_mask]
    print(f"过滤零值后的数据量: {len(df)}")
    
    # 处理数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 检查填充后的NaN值
    print("\n填充后的NaN值统计:")
    for col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"{col}: {nan_count} 个NaN值")
    
    print(f"清洗后的数据量: {len(df)}")
    
    # 打印每个目标变量的非零值数量
    print("\n目标变量非零值统计:")
    for target in target_variables:
        if target in df.columns:
            non_zero_count = (df[target] != 0).sum()
            print(f"{target}: {non_zero_count} 个非零值")
    
    return df

def prepare_data(df, target_var, selected_features=None):
    """准备训练数据，可选择只使用选定的特征"""
    # 1. 根据目标变量删除不需要的列
    # 参考cv_dataset.py
    drop_map = {
        'lst_day_c': ['sidewalk_MEAN', 'building_MEAN', 'vegetation_MEAN', 'sky_MEAN', 'nighttime_light_dnb', 'lst_night_c', 'uhi_night_c'],
        'lst_night_c': ['sidewalk_MEAN', 'building_MEAN', 'vegetation_MEAN', 'sky_MEAN', 'nighttime_light_dnb', 'lst_day_c', 'uhi_night_c'],
        'nighttime_': ['sidewalk_MEAN', 'building_MEAN', 'vegetation_MEAN', 'sky_MEAN', 'lst_day_c', 'lst_night_c', 'uhi_night_c']
    }
    drop_cols = drop_map.get(target_var, [])
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=col)
    
    # 2. 删除'不透水面比例'或'NDVI_MEAN'为0的所有行
    for col in ['不透水面比例', 'NDVI_MEAN']:
        if col in df.columns:
            df = df[df[col] != 0]
    
    # 3. 添加'POI总数'和'平均建筑高度'两列
    # 添加POI总数
    poi_columns = [col for col in df.columns if col.startswith('POI') and col != 'POI总数']
    if poi_columns:
        df['POI总数'] = df[poi_columns].sum(axis=1)
    # 添加平均建筑高度
    if '容积率' in df.columns and '建筑密度' in df.columns:
        df['平均建筑高度'] = df.apply(
            lambda row: row['容积率'] / row['建筑密度'] if row['建筑密度'] != 0 and not pd.isnull(row['建筑密度']) else 0,
            axis=1
        )
    
    # 首先过滤掉当前目标变量为0的行
    df_filtered = df[df[target_var] != 0].copy()
    print(f"\n目标变量 {target_var} 的非零值数量: {len(df_filtered)}")
    
    # 检查目标变量中的NaN值
    nan_count = df_filtered[target_var].isna().sum()
    if nan_count > 0:
        print(f"警告：过滤后的目标变量 {target_var} 中仍存在 {nan_count} 个NaN值")
        # 使用中位数填充NaN值
        df_filtered[target_var] = df_filtered[target_var].fillna(df_filtered[target_var].median())
        print(f"已使用中位数填充NaN值")
    
    # 选择特征列（排除目标变量）
    exclude_cols = ['lst_day_c', 'lst_night_c', 'nighttime_', 'uhi_day_c', 'uhi_night_c']
    print(f"\n排除的目标变量相关列: {exclude_cols}")
    
    if selected_features is None:
        # 使用所有特征
        feature_cols = [col for col in df_filtered.columns if col not in exclude_cols]
        print(f"\n排除目标变量后的特征数量: {len(feature_cols)}")
        
        # 只保留数值类型的特征
        numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df_filtered[col])]
        non_numeric_cols = [col for col in feature_cols if col not in numeric_cols]
        if non_numeric_cols:
            print(f"移除的非数值特征: {non_numeric_cols}")
        feature_cols = numeric_cols
        print(f"保留的数值特征数量: {len(feature_cols)}")
        
        # 移除常数特征
        constant_features = []
        for col in feature_cols:
            if df_filtered[col].nunique() <= 1:
                constant_features.append(col)
        if constant_features:
            print(f"\n移除的常数特征: {constant_features}")
            feature_cols = [col for col in feature_cols if col not in constant_features]
            print(f"移除常数特征后的特征数量: {len(feature_cols)}")
        
        print(f"\n最终使用的特征列表:")
        for i, col in enumerate(feature_cols, 1):
            print(f"{i}. {col}")
    else:
        # 使用选定的特征
        feature_cols = selected_features
        print(f"\n使用筛选后的 {len(feature_cols)} 个重要特征进行分析:")
        print(", ".join(feature_cols))
    
    # 准备特征矩阵
    X = df_filtered[feature_cols].values
    y = df_filtered[target_var].values
    
    # 数据检查
    print("\n数据检查:")
    print(f"X 形状: {X.shape}")
    print(f"X 中的 NaN 数量: {np.isnan(X).sum()}")
    print(f"X 中的 Inf 数量: {np.isinf(X).sum()}")
    print(f"y 中的 NaN 数量: {np.isnan(y).sum()}")
    print(f"y 中的 Inf 数量: {np.isinf(y).sum()}")
    
    # 处理异常值
    X = np.clip(X, -1e6, 1e6)
    y = np.clip(y, -1e6, 1e6)
    
    # 使用StandardScaler进行标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # 保存特征标准化参数
    scaler_path = f'simplified_results/models/{target_var}_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\n特征标准化参数已保存至: {scaler_path}")
    
    # 对目标变量进行标准化
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
    # 保存目标变量标准化参数
    y_scaler_path = f'simplified_results/models/{target_var}_y_scaler.pkl'
    with open(y_scaler_path, 'wb') as f:
        pickle.dump(y_scaler, f)
    print(f"目标变量标准化参数已保存至: {y_scaler_path}")
    
    # 划分训练集和测试集（80%训练，20%测试，随机）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # 转换为PyTorch张量
    return {
        'train_input': torch.tensor(X_train, dtype=torch.float32).to(device),
        'train_label': torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device),
        'test_input': torch.tensor(X_test, dtype=torch.float32).to(device),
        'test_label': torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
    }, feature_cols

def train_model(dataset, input_dim, phase=1, target_var=None):
    """训练KAN模型，phase=1为特征选择阶段，phase=2为最终模型训练阶段"""
    # === 补全参数设置，防止KeyError ===
    # 可根据实际需要调整默认值
    params = {
        'hidden_dim': 8,
        'grid': 6,
        'k': 3,
        'steps': 50,
        'lamb_l1': 0.01,
        'lamb_entropy': 0.01,
        'lr': 0.0001
    }
    # 你可以根据target_var或phase自定义参数
    if target_var == 'lst_day_c':
        params.update({'steps': 50, 'lamb_l1': 0.01, 'lamb_entropy': 0.01, 'lr': 0.0001})
    elif target_var == 'lst_night_c':
        params.update({'steps': 50, 'lamb_l1': 0.01, 'lamb_entropy': 0.01, 'lr': 0.0001})
    elif target_var == 'nighttime_':
        params.update({'steps': 50, 'lamb_l1': 0.01, 'lamb_entropy': 0.01, 'lr': 0.0001})
    print(f"\n训练参数: {params}")
    
    # 检查输入数据
    print("\n检查训练数据:")
    train_input = dataset['train_input']
    train_label = dataset['train_label']
    print(f"训练输入形状: {train_input.shape}")
    print(f"训练标签形状: {train_label.shape}")
    print(f"训练输入中的NaN数量: {torch.isnan(train_input).sum().item()}")
    print(f"训练标签中的NaN数量: {torch.isnan(train_label).sum().item()}")
    
    # 训练前再次自动排查和修复nan/inf
    for k, v in dataset.items():
        n_nan = torch.isnan(v).sum().item()
        n_inf = torch.isinf(v).sum().item()
        if n_nan > 0 or n_inf > 0:
            print(f"警告：训练前 {k} 中存在 {n_nan} 个NaN和 {n_inf} 个Inf，已自动用0填充！")
            dataset[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    # 初始化模型
    model = KAN(
        width=[input_dim, params['hidden_dim'], 1],
        grid=params['grid'],
        k=params['k'],
        symbolic_enabled=True
    ).to(device)

    print(f"阶段{phase} - 模型已配置并移至设备 {device}")

    # 自动多轮修复训练loss为nan的情况
    max_retry = 3
    retry = 0
    while retry < max_retry:
        try:
            print(f"\n开始初步训练...（第{retry+1}次尝试）")
            history = model.fit(
                dataset=dataset,
                opt="LBFGS",
                steps=params['steps'],
                lamb_l1=params['lamb_l1'],
                lamb_entropy=params['lamb_entropy'],
                lr=params['lr']
            )
            if np.isnan(history['train_loss']).any():
                print(f"警告：训练过程中出现NaN值，第{retry+1}次尝试失败，自动调整参数重试...")
                # 自动减小学习率、增大正则化
                params['lr'] *= 0.1
                params['lamb_l1'] *= 10
                params['lamb_entropy'] *= 10
                retry += 1
                continue
            else:
                break
        except Exception as e:
            print(f"训练过程中出错: {str(e)}，第{retry+1}次尝试失败，自动调整参数重试...")
            params['lr'] *= 0.1
            params['lamb_l1'] *= 10
            params['lamb_entropy'] *= 10
            retry += 1
    if retry == max_retry:
        print("多次尝试后仍有NaN，建议检查数据分布或进一步清洗数据！")

    print(f"阶段{phase} - 模型训练完成")
    return model

def plot_model_structure(model, target_var, phase):
    """绘制KAN模型结构图"""
    try:
        # 尝试使用KAN的内置绘图方法
        plt.figure(figsize=(12, 8))
        try:
            # 方法1：直接调用KAN模型的plot方法
            model.plot()
            plot_success = True
        except Exception as e1:
            print(f"标准model.plot()方法出错: {str(e1)}")
            plot_success = False
            
            # 方法2：尝试使用另一种方式调用plot方法
            try:
                plt.clf()  # 清除之前的图形
                # 有些版本可能需要传入参数
                model.plot(show_weights=True, figsize=(12, 8))
                plot_success = True
            except Exception as e2:
                print(f"带参数model.plot()方法也出错: {str(e2)}")
                
                # 方法3：尝试手动绘制模型结构和曲线
                try:
                    plt.clf()  # 清除之前的图形
                    plot_success = plot_custom_kan_structure(model)
                except Exception as e3:
                    print(f"自定义绘图方法也出错: {str(e3)}")
                    plot_success = False
        
        # 如果所有绘图方法都失败，创建简单的替代图
        if not plot_success:
            plt.clf()  # 清除之前的图形
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"KAN模型结构 - {target_var} (阶段{phase})\n无法绘制详细结构\n输入维度: {model.width[0]}\n隐藏层: {model.width[1]}", 
                    ha='center', va='center', fontsize=14)
            plt.axis('off')
        
        plt.title(f"KAN模型结构 - {target_var} (阶段{phase})")
        plt.tight_layout()
        
        # 保存图片
        file_path = f'simplified_results/figures/{target_var}_model_phase{phase}.png'
        plt.savefig(file_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"模型结构图已保存至 {file_path}")
        return True
    except Exception as e:
        print(f"绘制模型结构图时出错: {str(e)}")
        # 尝试创建一个非常简单的结构图
        try:
            plt.figure(figsize=(8, 5))
            plt.text(0.5, 0.5, f"KAN模型 - {target_var}\n阶段 {phase}", 
                    ha='center', va='center', fontsize=16)
            plt.axis('off')
            
            file_path = f'simplified_results/figures/{target_var}_model_phase{phase}_simple.png'
            plt.savefig(file_path, dpi=150)
            plt.close()
            print(f"创建了简化的模型图: {file_path}")
        except:
            print("创建简化图也失败了")
        return False

def plot_custom_kan_structure(model):
    """自定义绘制KAN模型结构，包括节点、连接和激活函数曲线"""
    # 获取模型的宽度配置
    widths = model.width
    n_layers = len(widths)
    
    # 创建画布和子图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制节点
    nodes = {}
    max_width = max(widths)
    
    # 计算每层的垂直偏移，使节点居中
    offsets = [max_width - w for w in widths]
    
    # 绘制网络结构
    for layer in range(n_layers):
        layer_width = widths[layer]
        for i in range(layer_width):
            # 计算节点位置
            x = layer
            y = i + offsets[layer] / 2
            
            # 存储节点位置
            nodes[(layer, i)] = (x, y)
            
            # 绘制节点
            if layer == 0:  # 输入层
                circle = plt.Circle((x, y), 0.2, color='lightblue', ec='blue')
            elif layer == n_layers - 1:  # 输出层
                circle = plt.Circle((x, y), 0.2, color='lightgreen', ec='green')
            else:  # 隐藏层
                circle = plt.Circle((x, y), 0.2, color='lightcoral', ec='red')
            
            ax.add_artist(circle)
            
            # 添加节点标签
            if layer == 0:
                plt.text(x-0.3, y, f'x{i}', ha='center', va='center')
            elif layer == n_layers - 1:
                plt.text(x+0.3, y, f'y{i}', ha='center', va='center')
            else:
                plt.text(x, y, f'{i}', ha='center', va='center', color='white')
    
    # 绘制连接和权重曲线
    if hasattr(model, 'edges'):
        for edge in model.edges:
            try:
                # 获取边的起点和终点
                from_node = edge[0]
                to_node = edge[1]
                
                if from_node in nodes and to_node in nodes:
                    # 获取节点位置
                    x1, y1 = nodes[from_node]
                    x2, y2 = nodes[to_node]
                    
                    # 绘制直线连接
                    plt.plot([x1, x2], [y1, y2], 'gray', alpha=0.5)
                    
                    # 尝试绘制激活函数曲线
                    # 注意：这需要访问模型中的激活函数信息，可能在不同版本的KAN库中结构不同
                    if hasattr(model, 'activations') and hasattr(model, 'grid_points'):
                        try:
                            # 在两点之间绘制一个小的激活函数曲线示意图
                            mid_x = (x1 + x2) / 2
                            mid_y = (y1 + y2) / 2
                            
                            # 获取激活函数
                            layer_idx = to_node[0] - 1  # 目标节点的层索引
                            neuron_idx = to_node[1]     # 目标节点的神经元索引
                            
                            # 只在有足够空间的情况下绘制曲线
                            if x2 - x1 > 0.8:
                                # 绘制一个小的激活函数示意图
                                curve_x = np.linspace(-0.2, 0.2, 20)
                                curve_y = 0.1 * np.sin(curve_x * 10) # 简化的激活函数示例
                                plt.plot(mid_x + curve_x, mid_y + curve_y, 'r-', linewidth=1)
                        except:
                            pass  # 如果无法绘制激活函数曲线，则跳过
            except:
                continue  # 如果处理边时出错，则跳过
        
    # 设置图像属性
    ax.set_aspect('equal')
    plt.axis('off')
    plt.xlim(-0.5, n_layers - 0.5)
    plt.ylim(-0.5, max_width + 0.5)
    
    return True

def calculate_feature_importance(model, dataset, feature_cols, target_var):
    """计算特征重要性"""
    print("\n计算特征重要性...")
    
    try:
        # 使用KAN的内置方法计算特征重要性
        model.attribute()
        if hasattr(model, 'feature_score') and model.feature_score is not None:
            feature_importance = model.feature_score.detach().cpu().numpy()
            
            # 检查并替换NaN值
            if np.isnan(feature_importance).any():
                print("警告：特征重要性中存在NaN值，将被替换为0")
                feature_importance = np.nan_to_num(feature_importance, 0.0)
            
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': feature_importance
            })
            
            # 检查特征重要性是否全为0
            if np.all(importance_df['importance'] == 0):
                print("所有特征重要性都为0，将使用均匀分布")
                importance_df['importance'] = 1.0 / len(feature_cols)
            
            # 按重要性排序
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # 打印前10个最重要的特征
            print("\n前10个最重要的特征:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                print(f"{i+1}. {row['feature']}: {row['importance']:.4f}")
            
            # 将特征重要性可视化
            plt.figure(figsize=(10, 6))
            top_n = min(15, len(importance_df))
            top_features = importance_df.head(top_n)
            plt.barh(top_features['feature'][::-1], top_features['importance'][::-1])
            plt.xlabel('重要性')
            plt.title('特征重要性排序（前15个）')
            plt.tight_layout()
            plt.savefig(f'simplified_results/figures/feature_importance_{target_var}.png', dpi=200)
            plt.close()
            
            # 保存特征重要性数据
            importance_df.to_csv(f'simplified_results/feature_importance_{target_var}.csv', index=False)
            
            return importance_df
            
        else:
            print("无法获取特征重要性分数，使用均匀分布")
            # 创建一个均匀分布的特征重要性
            uniform_importance = np.ones(len(feature_cols)) / len(feature_cols)
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': uniform_importance
            })
            return importance_df
            
    except Exception as e:
        print(f"计算特征重要性时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # 创建均匀分布的特征重要性作为后备
        uniform_importance = np.ones(len(feature_cols)) / len(feature_cols)
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': uniform_importance
        })
        print("使用均匀特征重要性作为后备")
        return importance_df

def select_important_features(importance_df, threshold_percentile=60):
    """选择特征重要性高于阈值的特征"""
    try:
        # 计算特征重要性的百分位数作为阈值
        threshold = np.percentile(importance_df['importance'], threshold_percentile)
        print(f"\n使用{threshold_percentile}百分位作为阈值: {threshold:.4f}")
        
        # 筛选特征
        important_features = importance_df[importance_df['importance'] > threshold]
        selected_features = important_features['feature'].tolist()
        
        # 确保至少选择5个特征
        if len(selected_features) < 5:
            print("警告：选择的重要特征少于5个，将选择前5个最重要的特征")
            selected_features = importance_df.head(5)['feature'].tolist()
        
        # 确保不超过原始特征数量的70%
        max_features = int(len(importance_df) * 0.7)  # 增加最大特征数量
        if len(selected_features) > max_features:
            print(f"警告：选择的重要特征超过最大限制({max_features}个)，将只保留前{max_features}个最重要的特征")
            selected_features = importance_df.head(max_features)['feature'].tolist()
        
        print(f"\n筛选出 {len(selected_features)} 个重要特征:")
        for i, feature in enumerate(selected_features, 1):
            importance = importance_df[importance_df['feature'] == feature]['importance'].values[0]
            print(f"{i}. {feature}: {importance:.4f}")
        
        # 保存特征选择结果
        selection_info = pd.DataFrame({
            'feature': selected_features,
            'importance': [importance_df[importance_df['feature'] == f]['importance'].values[0] for f in selected_features],
            'kan_importance': [importance_df[importance_df['feature'] == f]['importance'].values[0] for f in selected_features]
        })
        
        # 保存到CSV文件
        selection_info.to_csv('simplified_results/selected_features.csv', index=False)
        
        return selected_features, threshold
        
    except Exception as e:
        print(f"选择重要特征时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # 如果出错，返回前10个最重要的特征
        print("使用前10个最重要的特征作为后备")
        selected_features = importance_df.head(10)['feature'].tolist()
        return selected_features, 0.0

def generate_simplified_equation(model, dataset, feature_cols, target_var):
    """生成基于重要特征的简化回归方程"""
    try:
        # 评估模型性能
        print("计算模型评估指标...")
        with torch.no_grad():
            y_pred = model(dataset['test_input']).cpu().numpy()
            y_true = dataset['test_label'].cpu().numpy()
            
        # 处理预测中的NaN值
        if np.isnan(y_pred).any():
            print("警告：预测结果中存在NaN值，将被替换为均值")
            y_mean = np.nanmean(y_true)
            y_pred = np.nan_to_num(y_pred, y_mean)
        
        # 计算评估指标
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        print("\n模型评估指标:")
        print(f"R² 决定系数: {r2:.4f}")
        print(f"RMSE 均方根误差: {rmse:.4f}")
        print(f"MAE 平均绝对误差: {mae:.4f}")
        
        # 使用KAN的符号化特征方程提取功能
        symbolic_eq = "无法生成符号化方程"
        try:
            # 定义符号库
            print("\n使用KAN的符号化特征方程提取功能...")
            lib = ['x', 'x^2', 'x^3', 'exp', 'log', 'sqrt', 'sin', 'cos', 'abs']
            
            # 应用自动符号化提取
            model.auto_symbolic(lib=lib)
            
            # 获取符号化公式
            print("提取符号化方程...")
            symbolic_result = model.symbolic_formula()
            
            # 处理提取的公式
            if isinstance(symbolic_result, tuple) and len(symbolic_result) > 0:
                formulas = symbolic_result[0]
                if isinstance(formulas, (list, tuple)) and len(formulas) > 0:
                    formula = formulas[0]  # 获取第一个公式
                    
                    # 函数用于四舍五入公式中的系数
                    def round_formula(f, digits=4):
                        if isinstance(f, (int, float)):
                            return round(f, digits)
                        elif isinstance(f, list):
                            return [round_formula(x, digits) for x in f]
                        else:
                            return f
                    
                    # 四舍五入处理公式系数
                    rounded_formula = round_formula(formula, 4)
                    
                    # 创建符号化方程字符串
                    symbolic_eq = f"{target_var} = {rounded_formula}"
                    print(f"提取的符号化方程: {symbolic_eq}")
                    
                    # 保存符号化方程到文件
                    with open(f'simplified_results/symbolic_equation_{target_var}.txt', 'w', encoding='utf-8') as f:
                        f.write(f"KAN符号化方程 ({target_var}):\n")
                        f.write(f"R² 决定系数: {r2:.4f}\n")
                        f.write(f"RMSE 均方根误差: {rmse:.4f}\n")
                        f.write(f"MAE 平均绝对误差: {mae:.4f}\n\n")
                        f.write(f"符号化方程:\n{symbolic_eq}")
                else:
                    print(f"未能从symbolic_result[0]提取公式")
            else:
                print(f"symbolic_result不是有效元组或为空")
                
        except Exception as e:
            print(f"提取符号化方程时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
        
        # 使用线性回归提取特征系数 (作为备用方案)
        print("\n生成基于线性回归的简化方程...")
        from sklearn.linear_model import LinearRegression
        X = dataset['train_input'].cpu().numpy()
        y = dataset['train_label'].cpu().numpy()
        
        # 拟合线性模型
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        
        # 提取系数
        coefficients = linear_model.coef_[0]
        intercept = linear_model.intercept_[0]
        
        # 构建回归方程
        equation = f"{target_var} = {intercept:.4f}"
        for i, (feat, coef) in enumerate(zip(feature_cols, coefficients)):
            if coef >= 0:
                equation += f" + {coef:.4f} * {feat}"
            else:
                equation += f" - {abs(coef):.4f} * {feat}"
        
        print("基于线性回归的简化方程:")
        print(equation)
        
        # 保存回归方程到文件
        with open(f'simplified_results/linear_equation_{target_var}.txt', 'w', encoding='utf-8') as f:
            f.write(f"基于线性回归的简化方程 ({target_var}):\n")
            f.write(f"R² 决定系数: {r2:.4f}\n")
            f.write(f"RMSE 均方根误差: {rmse:.4f}\n")
            f.write(f"MAE 平均绝对误差: {mae:.4f}\n\n")
            f.write(f"线性回归方程:\n{equation}")
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'equation': symbolic_eq if symbolic_eq != "无法生成符号化方程" else equation
        }
    except Exception as e:
        print(f"生成简化回归方程时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            'r2': 0,
            'rmse': 1,
            'mae': 1,
            'equation': "生成方程出错"
        }

def visualize_activation_functions(model, target_var, phase):
    """可视化KAN模型中的激活函数曲线"""
    try:
        print(f"\n尝试可视化 {target_var} (阶段{phase}) 模型的激活函数曲线...")
        
        # 创建保存激活函数曲线的目录
        os.makedirs('simplified_results/figures/activations', exist_ok=True)
        
        # 获取模型的宽度配置
        widths = model.width
        n_layers = len(widths)
        
        # 检查是否能访问激活函数相关属性
        if hasattr(model, 'activations'):
            # 尝试获取并可视化激活函数
            for layer_idx in range(n_layers - 1):  # 遍历除了输出层之外的所有层
                for neuron_idx in range(widths[layer_idx + 1]):  # 遍历下一层的所有神经元
                    try:
                        plt.figure(figsize=(8, 4))
                        # 生成x轴数据
                        x = np.linspace(-1, 1, 100)
                        
                        # 尝试不同方式获取激活函数
                        activation_found = False
                        
                        # 方法1：尝试直接访问激活函数属性
                        try:
                            if hasattr(model, 'get_activation'):
                                # 某些版本可能有get_activation方法
                                activation = model.get_activation(layer_idx, neuron_idx)
                                if activation is not None:
                                    y = [activation(xi) for xi in x]
                                    plt.plot(x, y, 'b-')
                                    activation_found = True
                        except Exception as e1:
                            print(f"方法1获取激活函数失败: {str(e1)}")
                        
                        # 方法2：尝试使用model.activations属性
                        if not activation_found and hasattr(model, 'activations'):
                            try:
                                activations = model.activations
                                if isinstance(activations, list) and layer_idx < len(activations):
                                    layer_activations = activations[layer_idx]
                                    if isinstance(layer_activations, list) and neuron_idx < len(layer_activations):
                                        activation_fn = layer_activations[neuron_idx]
                                        if callable(activation_fn):
                                            y = [activation_fn(xi) for xi in x]
                                            plt.plot(x, y, 'r-')
                                            activation_found = True
                            except Exception as e2:
                                print(f"方法2获取激活函数失败: {str(e2)}")
                        
                        # 方法3：如果前两种方法都失败，则绘制一个示例激活函数
                        if not activation_found:
                            # 绘制一个示例激活函数(样条函数)
                            def sample_activation(x):
                                # 一个简单的立方样条示例
                                return 0.5 * x**3 + 0.3 * x 
                            
                            y = [sample_activation(xi) for xi in x]
                            plt.plot(x, y, 'g-', alpha=0.5)
                            plt.text(0, 0, "示例激活函数\n(非实际函数)", 
                                     ha='center', va='center', fontsize=10,
                                     bbox=dict(facecolor='white', alpha=0.7))
                        
                        # 设置图形属性
                        plt.title(f"层{layer_idx}→层{layer_idx+1}，神经元{neuron_idx}的激活函数")
                        plt.xlabel('输入')
                        plt.ylabel('输出')
                        plt.grid(True, alpha=0.3)
                        plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
                        plt.axvline(x=0, color='k', linestyle='-', alpha=0.2)
                        
                        # 保存图像
                        file_path = f'simplified_results/figures/activations/{target_var}_phase{phase}_layer{layer_idx}_neuron{neuron_idx}.png'
                        plt.savefig(file_path, dpi=150)
                        plt.close()
                    except Exception as e:
                        print(f"可视化层{layer_idx}神经元{neuron_idx}的激活函数失败: {str(e)}")
            
            print(f"激活函数曲线已保存至 simplified_results/figures/activations 目录")
            return True
        else:
            print("无法访问模型的激活函数属性")
            return False
    except Exception as e:
        print(f"可视化激活函数时出错: {str(e)}")
        return False

def predict_with_kan(file_path, target_var, test_data=None):
    """
    使用KAN模型进行预测
    
    参数:
    file_path: str, 训练数据Excel文件路径
    target_var: str, 目标变量名称 ('lst_day_c', 'lst_night_c', 或 'nighttime_')
    test_data: pd.DataFrame, 可选，用于预测的新数据。如果为None，则使用训练数据进行预测
    
    返回:
    dict: 包含以下键值对：
        - 'model': 训练好的KAN模型
        - 'predictions': 预测结果
        - 'r2': R²决定系数
        - 'rmse': 均方根误差
        - 'mae': 平均绝对误差
        - 'feature_importance': 特征重要性DataFrame
        - 'selected_features': 选中的重要特征列表
    """
    print(f"\n开始训练和预测 {target_var}...")
    
    # 加载数据
    df = load_data(file_path)
    if df is None or len(df) == 0:
        raise ValueError("加载的数据为空")
    
    # 准备训练数据
    dataset, feature_cols = prepare_data(df, target_var)
    
    # 训练初步模型
    model_phase1 = train_model(dataset, len(feature_cols), phase=1, target_var=target_var)

    # 第一阶段模型预测对比
    model_phase1.eval()
    with torch.no_grad():
        y_pred1 = model_phase1(dataset['test_input']).cpu().numpy().ravel()
        y_true1 = dataset['test_label'].cpu().numpy().ravel()
    print("\n[阶段1] 前10个样本的真实值和预测值对比：")
    for i in range(min(10, len(y_true1))):
        print(f"样本 {i+1}: 真实值 = {y_true1[i]:.4f}, 预测值 = {y_pred1[i]:.4f}")

    # 绘制模型结构图
    plot_model_structure(model_phase1, target_var, phase=1)

    # 尝试可视化激活函数曲线
    success = visualize_activation_functions(model_phase1, target_var, phase=1)
    if not success:
        print("[说明] 阶段1模型plot没有激活函数曲线，原因可能有：\n1. KAN模型实现未保存activations属性；\n2. 训练步数较少或网络较浅，未触发激活函数的存储；\n3. 当前KAN库版本不支持激活函数可视化。\n这不会影响模型训练和预测，只是无法画出激活函数曲线。")

    # 计算特征重要性
    importance_df = calculate_feature_importance(model_phase1, dataset, feature_cols, target_var)
    
    # 选择重要特征
    selected_features, threshold = select_important_features(importance_df, threshold_percentile=60)
    
    # 阶段2：只使用重要特征重新训练模型
    print("\n## 阶段2：使用重要特征重新训练模型 ##")
    
    # 准备数据(只使用重要特征)
    dataset_selected, feature_cols_selected = prepare_data(df, target_var, selected_features)
    
    # 训练精细模型
    model_phase2 = train_model(dataset_selected, len(feature_cols_selected), phase=2, target_var=target_var)
    
    # 在测试集上进行预测并输出前10个样本的对比
    model_phase2.eval()
    with torch.no_grad():
        y_pred = model_phase2(dataset_selected['test_input']).cpu().numpy().ravel()
        y_true = dataset_selected['test_label'].cpu().numpy().ravel()
    print("\n前10个样本的真实值和预测值对比：")
    for i in range(min(10, len(y_true))):
        print(f"样本 {i+1}: 真实值 = {y_true[i]:.4f}, 预测值 = {y_pred[i]:.4f}")
    
    # 保存模型和特征信息
    model_path = f'simplified_results/models/{target_var}_model.pth'
    torch.save({
        'model_state_dict': model_phase2.state_dict(),
        'feature_cols': feature_cols_selected,
        'model_params': {
            'width': model_phase2.width,
            'grid': model_phase2.grid,
            'k': model_phase2.k
        }
    }, model_path)
    print(f"\n模型已保存至: {model_path}")
    
    # 保存特征列表
    with open(f'simplified_results/models/{target_var}_features.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(feature_cols_selected))
    print(f"特征列表已保存至: simplified_results/models/{target_var}_features.txt")
    
    # 绘制模型结构图
    plot_model_structure(model_phase2, target_var, phase=2)
    
    # 尝试可视化激活函数曲线
    visualize_activation_functions(model_phase2, target_var, phase=2)
    
    # 生成简化回归方程
    equation_result = generate_simplified_equation(
        model_phase2, dataset_selected, feature_cols_selected, target_var
    )
    
    # 保存结果
    results_df = pd.DataFrame({
        'true_value': y_true,
        'predicted_value': y_pred
    })
    
    # 如果有测试数据，添加原始特征值
    if test_data is not None:
        for feature in selected_features:
            results_df[feature] = test_data[feature].values
    
    # 保存预测结果到Excel
    output_file = f'simplified_results/predictions_{target_var}.xlsx'
    results_df.to_excel(output_file, index=False)
    print(f"\n预测结果已保存至: {output_file}")
    
    return {
        'model': model_phase2,
        'predictions': y_pred,
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'feature_importance': importance_df,
        'selected_features': selected_features
    }

def main():
    print("开始两阶段KAN模型分析程序...")
    
    try:
        # 创建结果目录
        os.makedirs('simplified_results', exist_ok=True)
        os.makedirs('simplified_results/figures', exist_ok=True)
        os.makedirs('simplified_results/models', exist_ok=True)  # 创建模型保存目录
        
        # 加载数据
        file_path = 'data_old.xlsx'  # 使用当前目录中的data.xlsx文件
        print(f"\n尝试加载数据文件: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件 {file_path} 不存在")
            
        df = load_data(file_path)
        if df is None or len(df) == 0:
            raise ValueError("加载的数据为空")
            
        print(f"\n成功加载数据，形状: {df.shape}")
        print("数据列名:", df.columns.tolist())
        
        # 分析目标变量
        target_vars = ['lst_day_c', 'lst_night_c', 'nighttime_']  # 分析城市热岛效应变量
        results = {}
        
        for target_var in target_vars:
            print(f"\n{'='*50}")
            print(f"分析目标变量: {target_var}")
            print(f"{'='*50}")
            
            try:
                # 检查目标变量是否存在
                if target_var not in df.columns:
                    print(f"警告：目标变量 {target_var} 不在数据集中，跳过")
                    continue
                    
                # 阶段1：使用所有特征进行初步分析
                print("\n## 阶段1：特征重要性分析 ##")
                
                # 准备数据(全部特征)
                dataset, feature_cols = prepare_data(df, target_var)
                if len(feature_cols) == 0:
                    print(f"警告：没有可用的特征用于分析 {target_var}，跳过")
                    continue
                
                # 训练初步模型
                model_phase1 = train_model(dataset, len(feature_cols), phase=1, target_var=target_var)

                # 第一阶段模型预测对比
                model_phase1.eval()
                with torch.no_grad():
                    y_pred1 = model_phase1(dataset['test_input']).cpu().numpy().ravel()
                    y_true1 = dataset['test_label'].cpu().numpy().ravel()
                print("\n[阶段1] 前10个样本的真实值和预测值对比：")
                for i in range(min(10, len(y_true1))):
                    print(f"样本 {i+1}: 真实值 = {y_true1[i]:.4f}, 预测值 = {y_pred1[i]:.4f}")

                # 绘制模型结构图
                plot_model_structure(model_phase1, target_var, phase=1)

                # 尝试可视化激活函数曲线
                success = visualize_activation_functions(model_phase1, target_var, phase=1)
                if not success:
                    print("[说明] 阶段1模型plot没有激活函数曲线，原因可能有：\n1. KAN模型实现未保存activations属性；\n2. 训练步数较少或网络较浅，未触发激活函数的存储；\n3. 当前KAN库版本不支持激活函数可视化。\n这不会影响模型训练和预测，只是无法画出激活函数曲线。")

                # 计算特征重要性
                importance_df = calculate_feature_importance(model_phase1, dataset, feature_cols, target_var)
                
                # 保存特征重要性到文件
                importance_df.to_csv(f'simplified_results/feature_importance_{target_var}.csv', index=False)
                
                # 选择重要特征
                selected_features, threshold = select_important_features(importance_df, threshold_percentile=60)
                
                # 阶段2：只使用重要特征重新训练模型
                print("\n## 阶段2：使用重要特征重新训练模型 ##")
                
                # 准备数据(只使用重要特征)
                dataset_selected, feature_cols_selected = prepare_data(df, target_var, selected_features)
                
                # 训练精细模型
                model_phase2 = train_model(dataset_selected, len(feature_cols_selected), phase=2, target_var=target_var)
                
                # 在测试集上进行预测并输出前10个样本的对比
                model_phase2.eval()
                with torch.no_grad():
                    y_pred = model_phase2(dataset_selected['test_input']).cpu().numpy().ravel()
                    y_true = dataset_selected['test_label'].cpu().numpy().ravel()
                print("\n前10个样本的真实值和预测值对比：")
                for i in range(min(10, len(y_true))):
                    print(f"样本 {i+1}: 真实值 = {y_true[i]:.4f}, 预测值 = {y_pred[i]:.4f}")
                
                # 保存模型和特征信息
                model_path = f'simplified_results/models/{target_var}_model.pth'
                torch.save({
                    'model_state_dict': model_phase2.state_dict(),
                    'feature_cols': feature_cols_selected,
                    'model_params': {
                        'width': model_phase2.width,
                        'grid': model_phase2.grid,
                        'k': model_phase2.k
                    }
                }, model_path)
                print(f"\n模型已保存至: {model_path}")
                
                # 保存特征列表
                with open(f'simplified_results/models/{target_var}_features.txt', 'w', encoding='utf-8') as f:
                    f.write('\n'.join(feature_cols_selected))
                print(f"特征列表已保存至: simplified_results/models/{target_var}_features.txt")
                
                # 绘制模型结构图
                plot_model_structure(model_phase2, target_var, phase=2)
                
                # 尝试可视化激活函数曲线
                visualize_activation_functions(model_phase2, target_var, phase=2)
                
                # 生成简化回归方程
                equation_result = generate_simplified_equation(
                    model_phase2, dataset_selected, feature_cols_selected, target_var
                )
                
                # 保存结果
                results[target_var] = {
                    'important_features': selected_features,
                    'threshold': threshold,
                    'equation_result': equation_result
                }
                
                print(f"\n{target_var} 分析完成！")
                
            except Exception as var_error:
                print(f"处理变量 {target_var} 时出错: {str(var_error)}")
                import traceback
                print(traceback.format_exc())
                continue
        
        # 保存分析结果摘要
        if results:
            with open('simplified_results/analysis_summary.txt', 'w', encoding='utf-8') as f:
                f.write("城市热岛效应两阶段分析摘要\n")
                f.write("="*50 + "\n\n")
                
                for target_var, result in results.items():
                    f.write(f"{target_var} 分析结果:\n")
                    f.write("-"*40 + "\n\n")
                    
                    eq_result = result['equation_result']
                    f.write(f"模型性能指标:\n")
                    f.write(f"R² 决定系数: {eq_result['r2']:.4f}\n")
                    f.write(f"RMSE 均方根误差: {eq_result['rmse']:.4f}\n")
                    f.write(f"MAE 平均绝对误差: {eq_result['mae']:.4f}\n\n")
                    
                    f.write(f"重要特征 ({len(result['important_features'])}个):\n")
                    f.write(', '.join(result['important_features']))
                    f.write(f"\n\n重要性阈值: > {result['threshold']:.4f}\n\n")
                    
                    f.write("方程:\n")
                    f.write(eq_result['equation'])
                    
                    # 判断是否为符号化方程
                    is_symbolic = eq_result['equation'].startswith(target_var) and not eq_result['equation'].startswith(target_var + " = " + target_var)
                    if is_symbolic:
                        f.write("\n\n(上述方程为KAN自动提取的符号化方程)")
                    else:
                        f.write("\n\n(上述方程为基于线性回归的简化方程)")
                    
                    f.write("\n\n" + "="*50 + "\n\n")
            
            print("\n分析完成！结果已保存至simplified_results目录。")
            print("- 特征重要性图表保存在 simplified_results/figures 目录")
            print("- 模型结构图保存在 simplified_results/figures 目录")
            print("- 符号化方程和分析摘要保存在 simplified_results 目录")
            print("- 训练好的模型保存在 simplified_results/models 目录")
        else:
            print("\n警告：没有成功完成任何目标变量的分析")
    
    except Exception as main_error:
        print(f"主程序执行出错: {str(main_error)}")
        import traceback
        print(traceback.format_exc())
        print("程序尝试处理部分结果...")
        
        # 尝试保存已有的结果
        if results:
            with open('simplified_results/partial_results.txt', 'w', encoding='utf-8') as f:
                f.write("部分分析结果:\n")
                for var, res in results.items():
                    f.write(f"{var}: {len(res.get('important_features', []))} 个重要特征\n")
        else:
            print("没有可保存的分析结果")

if __name__ == "__main__":
    main() 