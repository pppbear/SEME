import numpy as np
import matplotlib.pyplot as plt
import torch
from kan.MultKAN import KAN
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 检查CUDA是否可用并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建必要的目录
os.makedirs('simplified_results', exist_ok=True)
os.makedirs('simplified_results/figures', exist_ok=True)

def load_data(file_path):
    """加载Excel数据"""
    df = pd.read_excel(file_path)
    print(f"成功读取数据，共 {len(df)} 行")
    
    # 重命名列
    if 'lst_night_' in df.columns:
        df = df.rename(columns={'lst_night_': 'lst_night_c'})
    if 'uhi_night_' in df.columns:
        df = df.rename(columns={'uhi_night_': 'uhi_night_c'})
    
    # 只保留有用的列
    exclude_cols = ['FID', 'Shape *', 'FID_', 'Id', 'Shape_Leng', 'Shape_Le_1', 'Shape_Area']
    df = df.drop(columns=exclude_cols, errors='ignore')
    
    # 处理数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df

def prepare_data(df, target_var, selected_features=None):
    """准备训练数据，可选择只使用选定的特征"""
    # 选择特征列（排除目标变量）
    exclude_cols = ['lst_day_c', 'lst_night_c', 'uhi_day_c', 'uhi_night_c']
    if selected_features is None:
        # 使用所有特征
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        # 只保留数值类型的特征
        feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
        print(f"\n使用全部 {len(feature_cols)} 个数值特征进行分析")
    else:
        # 使用选定的特征
        feature_cols = selected_features
        print(f"\n使用筛选后的 {len(feature_cols)} 个重要特征进行分析:")
        print(", ".join(feature_cols))
    
    # 准备特征矩阵
    X = df[feature_cols].values
    y = df[target_var].values
    
    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 处理无效值
    X = np.nan_to_num(X, 0)
    y = np.nan_to_num(y, np.nanmedian(y))
    
    # 转换为PyTorch张量并移至设备
    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).reshape(-1, 1).to(device)
    
    # 创建数据集
    dataset = {
        'train_input': X,
        'train_label': y,
        'test_input': X,
        'test_label': y
    }
    
    return dataset, feature_cols

def train_model(dataset, input_dim, phase=1):
    """训练KAN模型，phase=1为特征选择阶段，phase=2为最终模型训练阶段"""
    # 根据训练阶段选择不同的模型参数
    if phase == 1:
        # 第一阶段：用于特征选择的初步模型
        model = KAN(
            width=[input_dim, 8, 1],     # 减少隐藏层节点数以避免过拟合
            grid=4,                      # 减少网格点数以提高稳定性
            k=2,                         # 减少每个输入连接的节点数
            symbolic_enabled=True        # 启用符号计算
        ).to(device)
        
        print(f"阶段1 - 初步模型已配置并移至设备 {device}")
        
        # 使用保守的训练设置
        history = model.fit(
            dataset=dataset,
            opt="Adam",                  # 使用Adam优化器
            steps=200,                    # 减少训练步数
            lamb_l1=0.5,                 # 适度的L1正则化
            lamb_entropy=0.5,            # 适度的熵正则化
            lr=0.0005                    # 使用非常小的学习率以确保稳定性
        )
    else:
        # 第二阶段：使用重要特征训练更精细的模型
        model = KAN(
            width=[input_dim, 10, 1],    # 增加隐藏层节点数以提高表达能力
            grid=5,                      # 适当增加网格点数
            k=3,                         # 增加每个输入连接的节点数
            symbolic_enabled=True        # 启用符号计算
        ).to(device)
        
        print(f"阶段2 - 精细模型已配置并移至设备 {device}")
        
        # 使用更精细的训练设置
        history = model.fit(
            dataset=dataset,
            opt="Adam",                  # 使用Adam优化器
            steps=300,                    # 增加训练步数以更好地学习
            lamb_l1=0.4,                 # 减小L1正则化以允许更多特征参与
            lamb_entropy=0.4,            # 减小熵正则化
            lr=0.001                     # 略微增加学习率以加快学习
        )
    
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

def calculate_feature_importance(model, dataset, feature_cols):
    """计算特征重要性"""
    print("\n计算特征重要性...")
    model.attribute()  # 使用KAN的内置方法计算特征重要性
    
    # 获取特征重要性分数
    if hasattr(model, 'feature_score') and model.feature_score is not None:
        try:
            # 获取特征重要性并处理可能的NaN值
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
            plt.savefig(f'simplified_results/figures/feature_importance.png', dpi=200)
            plt.close()
            
            return importance_df
        except Exception as e:
            print(f"处理特征重要性时出错: {str(e)}")
            # 创建一个均匀分布的特征重要性
            uniform_importance = np.ones(len(feature_cols)) / len(feature_cols)
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': uniform_importance
            })
            print("使用均匀特征重要性作为后备")
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

def select_important_features(importance_df, threshold_percentile=75):
    """选择特征重要性高于阈值的特征"""
    # 计算特征重要性的百分位数作为阈值
    threshold = np.percentile(importance_df['importance'], threshold_percentile)
    print(f"\n使用{threshold_percentile}百分位作为阈值: {threshold:.4f}")
    
    # 筛选特征
    important_features = importance_df[importance_df['importance'] > threshold]
    selected_features = important_features['feature'].tolist()
    
    print(f"筛选出 {len(selected_features)} 个重要特征（重要性 > {threshold:.4f}）")
    return selected_features, threshold

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

def main():
    print("开始两阶段KAN模型分析程序...")
    
    # 创建结果目录
    os.makedirs('simplified_results', exist_ok=True)
    os.makedirs('simplified_results/figures', exist_ok=True)
    
    # 加载数据
    file_path = 'data.xlsx'  # 使用当前目录中的data.xlsx文件
    df = load_data(file_path)
    
    # 分析目标变量
    target_vars = ['lst_day_c', 'lst_night_c', 'uhi_day_c', 'uhi_night_c']  # 分析城市热岛效应变量
    results = {}
    
    try:
        for target_var in target_vars:
            print(f"\n{'='*50}")
            print(f"分析目标变量: {target_var}")
            print(f"{'='*50}")
            
            try:
                # 阶段1：使用所有特征进行初步分析
                print("\n## 阶段1：特征重要性分析 ##")
                
                # 准备数据(全部特征)
                dataset, feature_cols = prepare_data(df, target_var)
                
                # 训练初步模型
                model_phase1 = train_model(dataset, len(feature_cols), phase=1)
                
                # 绘制模型结构图
                plot_model_structure(model_phase1, target_var, phase=1)
                
                # 尝试可视化激活函数曲线
                visualize_activation_functions(model_phase1, target_var, phase=1)
                
                # 计算特征重要性
                importance_df = calculate_feature_importance(model_phase1, dataset, feature_cols)
                
                # 保存特征重要性到文件
                importance_df.to_csv(f'simplified_results/feature_importance_{target_var}.csv', index=False)
                
                # 选择重要特征
                selected_features, threshold = select_important_features(importance_df, threshold_percentile=75)
                
                # 阶段2：只使用重要特征重新训练模型
                print("\n## 阶段2：使用重要特征重新训练模型 ##")
                
                # 准备数据(只使用重要特征)
                dataset_selected, feature_cols_selected = prepare_data(df, target_var, selected_features)
                
                # 训练精细模型
                model_phase2 = train_model(dataset_selected, len(feature_cols_selected), phase=2)
                
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
        
        # 保存分析结果摘要
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