import numpy as np
import matplotlib.pyplot as plt
import torch
from kan.MultKAN import KAN
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import sys
import warnings
import networkx as nx
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 检查CUDA是否可用并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建必要的目录
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

def load_data(file_path):
    """加载Excel数据"""
    try:
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
    except Exception as e:
        print(f"读取数据时出错: {str(e)}")
        sys.exit(1)

def prepare_data(df, target_var, feature_threshold=0.0):
    """准备训练数据，可选择根据特征重要性阈值筛选特征"""
    try:
        # 选择特征列（排除目标变量）
        exclude_cols = ['lst_day_c', 'lst_night_c', 'uhi_day_c', 'uhi_night_c']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 只保留数值类型的特征
        feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
        
        # 如果提供了特征重要性，筛选特征
        if isinstance(feature_threshold, dict) and target_var in feature_threshold:
            important_features = feature_threshold[target_var]
            feature_cols = [col for col in feature_cols if col in important_features]
            print(f"\n使用 {len(feature_cols)} 个重要特征进行分析")
            print(f"筛选后的特征: {', '.join(feature_cols)}")
        else:
            print(f"\n使用 {len(feature_cols)} 个数值特征进行分析")
        
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
    except Exception as e:
        print(f"准备数据时出错: {str(e)}")
        sys.exit(1)

def train_kan_model(dataset, input_dim):
    """训练KAN模型"""
    try:
        # 创建模型并移至设备
        model = KAN(
            width=[input_dim, 10, 1],    # 适当增加隐藏层节点数
            grid=5,                      # 适当设置网格点数
            k=3,                         # 设置每个输入连接的节点数
            symbolic_enabled=True        # 启用符号计算以提高可解释性
        ).to(device)
        
        print(f"模型已配置并移至设备 {device}，输入维度: {input_dim}, 隐藏层节点: 10, 网格点: 5")
        
        # 使用更保守的训练设置
        history = model.fit(
            dataset=dataset,
            opt="Adam",                  # 使用Adam优化器
            steps=50,                    # 增加训练步数以更好地收敛
            lamb_l1=0.5,                 # 设置适当的L1正则化强度
            lamb_entropy=0.5,            # 设置适当的熵正则化强度
            lr=0.001                     # 使用较小的学习率以避免不稳定
        )
        
        # 创建新的图形并保存，避免直接使用model.plot()可能导致的问题
        plt.figure(figsize=(10, 6))
        
        # 尝试绘制模型结构，如果失败则创建简单的替代图
        try:
            model.plot()
            plt.title(f"KAN模型结构 - 隐藏层: 10, 网格点: 5")
        except Exception as plot_error:
            print(f"绘制模型结构时出错: {str(plot_error)}")
            # 创建一个简单的替代图
            plt.text(0.5, 0.5, f"KAN模型结构\n输入维度: {input_dim}\n隐藏层: 10\n网格点: 5", 
                    horizontalalignment='center', verticalalignment='center', fontsize=12)
            plt.axis('off')
        
        plt.tight_layout()
        
        return model, plt.gcf()  # 返回模型和当前图形对象
    except Exception as e:
        print(f"训练模型时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

def analyze_kan_model(model, dataset, feature_cols, target_var):
    """分析KAN模型并提取特征重要性"""
    try:
        # 计算特征重要性 - 保持在GPU上
        print("计算特征重要性...")
        model.attribute()  # 使用KAN的内置方法计算特征重要性
        
        # 获取特征重要性分数
        if hasattr(model, 'feature_score') and model.feature_score is not None:
            feature_importance = model.feature_score.detach().cpu().numpy()
            
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': feature_importance
            })
            
            # 按重要性排序
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # 保存完整的特征重要性到Excel
            importance_df.to_excel(f'results/feature_importance_{target_var}.xlsx', index=False)
            
            # 绘制特征重要性条形图 - 只显示前10个
            top_features = importance_df.head(10)
            plt.figure(figsize=(10, 6))
            plt.barh(top_features['feature'], top_features['importance'])
            plt.xlabel('特征重要性')
            plt.ylabel('特征')
            plt.title(f'{target_var} - 前10个最重要特征')
            plt.tight_layout()
            plt.savefig(f'results/feature_importance_{target_var}.png', dpi=200)
            plt.close()
            
            # 打印前5个最重要的特征
            print("\n前5个最重要的特征:")
            for _, row in importance_df.head(5).iterrows():
                print(f"{row['feature']}: {row['importance']:.4f}")
                
            return importance_df
        else:
            print("无法获取特征重要性分数")
            return None
    except Exception as e:
        print(f"分析模型时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def evaluate_model(model, dataset, feature_cols, target_var):
    """评估模型性能并输出回归方程"""
    try:
        # 预测值 - 在GPU上计算后移至CPU
        with torch.no_grad():
            y_pred = model(dataset['test_input']).cpu().numpy()
            y_true = dataset['test_label'].cpu().numpy()
        
        # 计算评估指标
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        print("\n模型评估指标:")
        print(f"R² 决定系数: {r2:.4f}")
        print(f"RMSE 均方根误差: {rmse:.4f}")
        print(f"MAE 平均绝对误差: {mae:.4f}")
        
        # 使用auto_symbolic函数生成符号化回归方程
        try:
            print("\n生成KAN符号化回归方程...")
            # 定义符号库，用于生成回归方程
            lib = ['x', 'x^2', 'x^3', 'exp', 'log', 'sqrt', 'sin', 'cos', 'abs']
            model.auto_symbolic(lib=lib)
            
            # 获取符号化公式
            print("获取model.symbolic_formula()的原始输出...")
            symbolic_result = model.symbolic_formula()
            print(f"symbolic_formula()返回值: {symbolic_result}")
            
            # 仿照示例中的方式提取公式
            print("尝试按照formula1, formula2 = model.symbolic_formula()[0]的方式提取...")
            
            try:
                if isinstance(symbolic_result, tuple) and len(symbolic_result) > 0:
                    formulas = symbolic_result[0]
                    print(f"symbolic_result[0]: {formulas}")
                    
                    if isinstance(formulas, (list, tuple)) and len(formulas) > 0:
                        formula1 = formulas[0]
                        print(f"提取的formula1: {formula1}")
                        
                        if len(formulas) > 1:
                            formula2 = formulas[1]
                            print(f"提取的formula2: {formula2}")
                        else:
                            formula2 = None
                            print("没有formula2")
                        
                        # 四舍五入系数
                        def round_formula(formula, digits=4):
                            if isinstance(formula, (int, float)):
                                return round(formula, digits)
                            elif isinstance(formula, list):
                                return [round_formula(x, digits) for x in formula]
                            else:
                                return formula
                        
                        # 处理并显示公式
                        rounded_formula1 = round_formula(formula1, 4)
                        print(f"处理后的formula1: {rounded_formula1}")
                        
                        if formula2 is not None:
                            rounded_formula2 = round_formula(formula2, 4)
                            print(f"处理后的formula2: {rounded_formula2}")
                        
                        # 构建回归方程字符串
                        symbolic_eq = f"{target_var} = {rounded_formula1}"
                        
                        # 保存符号化回归方程到文件
                        with open(f'results/kan_symbolic_equation_{target_var}.txt', 'w', encoding='utf-8') as f:
                            f.write(f"KAN符号化回归方程 ({target_var}):\n")
                            f.write(f"R² 决定系数: {r2:.4f}\n")
                            f.write(f"RMSE 均方根误差: {rmse:.4f}\n")
                            f.write(f"MAE 平均绝对误差: {mae:.4f}\n\n")
                            f.write(f"公式1: {rounded_formula1}\n")
                            if formula2 is not None:
                                f.write(f"公式2: {rounded_formula2}\n")
                    else:
                        print(f"symbolic_result[0]不是可迭代对象或为空: {formulas}")
                        symbolic_eq = "无法提取公式"
                else:
                    print(f"symbolic_result不是元组或为空: {symbolic_result}")
                    symbolic_eq = "无法提取公式"
            except Exception as e2:
                print(f"处理符号化公式时出错: {str(e2)}")
                import traceback
                print(traceback.format_exc())
                symbolic_eq = "处理公式出错"
                
        except Exception as e:
            print(f"生成符号化回归方程时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
            symbolic_eq = None
        
        # 计算特征重要性
        importance_df = analyze_kan_model(model, dataset, feature_cols, target_var)
        
        if importance_df is not None:
            # 获取前5个最重要的特征
            top_indices = importance_df.index[:5].tolist()
            top_features = importance_df['feature'].iloc[top_indices].tolist()
            top_importance = importance_df['importance'].iloc[top_indices].tolist()
            
            # 构建KAN回归方程
            equation = f"{target_var} = "
            for i, (feat, imp) in enumerate(zip(top_features, top_importance)):
                if i > 0:
                    equation += " + "
                equation += f"{imp:.4f} * {feat}"
            
            equation += " + ..."
            print(f"\nKAN模型回归方程（基于前5个最重要特征）:")
            print(equation)
            
            # 保存回归方程到文件
            with open(f'results/kan_regression_{target_var}.txt', 'w', encoding='utf-8') as f:
                f.write(f"KAN模型回归方程 ({target_var}):\n")
                f.write(f"R² 决定系数: {r2:.4f}\n")
                f.write(f"RMSE 均方根误差: {rmse:.4f}\n")
                f.write(f"MAE 平均绝对误差: {mae:.4f}\n\n")
                f.write(equation)
                
                # 如果存在符号化方程，也添加到文件中
                if symbolic_eq is not None:
                    f.write("\n\n符号化回归方程:\n")
                    f.write(str(symbolic_eq))
        
        # 构建线性回归模型作为对比基准
        print("\n构建参考线性回归模型...")
        from sklearn.linear_model import LinearRegression
        X = dataset['train_input'].detach().cpu().numpy()
        y = dataset['train_label'].detach().cpu().numpy()
        
        # 训练线性回归模型
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        
        # 计算线性模型的R²
        lr_pred = lr_model.predict(X)
        lr_r2 = r2_score(y, lr_pred)
        print(f"线性回归模型 R²: {lr_r2:.4f}")
        
        # 生成线性回归方程
        lr_coeffs = lr_model.coef_[0]
        lr_intercept = lr_model.intercept_[0]
        
        # 选择前5个最大系数
        coef_indices = np.argsort(np.abs(lr_coeffs))[-5:][::-1]
        
        # 构建方程
        lr_equation = f"{target_var} = {lr_intercept:.4f}"
        for idx in coef_indices:
            lr_equation += f" + {lr_coeffs[idx]:.4f} * {feature_cols[idx]}"
        lr_equation += " + ..."
        
        print(f"\n参考线性回归方程（前5个最重要系数）:")
        print(lr_equation)
        
        # 保存线性回归方程到文件
        with open(f'results/linear_regression_{target_var}.txt', 'w', encoding='utf-8') as f:
            f.write(f"线性回归方程 ({target_var}):\n")
            f.write(f"R²: {lr_r2:.4f}\n\n")
            f.write(lr_equation)
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'lr_r2': lr_r2
        }
    except Exception as e:
        print(f"评估模型时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def visualize_kan_structure(model, target_var):
    """自定义KAN模型结构可视化函数"""
    plt.figure(figsize=(12, 8))
    G = nx.DiGraph()
    
    # 添加节点
    input_size = model.width[0]
    hidden_size = model.width[1]
    output_size = model.width[2]
    
    # 添加输入层节点
    for i in range(input_size):
        G.add_node(f'i{i}', pos=(0, i - input_size/2), node_type='input')
    
    # 添加隐藏层节点
    for i in range(hidden_size):
        G.add_node(f'h{i}', pos=(1, i - hidden_size/2), node_type='hidden')
    
    # 添加输出层节点
    for i in range(output_size):
        G.add_node(f'o{i}', pos=(2, i - output_size/2), node_type='output')
    
    # 添加边
    for i in range(input_size):
        for j in range(hidden_size):
            G.add_edge(f'i{i}', f'h{j}')
    
    for i in range(hidden_size):
        for j in range(output_size):
            G.add_edge(f'h{i}', f'o{j}')
    
    # 绘制网络
    pos = nx.get_node_attributes(G, 'pos')
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos,
                          nodelist=[n for n, d in G.nodes(data=True) if d['node_type']=='input'],
                          node_color='lightblue', node_size=500, node_shape='s')
    nx.draw_networkx_nodes(G, pos,
                          nodelist=[n for n, d in G.nodes(data=True) if d['node_type']=='hidden'],
                          node_color='lightgreen', node_size=500, node_shape='o')
    nx.draw_networkx_nodes(G, pos,
                          nodelist=[n for n, d in G.nodes(data=True) if d['node_type']=='output'],
                          node_color='lightpink', node_size=500, node_shape='s')
    
    # 绘制边
    nx.draw_networkx_edges(G, pos)
    
    plt.title(f'KAN模型结构 - {target_var}')
    plt.axis('off')
    plt.savefig(f'results/model_structure_{target_var}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        print("开始分析程序...")
        
        # 创建结果目录
        os.makedirs('results', exist_ok=True)
        os.makedirs('figures', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # 清理旧的图片文件
        for f in os.listdir('figures'):
            if f.startswith('sp_'):
                os.remove(os.path.join('figures', f))
        
        # 加载数据
        file_path = 'G:/软管项目/data.xlsx'
        df = load_data(file_path)
        
        # 分析每个目标变量
        target_vars = ['lst_day_c', 'lst_night_c', 'uhi_day_c', 'uhi_night_c']
        metrics = {}
        important_features = {}
        
        # 第一轮：训练完整模型并确定重要特征
        print("\n开始分析所有目标变量")
        for target_var in target_vars:
            print(f"\n{'='*50}")
            print(f"分析目标变量: {target_var}")
            print(f"{'='*50}")
            
            # 准备数据（使用所有特征）
            dataset, feature_cols = prepare_data(df, target_var)
            
            # 训练模型并获取结构图
            model, fig = train_kan_model(dataset, len(feature_cols))
            
            # 保存模型结构图
            print("\n保存模型结构图...")
            fig.savefig(f'figures/model_structure_{target_var}.png', dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            # 清理中间文件
            for f in os.listdir('figures'):
                if f.startswith('sp_'):
                    os.remove(os.path.join('figures', f))
            
            # 分析模型
            importance_df = analyze_kan_model(model, dataset, feature_cols, target_var)
            
            # 评估模型
            print(f"\n评估模型 - {target_var}")
            metrics[target_var] = evaluate_model(model, dataset, feature_cols, target_var)
            
            # 保存模型
            torch.save(model.state_dict(), f'models/{target_var}_model.pt')
            
            # 确定重要特征（重要性大于平均值的特征）
            if importance_df is not None:
                threshold = importance_df['importance'].mean()
                important_features[target_var] = importance_df[importance_df['importance'] > threshold]['feature'].tolist()
                print(f"\n筛选出 {len(important_features[target_var])} 个重要特征（重要性 > {threshold:.4f}）")
        
        # 保存所有评估指标
        metrics_df = pd.DataFrame(metrics).T
        metrics_df.to_excel('results/model_metrics.xlsx')
        
        # 保存重要特征列表
        with open('results/important_features.txt', 'w', encoding='utf-8') as f:
            for target_var, features in important_features.items():
                f.write(f"{target_var} 重要特征 ({len(features)}个):\n")
                f.write(', '.join(features))
                f.write('\n\n')
        
        print("\n分析完成！结果已保存至results目录。")
        print("- 模型结构图保存在 figures 目录")
        print("- 特征重要性图表和回归方程保存在 results 目录")
        print("- 模型保存在 models 目录")
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 