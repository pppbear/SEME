import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kan.kan.MultKAN import KAN
import warnings
import pickle
warnings.filterwarnings('ignore')

# 检查CUDA是否可用并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

def preprocess_data(df, target_var=None, feature_cols=None):
    """预处理数据，与训练时使用相同的步骤"""
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
    
    # 如果提供了特征列表，只保留这些特征
    if feature_cols is not None:
        # 检查所有特征是否存在
        missing_features = [f for f in feature_cols if f not in df.columns]
        if missing_features:
            raise ValueError(f"数据中缺少以下特征: {missing_features}")
        df = df[feature_cols + ([target_var] if target_var in df.columns else [])]
    else:
        # 只保留有用的列
        exclude_cols = ['FID', 'Shape *', 'FID_', 'Id', 'Shape_Leng', 'Shape_Le_1', 'Shape_Area','Longitude','Latitude']
        df = df.drop(columns=exclude_cols, errors='ignore')
    
    # 如果指定了目标变量，过滤掉目标变量为0的值
    if target_var is not None and target_var in df.columns:
        df = df[df[target_var] != 0].copy()
        print(f"过滤零值后的数据量: {len(df)}")
    
    # 处理数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 检查填充后的NaN值
    print("\n填充后的NaN值统计:")
    for col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"{col}: {nan_count} 个NaN值")
    
    print(f"预处理后的数据量: {len(df)}")
    
    return df

def load_model(model_path, features_path):
    """加载训练好的KAN模型和特征列表"""
    # 加载模型参数
    checkpoint = torch.load(model_path, map_location=device)
    print("\n加载的模型参数:")
    print(f"width: {checkpoint['model_params']['width']}")
    print(f"grid: {checkpoint['model_params']['grid']}")
    print(f"k: {checkpoint['model_params']['k']}")
    
    # 创建新的KAN模型
    model = KAN(
        width=checkpoint['model_params']['width'],
        grid=checkpoint['model_params']['grid'],
        k=checkpoint['model_params']['k'],
        symbolic_enabled=True
    ).to(device)
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置为评估模式
    
    # 打印模型参数统计
    print("\n模型参数统计:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}:")
            print(f"  形状: {param.shape}")
            print(f"  均值: {param.data.mean().item():.6f}")
            print(f"  标准差: {param.data.std().item():.6f}")
            print(f"  最小值: {param.data.min().item():.6f}")
            print(f"  最大值: {param.data.max().item():.6f}")
    
    # 验证模型是否真的处于评估模式
    print(f"\n模型是否处于评估模式: {not model.training}")
    
    # 加载特征列表，尝试不同的编码方式
    try:
        # 首先尝试UTF-8编码
        with open(features_path, 'r', encoding='utf-8') as f:
            feature_cols = f.read().splitlines()
    except UnicodeDecodeError:
        try:
            # 如果失败，尝试GBK编码
            with open(features_path, 'r', encoding='gbk') as f:
                feature_cols = f.read().splitlines()
        except UnicodeDecodeError:
            # 如果还是失败，尝试GB2312编码
            with open(features_path, 'r', encoding='gb2312') as f:
                feature_cols = f.read().splitlines()
    
    print("\n加载的特征列表:")
    for i, feature in enumerate(feature_cols, 1):
        print(f"{i}. {feature}")
    
    return model, feature_cols

def predict_with_kan(data_path, target_var, model_dir='D:\\vscode\\Models_Compare\\kan_new\\models'):
    """
    使用训练好的KAN模型进行预测
    
    参数:
    data_path: str, 数据文件路径
    target_var: str, 目标变量名称 ('lst_day_c', 'lst_night_c', 或 'nighttime_')
    model_dir: str, 模型文件所在目录
    
    返回:
    dict: 包含预测结果和评估指标
    """
    print(f"\n开始预测 {target_var}...")
    
    # 检查模型文件是否存在
    model_path = os.path.join(model_dir, f'{target_var}_model.pth')
    features_path = os.path.join(model_dir, f'{target_var}_features.txt')
    scaler_path = os.path.join(model_dir, f'{target_var}_scaler.pkl')
    y_scaler_path = os.path.join(model_dir, f'{target_var}_y_scaler.pkl')
    
    print(f"检查模型文件:")
    print(f"模型文件: {model_path}")
    print(f"特征文件: {features_path}")
    print(f"标准化参数文件: {scaler_path}")
    print(f"目标变量标准化参数文件: {y_scaler_path}")
    
    print("模型文件绝对路径：", os.path.abspath(model_path))
    print("文件是否存在：", os.path.exists(model_path))
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"特征列表文件不存在: {features_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"标准化参数文件不存在: {scaler_path}")
    if not os.path.exists(y_scaler_path):
        raise FileNotFoundError(f"目标变量标准化参数文件不存在: {y_scaler_path}")
    
    # 加载模型和特征
    model, feature_cols = load_model(model_path, features_path)
    print(f"成功加载模型和特征列表，特征数量: {len(feature_cols)}")
    
    # 加载标准化参数
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(y_scaler_path, 'rb') as f:
        y_scaler = pickle.load(f)
    print("成功加载标准化参数")
    
    # 打印标准化参数统计
    print("\n标准化参数统计:")
    print("特征标准化参数:")
    print(f"  均值: {scaler.mean_}")
    print(f"  标准差: {scaler.scale_}")
    print("目标变量标准化参数:")
    print(f"  均值: {y_scaler.mean_}")
    print(f"  标准差: {y_scaler.scale_}")
    
    # 加载并预处理数据
    print(f"\n加载数据文件: {data_path}")
    df = pd.read_excel(data_path)
    print(f"加载数据，形状: {df.shape}")
    
    # 打印数据集的列名，用于调试
    print("\n数据集中的列名:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")
    
    # 预处理数据，只使用训练时使用的特征
    df = preprocess_data(df, target_var, feature_cols)
    print(f"预处理后的数据形状: {df.shape}")
    
    # 准备预测数据
    X = df[feature_cols].values
    print("\n特征数据统计:")
    print(f"特征数据形状: {X.shape}")
    print(f"特征数据范围: [{X.min():.4f}, {X.max():.4f}]")
    print(f"特征数据均值: {X.mean():.4f}")
    print(f"特征数据标准差: {X.std():.4f}")
    
    # 使用保存的标准化参数进行标准化
    X = scaler.transform(X)
    print("\n标准化后的特征数据统计:")
    print(f"标准化后数据范围: [{X.min():.4f}, {X.max():.4f}]")
    print(f"标准化后数据均值: {X.mean():.4f}")
    print(f"标准化后数据标准差: {X.std():.4f}")
    
    # 转换为PyTorch张量
    X = torch.FloatTensor(X).to(device)
    
    # 进行预测
    model.eval()  # 确保模型处于评估模式
    with torch.no_grad():
        y_pred = model(X)
        print("\n原始预测结果统计:")
        print(f"预测结果形状: {y_pred.shape}")
        print(f"预测结果范围: [{y_pred.min().item():.4f}, {y_pred.max().item():.4f}]")
        print(f"预测结果均值: {y_pred.mean().item():.4f}")
        print(f"预测结果标准差: {y_pred.std().item():.4f}")
    
    # 将预测结果转换回numpy数组
    y_pred = y_pred.cpu().numpy().flatten()  # 确保是一维数组
    
    # 反标准化预测结果
    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    print("\n反标准化后的预测结果统计:")
    print(f"反标准化后预测结果范围: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    print(f"反标准化后预测结果均值: {y_pred.mean():.4f}")
    print(f"反标准化后预测结果标准差: {y_pred.std():.4f}")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'predicted_value': y_pred
    })
    
    # 如果有真实值，计算评估指标
    if target_var in df.columns:
        y_true = df[target_var].values
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        results_df['true_value'] = y_true
        
        print("\n预测结果评估:")
        print(f"R²: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        # 打印前5个样本的预测结果和真实值对比
        print("\n前5个样本的预测结果和真实值对比:")
        for i in range(min(5, len(y_true))):
            print(f"样本 {i+1}: 真实值 = {y_true[i]:.4f}, 预测值 = {y_pred[i]:.4f}")
    
    # 保存预测结果到Excel文件
    output_file = f'prediction_results/predictions_{target_var}.xlsx'
    results_df.to_excel(output_file, index=False)
    print(f"\n预测结果已保存到: {output_file}")
    
    return results_df

def main():
    print("开始KAN模型预测程序...")
    
    try:
        # 创建预测结果目录
        os.makedirs('prediction_results', exist_ok=True)
        
        # 预测目标变量
        target_vars = ['lst_day_c', 'lst_night_c', 'nighttime_']
        results = {}
        
        for target_var in target_vars:
            print(f"\n{'='*50}")
            print(f"预测目标变量: {target_var}")
            print(f"{'='*50}")
            
            try:
                # 使用训练好的模型进行预测
                result = predict_with_kan('data_old.xlsx', target_var, model_dir='D:\\vscode\\Models_Compare\\kan_new\\models')
                results[target_var] = result
                
            except Exception as var_error:
                print(f"预测变量 {target_var} 时出错: {str(var_error)}")
                import traceback
                print(traceback.format_exc())
                continue
        
        # 保存预测结果摘要
        if results:
            with open('prediction_results/prediction_summary.txt', 'w', encoding='utf-8') as f:
                f.write("KAN模型预测结果摘要\n")
                f.write("="*50 + "\n\n")
                
                for target_var, result in results.items():
                    f.write(f"{target_var} 预测结果:\n")
                    f.write("-"*40 + "\n\n")
                    
                    if 'true_value' in result.columns:
                        r2 = r2_score(result['true_value'], result['predicted_value'])
                        rmse = np.sqrt(mean_squared_error(result['true_value'], result['predicted_value']))
                        mae = mean_absolute_error(result['true_value'], result['predicted_value'])
                        
                        f.write(f"模型性能指标:\n")
                        f.write(f"R²: {r2:.4f}\n")
                        f.write(f"RMSE: {rmse:.4f}\n")
                        f.write(f"MAE: {mae:.4f}\n\n")
                    
                    f.write("="*50 + "\n\n")
            
            print("\n预测完成！结果已保存至prediction_results目录。")
            print("- 预测结果保存在 prediction_results/predictions_*.xlsx 文件")
            print("- 预测摘要保存在 prediction_results/prediction_summary.txt")
        else:
            print("\n警告：没有成功完成任何目标变量的预测")
    
    except Exception as main_error:
        print(f"主程序执行出错: {str(main_error)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 