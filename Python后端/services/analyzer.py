import pandas as pd

def analyze_variables_with_kan(file_path, target):
    df = pd.read_excel(file_path)
    if target not in df.columns:
        raise ValueError('目标因变量不存在于Excel中')
    X = df.drop(columns=[target])
    y = df[target]
    importances = []
    for col in X.columns:
        # 这里用相关系数模拟KAN模型的解释性分数，实际请替换为KAN分析
        importance = abs(X[col].corr(y))
        importances.append({"variable": col, "importance": float(importance)})
    return importances 