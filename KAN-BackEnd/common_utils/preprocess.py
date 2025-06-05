import pandas as pd

# 模型调用预处理
def preprocess_for_model(df, feature_cols):
    """
    通用数据预处理：
    1. 过滤NDVI_MEAN和不透水面比例为0的行
    2. 自动补齐POI总数
    3. 自动生成平均建筑高度
    """
    # 自动去除无用列
    exclude_cols = ['FID', 'Shape *', 'FID_', 'Id', 'Shape_Leng', 'Shape_Le_1', 'Shape_Area', 'Longitude', 'Latitude']
    df = df.drop(columns=exclude_cols, errors='ignore')

    # 过滤NDVI_MEAN和不透水面比例为0的行
    for col in ['NDVI_MEAN', '不透水面比例']:
        if col in df.columns:
            before = len(df)
            df = df[df[col] != 0]
            print(f"已过滤{col}为0的行，剩余{len(df)}/{before}")

    # 自动补齐POI总数
    if 'POI总数' in feature_cols and 'POI总数' not in df.columns:
        poi_cols = [col for col in df.columns if col.startswith('POI')]
        if poi_cols:
            df['POI总数'] = df[poi_cols].sum(axis=1)
            print(f"已自动补齐POI总数列，使用{len(poi_cols)}个POI相关列加和。")
        else:
            print("警告：特征文件要求POI总数，但数据中没有任何POI相关列，POI总数将为NaN。")

    # 自动生成平均建筑高度
    if '平均建筑高度' in feature_cols and '平均建筑高度' not in df.columns:
        if '容积率' in df.columns and '建筑密度' in df.columns:
            df['平均建筑高度'] = df.apply(
                lambda row: row['容积率'] / row['建筑密度'] if row['建筑密度'] != 0 and not pd.isnull(row['建筑密度']) else 0,
                axis=1
            )
            print("已自动生成平均建筑高度列。")
            
    return df 

# 关键特征值分析预处理
def preprocess_for_analyze(df):
    """
    通用数据预处理：
    1. 自动去除无用列
    2. 过滤NDVI_MEAN和不透水面比例为0的行
    3. 自动补齐POI总数
    4. 自动生成平均建筑高度
    """

    # 自动去除无用列
    exclude_cols = ['FID', 'Shape *', 'FID_', 'Id', 'Shape_Leng', 'Shape_Le_1', 'Shape_Area', 'Longitude', 'Latitude']
    df = df.drop(columns=exclude_cols, errors='ignore')

    # 过滤NDVI_MEAN和不透水面比例为0的行
    for col in ['NDVI_MEAN', '不透水面比例']:
        if col in df.columns:
            before = len(df)
            df = df[df[col] != 0]
            print(f"已过滤{col}为0的行，剩余{len(df)}/{before}")

    # 自动补齐POI总数
    if 'POI总数' not in df.columns:
        poi_cols = [col for col in df.columns if col.startswith('POI')]
        if poi_cols:
            df['POI总数'] = df[poi_cols].sum(axis=1)
            print(f"已自动补齐POI总数列，使用{len(poi_cols)}个POI相关列加和。")
        else:
            print("警告：数据中没有任何POI相关列，POI总数将为NaN。")

    # 自动生成平均建筑高度
    if '平均建筑高度' not in df.columns:
        if '容积率' in df.columns and '建筑密度' in df.columns:
            df['平均建筑高度'] = df.apply(
                lambda row: row['容积率'] / row['建筑密度'] if row['建筑密度'] != 0 and not pd.isnull(row['建筑密度']) else 0,
                axis=1
            )
            print("已自动生成平均建筑高度列。")
            
    return df