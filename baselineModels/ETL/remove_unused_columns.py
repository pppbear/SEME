import os
import pandas as pd

# 配置路径
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'shanghai_nozero.xlsx')
OUTPUT_FILE = os.path.join(DATA_DIR, 'shanghai_nozero_cleaned.xlsx')

# 需要删除的无用列
unused_columns = [
    "FID", "Shape *", "FID_", "Id", "Shape_Leng", "SUM_1", "LENGTH", "LENGTH_1", "Land02", "Land50234", "Land505", "sidewalk_M", "building_M", "vegetation",
    "sky_MEAN", "POI餐饮", "POI风景", "POI公司", "POI购物", "POI科教", "POI医疗", "POI政府", "railway_m", "Subway_m", "car_road_m", "high_grade", "Shape_Le_1",
    "Shape_Area", "Longitude", "Latitude"
]

# 目标变量列（也可选删除）
target_columns = ["nighttime_", "lst_day_c", "lst_night_"]

if __name__ == '__main__':
    # 读取原始数据
    df = pd.read_excel(INPUT_FILE)
    print(f"原始数据形状: {df.shape}")

    # 删除无用列和目标列（如果存在）
    drop_cols = [col for col in unused_columns + target_columns if col in df.columns]
    df_cleaned = df.drop(columns=drop_cols)
    print(f"已删除 {len(drop_cols)} 列: {drop_cols}")
    print(f"清洗后数据形状: {df_cleaned.shape}")

    # 保存为新文件
    df_cleaned.to_excel(OUTPUT_FILE, index=False)
    print(f"已保存清洗后的数据到: {OUTPUT_FILE}") 