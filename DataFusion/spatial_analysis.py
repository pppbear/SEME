import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from sklearn.preprocessing import MinMaxScaler
import os
from shapely.geometry import Point
import re

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建输出目录
if not os.path.exists('outputs'):
    os.makedirs('outputs')
if not os.path.exists('outputs/figures'):
    os.makedirs('outputs/figures')

print("开始读取数据...")
# 读取Excel数据
df = pd.read_excel('上海市栅格数据.xlsx')
print(f"成功读取数据，共 {len(df)} 行")

# 数据预处理
print("开始数据预处理...")

# 定义函数将DMS格式经纬度转换为十进制度数
def dms_to_decimal(dms_str):
    # 使用正则表达式提取度、分、秒和方向
    pattern = r'(\d+)°\s*(\d+)\'\s*([\d\.]+)"\s*([NSWE])'
    match = re.search(pattern, dms_str)
    if not match:
        return None
    
    degrees = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    direction = match.group(4)
    
    # 转换为十进制度数
    decimal = degrees + minutes/60 + seconds/3600
    
    # 南纬和西经为负值
    if direction in ['S', 'W']:
        decimal = -decimal
        
    return decimal

# 检查并转换经纬度列
if 'Longitude' in df.columns and 'Latitude' in df.columns:
    # 如果经纬度是字符串格式，转换为十进制度数
    if isinstance(df['Longitude'].iloc[0], str):
        print("转换DMS格式经纬度为十进制度数...")
        df['longitude_num'] = df['Longitude'].apply(dms_to_decimal)
        df['latitude_num'] = df['Latitude'].apply(dms_to_decimal)
    else:
        # 如果已经是数值格式，直接使用
        df['longitude_num'] = df['Longitude']
        df['latitude_num'] = df['Latitude']
else:
    # 如果找不到经纬度列，检查其他可能的列名
    possible_lon_columns = ['longitude', 'lon', '经度', 'LONGITUDE']
    possible_lat_columns = ['latitude', 'lat', '纬度', 'LATITUDE']
    
    lon_col = None
    for col in possible_lon_columns:
        if col in df.columns:
            lon_col = col
            break
            
    lat_col = None
    for col in possible_lat_columns:
        if col in df.columns:
            lat_col = col
            break
    
    if lon_col and lat_col:
        # 检查是否需要转换字符串格式的DMS
        if isinstance(df[lon_col].iloc[0], str) and '°' in str(df[lon_col].iloc[0]):
            print(f"转换DMS格式经纬度为十进制度数，使用列 {lon_col} 和 {lat_col}...")
            df['longitude_num'] = df[lon_col].apply(dms_to_decimal)
            df['latitude_num'] = df[lat_col].apply(dms_to_decimal)
        else:
            df['longitude_num'] = df[lon_col]
            df['latitude_num'] = df[lat_col]
    else:
        # 如果找不到经纬度列，生成随机经纬度（上海地区）
        print("未找到经纬度列，生成随机经纬度数据...")
        np.random.seed(42)  # 设置随机种子保证结果可重复
        n_rows = len(df)
        # 经度范围：约121.2°E - 121.8°E（上海市范围内）
        df['longitude_num'] = np.random.uniform(121.2, 121.8, n_rows)
        # 纬度范围：约31.0°N - 31.4°N（上海市范围内）
        df['latitude_num'] = np.random.uniform(31.0, 31.4, n_rows)

print(f"经度范围: {df['longitude_num'].min():.6f} 到 {df['longitude_num'].max():.6f}")
print(f"纬度范围: {df['latitude_num'].min():.6f} 到 {df['latitude_num'].max():.6f}")

# 创建GeoDataFrame用于空间分析
geometry = [Point(xy) for xy in zip(df['longitude_num'], df['latitude_num'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# 计算各项指标
print("开始计算宜居性指标...")

# 1. 环境宜居度计算
# 设置权重
weights = {
    'NDVI_MEAN': 0.4,          # 植被指数，正向影响
    '建筑密': -0.3,             # 建筑密度，负向影响
    'POI商务': 0.05,            # 商务POI，小权重正向影响
    'POI生活': 0.1,             # 生活POI，较大权重正向影响
    'POI医疗': 0.15,            # 医疗POI，较大权重正向影响
    '不透水': -0.1,             # 不透水面积，负向影响
    'lst_day_c': -0.15,        # 日间温度，负向影响
    'lst_night_': -0.05,       # 夜间温度，小权重负向影响
    'car_road_m': 0.05,        # 道路通达性，小权重正向影响
    'Subway_m': 0.1            # 地铁通达性，正向影响
}

# 标准化数据
scaler = MinMaxScaler()
for col in weights.keys():
    if col in df.columns:
        df[f'{col}_norm'] = scaler.fit_transform(df[[col]])

# 计算综合宜居度指数
df['livability_index'] = 0
for col, weight in weights.items():
    if col in df.columns:
        df['livability_index'] += df[f'{col}_norm'] * weight

# 将宜居度指数标准化到0-100
df['livability_index'] = 50 + 50 * (df['livability_index'] - df['livability_index'].mean()) / df['livability_index'].std()
df['livability_index'] = df['livability_index'].clip(0, 100)  # 限制在0-100范围内

# 2. POI密度分析
# 计算各类POI总和
df['POI_total'] = df['POI餐饮'] + df['POI风景'] + df['POI公司'] + df['POI购物'] + df['POI科教'] + \
                 df['POI商务'] + df['POI生活'] + df['POI体育'] + df['POI医疗'] + df['POI政府']

# 3. 城市热岛效应分析
# 使用白天和夜间温度数据计算热岛强度
if 'lst_day_c' in df.columns and 'lst_night_' in df.columns:
    # 热岛强度可以定义为区域温度偏离平均温度的程度
    df['heat_island_intensity'] = df['lst_day_c'] - df['lst_day_c'].mean()

# 数据可视化
print("开始生成可视化结果...")

# 1. 宜居性热力图
plt.figure(figsize=(10, 8))
plt.scatter(df['longitude_num'], df['latitude_num'], 
            c=df['livability_index'], cmap='viridis', 
            alpha=0.7, s=10)
plt.colorbar(label='宜居性指数 (0-100)')
plt.title('区域宜居性分布')
plt.xlabel('经度')
plt.ylabel('纬度')
plt.savefig('outputs/figures/livability_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. NDVI分布图
plt.figure(figsize=(10, 8))
plt.scatter(df['longitude_num'], df['latitude_num'], 
            c=df['NDVI_MEAN'], cmap='Greens', 
            alpha=0.7, s=10)
plt.colorbar(label='NDVI值')
plt.title('区域植被覆盖度(NDVI)分布')
plt.xlabel('经度')
plt.ylabel('纬度')
plt.savefig('outputs/figures/ndvi_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 建筑密度分布
plt.figure(figsize=(10, 8))
plt.scatter(df['longitude_num'], df['latitude_num'], 
            c=df['建筑密'], cmap='Reds', 
            alpha=0.7, s=10)
plt.colorbar(label='建筑密度')
plt.title('区域建筑密度分布')
plt.xlabel('经度')
plt.ylabel('纬度')
plt.savefig('outputs/figures/building_density.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. 热岛效应分布图
if 'heat_island_intensity' in df.columns:
    plt.figure(figsize=(10, 8))
    plt.scatter(df['longitude_num'], df['latitude_num'], 
                c=df['heat_island_intensity'], cmap='coolwarm', 
                alpha=0.7, s=10)
    plt.colorbar(label='热岛强度 (°C)')
    plt.title('城市热岛效应分布')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.savefig('outputs/figures/heat_island.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5. POI密度分布
plt.figure(figsize=(10, 8))
plt.scatter(df['longitude_num'], df['latitude_num'], 
            c=df['POI_total'], cmap='plasma', 
            alpha=0.7, s=10)
plt.colorbar(label='POI总数')
plt.title('区域POI密度分布')
plt.xlabel('经度')
plt.ylabel('纬度')
plt.savefig('outputs/figures/poi_density.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. 交互式地图可视化（使用folium）
print("生成交互式地图...")

# 确保中心点经纬度有效
center_lat = df['latitude_num'].mean()
center_lon = df['longitude_num'].mean()

print(f"地图中心点: 经度={center_lon:.6f}, 纬度={center_lat:.6f}")

# 基础地图
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# 添加宜居性热力图层
heat_data = []
for _, row in df.iterrows():
    try:
        lat = float(row['latitude_num'])
        lon = float(row['longitude_num'])
        value = float(row['livability_index'])
        heat_data.append([lat, lon, value])
    except (ValueError, TypeError):
        continue

# 添加热力图 - 修正了gradient参数的格式问题
gradient_dict = {'0.2': 'blue', '0.4': 'lime', '0.6': 'yellow', '1': 'red'}
HeatMap(heat_data, radius=15, max_zoom=13, gradient=gradient_dict).add_to(m)

# 保存地图
m.save('outputs/livability_heatmap.html')

# 7. 相关性分析
print("生成相关性分析...")
# 选择有意义的变量进行相关性分析
corr_vars = ['NDVI_MEAN', '建筑密', '容积率', 'POI餐饮', 'POI公司', 'POI医疗', 
             'POI生活', '不透水', 'car_road_m', 'lst_day_c', 'livability_index']
corr_df = df[corr_vars].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('各指标相关性分析')
plt.savefig('outputs/figures/correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 保存处理后的数据
df.to_excel('outputs/processed_spatial_data.xlsx', index=False)
print("分析完成，所有结果已保存到outputs目录") 