import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 确保输出目录存在
if not os.path.exists('outputs/kernel_density'):
    os.makedirs('outputs/kernel_density')

print("开始核密度分析...")
# 读取处理后的数据
df = pd.read_excel('outputs/processed_spatial_data.xlsx')
print(f"成功读取数据，共 {len(df)} 行")

# 各类POI核密度分析
poi_columns = ['POI餐饮', 'POI风景', 'POI公司', 'POI购物', 'POI科教', 'POI商务', 'POI生活', 'POI体育', 'POI医疗', 'POI政府']

# 创建网格
xmin, xmax = df['longitude_num'].min(), df['longitude_num'].max()
ymin, ymax = df['latitude_num'].min(), df['latitude_num'].max()

# 扩大一点边界以便更好地可视化
margin = 0.01
xmin -= margin
xmax += margin
ymin -= margin
ymax += margin

# 创建网格点
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])

print("计算各类POI的核密度...")
for poi_type in poi_columns:
    print(f"分析 {poi_type}...")
    
    # 过滤掉值为零的点，只考虑有POI的位置
    valid_points = df[df[poi_type] > 0]
    
    if len(valid_points) > 5:  # 确保有足够的数据点进行核密度估计
        # 准备数据
        values = np.vstack([valid_points['longitude_num'], valid_points['latitude_num']])
        
        # 计算核密度
        kernel = gaussian_kde(values)
        Z = np.reshape(kernel(positions), X.shape)
        
        # 绘制核密度图
        plt.figure(figsize=(10, 8))
        plt.imshow(np.rot90(Z), cmap='YlOrRd', extent=[xmin, xmax, ymin, ymax])
        plt.scatter(valid_points['longitude_num'], valid_points['latitude_num'], 
                   c='b', s=5, alpha=0.3)
        plt.colorbar(label='密度')
        plt.title(f'{poi_type}核密度分布')
        plt.xlabel('经度')
        plt.ylabel('纬度')
        plt.savefig(f'outputs/kernel_density/{poi_type}_kernel_density.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{poi_type} 核密度图已生成")
    else:
        print(f"{poi_type} 数据点不足，跳过核密度分析")

# 服务可达性分析 - 主要考虑医疗、商业和生活服务设施的综合密度
service_cols = ['POI医疗', 'POI购物', 'POI生活']
print("计算服务可达性核密度...")

# 为每个点创建一个服务可达性得分
df['service_accessibility'] = df[service_cols].sum(axis=1)

# 过滤掉值为零的点
valid_points = df[df['service_accessibility'] > 0]

if len(valid_points) > 5:
    # 准备数据
    values = np.vstack([valid_points['longitude_num'], valid_points['latitude_num']])
    
    # 使用服务可达性得分作为权重
    weights = valid_points['service_accessibility']
    
    # 计算核密度
    kernel = gaussian_kde(values, weights=weights)
    Z = np.reshape(kernel(positions), X.shape)
    
    # 绘制核密度图
    plt.figure(figsize=(10, 8))
    plt.imshow(np.rot90(Z), cmap='viridis', extent=[xmin, xmax, ymin, ymax])
    plt.colorbar(label='服务可达性')
    plt.title('服务可达性核密度分布')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.savefig('outputs/kernel_density/service_accessibility.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("服务可达性核密度图已生成")
else:
    print("服务可达性数据点不足，跳过核密度分析")

print("核密度分析完成，所有结果已保存到outputs/kernel_density目录") 