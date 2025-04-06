# 城市宜居性空间分析与可视化

本项目利用空间数据分析技术，对城市各区域的宜居性进行评估与可视化。

## 项目结构

```
.
├── spatial_analysis.py       # 主要空间分析和可视化脚本
├── kernel_density_analysis.py # 核密度分析脚本
├── spatial_data.xlsx         # 原始数据
├── outputs/                  # 输出结果
│   ├── figures/              # 静态图表
│   ├── kernel_density/       # 核密度分析结果
│   ├── processed_spatial_data.xlsx # 处理后的数据
│   └── livability_heatmap.html     # 交互式宜居性热力图
└── README.md                 # 项目说明文档
```

## 功能描述

1. **数据处理**
   - 数据清洗与预处理
   - 数据标准化
   - 经纬度处理与空间数据转换

2. **空间分析**
   - 环境宜居度计算
   - POI密度分析
   - 服务可达性分析
   - 城市热岛效应分析

3. **可视化结果**
   - 宜居性热力图
   - POI分布密度图
   - 植被覆盖度(NDVI)分布图
   - 建筑密度分布图
   - 城市热岛效应分布图
   - 交互式Web地图

## 使用方法

1. 创建并激活conda环境：
   ```
   conda create -n spatial_analysis python=3.9
   conda activate spatial_analysis
   ```

2. 安装依赖包：
   ```
   conda install -y pandas numpy matplotlib seaborn geopandas folium scikit-learn openpyxl
   ```

3. 运行分析脚本：
   ```
   python spatial_analysis.py
   python kernel_density_analysis.py
   ```

4. 查看结果：
   - 静态图表位于 `outputs/figures/` 目录
   - 核密度分析结果位于 `outputs/kernel_density/` 目录
   - 交互式热力图可在浏览器中打开 `outputs/livability_heatmap.html`

## 宜居性评估方法

宜居性指数通过以下指标加权计算得出：

- NDVI植被指数 (正向影响)
- 建筑密度 (负向影响)
- POI设施多样性 (正向影响)
- 不透水面积 (负向影响)
- 温度数据 (负向影响)
- 交通通达性 (正向影响)

## 注意事项

- 本项目使用的坐标系为WGS84 (EPSG:4326)
- 分析结果的精度取决于输入数据的质量和完整性 