# SEME 基线模型目录说明文档

本目录包含了用于城市环境指标预测的基线模型实现，主要通过多层感知器（MLP）和随机森林（RF）模型对夜间光、白天地表温度和夜间地表温度三个指标进行预测。

## 文件结构

```
baselineModel/
├── data/                    # 数据目录
│   └── shanghai.xlsx        # 原始数据集
├── models/                  # 模型保存目录
│   ├── mlp_model.pth        # 预训练的MLP模型权重
│   └── rf_model.joblib      # 预训练的随机森林模型
├── results/                 # 结果输出目录
│   ├── model_comparison.png # 模型预测效果对比图
│   ├── metrics_comparison.png # 模型性能指标对比图
│   ├── mlp_predictions.png  # MLP模型预测结果图
│   ├── rf_predictions.png   # 随机森林模型预测结果图
│   └── nested_cv_comparison.png # 嵌套交叉验证结果图
├── dataset.py               # 数据加载和预处理模块
├── mlp.py                   # 多层感知器模型实现
├── rf.py                    # 随机森林模型实现
├── compare_models.py        # 模型性能对比工具
├── nested_cv_evaluation.py  # 十折嵌套交叉验证评估系统
└── __pycache__/             # Python缓存目录
```

## 核心文件功能详解

### 1. 数据处理模块 (dataset.py)

- **主要功能**：加载和预处理上海城市环境数据
- **关键组件**：
  - `LightHeatDataset` 类：自定义PyTorch数据集类，用于加载和处理光热数据
  - `load_dataset` 函数：读取Excel数据，删除无用列，分离特征和目标变量，进行标准化，划分训练集和测试集
- **目标变量**：夜间光(`nighttime_`)、白天地表温度(`lst_day_c`)和夜间地表温度(`lst_night_`)

### 2. 多层感知器模型 (mlp.py)

- **主要功能**：构建和训练基于PyTorch的神经网络模型
- **模型结构**：输入层 → 64节点隐藏层 → 32节点隐藏层 → 输出层
- **核心实现**：
  - 数据加载和张量转换
  - 模型定义与初始化
  - 模型训练过程（使用Adam优化器和MSE损失函数）
  - 模型评估和性能可视化
  - 模型保存到models目录
  - 结果图表保存到results目录

### 3. 随机森林模型 (rf.py)

- **主要功能**：构建和训练基于scikit-learn的随机森林模型
- **模型参数**：使用100棵决策树的随机森林回归器
- **核心实现**：
  - 数据加载和预处理
  - 模型训练与预测
  - 性能评估（MSE和R²计算）
  - 模型保存到models目录
  - 结果图表保存到results目录

### 4. 模型对比工具 (compare_models.py)

- **主要功能**：直接对比MLP和随机森林模型在相同数据上的性能
- **核心实现**：
  - 加载并预处理数据集
  - 训练MLP模型（或加载预训练模型）
  - 训练随机森林模型
  - 比较两个模型在各目标变量上的MSE和R²指标
  - 生成散点图和条形图对比可视化，保存到results目录

### 5. 嵌套交叉验证评估系统 (nested_cv_evaluation.py)

- **主要功能**：通过十折嵌套交叉验证全面评估模型性能
- **评估流程**：
  - **外部循环**：将数据划分为10折，每次选1折作为测试集
  - **内部循环**：在剩余9折上进行参数网格搜索
  - **超参数优化**：通过内部交叉验证选择最佳参数
  - **模型评估**：使用最佳参数在测试集上评估性能
- **支持模型**：MLP、随机森林
- **可视化输出**：生成性能对比图表，保存到results目录

## 使用指南

### 基本模型评估

```python
# 运行单个MLP模型
python baselineModel/mlp.py

# 运行单个随机森林模型
python baselineModel/rf.py

# 直接对比两个模型性能
python baselineModel/compare_models.py
```

### 高级评估（嵌套交叉验证）

```python
python baselineModel/nested_cv_evaluation.py
```
运行后会提示选择评估的模型类型：
- `mlp`：只评估MLP模型
- `rf`：只评估随机森林模型
- `both`：同时评估两种模型并进行比较

## 扩展指南

### 添加新模型（如KAN模型）

1. 创建新的模型实现文件（如`kan.py`）
2. 在`nested_cv_evaluation.py`中添加对应的训练函数和参数网格
3. 更新可视化函数以包含新模型的结果
4. 将模型保存到models目录，结果图表保存到results目录

### 添加新数据特征

1. 修改`dataset.py`中的数据加载函数，保留需要的特征列
2. 调整特征工程和预处理步骤
3. 将新数据文件放入data目录

## 性能指标说明

- **MSE (均方误差)**：预测值与实际值差异的平方平均值，越低越好
- **R² (决定系数)**：模型解释数据变异性的程度，范围0-1，越高越好 