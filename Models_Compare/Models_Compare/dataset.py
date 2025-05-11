import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
import os

# 自定义 PyTorch 数据集类，用于加载和处理光热数据
class LightHeatDataset(Dataset):
    def __init__(self, X, y):
        # 将输入特征和目标变量转换为PyTorch张量
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # 返回数据集的样本数量
        return len(self.X)

    def __getitem__(self, idx):
        # 根据索引返回对应的特征和标签对
        return self.X[idx], self.y[idx]

# 加载数据集函数，处理Excel数据并进行预处理
def load_dataset(filepath=None, batch_size=32):
    # 如果未指定文件路径，使用默认路径
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "data", "shanghai.xlsx")
    
    # 1. 读取Excel文件数据
    df = pd.read_excel(filepath)

    # 2. 删除无用列，保留有用的特征
    df.drop(columns=["FID", "Shape *", "FID_", "Id", "Shape_Leng", "SUM_1", "LENGTH", "LENGTH_1", "Land02", "Land50234", "Land505", "sidewalk_M", "building_M", "vegetation",
                    "sky_MEAN", "POI餐饮", "POI风景", "POI公司", "POI购物", "POI科教", "POI医疗", "POI政府", "railway_m", "Subway_m", "car_road_m", "high_grade", "Shape_Le_1",
                    "Shape_Area", "Longitude", "Latitude"], inplace=True)

    # 3. 分离输入特征和目标变量
    target_cols = ["nighttime_", "lst_day_c", "lst_night_"]  # 目标变量: 夜间光、白天地表温度、夜间地表温度
    y = df[target_cols]
    X = df.drop(columns=target_cols)

    # 4. 对输入特征和目标变量进行标准化处理
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # 5. 将数据集划分为训练集和测试集(80%训练, 20%测试)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    # 返回处理后的数据集和y的缩放器(用于后续反标准化)
    return X_train, X_test, y_train, y_test, scaler_y