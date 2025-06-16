import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
# 自定义 PyTorch 数据集类
class LightHeatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 加载数据
def load_dataset(filepath, batch_size=32):
    # 1. 读取 Excel
    df = pd.read_excel(filepath)

    # 2. 删除无用列
    df.drop(columns=["FID", "Shape *", "FID_", "Id", "Shape_Leng", "SUM_1", "LENGTH", "LENGTH_1", "Land02", "Land50234", "Land505", "sidewalk_M", "building_M", "vegetation",
                    "sky_MEAN", "POI餐饮", "POI风景", "POI公司", "POI购物", "POI科教", "POI医疗", "POI政府", "railway_m", "Subway_m", "car_road_m", "high_grade", "Shape_Le_1",
                    "Shape_Area", "Longitude", "Latitude"], inplace=True)

    # 3. 分出输入与输出列
    target_cols = ["nighttime_", "lst_day_c", "lst_night_"]
    y = df[target_cols]
    X = df.drop(columns=target_cols)

    # 4. 标准化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # 5. 划分训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler_y