import pandas as pd
import re
from data_service.app.core.config import settings

def dms_str_to_float(dms_str):
    match = re.match(r"(\d+)°\s*(\d+)'[\s]*(\d+(?:\.\d+)?)\"?\s*([NSEW])", dms_str.strip())
    if not match:
        raise ValueError(f"经纬度格式不正确: {dms_str}")
    deg, minute, sec, direction = match.groups()
    value = float(deg) + float(minute) / 60 + float(sec) / 3600
    if direction in ['S', 'W']:
        value = -value
    return value

def load_data_by_target(target: str, file_path: str = settings.DATA_PATH):
    df = pd.read_excel(file_path)
    if target not in df.columns:
        raise ValueError("目标因变量不存在于Excel中")
    if "Longitude" not in df.columns or "Latitude" not in df.columns:
        raise ValueError("Excel中缺少Longitude或Latitude列")
    data = []
    for _, row in df.iterrows():
        lon = dms_str_to_float(row["Longitude"])
        lat = dms_str_to_float(row["Latitude"])
        data.append({
            "longitude": lon,
            "latitude": lat,
            "value": float(row[target])
        })
    return data 