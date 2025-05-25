import pandas as pd

def predict_from_excel(file_path, target):
    """
    读取Excel文件并根据target返回预测结果
    :param file_path: Excel文件路径
    :param target: 目标因变量（nighttime_、lst_day_c、lst_night_）
    :return: 预测结果列表
    """
    df = pd.read_excel(file_path)
    if df.shape[1] == 0:
        raise ValueError('Excel 文件无数据')
    if target == 'nighttime_':
        predictions = (df.iloc[:, 0] + 2).tolist()
    elif target == 'lst_day_c':
        predictions = (df.iloc[:, 0] * 2).tolist()
    elif target == 'lst_night_':
        predictions = (df.iloc[:, 0] - 2).tolist()
    else:
        raise ValueError('目标因变量不合法')
    return predictions 