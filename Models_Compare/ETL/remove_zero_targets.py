import pandas as pd
import os

def remove_zero_targets(input_path='data.xlsx', output_path=None):
    # 读取数据
    df = pd.read_excel(input_path)
    # 自动检测目标列
    target_cols = [col for col in df.columns if any(key in col for key in ['nighttime_', 'lst_day_c', 'lst_night_'])]
    if len(target_cols) < 3:
        print(f"警告：未检测到全部三个目标变量列，实际检测到: {target_cols}")
    # 删除任一目标变量为0的行
    before = len(df)
    df_clean = df[~(df[target_cols] == 0).any(axis=1)]
    after = len(df_clean)
    print(f"已删除{before - after}行，剩余{after}行")
    # 保存新文件
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + '_nozero' + ext
    df_clean.to_excel(output_path, index=False)
    print(f"已保存清洗后的文件到: {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='删除三个因变量中有0的行')
    parser.add_argument('--input', type=str, default='data.xlsx', help='输入Excel文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出Excel文件路径')
    args = parser.parse_args()
    remove_zero_targets(args.input, args.output) 