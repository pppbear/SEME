import os
import time
import subprocess

def run_script(script_name, description):
    """运行指定的Python脚本并输出状态信息"""
    print(f"\n{'=' * 60}")
    print(f"开始执行 {description}")
    print(f"{'=' * 60}")
    
    start_time = time.time()
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        print(f"\n{description}执行成功！耗时: {time.time() - start_time:.2f} 秒")
    else:
        print(f"错误: {description}执行失败")
        print(f"错误信息:\n{result.stderr}")
    
    return result.returncode == 0

def main():
    """主函数，按顺序执行所有分析步骤"""
    print("城市宜居性空间分析与可视化系统")
    print("=" * 50)
    
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        print("创建输出目录: outputs/")
    
    if not os.path.exists('outputs/figures'):
        os.makedirs('outputs/figures')
        print("创建输出目录: outputs/figures/")
    
    if not os.path.exists('outputs/kernel_density'):
        os.makedirs('outputs/kernel_density')
        print("创建输出目录: outputs/kernel_density/")
    
    # 步骤1: 生成示例数据（如果不存在）
    if not os.path.exists('spatial_data.xlsx'):
        if run_script('create_sample_data.py', '生成示例数据'):
            print("示例数据生成成功: spatial_data.xlsx")
        else:
            print("数据生成失败，终止执行")
            return
    else:
        print("示例数据已存在，跳过生成步骤")
    
    # 步骤2: 执行主空间分析
    if run_script('spatial_analysis.py', '主空间分析与基础可视化'):
        print("主空间分析完成")
    else:
        print("主空间分析失败，终止执行")
        return
    
    # 步骤3: 执行核密度分析
    if run_script('kernel_density_analysis.py', '核密度与可达性分析'):
        print("核密度分析完成")
    else:
        print("核密度分析失败")
    
    print("\n" + "=" * 50)
    print("所有分析步骤执行完毕！")
    print("=" * 50)
    print("\n结果输出目录:")
    print("  - 处理后的数据: outputs/processed_spatial_data.xlsx")
    print("  - 静态图表: outputs/figures/")
    print("  - 核密度分析: outputs/kernel_density/")
    print("  - 交互式热力图: outputs/livability_heatmap.html")

if __name__ == "__main__":
    main() 