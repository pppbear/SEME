# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from kan.MultKAN import KAN
# import pandas as pd

# # 1. 生成示例数据
# def create_dataset():
#     # 设置随机种子以确保结果可重现
#     torch.manual_seed(42)
#     np.random.seed(42)
    
#     # 生成训练数据
#     x = torch.rand(1000, 2) * 4 - 2  # 生成[-2, 2]范围内的1000个2维随机点
#     y = torch.sin(x[:, 0]) + 0.1 * x[:, 0]**2 + 0.5 * x[:, 1]  # 添加第二个变量的影响
    
#     # 创建数据集字典
#     dataset = {
#         'train_input': x,
#         'train_label': y.reshape(-1, 1),
#         'test_input': x,
#         'test_label': y.reshape(-1, 1)
#     }
    
#     # 打印数据集信息
#     print("\n数据集信息:")
#     print("训练数据形状:", x.shape)
#     print("标签数据形状:", y.reshape(-1, 1).shape)
    
#     return dataset

# # 2. 创建并训练KAN模型
# def train_model(dataset):
#     try:
#         print("\n开始训练模型...")
#         # 创建一个KAN模型，增加输入维度和隐藏层节点
#         model = KAN(
#             width=[2, 5, 1],     # 输入层2个节点，隐藏层5个节点，输出层1个节点
#             grid=5,              # 5个网格点
#             k=3,                 # 3阶样条
#             symbolic_enabled=True # 启用符号计算
#         )
        
#         # 训练模型
#         print("正在训练模型，这可能需要一些时间...")
#         history = model.fit(
#             dataset=dataset,
#             opt="LBFGS",        # 优化器
#             steps=50,           # 训练步数
#             lamb_l1=1.,         # L1正则化系数
#             lamb_entropy=2.     # 熵正则化系数
#         )
        
#         print("模型训练完成！")
#         return model
        
#     except Exception as e:
#         print("模型训练过程中出现错误：", str(e))
#         raise

# # 3. 分析模型
# def analyze_model(model, dataset):
#     try:
#         print("\n" + "="*50)
#         print("模型分析结果：")
#         print("="*50)
        
#         # 1. 获取符号表达式
#         print("\n1. 数学表达式分析：")
#         try:
#             print("\n1.1 模型结构分析：")
#             # 使用auto_symbolic获取详细的数学关系
#             print("正在分析模型结构...")
#             model.auto_symbolic(verbose=1, weight_simple=0.8, r2_threshold=0.90)
            
#             print("\n1.2 各层的数学关系：")
#             # 对每个输出节点进行分析
#             for l in range(model.depth):
#                 print(f"\n第{l+1}层的数学关系：")
#                 for i in range(model.width_out[l+1]):
#                     try:
#                         # 获取该节点的函数关系
#                         fun = model.get_fun(l, 0, i)
#                         if fun is not None:
#                             print(f"节点 {i}: {fun}")
#                     except Exception as e:
#                         print(f"分析节点 {i} 时出错：{str(e)}")
            
#         except Exception as e:
#             print("获取数学表达式时出错：", str(e))
        
#         # 2. 特征重要性分析
#         print("\n2. 特征重要性分析：")
#         try:
#             # 2.1 基于模型的特征重要性
#             print("\n2.1 基于模型的特征重要性：")
#             importance = model.attribute()
#             if isinstance(importance, (list, np.ndarray)):
#                 importance = np.array(importance)
#                 total = np.sum(importance) if np.sum(importance) != 0 else 1
                
#                 # 保存特征重要性得分
#                 model.feature_score = importance
                
#                 print(f"X1 (非线性项) 重要性：{importance[0]:.4f} ({(importance[0]/total*100):.2f}%)")
#                 print(f"X2 (线性项) 重要性：{importance[1]:.4f} ({(importance[1]/total*100):.2f}%)")
                
#                 # 计算相对重要性
#                 relative_importance = importance / total
#                 print("\n相对重要性比例：")
#                 print(f"X1 : X2 = {relative_importance[0]:.4f} : {relative_importance[1]:.4f}")
#             else:
#                 print("无法计算基于模型的特征重要性")
            
#             # 2.2 基于梯度的特征重要性
#             print("\n2.2 基于梯度的特征重要性：")
#             try:
#                 X = dataset['train_input']
#                 with torch.no_grad():
#                     gradients = []
#                     for i in range(X.shape[1]):
#                         X_temp = X.clone()
#                         X_temp.requires_grad = True
#                         output = model(X_temp)
#                         grad = torch.autograd.grad(output.sum(), X_temp)[0]
#                         feature_importance = torch.mean(torch.abs(grad[:, i])).item()
#                         gradients.append(feature_importance)
                
#                 total_grad = sum(gradients)
#                 print(f"X1 梯度重要性：{gradients[0]:.4f} ({(gradients[0]/total_grad*100):.2f}%)")
#                 print(f"X2 梯度重要性：{gradients[1]:.4f} ({(gradients[1]/total_grad*100):.2f}%)")
#             except Exception as e:
#                 print("计算梯度重要性时出错：", str(e))
            
#             # 2.3 基于预测值变化的重要性分析
#             print("\n2.3 基于预测值变化的重要性分析：")
#             try:
#                 variations = []
#                 X_base = X.clone()
#                 base_pred = model(X_base).detach().mean().item()
                
#                 for i in range(X.shape[1]):
#                     X_temp = X_base.clone()
#                     X_temp[:, i] = X_temp[:, i] + torch.std(X_temp[:, i])
#                     new_pred = model(X_temp).detach().mean().item()
#                     variation = abs(new_pred - base_pred)
#                     variations.append(variation)
                
#                 total_var = sum(variations)
#                 print(f"X1 变化重要性：{variations[0]:.4f} ({(variations[0]/total_var*100):.2f}%)")
#                 print(f"X2 变化重要性：{variations[1]:.4f} ({(variations[1]/total_var*100):.2f}%)")
#             except Exception as e:
#                 print("计算变化重要性时出错：", str(e))
                
#         except Exception as e:
#             print("特征重要性分析时出错：", str(e))
        
#         # 3. 特征交互分析
#         print("\n3. 特征交互分析：")
#         try:
#             interactions = model.feature_interaction(l=0, neuron_th=1e-2, feature_th=1e-2)
#             print("特征交互矩阵：")
#             if isinstance(interactions, dict):
#                 for key, value in interactions.items():
#                     if len(key) == 1:
#                         print(f"特征 X{key[0]+1} 的独立影响强度：{value:.4f}")
#                     elif len(key) == 2:
#                         print(f"特征 X{key[0]+1} 和 X{key[1]+1} 的交互强度：{value:.4f}")
#             else:
#                 print(interactions)
#         except Exception as e:
#             print("特征交互分析时出错：", str(e))
            
#     except Exception as e:
#         print("模型分析过程中出现错误：", str(e))

# # 4. 可视化结果
# def plot_results(dataset, model):
#     try:
#         print("\n生成可视化结果...")
        
#         # 创建一个大的图形
#         plt.figure(figsize=(20, 6))
        
#         # 1. 绘制预测结果的3D图
#         try:
#             # 创建测试点网格
#             x1 = np.linspace(-2, 2, 50)
#             x2 = np.linspace(-2, 2, 50)
#             X1, X2 = np.meshgrid(x1, x2)
#             X_test = torch.tensor(np.column_stack((X1.flatten(), X2.flatten())), dtype=torch.float32)
            
#             # 获取预测值
#             Y_pred = model(X_test).detach().numpy().reshape(50, 50)
            
#             # 计算真实值
#             Y_true = np.sin(X1) + 0.1 * X1**2 + 0.5 * X2
            
#             # 3D表面图 - 预测值
#             ax1 = plt.subplot(131, projection='3d')
#             surf1 = ax1.plot_surface(X1, X2, Y_pred, cmap='viridis')
#             ax1.set_xlabel('X1')
#             ax1.set_ylabel('X2')
#             ax1.set_zlabel('Y预测')
#             ax1.set_title('模型预测的3D表面图')
#             plt.colorbar(surf1, ax=ax1)
            
#         except Exception as e:
#             print("绘制3D预测图时出错：", str(e))
        
#         # 2. 绘制预测误差
#         try:
#             ax2 = plt.subplot(132)
#             error = np.abs(Y_pred - Y_true)
#             im = ax2.imshow(error, extent=[-2, 2, -2, 2], origin='lower', cmap='hot')
#             plt.colorbar(im, ax=ax2)
#             ax2.set_xlabel('X1')
#             ax2.set_ylabel('X2')
#             ax2.set_title('预测误差热力图')
            
#         except Exception as e:
#             print("绘制误差图时出错：", str(e))
        
#         # 3. 绘制特征重要性
#         try:
#             ax3 = plt.subplot(133)
#             feature_scores = model.feature_score
#             if feature_scores is not None:
#                 bars = ax3.bar(['X1\n(sin项和二次项)', 'X2\n(线性项)'], feature_scores)
#                 ax3.set_title('特征重要性分析')
#                 ax3.set_ylabel('重要性得分')
                
#                 # 添加数值标签
#                 for bar in bars:
#                     height = bar.get_height()
#                     ax3.text(bar.get_x() + bar.get_width()/2., height,
#                             f'{height:.3f}',
#                             ha='center', va='bottom')
#             else:
#                 print("无法获取特征重要性得分")
                
#         except Exception as e:
#             print("绘制特征重要性图时出错：", str(e))
        
#         plt.tight_layout()
#         plt.show()
#         print("可视化完成！")
        
#     except Exception as e:
#         print("可视化过程中出现错误：", str(e))

# def main():
#     try:
#         # 生成数据
#         print("开始生成数据...")
#         dataset = create_dataset()
        
#         # 训练模型
#         model = train_model(dataset)
        
#         # 分析模型
#         analyze_model(model, dataset)
        
#         # 可视化结果
#         plot_results(dataset, model)
        
#     except Exception as e:
#         print("程序执行过程中出现错误：", str(e))

# if __name__ == "__main__":
#     main() 


import matplotlib
# 设置后端，优先使用TkAgg
try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        matplotlib.use('Agg')  # 如果没有GUI后端，使用Agg
import matplotlib.pyplot as plt

from kan import *
import torch

# 检查并打印当前后端
print(f"使用的Matplotlib后端: {matplotlib.get_backend()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用的设备: {device}")

# 创建并配置图形
plt.ion()  # 打开交互模式
model = KAN(width=[2,3,2,1], noise_scale=0.3, device=device)
x = torch.normal(0,1,size=(100,2)).to(device)
model(x)
model = model.prune()

# 创建新的图形窗口
plt.figure(figsize=(10, 8))
model.plot(beta=100)

# 确保图形显示
plt.draw()
input("按回车键继续...")  # 等待用户输入防止图形窗口立即关闭
plt.close()