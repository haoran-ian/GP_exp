import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors

# 假设你的原始数据名为 original_data
# original_data.shape = (2776, 18)

def process_data_to_matrices(original_data):
    """
    将原始数据转换为64x64矩阵序列
    """
    # 提取后16列
    last_16_columns = original_data[:, 2:18]  # 索引2-17对应后16列
    
    processed_matrices = []
    
    for i in range(len(original_data)):
        # 获取当前行的16个数据
        row_data = last_16_columns[i]
        
        # 重塑为4x4矩阵
        matrix_4x4 = row_data.reshape(4, 4)
        
        # 将每个元素重复为16x16矩阵然后组合成64x64矩阵
        final_matrix = np.zeros((64, 64))
        
        for row_idx in range(4):
            for col_idx in range(4):
                # 当前元素值
                element_value = matrix_4x4[row_idx, col_idx]
                
                # 创建16x16的块所有元素值相同
                block = np.full((16, 16), element_value)
                
                # 将块放入最终矩阵的对应位置
                start_row = row_idx * 16
                end_row = start_row + 16
                start_col = col_idx * 16
                end_col = start_col + 16
                
                final_matrix[start_row:end_row, start_col:end_col] = block
        
        processed_matrices.append(final_matrix)
    
    return np.array(processed_matrices)

def create_heatmap_gif(processed_matrices, output_filename='heatmap_animation.gif', 
                      fps=10, dpi=100, figsize=(8, 8)):
    # 设置中文字体（如果需要显示中文）
    # plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    # plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 计算全局颜色范围保持所有帧颜色一致
    vmin = np.min(processed_matrices)
    vmax = np.max(processed_matrices)
    
    # 初始化第一帧
    im = ax.imshow(processed_matrices[0], cmap='viridis', 
                   vmin=vmin, vmax=vmax, aspect='equal')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('value', rotation=270, labelpad=15)
    
    # 设置标题和坐标轴
    ax.set_title('0/{}'.format(len(processed_matrices)))
    # ax.set_xlabel('X 坐标')
    # ax.set_ylabel('Y 坐标')
    
    # 移除坐标轴刻度
    ax.set_xticks([])
    ax.set_yticks([])
    
    def update(frame):
        im.set_array(processed_matrices[frame])
        ax.set_title('{}/{}'.format(frame+1, len(processed_matrices)))
        return [im]
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(processed_matrices),
                        interval=1000/fps, blit=True)
    
    # 保存为GIF
    anim.save(output_filename, writer='pillow', fps=fps, dpi=dpi)
    
    plt.close(fig)
    
    return anim

# 高级版本：带有更多自定义选项
def create_advanced_heatmap_gif(processed_matrices, output_filename='advanced_heatmap.gif',
                               fps=15, dpi=150, figsize=(10, 8), 
                               colormap='plasma', show_progress=True):
    """
    高级版本更多自定义选项的热图GIF
    """
    # plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    # plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 计算颜色范围
    vmin = np.min(processed_matrices)
    vmax = np.max(processed_matrices)
    
    # 初始化
    im = ax.imshow(processed_matrices[0], cmap=colormap, 
                   vmin=vmin, vmax=vmax, aspect='equal')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('value', rotation=270, labelpad=20, fontsize=12)
    
    # 美化图形
    ax.set_title('1/{}'.format(len(processed_matrices)), 
                fontsize=14, pad=20)
    # ax.set_xlabel('', fontsize=10)
    # ax.set_ylabel('', fontsize=10)
    
    # 添加网格线显示4x4结构
    for i in range(1, 4):
        ax.axhline(y=i*16-0.5, color='white', linewidth=1, alpha=0.5)
        ax.axvline(x=i*16-0.5, color='white', linewidth=1, alpha=0.5)
    
    def update(frame):
        im.set_array(processed_matrices[frame])
        ax.set_title('{}/{}'.format(frame+1, len(processed_matrices)), 
                    fontsize=14, pad=20)
        
        if show_progress and (frame+1) % 100 == 0:
            print(f"{frame+1}/{len(processed_matrices)}")
        
        return [im]
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(processed_matrices),
                        interval=1000/fps, blit=True)
    
    # 保存GIF
    anim.save(output_filename, writer='pillow', fps=fps, dpi=dpi)
    
    plt.close(fig)
    return anim

# 主执行函数
def main(original_data):
    # 假设你的数据已经加载为 original_data
    # original_data = np.load('your_data.npy')  # 或者你的数据加载方式
    
    # 这里用随机数据演示（替换为你的实际数据）
    print("创建演示数据...")
    # original_data = np.random.randn(2776, 18)
    
    print("处理数据为64x64矩阵...")
    # 处理数据
    processed_matrices = process_data_to_matrices(original_data)
    print(f"处理完成生成 {len(processed_matrices)} 个64x64矩阵")
    
    # 创建基础版本GIF
    print("\n创建基础版本GIF...")
    create_heatmap_gif(processed_matrices, 
                      output_filename='basic_heatmap.gif',
                      fps=10, figsize=(8, 8))
    
    # 创建高级版本GIF
    print("\n创建高级版本GIF...")
    create_advanced_heatmap_gif(processed_matrices[:500],  # 只使用前500帧演示避免文件过大
                               output_filename='advanced_heatmap.gif',
                               fps=15, figsize=(10, 8),
                               colormap='coolwarm')
    
    print("\n所有GIF生成完成")

# 如果只需要生成部分帧的GIF（避免文件过大）
def create_partial_gif(original_data, start_frame=0, num_frames=100, 
                      output_filename='partial_heatmap.gif'):
    """
    生成部分帧的GIF避免文件过大
    """
    # 处理指定范围的数据
    partial_data = original_data[start_frame:start_frame+num_frames]
    processed_matrices = process_data_to_matrices(partial_data)
    
    # 创建GIF
    create_advanced_heatmap_gif(processed_matrices, 
                               output_filename=output_filename,
                               fps=10, figsize=(10, 8),
                               colormap='viridis')

if __name__ == "__main__":
    runs = []
    data_path = "problems_ela_compare/data_f1122_regressor_p_diff/IOHprofiler_f1122_DIM16.dat"
    f = open(data_path, "r")
    lines = f.readlines()
    run = []
    for line in lines:
        if line[0] == "e":
            runs += [np.array(run)]
            run = []
        else:
            record = line[:-1].split(" ")
            for i in range(len(record)):
                record[i] = float(record[i])
            run += [record]
    runs += [np.array(run)]
    runs = runs[1:]
    original_data = runs[0]
    main(original_data)
    # 如果只想生成部分帧：
    create_partial_gif(original_data, start_frame=0, num_frames=200, 
                      output_filename='partial_animation.gif')