# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines

# # --- 数据定义 ---

# # 图1: Low-Frequency Task 的数据
# data_low = {
#     'DiffuseIT':         {'x': 0.28,  'y': 0.94,  'color': 'blue'},
#     'Plug-and-Play':     {'x': 0.339, 'y': 0.959, 'color': 'red'},
#     'Prompt-to-Prompt':  {'x': 0.352, 'y': 0.966, 'color': 'green'},
#     'VQGAN-CLIP':        {'x': 0.379, 'y': 0.91,  'color': 'purple'},
#     'SDEdit(0.5)':       {'x': 0.321, 'y': 0.95,  'color': 'orange'},
#     'SDEdit(0.85)':      {'x': 0.36,  'y': 0.921, 'color': 'cyan'},
#     'Text2Live':         {'x': 0.3,   'y': 0.956, 'color': 'magenta'},
#     'InstructPix2Pix':   {'x': 0.319, 'y': 0.961, 'color': 'brown'},
#     'FCDiffusion':       {'x': 0.358, 'y': 0.952, 'color': '#CCCC00'}, # Yellow
#     'DKD-low(ours)':     {'x': 0.336, 'y': 0.932, 'color': 'black', 'size': 80, 'weight': 'bold'}
# }

# # 图2: High-Frequency Task 的数据
# data_high = {
#     'DiffuseIT':         {'x': 0.26,  'y': 0.95,  'color': 'blue'},
#     'Plug-and-Play':     {'x': 0.318, 'y': 0.957, 'color': 'red'},
#     'Prompt-to-Prompt':  {'x': 0.278, 'y': 0.95,  'color': 'green'},
#     'VQGAN-CLIP':        {'x': 0.361, 'y': 0.925, 'color': 'purple'},
#     'SDEdit(0.5)':       {'x': 0.262, 'y': 0.954, 'color': 'orange'},
#     'SDEdit(0.85)':      {'x': 0.358, 'y': 0.924, 'color': 'cyan'},
#     'Text2Live':         {'x': 0.298, 'y': 0.967, 'color': 'magenta'},
#     'InstructPix2Pix':   {'x': 0.27,  'y': 0.963, 'color': 'brown'},
#     'FCDiffusion':       {'x': 0.329, 'y': 0.965, 'color': '#CCCC00'}, # Yellow
#     'DKD-high(ours)':    {'x': 0.321, 'y': 0.943, 'color': 'black', 'size': 80, 'weight': 'bold'}
# }

# # 图3: Mini-Frequency Task 的数据
# data_mini = {
#     'Prompt-to-Prompt':  {'x': 0.28,  'y': 0.068, 'color': 'green'},
#     'VQGAN-CLIP':        {'x': 0.362, 'y': 0.044, 'color': 'purple'},
#     'SDEdit(0.5)':       {'x': 0.298, 'y': 0.102, 'color': 'orange'},
#     'SDEdit(0.85)':      {'x': 0.318, 'y': 0.012, 'color': 'cyan'},
#     'Text2Live':         {'x': 0.271, 'y': 0.053, 'color': 'magenta'},
#     'InstructPix2Pix':   {'x': 0.3,   'y': 0.082, 'color': 'brown'},
#     'FCDiffusion':       {'x': 0.306, 'y': 0.075, 'color': '#CCCC00'}, # Yellow
#     'DKD-mini(ours)':    {'x': 0.337, 'y': 0.137, 'color': 'black', 'size': 80, 'weight': 'bold'}
# }

# # 通用绘图函数
# def create_scatter_plot(data, title, xlabel, ylabel, filename):
#     """
#     根据输入的数据和配置生成散点图。
#     """
#     fig, ax = plt.subplots(figsize=(8, 3))

#     # 绘制每个数据点
#     for label, values in data.items():
#         ax.scatter(
#             values['x'], values['y'],
#             c=values['color'],
#             s=values.get('size', 40), # 默认大小为40，突出点使用自定义大小
#             label=label,
#             marker='o',
#             edgecolors='w' if values.get('weight') != 'bold' else 'black', # 让我们的点更突出
#             linewidths=1
#         )

#     # 设置图表标题和坐标轴标签
#     ax.set_title(title, fontsize=14, fontweight='bold')
#     ax.set_xlabel(xlabel, fontsize=12)
#     ax.set_ylabel(ylabel, fontsize=12)

#     # 设置网格
#     ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

#     # 创建并设置图例
#     # 为了在图例中实现加粗效果，我们手动创建图例句柄
#     legend_handles = []
#     for label, values in data.items():
#         handle = mlines.Line2D([], [], color=values['color'], marker='o', linestyle='None',
#                                markersize=8, label=label)
#         legend_handles.append(handle)
        
#     legend = ax.legend(handles=legend_handles, title='Models', bbox_to_anchor=(1.03, 1), loc='upper left')
    
#     # 将图例中的 'ours' 标签设置为粗体
#     for text in legend.get_texts():
#         if 'ours' in text.get_text():
#             text.set_fontweight('bold')

#     # 调整布局以防止标签被裁剪
#     plt.tight_layout(rect=[0, 0, 0.85, 1]) # 为图例留出空间

#     # 保存图像
#     plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
#     print(f"图像已保存为 {filename}.png")

#     # 显示图像
#     plt.show()


# # --- 生成三个图 ---

# # 1. 生成 Low-Frequency 图
# create_scatter_plot(
#     data=data_low,
#     title='Quantitative comparison for the low-frequency task',
#     xlabel='Text-Image Similarity',
#     ylabel='Structure Similarity',
#     filename='low_frequency_comparison'
# )

# # 2. 生成 High-Frequency 图
# create_scatter_plot(
#     data=data_high,
#     title='Quantitative comparison for the high-frequency style translation task',
#     xlabel='Text-Image Similarity',
#     ylabel='Structure Similarity',
#     filename='high_frequency_comparison'
# )

# # 3. 生成 Mini-Frequency 图
# create_scatter_plot(
#     data=data_mini,
#     title='Quantitative comparison for the mini-frequency content creation task',
#     xlabel='Text-Image Similarity',
#     ylabel='Structure Distance',
#     filename='mini_frequency_comparison'
# )

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# --- 字体设置 (用于正确显示中文) ---
# 确保你的系统中有支持中文的字体，例如 'SimHei', 'Microsoft YaHei' 等
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


# --- 数据定义 (与之前相同) ---

# 图1: Low-Frequency Task 的数据
data_low = {
    'DiffuseIT':         {'x': 0.28,  'y': 0.94,  'color': 'blue'},
    'Plug-and-Play':     {'x': 0.339, 'y': 0.959, 'color': 'red'},
    'Prompt-to-Prompt':  {'x': 0.352, 'y': 0.966, 'color': 'green'},
    'VQGAN-CLIP':        {'x': 0.379, 'y': 0.91,  'color': 'purple'},
    'SDEdit(0.5)':       {'x': 0.321, 'y': 0.95,  'color': 'orange'},
    'SDEdit(0.85)':      {'x': 0.36,  'y': 0.921, 'color': 'cyan'},
    'Text2Live':         {'x': 0.3,   'y': 0.956, 'color': 'magenta'},
    'InstructPix2Pix':   {'x': 0.319, 'y': 0.961, 'color': 'brown'},
    'FCDiffusion':       {'x': 0.358, 'y': 0.952, 'color': '#CCCC00'}, # Yellow
    'DKD-low(ours)':     {'x': 0.336, 'y': 0.932, 'color': 'black', 'size': 80, 'weight': 'bold'}
}

# 图2: High-Frequency Task 的数据
data_high = {
    'DiffuseIT':         {'x': 0.26,  'y': 0.95,  'color': 'blue'},
    'Plug-and-Play':     {'x': 0.318, 'y': 0.957, 'color': 'red'},
    'Prompt-to-Prompt':  {'x': 0.278, 'y': 0.95,  'color': 'green'},
    'VQGAN-CLIP':        {'x': 0.361, 'y': 0.925, 'color': 'purple'},
    'SDEdit(0.5)':       {'x': 0.262, 'y': 0.954, 'color': 'orange'},
    'SDEdit(0.85)':      {'x': 0.358, 'y': 0.924, 'color': 'cyan'},
    'Text2Live':         {'x': 0.298, 'y': 0.967, 'color': 'magenta'},
    'InstructPix2Pix':   {'x': 0.27,  'y': 0.963, 'color': 'brown'},
    'FCDiffusion':       {'x': 0.329, 'y': 0.965, 'color': '#CCCC00'}, # Yellow
    'DKD-high(ours)':    {'x': 0.321, 'y': 0.943, 'color': 'black', 'size': 80, 'weight': 'bold'}
}

# 图3: Mini-Frequency Task 的数据
data_mini = {
    'Prompt-to-Prompt':  {'x': 0.28,  'y': 0.068, 'color': 'green'},
    'VQGAN-CLIP':        {'x': 0.362, 'y': 0.044, 'color': 'purple'},
    'SDEdit(0.5)':       {'x': 0.298, 'y': 0.102, 'color': 'orange'},
    'SDEdit(0.85)':      {'x': 0.318, 'y': 0.012, 'color': 'cyan'},
    'Text2Live':         {'x': 0.271, 'y': 0.053, 'color': 'magenta'},
    'InstructPix2Pix':   {'x': 0.3,   'y': 0.082, 'color': 'brown'},
    'FCDiffusion':       {'x': 0.306, 'y': 0.075, 'color': '#CCCC00'}, # Yellow
    'DKD-mini(ours)':    {'x': 0.337, 'y': 0.137, 'color': 'black', 'size': 80, 'weight': 'bold'}
}

# 通用绘图函数
def create_scatter_plot(data, title, xlabel, ylabel, filename):
    """
    根据输入的数据和配置生成散点图。
    """
    # --- 修改点: 调整 figsize 的高度为 3 ---
    # 将宽度调整为6，以保持合理的宽高比
    fig, ax = plt.subplots(figsize=(8, 3))

    # 绘制每个数据点
    for label, values in data.items():
        ax.scatter(
            values['x'], values['y'],
            c=values['color'],
            s=values.get('size', 40),
            label=label,
            marker='o',
            edgecolors='w' if values.get('weight') != 'bold' else 'black',
            linewidths=1
        )

    # 设置图表标题和坐标轴标签
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # 设置网格
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    # 创建并设置图例
    legend_handles = []
    for label, values in data.items():
        handle = mlines.Line2D([], [], color=values['color'], marker='o', linestyle='None',
                               markersize=6, label=label)
        legend_handles.append(handle)
        
    legend = ax.legend(handles=legend_handles, title='Models', bbox_to_anchor=(1.03, 1), loc='upper left')
    
    for text in legend.get_texts():
        if 'ours' in text.get_text():
            text.set_fontweight('bold')

    # 调整布局以防止标签被裁剪
    plt.tight_layout(rect=[0, 0, 0.75, 1]) # 为图例留出空间

    # 保存图像
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    print(f"图像已保存为 {filename}.png")

    # 显示图像
    plt.show()


# --- 生成三个图 (使用中文标题) ---

# 1. 生成 Low-Frequency 图
create_scatter_plot(
    data=data_low,
    title='低频任务量化比较', # <-- 修改点
    xlabel='文本-图像相似度',
    ylabel='结构相似度',
    filename='low_frequency_comparison_zh'
)

# 2. 生成 High-Frequency 图
create_scatter_plot(
    data=data_high,
    title='高频风格转换任务量化比较', # <-- 修改点
    xlabel='文本-图像相似度',
    ylabel='结构相似度',
    filename='high_frequency_comparison_zh'
)

# 3. 生成 Mini-Frequency 图
create_scatter_plot(
    data=data_mini,
    title='微频内容创作任务量化比较', # <-- 修改点
    xlabel='文本-图像相似度',
    ylabel='结构距离',
    filename='mini_frequency_comparison_zh'
)