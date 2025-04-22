# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns

# # # 数据
# # data = {
# #     'low': {
# #         'test_mini': -0.0572,
# #         'test_low': -0.0584,
# #         'test_mid': -0.0065,
# #         'test_high': 0.0166
# #     },
# #     'mini': {
# #         'test_mini': -0.0910,
# #         'test_low': -0.0522,
# #         'test_mid': -0.0026,
# #         'test_high': 0.0190
# #     },
# #     'mid': {
# #         'test_mini': 2.0963,
# #         'test_low': 0.5250,
# #         'test_mid': -0.1194,
# #         'test_high': -0.0063
# #     },
# #     'high': {
# #         'mini': -0.2604,
# #         'low': -0.0453,
# #         'mid': 0.0567,
# #         'high': 1.1685
# #     }
# # }

# # # 颜色映射
# # cmap = 'plasma'

# # # 生成每个频带的热力图
# # band_labels = {
# #     'low': 'Low-Pass Control (ΔE_low)',
# #     'mini': 'Mini-Pass Control (ΔE_mini)',
# #     'mid': 'Mid-Pass Control (ΔE_mid)',
# #     'high': 'High-Pass Control (ΔE_high)'
# # }

# # for band in data:
# #     values = list(data[band].values())
# #     operations = list(data[band].keys())
    
# #     # 创建一个矩阵，其中每个操作对应一个行，每个频带对应一个列
# #     matrix = np.array(values).reshape(len(operations), 1)
    
# #     # 创建热力图
# #     plt.figure(figsize=(10, 8))
# #     ax = sns.heatmap(matrix, annot=True, cmap=cmap, xticklabels=[band], yticklabels=operations,
# #                      cbar=True, cbar_kws={'label': 'Energy Difference Ratio (ΔE)'})
    
# #     # 设置标题
# #     ax.set_title(band_labels[band])
# #     ax.set_xlabel('Frequency Band')
# #     ax.set_ylabel('Operations')
    
# #     # 调整颜色条的位置和大小
# #     cbar = ax.collections[0].colorbar
# #     cbar.ax.tick_params(labelsize=8)
# #     cbar.set_label('Energy Difference Ratio (ΔE)', fontsize=8)
    
# #     # 保存热力图
# #     plt.savefig(f'{band}_frequency_heatmap.png', dpi=300, bbox_inches='tight')
# #     plt.close()



# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 数据
# data = {
#     'low': {
#         'mini': 0.0184,
#         'low': 0.0111,
#         'mid': 0.0052,
#         'high': -0.0024
#     },
#     'mini': {
#         'mini': 0.0525,
#         'low': 0.0185,
#         'mid': 0.0137,
#         'high': 0.0289
#     },
#     'mid': {
#         'mini': -0.9478,
#         'low': -0.4454,
#         'mid': -0.1109,
#         'high': -0.8976
#     },
#     'high': {
#         'mini': -0.2604,
#         'low': -0.0453,
#         'mid': 0.0567,
#         'high': 1.1685
#     }
# }

# # 颜色映射
# cmap = 'plasma'

# # 生成每个频带的热力图
# band_labels = {
#     'low': 'Low-Pass Control (ΔE_low)',
#     'mini': 'Mini-Pass Control (ΔE_mini)',
#     'mid': 'Mid-Pass Control (ΔE_mid)',
#     'high': 'High-Pass Control (ΔE_high)'
# }

# for band in data:
#     values = list(data[band].values())
#     operations = list(data[band].keys())
    
#     # 创建一个矩阵，其中每个操作对应一个行，每个频带对应一个列
#     matrix = np.array(values).reshape(len(operations), 1)
    
#     # 创建热力图
#     plt.figure(figsize=(10, 8))
#     ax = sns.heatmap(matrix, annot=True, cmap=cmap, xticklabels=[band], yticklabels=operations,
#                      cbar=True, cbar_kws={'label': 'Energy Difference Ratio (ΔE)'})
    
#     # 设置标题
#     ax.set_title(band_labels[band])
#     ax.set_xlabel('Frequency Band')
#     ax.set_ylabel('Operations')
    
#     # 调整颜色条的位置和大小
#     cbar = ax.collections[0].colorbar
#     cbar.ax.tick_params(labelsize=8)
#     cbar.set_label('Energy Difference Ratio (ΔE)', fontsize=8)
    
#     # 保存热力图
#     plt.savefig(f'{band}_frequency_heatmap.png', dpi=300, bbox_inches='tight')
#     plt.close()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 数据
data = {
    'low': {
        'mini': 0.0184,
        'low': 0.0111,
        'mid': 0.0052,
        'high': -0.0024
    },
    'mini': {
        'mini': 0.0525,
        'low': 0.0185,
        'mid': 0.0137,
        'high': 0.0289
    },
    'mid': {
        'mini': -0.9478,
        'low': -0.4454,
        'mid': -0.1109,
        'high': -0.8976
    },
    'high': {
        'mini': -0.2604,
        'low': -0.0453,
        'mid': 0.0567,
        'high': 1.1685
    }
}

# 颜色映射
cmap = 'plasma'

# 生成每个频带的热力图
band_labels = {
    'low': 'Low-Pass Control (ΔE_low)',
    'mini': 'Mini-Pass Control (ΔE_mini)',
    'mid': 'Mid-Pass Control (ΔE_mid)',
    'high': 'High-Pass Control (ΔE_high)'
}

for band in data:
    values = list(data[band].values())
    operations = list(data[band].keys())
    
    # 创建一个矩阵，其中每个操作对应一个行，每个频带对应一个列
    matrix = np.array(values).reshape(len(operations), 1)
    
    # 创建热力图
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(matrix, annot=True, annot_kws={"size": 26}, cmap=cmap, xticklabels=[band], yticklabels=operations,
                     cbar=True, cbar_kws={'label': 'Energy Difference Ratio (ΔE)'})
    
    # 设置标题
    ax.set_title(band_labels[band], fontsize=26)
    ax.set_xlabel('Frequency Band', fontsize=20)
    ax.set_ylabel('Operations', fontsize=20)
    
    # 调整颜色条的位置和大小
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)
    cbar.set_label('Energy Difference Ratio (ΔE)', fontsize=22)
    
    # 调整 x 轴和 y 轴的刻度标签字体大小
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    # 保存热力图
    plt.savefig(f'{band}_frequency_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()