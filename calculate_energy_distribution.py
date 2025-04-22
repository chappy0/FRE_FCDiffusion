# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from torchvision import transforms
# from tools.dct_util import dct_2d, high_pass, low_pass, low_pass_and_shuffle

# def load_image(image_path, transform=None):
#     """
#     加载图像并转换为 Tensor。
#     """
#     image = Image.open(image_path).convert('RGB')
#     if transform is None:
#         transform = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#         ])
#     return transform(image).unsqueeze(0)  # 添加 batch 维度

# def extract_dct_features(image_tensor, control_mode='low_pass', threshold_low=30, threshold_mid=20, threshold_high=50):
#     """
#     对输入图像提取指定控制模式下的 DCT 特征。
#     控制模式:
#         - low_pass: 低频信息
#         - mini_pass: 低频信息并随机打乱部分系数
#         - mid_pass: 先低通再高通（提取中频）
#         - high_pass: 高频信息
#     """
#     # 转换为灰度图（对每个通道求均值）
#     image_gray = torch.mean(image_tensor, dim=1, keepdim=True)
#     # 计算二维 DCT 变换
#     # dct_features = dct_2d(image_gray.squeeze(0), norm='ortho').cuda()
#     dct_features = dct_2d(image_gray, norm='ortho').cuda()
    
#     if control_mode == 'low_pass':
#         filtered_dct = low_pass(dct_features, threshold_low)
#     elif control_mode == 'mini_pass':
#         filtered_dct = low_pass_and_shuffle(dct_features, threshold_low)
#     elif control_mode == 'mid_pass':
#         filtered_dct = high_pass(low_pass(dct_features, threshold_low), threshold_mid)
#     elif control_mode == 'high_pass':
#         filtered_dct = high_pass(dct_features, threshold_high)
#     else:
#         filtered_dct = dct_features
        
#     return filtered_dct.cpu().numpy()

# def plot_folder_dct_energy_distribution(folder_path, modes=['low_pass', 'mini_pass', 'mid_pass', 'high_pass'],
#                                           threshold_low=30, threshold_mid=20, threshold_high=50):
#     """
#     对文件夹中的所有图片计算四种控制模式下的 DCT 频域能量分布，
#     并对每种模式下的能量分布取平均后绘图展示。
#     """
#     # 获取文件夹中所有图片路径（支持 jpg/png/jpeg 格式）
#     image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path)
#                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
#     if len(image_files) == 0:
#         print("文件夹中没有找到图片！")
#         return

#     # 用于累加每个模式的能量图，初始化累加数组
#     accum_energy = {mode: None for mode in modes}
#     count = 0

#     for img_path in image_files:
#         try:
#             image_tensor = load_image(img_path)
#         except Exception as e:
#             print(f"加载图片 {img_path} 失败: {e}")
#             continue

#         for mode in modes:
#             dct_features = extract_dct_features(image_tensor, control_mode=mode,
#                                                 threshold_low=threshold_low,
#                                                 threshold_mid=threshold_mid,
#                                                 threshold_high=threshold_high)
#             # 计算能量（模平方）
#             energy = np.abs(dct_features) ** 2
#             # 将低频移到中心，便于观察
#             energy_shifted = np.fft.fftshift(energy)
#             if accum_energy[mode] is None:
#                 accum_energy[mode] = energy_shifted
#             else:
#                 accum_energy[mode] += energy_shifted
#         count += 1

#     # 计算平均能量分布
#     avg_energy = {mode: (accum_energy[mode] / count) for mode in modes}

#     # 对数变换（便于显示能量差异）
#     avg_energy_log = {mode: np.log(avg_energy[mode] + 1) for mode in modes}

#     # 绘制平均能量分布图
#     titles = {
#         'low_pass': 'Low Pass',
#         'mini_pass': 'Mini Pass',
#         'mid_pass': 'Mid Pass',
#         'high_pass': 'High Pass'
#     }
    
#     fig, axes = plt.subplots(2, 2, figsize=(10, 10))
#     axes = axes.flatten()
    
#     for i, mode in enumerate(modes):
#         # axes[i].imshow(avg_energy_log[mode], cmap='jet')
#         axes[i].imshow(np.squeeze(avg_energy_log[mode]), cmap='jet')
#         axes[i].set_title(titles.get(mode, mode))
#         axes[i].axis('off')
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     # 替换为实际图片文件夹路径
#     folder_path = r"D:\paper\FCDiffusion_code-main\datasets\test_high_interpre"
#     plot_folder_dct_energy_distribution(folder_path)

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tools.dct_util import dct_2d, high_pass, low_pass, low_pass_and_shuffle

def load_image(image_path, transform=None):
    """
    加载图像并转换为 Tensor。
    """
    image = Image.open(image_path).convert('RGB')
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    return transform(image).unsqueeze(0)  # 添加 batch 维度

def extract_dct_features(image_tensor, control_mode='low_pass', threshold_low=30, threshold_mid=20, threshold_high=50):
    """
    对输入图像提取指定控制模式下的 DCT 特征。
    控制模式:
        - low_pass: 低频信息
        - mini_pass: 低频信息并随机打乱部分系数
        - mid_pass: 先低通再高通（提取中频）
        - high_pass: 高频信息
    """
    # 转换为灰度图（对每个通道求均值）
    image_gray = torch.mean(image_tensor, dim=1, keepdim=True)
    # 计算二维 DCT 变换（保持批量维度）
    dct_features = dct_2d(image_gray, norm='ortho').cuda()
    
    if control_mode == 'low_pass':
        filtered_dct = low_pass(dct_features, 30)
    elif control_mode == 'mini_pass':
        filtered_dct = low_pass_and_shuffle(dct_features, 10)
    elif control_mode == 'mid_pass':
        filtered_dct = high_pass(low_pass(dct_features, 20), 40)
    elif control_mode == 'high_pass':
        filtered_dct = high_pass(dct_features, 50)
    else:
        filtered_dct = dct_features
        
    return filtered_dct.cpu().numpy()

def radial_profile(data):
    """
    计算二维数据的径向平均分布。
    data: 2D numpy 数组
    返回：一维数组，代表各半径的平均能量
    """
    # 创建坐标网格
    y, x = np.indices(data.shape)
    center = np.array([(x.max() - x.min())/2.0, (y.max() - y.min())/2.0])
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r_int = r.astype(int)
    
    # 累加每个半径上的值并求平均
    tbin = np.bincount(r_int.ravel(), data.ravel())
    nr = np.bincount(r_int.ravel())
    radialprofile = tbin / (nr + 1e-10)
    return radialprofile

def get_radial_energy_distribution(image_tensor, control_mode='low_pass',
                                   threshold_low=30, threshold_mid=20, threshold_high=50):
    """
    针对一张图像，在指定控制模式下计算径向能量分布。
    返回：一维能量曲线（横轴代表频率半径）
    """
    dct_features = extract_dct_features(image_tensor, control_mode, threshold_low, threshold_mid, threshold_high)
    energy = np.abs(dct_features)**2
    # fftshift 后中心区域为低频
    energy_shifted = np.fft.fftshift(energy)
    radial_energy = radial_profile(energy_shifted.squeeze())
    return radial_energy

def plot_folder_radial_energy_distribution(folder_path, modes=['low_pass', 'mini_pass', 'mid_pass', 'high_pass'],
                                           threshold_low=30, threshold_mid=20, threshold_high=50):
    """
    对文件夹中所有图片，在各控制模式下计算径向能量分布，求平均后绘制能量曲线。
    """
    # 获取所有图片路径
    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path)
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    if len(image_files) == 0:
        print("文件夹中没有找到图片！")
        return

    # 为每个模式累加径向能量曲线
    accum_profiles = {mode: None for mode in modes}
    count = 0

    for img_path in image_files:
        try:
            image_tensor = load_image(img_path)
        except Exception as e:
            print(f"加载图片 {img_path} 失败: {e}")
            continue

        for mode in modes:
            radial_energy = get_radial_energy_distribution(image_tensor, control_mode=mode,
                                                           threshold_low=threshold_low,
                                                           threshold_mid=threshold_mid,
                                                           threshold_high=threshold_high)
            # 累加时注意不同图片的径向长度可能不一致，这里取最短的长度
            if accum_profiles[mode] is None:
                accum_profiles[mode] = radial_energy
            else:
                min_len = min(len(accum_profiles[mode]), len(radial_energy))
                accum_profiles[mode] = accum_profiles[mode][:min_len] + radial_energy[:min_len]
        count += 1

    avg_profiles = {mode: (accum_profiles[mode] / count) for mode in modes}

    # 绘图
    plt.figure(figsize=(10, 6))
    for mode in modes:
        plt.plot(avg_profiles[mode], label=mode)
    plt.xlabel("Frequency radius (a.u.)")
    plt.ylabel("Average energy")
    plt.title("Average Radial Energy Distribution")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 替换为实际图片文件夹路径
    folder_path = r"D:\paper\FCDiffusion_code-main\datasets\test_mini_interpre"
    plot_folder_radial_energy_distribution(folder_path)
