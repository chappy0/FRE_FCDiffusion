# import os
# import torch
# import numpy as np
# from scipy.stats import entropy
# from PIL import Image
# from torchvision import transforms
# from tools.dct_util import dct_2d, high_pass, idct_2d, low_pass, low_pass_and_shuffle

# def compute_kl_divergence(p, q):
#     """
#     Computes the Kullback-Leibler divergence between two probability distributions p and q.
#     """
#     p = p + 1e-10  # To prevent log(0)
#     q = q + 1e-10  # To prevent log(0)
#     return np.sum(p * np.log(p / q), axis=-1)

# def js_divergence(p, q):
#     """
#     Computes the Jensen-Shannon divergence between two distributions p and q.
#     """
#     m = 0.5 * (p + q)
#     return 0.5 * (compute_kl_divergence(p, m) + compute_kl_divergence(q, m))

# # def compute_mode_distinction(features_mode1, features_mode2):
# #     """
# #     Computes the mode distinction (JS divergence) between features of two control modes.
# #     :param features_mode1: Features extracted from images generated with control mode 1.
# #     :param features_mode2: Features extracted from images generated with control mode 2.
# #     :return: JS divergence between the features of two control modes.
# #     """
# #     # Flatten the features to 1D for histogram computation
# #     hist1 = np.histogram(features_mode1.flatten(), bins=50, density=True)[0]
# #     hist2 = np.histogram(features_mode2.flatten(), bins=50, density=True)[0]
    
# #     # Compute JS Divergence
# #     js_div = js_divergence(hist1, hist2)
# #     return js_div

# def compute_mode_distinction(features_mode1, features_mode2):
#     """
#     Computes the mode distinction (JS divergence) between features of two control modes.
#     :param features_mode1: Features extracted from images generated with control mode 1.
#     :param features_mode2: Features extracted from images generated with control mode 2.
#     :return: JS divergence between the features of two control modes.
#     """
#     # Flatten the features to 1D for histogram computation
#     features_mode1 = features_mode1.flatten()
#     features_mode2 = features_mode2.flatten()
    
#     # Normalize features to [0, 1] range
#     min_val = min(features_mode1.min(), features_mode2.min())
#     max_val = max(features_mode1.max(), features_mode2.max())
#     features_mode1 = (features_mode1 - min_val) / (max_val - min_val + 1e-10)
#     features_mode2 = (features_mode2 - min_val) / (max_val - min_val + 1e-10)
    
#     # Compute histograms with consistent range
#     hist1, _ = np.histogram(features_mode1, bins=100, range=(0, 1), density=True)
#     hist2, _ = np.histogram(features_mode2, bins=100, range=(0, 1), density=True)
    
#     # Compute JS Divergence
#     js_div = js_divergence(hist1, hist2)
#     return js_div

# def load_image(image_path, transform=None):
#     """
#     Load an image from a file path and apply transformations.
#     :param image_path: Path to the image file.
#     :param transform: Optional transformations to apply.
#     :return: Image tensor.
#     """
#     image = Image.open(image_path).convert('RGB')
#     if transform is None:
#         transform = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#         ])
#     return transform(image).unsqueeze(0)  # Add batch dimension

# def extract_dct_features(image_tensor, control_mode='low_pass', threshold_low=30, threshold_mid=20, threshold_high=50):
#     """
#     Extracts DCT features from an image tensor based on the specified control mode.
#     :param image_tensor: Input image tensor.
#     :param control_mode: The control mode for frequency band extraction ('low_pass', 'mini_pass', 'mid_pass', 'high_pass').
#     :param threshold_low: The threshold for low-pass filtering.
#     :param threshold_mid: The threshold for mid-pass filtering.
#     :param threshold_high: The threshold for high-pass filtering.
#     :return: Extracted DCT features.
#     """
#     # Convert image to grayscale for DCT processing
#     image_gray = torch.mean(image_tensor, dim=1, keepdim=True)
    
#     # Apply DCT
#     dct_features = dct_2d(image_gray.squeeze(0), norm='ortho')
    
#     # Extract frequency bands
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
    
#     return filtered_dct.numpy()

# def compute_folder_js_divergence(generated_folder, target_folder, control_mode='low_pass', threshold_low=30, threshold_mid=20, threshold_high=50):
#     """
#     Compute the JS divergence for all images in two folders.
#     :param generated_folder: Path to the folder containing generated images.
#     :param target_folder: Path to the folder containing target images.
#     :param control_mode: The control mode for frequency band extraction.
#     :param threshold_low: The threshold for low-pass filtering.
#     :param threshold_mid: The threshold for mid-pass filtering.
#     :param threshold_high: The threshold for high-pass filtering.
#     :return: Average JS divergence for the specified control mode.
#     """
#     # # Get list of image files in both folders
#     # generated_files = os.listdir(generated_folder)
#     # target_files = os.listdir(target_folder)
    
#     # # Ensure the same number of images in both folders
#     # assert len(generated_files) == len(target_files), "Folders must contain the same number of images"
    
#     # Initialize average JS divergence
#     avg_js_div = 0.0
#     count = 0

#     images2 = sorted([os.path.join(generated_folder, f) for f in os.listdir(generated_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
#     # 获取文件夹2中的所有图片文件，并确保它们以 "re_" 开头并排序
#     images1 = sorted([os.path.join(target_folder, f) for f in os.listdir(target_folder) if f.endswith(('.jpg', '.png', '.jpeg') )])

#     results = []
#     for img1_path in images1:
#         base_img1 = os.path.splitext(os.path.basename(img1_path))[0]  # 去掉文件扩展名

#         # found_match = False
#         for img2_path in images2:
#             if os.path.splitext(os.path.basename(img2_path))[0] ==   base_img1:
#                 generated_image = load_image(img2_path)
#                 target_image = load_image(img1_path)
    
#     # # Process each pair of images
#     # for generated_file, target_file in zip(generated_files, target_files):
#     #     # Load images
#     #     generated_image = load_image(os.path.join(generated_folder, generated_file))
#     #     target_image = load_image(os.path.join(target_folder, target_file))
        
#                 # Extract DCT features
#                 generated_features = extract_dct_features(generated_image, control_mode, threshold_low, threshold_mid, threshold_high)
#                 target_features = extract_dct_features(target_image, control_mode, threshold_low, threshold_mid, threshold_high)
                
#                 # Compute JS divergence
#                 js_div = compute_mode_distinction(generated_features, target_features)
#                 print(f"js_div:{js_div}")
#                 avg_js_div += js_div
                
#                 count += 1
    
#     # Compute average JS divergence
#     if count > 0:
#         avg_js_div /= count
    
#     return avg_js_div


# def compute_mode_distinction_across_modes(image_tensor, modes=['low_pass', 'mini_pass', 'mid_pass', 'high_pass'],
#                                           threshold_low=30, threshold_mid=20, threshold_high=50):
#     """
#     对同一张图像计算不同频域控制模式之间的模式区分度（JS散度）。
#     对于每个控制模式，先提取其 DCT 特征，再计算任意两个模式之间的 JS 散度。
#     返回一个字典，键为 "mode1 vs mode2"，值为对应的 JS 散度。
#     """
#     # 分别提取各控制模式的 DCT 特征
#     features_dict = {}
#     for mode in modes:
#         features_dict[mode] = extract_dct_features(image_tensor, mode, threshold_low, threshold_mid, threshold_high)
    
#     # 计算两两控制模式之间的 JS 散度
#     divergences = {}
#     for i in range(len(modes)):
#         for j in range(i + 1, len(modes)):
#             mode1 = modes[i]
#             mode2 = modes[j]
#             js_div = compute_mode_distinction(features_dict[mode1], features_dict[mode2])
#             divergences[f"{mode1} vs {mode2}"] = js_div
#     return divergences

# # 举例：对单张图像计算四个频段的模式区分度
# if __name__ == "__main__":
#     # 读取单张图像
#     image_path = r"path/to/your/image.jpg"  # 替换为实际图片路径
#     image_tensor = load_image(image_path)
    
#     divergences = compute_mode_distinction_across_modes(image_tensor)
#     for pair, js_div in divergences.items():
#         print(f"{pair}: JS Divergence = {js_div:.4f}")


#     generated_folder = r"D:\paper\FCDiffusion_code-main\datasets\test_low_interpre"   # Path to generated images folder
#     target_folder = r"D:\paper\FCDiffusion_code-main\datasets\test_mid_interpre"        # Path to target images folder
#     control_mode = 'low_pass'  # Choose 'low_pass', 'mini_pass', 'mid_pass', or 'high_pass'
    
#     avg_js_div = compute_folder_js_divergence(generated_folder, target_folder, control_mode)
#     print(f"Average JS Divergence for {control_mode} frequency band: {avg_js_div}")


import os
import torch
import numpy as np
from scipy.stats import entropy
from PIL import Image
from torchvision import transforms
from tools.dct_util import dct_2d, high_pass, idct_2d, low_pass, low_pass_and_shuffle

def compute_kl_divergence(p, q):
    """
    Computes the Kullback-Leibler divergence between two probability distributions p and q.
    """
    p = p + 1e-10  # To prevent log(0)
    q = q + 1e-10  # To prevent log(0)
    return np.sum(p * np.log(p / q), axis=-1)

def js_divergence(p, q):
    """
    Computes the Jensen-Shannon divergence between two distributions p and q.
    """
    m = 0.5 * (p + q)
    return 0.5 * (compute_kl_divergence(p, m) + compute_kl_divergence(q, m))

def compute_mode_distinction(features_mode1, features_mode2):
    """
    Computes the mode distinction (JS divergence) between features of two control modes.
    对输入特征先归一化到 [0, 1]，再计算直方图后得到 JS 散度。
    :param features_mode1: Features extracted for control mode 1.
    :param features_mode2: Features extracted for control mode 2.
    :return: JS divergence between the features of two control modes.
    """
    # Flatten特征为1D
    features_mode1 = features_mode1.flatten()
    features_mode2 = features_mode2.flatten()
    
    # 归一化到 [0, 1]
    min_val = min(features_mode1.min(), features_mode2.min())
    max_val = max(features_mode1.max(), features_mode2.max())
    features_mode1 = (features_mode1 - min_val) / (max_val - min_val + 1e-10)
    features_mode2 = (features_mode2 - min_val) / (max_val - min_val + 1e-10)
    
    # 计算直方图，确保使用相同的 bins 和范围
    hist1, _ = np.histogram(features_mode1, bins=100, range=(0, 1), density=True)
    hist2, _ = np.histogram(features_mode2, bins=100, range=(0, 1), density=True)
    
    # 计算 JS 散度
    js_div = js_divergence(hist1, hist2)
    return js_div

def load_image(image_path, transform=None):
    """
    Load an image from a file path and apply transformations.
    :param image_path: Path to the image file.
    :param transform: Optional transformations to apply.
    :return: Image tensor with batch dimension.
    """
    image = Image.open(image_path).convert('RGB')
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# def extract_dct_features(image_tensor, control_mode='low_pass', threshold_low=30, threshold_mid=20, threshold_high=50):
#     """
#     Extracts DCT features from an image tensor based on the specified control mode.
#     :param image_tensor: Input image tensor.
#     :param control_mode: The control mode for frequency band extraction ('low_pass', 'mini_pass', 'mid_pass', 'high_pass').
#     :param threshold_low: The threshold for low-pass filtering.
#     :param threshold_mid: The threshold for mid-pass filtering.
#     :param threshold_high: The threshold for high-pass filtering.
#     :return: Extracted DCT features (numpy array).
#     """
#     # 转换为灰度图（对每个通道求均值）
#     image_gray = torch.mean(image_tensor, dim=1, keepdim=True)
    
#     # 计算 DCT 变换
#     dct_features = dct_2d(image_gray.squeeze(0), norm='ortho').cuda()
    
#     # 根据控制模式提取对应的频域信息
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

def extract_dct_features(image_tensor, control_mode='low_pass', threshold_low=30, threshold_mid=20, threshold_high=50):
    """
    Extracts DCT features from an image tensor based on the specified control mode.
    :param image_tensor: Input image tensor.
    :param control_mode: The control mode for frequency band extraction ('low_pass', 'mini_pass', 'mid_pass', 'high_pass').
    :param threshold_low: The threshold for low-pass filtering.
    :param threshold_mid: The threshold for mid-pass filtering.
    :param threshold_high: The threshold for high-pass filtering.
    :return: Extracted DCT features (numpy array).
    """
    # 将图像张量移动到 GPU 上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    
    # 计算 DCT 变换
    dct_features = dct_2d(image_tensor, norm='ortho')
    
    # 根据控制模式提取对应的频域信息
    if control_mode == 'low_pass':
        filtered_dct = low_pass(dct_features, threshold_low)
    elif control_mode == 'mini_pass':
        filtered_dct = low_pass_and_shuffle(dct_features, threshold_low)
    elif control_mode == 'mid_pass':
        filtered_dct = high_pass(low_pass(dct_features, threshold_low), threshold_mid)
    elif control_mode == 'high_pass':
        filtered_dct = high_pass(dct_features, threshold_high)
    else:
        filtered_dct = dct_features
    
    return filtered_dct.cpu().numpy()

def compute_mode_distinction_across_modes(image_tensor, modes=['low_pass', 'mini_pass', 'mid_pass', 'high_pass'],
                                          threshold_low=30, threshold_mid=20, threshold_high=50,control_mode='mid_pass'):
    """
    对同一张图像计算不同频域控制模式之间的模式区分度（JS散度）。
    对于每个控制模式，提取其 DCT 特征，再计算任意两个模式之间的 JS 散度。
    返回一个字典，键为 "mode1 vs mode2"，值为对应的 JS 散度。
    """
    # 提取各控制模式的特征
    features_dict = {}
    for mode in modes:
        features_dict[mode] = extract_dct_features(image_tensor, mode, threshold_low, threshold_mid, threshold_high)
    
    # 两两计算 JS 散度
    divergences = {}
    for i in range(len(modes)):
        # for j in range(i + 1, len(modes)):
        #     mode1 = modes[i]
            mode1 = control_mode
            mode2 = modes[i]
            js_div = compute_mode_distinction(features_dict[mode1], features_dict[mode2])
            divergences[f"{mode1} vs {mode2}"] = js_div
    return divergences

def compute_folder_mode_distinction(folder_path, modes=['low_pass', 'mini_pass', 'mid_pass', 'high_pass'],
                                    threshold_low=30, threshold_mid=20, threshold_high=50, control_mode= ''):
    """
    对文件夹中的每张图像计算不同频域控制模式之间的模式区分度（JS散度），
    并计算各模式对的平均 JS 散度。
    :param folder_path: 包含图片的文件夹路径。
    :return: 一个字典，键为 "mode1 vs mode2"，值为所有图片的平均 JS 散度。
    """
    # 获取文件夹中的所有图片路径（支持 jpg/png/jpeg 格式）
    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path)
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    # 用于累加每对模式的 JS 散度
    sum_divergences = {}
    count = 0
    
    for img_path in image_files:
        try:
            image_tensor = load_image(img_path)
        except Exception as e:
            print(f"加载图片 {img_path} 失败: {e}")
            continue
        
        divergences = compute_mode_distinction_across_modes(image_tensor, modes, threshold_low, threshold_mid, threshold_high,control_mode)
        for key, value in divergences.items():
            sum_divergences[key] = sum_divergences.get(key, 0) + value
        count += 1
    
    # 计算平均值
    avg_divergences = {key: value / count for key, value in sum_divergences.items()} if count > 0 else {}
    return avg_divergences

if __name__ == "__main__":
    # 示例：指定图片文件夹路径，对文件夹中所有图片计算不同频域控制模式之间的平均 JS 散度
    folder_path = r"D:\paper\FCDiffusion_code-main\datasets\test_mid"   # 替换为实际图片文件夹路径
    avg_divergences = compute_folder_mode_distinction(folder_path,control_mode='mid_pass')
    print("各频域控制模式之间的平均模式区分度（JS 散度）：")
    for pair, js_div in avg_divergences.items():
        print(f"{pair}: {js_div:.4f}")
