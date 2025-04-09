# import os
# import torch
# import numpy as np
# from scipy.stats import entropy
# from PIL import Image
# from torchvision import transforms
# from tools.dct_util import dct_2d, high_pass, idct_2d, low_pass, low_pass_and_shuffle

# def calculate_frequency_band_energy(dct_features, band, threshold_low=30, threshold_mid=20, threshold_high=50):
#     """
#     计算指定频带的能量。
#     :param dct_features: DCT 特征。
#     :param band: 频带范围（'low', 'mini', 'mid', 'high'）。
#     :param threshold_low: 低通滤波的阈值。
#     :param threshold_mid: 中通滤波的阈值。
#     :param threshold_high: 高通滤波的阈值。
#     :return: 指定频带的能量。
#     """
#     h, w = dct_features.shape[-2], dct_features.shape[-1]
#     vertical = torch.arange(0, h).reshape(-1, 1).repeat(1, w)
#     horizontal = torch.arange(0, w).reshape(1, -1).repeat(h, 1)
#     mask = vertical + horizontal

#     if band == 'low':
#         mask = mask <= threshold_low
#     elif band == 'mini':
#         mask_low = mask <= threshold_low
#         mask_mid = mask <= threshold_mid
#         mask = mask_low ^ mask_mid  # 低频到中频之间的区域
#     elif band == 'mid':
#         mask = (mask > threshold_low) & (mask <= threshold_mid)
#     elif band == 'high':
#         mask = mask > threshold_high
#     else:
#         raise ValueError("Invalid band. Choose 'low', 'mini', 'mid', or 'high'.")

#     band_mask = mask.to(dct_features.device)
#     band_energy = torch.sum(torch.abs(dct_features) ** 2 * band_mask)
#     total_energy = torch.sum(torch.abs(dct_features) ** 2)
#     return band_energy / total_energy if total_energy != 0 else 0

# def calculate_energy_difference_ratio(input_dct, output_dct, band, threshold_low=30, threshold_mid=20, threshold_high=50):
#     """
#     计算指定频带的能量差异比。
#     :param input_dct: 输入图像的 DCT 特征。
#     :param output_dct: 输出图像的 DCT 特征。
#     :param band: 频带范围。
#     :param threshold_low: 低通滤波的阈值。
#     :param threshold_mid: 中通滤波的阈值。
#     :param threshold_high: 高通滤波的阈值。
#     :return: 能量差异比。
#     """
#     input_energy = calculate_frequency_band_energy(input_dct, band, threshold_low, threshold_mid, threshold_high)
#     output_energy = calculate_frequency_band_energy(output_dct, band, threshold_low, threshold_mid, threshold_high)
#     return (output_energy - input_energy) / input_energy if input_energy != 0 else 0

# def evaluate_frequency_band_energy(original_folder, generated_folders, band='low', threshold_low=30, threshold_mid=20, threshold_high=50):
#     """
#     评估不同操作输出图像与原始图像在指定频带的能量差异比。
#     :param original_folder: 原始图像文件夹。
#     :param generated_folders: 不同操作输出图像的文件夹列表。
#     :param band: 频带范围。
#     :param threshold_low: 低通滤波的阈值。
#     :param threshold_mid: 中通滤波的阈值。
#     :param threshold_high: 高通滤波的阈值。
#     :return: 各操作在指定频带的能量差异比。
#     """
#     original_files = sorted([os.path.join(original_folder, f) for f in os.listdir(original_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
#     results = {}

#     for generated_folder in generated_folders:
#         generated_files = sorted([os.path.join(generated_folder, f) for f in os.listdir(generated_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
#         avg_delta_e = 0.0
#         count = 0

#         for original_file, generated_file in zip(original_files, generated_files):
#             original_image = load_image(original_file)
#             generated_image = load_image(generated_file)

#             original_dct = dct_2d(original_image.squeeze(0), norm='ortho')
#             generated_dct = dct_2d(generated_image.squeeze(0), norm='ortho')

#             delta_e = calculate_energy_difference_ratio(original_dct, generated_dct, band, threshold_low, threshold_mid, threshold_high)
#             avg_delta_e += delta_e
#             count += 1

#         if count > 0:
#             avg_delta_e /= count
#         results[os.path.basename(generated_folder)] = avg_delta_e

#     return results

# def calculate_spectral_similarity(spectrum1, spectrum2):
#     """
#     计算两个频谱的余弦相似度。
#     :param spectrum1: 第一个频谱。
#     :param spectrum2: 第二个频谱。
#     :return: 余弦相似度。
#     """
#     spectrum1_flat = spectrum1.flatten()
#     spectrum2_flat = spectrum2.flatten()
#     dot_product = np.dot(spectrum1_flat, spectrum2_flat)
#     norm1 = np.linalg.norm(spectrum1_flat)
#     norm2 = np.linalg.norm(spectrum2_flat)
#     return dot_product / (norm1 * norm2 + 1e-10)

# def evaluate_spectral_similarity(folder1, folder2):
#     """
#     评估两个文件夹中图像的频谱相似性。
#     :param folder1: 第一个文件夹。
#     :param folder2: 第二个文件夹。
#     :return: 平均频谱相似性。
#     """
#     files1 = sorted([os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('.jpg', '.png', '.jpeg'))])
#     files2 = sorted([os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('.jpg', '.png', '.jpeg'))])
#     avg_similarity = 0.0
#     count = 0

#     for file1, file2 in zip(files1, files2):
#         image1 = load_image(file1)
#         image2 = load_image(file2)

#         dct1 = dct_2d(image1.squeeze(0), norm='ortho').cpu().numpy()
#         dct2 = dct_2d(image2.squeeze(0), norm='ortho').cpu().numpy()

#         similarity = calculate_spectral_similarity(dct1, dct2)
#         avg_similarity += similarity
#         count += 1

#     if count > 0:
#         avg_similarity /= count
#     return avg_similarity

# def evaluate_across_operations_similarity(generated_folders):
#     """
#     评估不同操作输出图像之间的频谱相似性。
#     :param generated_folders: 不同操作输出图像的文件夹列表。
#     :return: 各操作对之间的平均频谱相似性。
#     """
#     similarities = {}
#     num_folders = len(generated_folders)

#     for i in range(num_folders):
#         for j in range(i + 1, num_folders):
#             folder1 = generated_folders[i]
#             folder2 = generated_folders[j]
#             similarity = evaluate_spectral_similarity(folder1, folder2)
#             key = f"{os.path.basename(folder1)} vs {os.path.basename(folder2)}"
#             similarities[key] = similarity

#     return similarities


# def load_image(image_path, transform=None):
#     """
#     Load an image from a file path and apply transformations.
#     :param image_path: Path to the image file.
#     :param transform: Optional transformations to apply.
#     :return: Image tensor with batch dimension.
#     """
#     image = Image.open(image_path).convert('RGB')
#     if transform is None:
#         transform = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#         ])
#     return transform(image).unsqueeze(0)  # Add batch dimension

# if __name__ == "__main__":
#     original_folder = r"D:\paper\FCDiffusion_code-main\datasets\original_images"  # 原始图像文件夹
#     generated_folders = [
#         r"D:\paper\FCDiffusion_code-main\datasets\test_mini_interpre",
#         r"D:\paper\FCDiffusion_code-main\datasets\test_low_interpre",  # 低通操作输出
#         r"D:\paper\FCDiffusion_code-main\datasets\test_mid_interpre",  # 中通操作输出
#         r"D:\paper\FCDiffusion_code-main\datasets\test_high_interpre"  # 高通操作输出
#     ]

#     # 评估频带能量对比
#     bands = ['low', 'mini', 'mid', 'high']
#     thresholds = {'threshold_low': 30, 'threshold_mid': 20, 'threshold_high': 50}
    
#     for band in bands:
#         energy_results = evaluate_frequency_band_energy(
#             original_folder, generated_folders, band,
#             thresholds['threshold_low'], thresholds['threshold_mid'], thresholds['threshold_high']
#         )
#         print(f"Energy Difference Ratio for {band} band:")
#         for op, delta_e in energy_results.items():
#             print(f"Operation {op}: ΔE = {delta_e:.4f}")
#         print()

#     # 评估跨操作频谱相似性
#     similarity_results = evaluate_across_operations_similarity(generated_folders)
#     print("Spectral Similarity across Operations:")
#     for pair, similarity in similarity_results.items():
#         print(f"{pair}: Similarity = {similarity:.4f}")



import os
import torch
import numpy as np
from scipy.stats import entropy
from PIL import Image
from torchvision import transforms
from tools.dct_util import dct_2d, high_pass, idct_2d, low_pass, low_pass_and_shuffle

def calculate_frequency_band_energy(dct_features, band):
    """
    计算指定频带的能量。
    :param dct_features: DCT 特征。
    :param band: 频带范围（'low', 'mini', 'mid', 'high'）。
    :return: 指定频带的能量。
    """
    h, w = dct_features.shape[-2], dct_features.shape[-1]
    vertical = torch.arange(0, h).reshape(-1, 1).repeat(1, w)
    horizontal = torch.arange(0, w).reshape(1, -1).repeat(h, 1)
    mask = vertical + horizontal

    if band == 'low':
        mask = mask <= 30
    elif band == 'mini':
        mask = mask <= 10
    elif band == 'mid':
        mask_low = mask <= 40
        mask_high = mask > 20
        mask = mask_low & mask_high
    elif band == 'high':
        mask = mask > 50
    else:
        raise ValueError("Invalid band. Choose 'low', 'mini', 'mid', or 'high'.")

    band_mask = mask.to(dct_features.device)
    band_energy = torch.sum(torch.abs(dct_features) ** 2 * band_mask)
    total_energy = torch.sum(torch.abs(dct_features) ** 2)
    return band_energy / total_energy if total_energy != 0 else 0

def calculate_energy_difference_ratio(input_dct, output_dct, band):
    """
    计算指定频带的能量差异比。
    :param input_dct: 输入图像的 DCT 特征。
    :param output_dct: 输出图像的 DCT 特征。
    :param band: 频带范围。
    :return: 能量差异比。
    """
    input_energy = calculate_frequency_band_energy(input_dct, band)
    output_energy = calculate_frequency_band_energy(output_dct, band)
    return (output_energy - input_energy) / input_energy if input_energy != 0 else 0

def evaluate_frequency_band_energy(original_folder, generated_folders, band='low'):
    """
    评估不同操作输出图像与原始图像在指定频带的能量差异比。
    :param original_folder: 原始图像文件夹。
    :param generated_folders: 不同操作输出图像的文件夹列表。
    :param band: 频带范围。
    :return: 各操作在指定频带的能量差异比。
    """
    original_files = sorted([os.path.join(original_folder, f) for f in os.listdir(original_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    results = {}

    for generated_folder in generated_folders:
        generated_files = sorted([os.path.join(generated_folder, f) for f in os.listdir(generated_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
        avg_delta_e = 0.0
        count = 0

        for original_file, generated_file in zip(original_files, generated_files):
            original_image = load_image(original_file)
            generated_image = load_image(generated_file)

            original_dct = dct_2d(original_image.squeeze(0), norm='ortho')
            generated_dct = dct_2d(generated_image.squeeze(0), norm='ortho')

            delta_e = calculate_energy_difference_ratio(original_dct, generated_dct, band)
            avg_delta_e += delta_e
            count += 1

        if count > 0:
            avg_delta_e /= count
        results[os.path.basename(generated_folder)] = avg_delta_e

    return results

def calculate_spectral_similarity(spectrum1, spectrum2):
    """
    计算两个频谱的余弦相似度。
    :param spectrum1: 第一个频谱。
    :param spectrum2: 第二个频谱。
    :return: 余弦相似度。
    """
    spectrum1_flat = spectrum1.flatten()
    spectrum2_flat = spectrum2.flatten()
    dot_product = np.dot(spectrum1_flat, spectrum2_flat)
    norm1 = np.linalg.norm(spectrum1_flat)
    norm2 = np.linalg.norm(spectrum2_flat)
    return dot_product / (norm1 * norm2 + 1e-10)

def evaluate_spectral_similarity(folder1, folder2):
    """
    评估两个文件夹中图像的频谱相似性。
    :param folder1: 第一个文件夹。
    :param folder2: 第二个文件夹。
    :return: 平均频谱相似性。
    """
    files1 = sorted([os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('.jpg', '.png', '.jpeg'))])
    files2 = sorted([os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('.jpg', '.png', '.jpeg'))])
    avg_similarity = 0.0
    count = 0

    for file1, file2 in zip(files1, files2):
        image1 = load_image(file1)
        image2 = load_image(file2)

        dct1 = dct_2d(image1.squeeze(0), norm='ortho').cpu().numpy()
        dct2 = dct_2d(image2.squeeze(0), norm='ortho').cpu().numpy()

        similarity = calculate_spectral_similarity(dct1, dct2)
        avg_similarity += similarity
        count += 1

    if count > 0:
        avg_similarity /= count
    return avg_similarity

def evaluate_across_operations_similarity(generated_folders):
    """
    评估不同操作输出图像之间的频谱相似性。
    :param generated_folders: 不同操作输出图像的文件夹列表。
    :return: 各操作对之间的平均频谱相似性。
    """
    similarities = {}
    num_folders = len(generated_folders)

    for i in range(num_folders):
        for j in range(i + 1, num_folders):
            folder1 = generated_folders[i]
            folder2 = generated_folders[j]
            similarity = evaluate_spectral_similarity(folder1, folder2)
            key = f"{os.path.basename(folder1)} vs {os.path.basename(folder2)}"
            similarities[key] = similarity

    return similarities


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

if __name__ == "__main__":
    original_folder = r"D:\paper\FCDiffusion_code-main\datasets\original_images"  # 原始图像文件夹
    generated_folders = [
        r"D:\paper\FCDiffusion_code-main\datasets\test_mini_interpre",
        r"D:\paper\FCDiffusion_code-main\datasets\test_low_interpre",  # 低通操作输出
        r"D:\paper\FCDiffusion_code-main\datasets\test_mid_interpre",  # 中通操作输出
        r"D:\paper\FCDiffusion_code-main\datasets\test_high_interpre"  # 高通操作输出
    ]

    # 评估频带能量对比
    bands = ['low', 'mini', 'mid', 'high']
    
    for band in bands:
        energy_results = evaluate_frequency_band_energy(original_folder, generated_folders, band)
        print(f"Energy Difference Ratio for {band} band:")
        for op, delta_e in energy_results.items():
            print(f"Operation {op}: ΔE = {delta_e:.4f}")
        print()

    # 评估跨操作频谱相似性
    similarity_results = evaluate_across_operations_similarity(generated_folders)
    print("Spectral Similarity across Operations:")
    for pair, similarity in similarity_results.items():
        print(f"{pair}: Similarity = {similarity:.4f}")