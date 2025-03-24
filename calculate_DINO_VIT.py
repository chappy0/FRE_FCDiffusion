import os
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np

import os
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:11304'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:11304'
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class DinoVitExtractor:
    def __init__(self, model_name="facebook/dino-vits16"):
        from transformers import ViTFeatureExtractor, AutoModel
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def extract_features(self, image):
        """
        Extract DINO features for a single image.
        Args:
            image (PIL.Image): Input image.
        Returns:
            features (torch.Tensor): Extracted feature map.
        """
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state  # Extract the token embeddings


def compute_structure_distance(features1, features2):
    """
    Compute structure distance between two sets of features.

    Args:
    - features1 (torch.Tensor): Feature map from image1.
    - features2 (torch.Tensor): Feature map from image2.

    Returns:
    - distance (float): Structure distance.
    """
    features1 = features1.mean(dim=1)  # Average over tokens
    features2 = features2.mean(dim=1)  # Average over tokens
    similarity = torch.cosine_similarity(features1, features2)
    return 1.0 - similarity.item()  # Structure distance


def compute_structure_similarity(image1, image2, dino_extractor):
    """
    Compute structure similarity between two images using DINO-ViT features.

    Args:
    - image1 (PIL.Image): First input image.
    - image2 (PIL.Image): Second input image.
    - dino_extractor (DinoVitExtractor): DINO-ViT feature extractor.

    Returns:
    - similarity (float): Structure similarity (1 - distance).
    """
    features1 = dino_extractor.extract_features(image1)
    features2 = dino_extractor.extract_features(image2)
    distance = compute_structure_distance(features1, features2)
    return 1.0 - distance  # Structure similarity



# def compute_folder_similarity(folder1, folder2, dino_extractor):
#     images1 = [os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('.jpg', '.png', '.jpeg'))]
#     images2 = [os.path.join(folder2, f) for f in os.listdir(folder2) 
#           if f.endswith(('.jpg', '.png', '.jpeg')) and f.startswith('re_')]

#     results = []
#     similarities = []

#     # print(f"images2:{images2}")
#     for img1_path in tqdm(images1, desc="Processing Folder 1"):
#         base_img1 = os.path.splitext(os.path.basename(img1_path))[0]
#         # print(f"base_img1:{base_img1}")
        
        
    
#     # Find matching image in folder2
#     match_img2 = next((img2 for img2 in images2 if os.path.splitext(os.path.basename(img2))[0].replace('re_', '') == base_img1), None)
#     # print(f"match_img2:{match_img2}")
#     if match_img2:
#         img2_path = os.path.join(folder2, match_img2)
#         image1 = Image.open(img1_path).convert("RGB")
#         image2 = Image.open(img2_path).convert("RGB")

#         # Compute structure similarity
#         similarity = compute_structure_similarity(image1, image2, dino_extractor)
#         similarities.append(similarity)

#         # Store results
#         results.append((os.path.basename(img1_path), os.path.basename(img2_path), similarity))

#     average_similarity = np.mean(similarities)
#     return results, average_similarity






def compute_folder_similarity(folder1, folder2, dino_extractor):
    # 获取文件夹1中的所有图片文件并排序
    images1 = sorted([os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    # 获取文件夹2中的所有图片文件，并确保它们以 "re_" 开头并排序
    images2 = sorted([os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('.jpg', '.png', '.jpeg') )])

    results = []
    for img1_path in tqdm(images1, desc="Processing Folder 1"):
        base_img1 = os.path.splitext(os.path.basename(img1_path))[0]  # 去掉文件扩展名

        found_match = False
        for img2_path in images2:
            if os.path.splitext(os.path.basename(img2_path))[0].split('_',1)[-1] ==   base_img1:
                # img2_path = os.path.join(folder2, img2)
                image1 = Image.open(img1_path).convert("RGB")
                image2 = Image.open(img2_path).convert("RGB")
                similarity = compute_structure_similarity(image1, image2, dino_extractor)
                results.append((os.path.basename(img1_path), os.path.basename(img2_path), similarity))
                found_match = True
                break
        
        if not found_match:
            print(f"No match found for {os.path.basename(img1_path)}")
            continue  # 如果没有找到匹配的文件，跳过当前迭代，继续下一个文件

    average_similarity = np.mean([x[2] for x in results])
    return results, average_similarity




def save_results_to_file(results, average_similarity, output_file):
    """
    Save structure similarity results to a file.

    Args:
    - results (list): List of tuples (image1, image2, similarity).
    - average_similarity (float): Average similarity across all pairs.
    - output_file (str): Path to the output file.
    """
    with open(output_file, "w") as f:
        f.write("Image1,Image2,Similarity\n")
        for img1, img2, similarity in results:
            f.write(f"{img1},{img2},{similarity:.4f}\n")
        f.write(f"\nAverage Similarity: {average_similarity:.4f}\n")


if __name__ == "__main__":
    # Define folders
    folder1 = r"D:\paper\FCDiffusion_code-main\datasets\test"
    folder2 = r"D:\paper\FCDiffusion_code-main\datasets\test_final"
    output_file = "similarity_results_dis_mid.csv"

    # Initialize DINO-ViT extractor
    dino_extractor = DinoVitExtractor()

    # Compute similarity for all images in the folders
    results, average_similarity = compute_folder_similarity(folder1, folder2, dino_extractor)

    # Save results to a file
    save_results_to_file(results, average_similarity, output_file)
    print(f"Results saved to {output_file}")
    print(f"Average Similarity: {average_similarity:.4f}")
