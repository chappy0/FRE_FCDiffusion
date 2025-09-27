
import os
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
from transformers import ViTFeatureExtractor, AutoModel

# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:11304'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:11304'
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- [MODIFICATION 1] ---
# Automatically select device: GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DinoVitExtractor:
    def __init__(self, model_name="../Super_FCD/dino-vits16/"):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # --- [MODIFICATION 2] ---
        # Move the model to the selected device (GPU)
        self.model.to(DEVICE)
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
        
        # --- [MODIFICATION 3] ---
        # Move the input tensors to the same device as the model
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

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
    # These operations will run on the device the features are on (GPU)
    features1 = features1.mean(dim=1)  # Average over tokens
    features2 = features2.mean(dim=1)  # Average over tokens
    similarity = torch.cosine_similarity(features1, features2)
    return 1.0 - similarity.item()  # .item() moves the result to CPU


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


def compute_folder_similarity(folder1, folder2, dino_extractor):
    # This function's logic remains unchanged
    images1 = sorted([os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('.jpg', '.png', '.jpeg'))])
    images2 = sorted([os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('.jpg', '.png', '.jpeg'))])

    results = []
    for img1_path in tqdm(images1, desc="Processing Folder 1"):
        base_img1 = os.path.splitext(os.path.basename(img1_path))[0]

        found_match = False
        for img2_path in images2:
            # if os.path.splitext(os.path.basename(img2_path))[0].replace("_output","") == base_img1:
            if os.path.splitext(os.path.basename(img2_path))[0].split('e_', 1)[-1] == base_img1:
                image1 = Image.open(img1_path).convert("RGB")
                image2 = Image.open(img2_path).convert("RGB")
                similarity = compute_structure_similarity(image1, image2, dino_extractor)
                results.append((os.path.basename(img1_path), os.path.basename(img2_path), similarity))
                found_match = True
                break
        
        if not found_match:
            print(f"No match found for {os.path.basename(img1_path)}")
            continue

    average_similarity = np.mean([x[2] for x in results]) if results else 0.0
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
    print(f"Using device: {DEVICE}") # Inform the user which device is being used
    
    # Define folders
    folder1 = "/home/apulis-dev/userdata/DGM/datasets/test_sub_200"
    # folder2 = "/home/apulis-dev/userdata/FCDiffusion_code/dynamic_outputs_task_conditioned_low/semantic_manipulation"
    folder2 = 'test_dha_perf_output0926'   #'single_lora_outputs2_nolora/high_pass' #'datasets/test_baseline'  #"dynamic_outputs_task_conditioned_low_0803_2/semantic_manipulation"
    # folder2 = "/home/apulis-dev/userdata/FCDiffusion_code/datasets/test_low_baseline_nfe4"
    output_file = "similarity_results_optim_low3.csv"


    # Initialize DINO-ViT extractor (it will now be on the GPU if available)
    dino_extractor = DinoVitExtractor()

    # Compute similarity for all images in the folders
    results, average_similarity = compute_folder_similarity(folder1, folder2, dino_extractor)

    # Save results to a file
    save_results_to_file(results, average_similarity, output_file)
    print(f"Results saved to {output_file}")
    print(f"Average Similarity: {average_similarity:.4f}")