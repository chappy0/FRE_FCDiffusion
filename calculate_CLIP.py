import os
from PIL import Image
import torch
import clip
from torchvision import transforms

# 1. load CLIP mode
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# define CLIP maxlength
MAX_CONTEXT_LENGTH = 77


def truncate_text(text):
    """Truncate the raw text to fit within the maximum context length for CLIP."""
    tokens = clip.tokenize([text])  # Tokenize the input text
    if tokens.shape[1] > MAX_CONTEXT_LENGTH:
        # Find the maximum number of characters that fit within the token limit
        while tokens.shape[1] > MAX_CONTEXT_LENGTH:
            text = text[:-1]  # Iteratively truncate the text by removing the last character
            tokens = clip.tokenize([text])  # Re-tokenize the truncated text
    return text

def load_images_and_texts(image_folder_path, text_folder_path, preprocess):
    image_text_pairs = []
    for filename in os.listdir(image_folder_path):
        base, ext = os.path.splitext(filename)
        if ext.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:  
            image_path = os.path.join(image_folder_path, filename)
            text_base = base.split("_",1)[-1]
            text_filename = f"{text_base}.txt"
            text_path = os.path.join(text_folder_path, text_filename)
            if os.path.exists(text_path):  
                try:
                    # load image
                    image = Image.open(image_path).convert("RGB")
                    image = preprocess(image)
                    # load text
                    with open(text_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        try:
                            text = truncate_text(text)  # Ensure text fits token limit
                            image_text_pairs.append((image, text))
                        except Exception as e:
                            print(f"Error truncating text for {text_path}: {e}")
                            continue

                except Exception as e:
                    print(f"Error processing pair ({filename}, {text_path}): {e}")
    return image_text_pairs

# 3. load dataset
folder_path = "/your/source/image/path"  #replace to your own path
text_folder_path = "/your/source/text/path"  #replace to your own path
image_text_pairs = load_images_and_texts(folder_path, text_folder_path,preprocess)

if not image_text_pairs:
    print("No valid image-text pairs found in the folder.")
    exit()

# 4. Batch calculate the similarity between images and their corresponding texts
cosine_similarities = []
with torch.no_grad():
    for image, text in image_text_pairs:
        # extract image features
        image = image.unsqueeze(0).to(device) 
        image_features = model.encode_image(image)  # [1, d]
        image_features /= image_features.norm(dim=-1, keepdim=True)  

        # extract text features
        text_tokens = clip.tokenize([text]).to(device)
        text_features = model.encode_text(text_tokens)  # [1, d]
        text_features /= text_features.norm(dim=-1, keepdim=True)  

        # calculate the similarity
        similarity = (image_features @ text_features.T).item()  
        cosine_similarities.append(similarity)

# 5. calculate the average similarity
average_similarity = sum(cosine_similarities) / len(cosine_similarities)




# 6. output the results

with open('cosine_similarities_mid_dis.txt', 'w') as file:
    file.write("Pairwise Cosine Similarities:\n")
    for i, (similarity, (_, text)) in enumerate(zip(cosine_similarities, image_text_pairs)):
        file.write(f"Pair {i + 1}: '{text}' -> Cosine Similarity: {similarity:.4f}\n")
    file.write(f"\nAverage Cosine Similarity: {average_similarity:.4f}\n")
