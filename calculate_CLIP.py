import os
from PIL import Image
import torch
import clip
from torchvision import transforms

# 1. 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 定义 CLIP 最大上下文长度
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



# def truncate_text(text):
#     """Truncate the text to ensure it fits within the CLIP tokenization context length."""
#     tokens = clip.tokenize([text])  # Tokenize the input text
#     while tokens.shape[1] > MAX_CONTEXT_LENGTH:  # Check token length
#         text = text[:-1]  # Remove the last character
#         print(f"text:{text}")
#         tokens = clip.tokenize([text])  # Re-tokenize
#     return text

from transformers import pipeline, GPT2Tokenizer

# Load tokenizer and summarization pipeline
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# summarizer = pipeline("summarization")

# max_tokens = 76
# def truncate_text(text):
#     """Truncate the text to fit within the specified max_tokens for CLIP."""
#     # Tokenize and check token count
#     tokens = tokenizer.encode(text)
#     if len(tokens) > max_tokens:
#         # Summarize text if too long
#         summaries = summarizer(text, max_length=max_tokens//2, min_length=max_tokens//4, do_sample=False)
#         text = summaries[0]['summary_text']
#         tokens = tokenizer.encode(text)
#         # Further truncate if necessary
#         while len(tokens) > max_tokens:
#             text = text[:-len(text.split()[-1])-1]  # Remove last word including space
#             tokens = tokenizer.encode(text)
#     return text




def load_images_and_texts(image_folder_path, text_folder_path, preprocess):
    image_text_pairs = []
    for filename in os.listdir(image_folder_path):
        base, ext = os.path.splitext(filename)
        if ext.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:  # 图像文件后缀
            image_path = os.path.join(image_folder_path, filename)
            text_base = base.split("e_",1)[-1]
            # text_base = base.replace("_output",'')
            text_filename = f"{text_base}.txt"
            text_path = os.path.join(text_folder_path, text_filename)
            if os.path.exists(text_path):  # 检查对应文本文件是否存在
                try:
                    # 加载图像
                    image = Image.open(image_path).convert("RGB")
                    image = preprocess(image)
                    # 加载文本
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

# 3. 加载文件夹数据
# folder_path = "/home/apulis-dev/userdata/FCDiffusion_code/dynamic_outputs_low_pass_mode_0708"  # 替换为你的文件夹路径
folder_path =   'test_dha_perf_output0926'  #'datasets/test_baseline/'
#dynamic_outputs_task_conditioned_mid_simple/Image_scene_translation
text_folder_path = "/home/apulis-dev/userdata/DGM/datasets/test_sub_200"
image_text_pairs = load_images_and_texts(folder_path, text_folder_path,preprocess)

if not image_text_pairs:
    print("No valid image-text pairs found in the folder.")
    exit()

# 4. 批量计算图像与对应文本的相似度
cosine_similarities = []

with torch.no_grad():
    for image, text in image_text_pairs:
        # 提取图像特征
        image = image.unsqueeze(0).to(device)  # 增加 batch 维度
        image_features = model.encode_image(image)  # [1, d]
        image_features /= image_features.norm(dim=-1, keepdim=True)  # 归一化

        # 提取文本特征
        text_tokens = clip.tokenize([text]).to(device)
        text_features = model.encode_text(text_tokens)  # [1, d]
        text_features /= text_features.norm(dim=-1, keepdim=True)  # 归一化

        # 计算余弦相似度
        similarity = (image_features @ text_features.T).item()  # 标量值
        cosine_similarities.append(similarity)

# 5. 计算平均相似度
average_similarity = sum(cosine_similarities) / len(cosine_similarities)




# 6. 输出结果
# print("Pairwise Cosine Similarities:")
# for i, (similarity, (_, text)) in enumerate(zip(cosine_similarities, image_text_pairs)):
#     print(f"Pair {i + 1}: '{text}' -> Cosine Similarity: {similarity:.4f}")

# print(f"\nAverage Cosine Similarity: {average_similarity:.4f}")
# 打开文件用于写入，如果文件不存在则创建
with open('cosine_similarities_low_baseline_nfe4.txt', 'w') as file:
    # 写入标题
    file.write("Pairwise Cosine Similarities:\n")
    
    # 遍历相似度和文本对，写入每一对的相似度
    for i, (similarity, (_, text)) in enumerate(zip(cosine_similarities, image_text_pairs)):
        file.write(f"Pair {i + 1}: '{text}' -> Cosine Similarity: {similarity:.4f}\n")
    
    # 写入平均相似度
    file.write(f"\nAverage Cosine Similarity: {average_similarity:.4f}\n")
    print(f"\nAverage Cosine Similarity: {average_similarity:.4f}\n")