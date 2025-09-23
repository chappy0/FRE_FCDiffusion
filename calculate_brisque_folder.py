import torch
import pyiqa
import argparse
import os
from tqdm import tqdm

def main(args):
    """
    主函数，用于批量计算一个文件夹内所有图片的平均BRISQUE分数。
    """
    # --- 1. 设置运行设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. 检查输入目录是否存在 ---
    if not os.path.isdir(args.input_dir):
        print(f"ERROR: Input directory not found at: {args.input_dir}")
        return

    # --- 3. 创建BRISQUE评估器实例 (在循环外创建一次即可) ---
    # device=device 可以让计算在GPU上进行，速度更快
    try:
        brisque_metric = pyiqa.create_metric('brisque', device=device)
        print(f"Successfully created BRISQUE metric. The score range is [0, 100]. Lower is better.")
    except Exception as e:
        print(f"Error creating IQA metric: {e}")
        return

    # --- 4. 查找文件夹中的所有图片文件 ---
    image_filenames = sorted([
        f for f in os.listdir(args.input_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ])
    
    if not image_filenames:
        print(f"ERROR: No image files found in '{args.input_dir}'.")
        return
        
    print(f"Found {len(image_filenames)} images to evaluate.")

    # --- 5. 循环计算所有图片的BRISQUE分数 ---
    total_brisque_score = 0.0
    processed_count = 0

    # 使用 tqdm 创建一个进度条
    for filename in tqdm(image_filenames, desc="Calculating BRISQUE scores"):
        image_path = os.path.join(args.input_dir, filename)
        
        try:
            # pyiqa可以直接处理图片路径
            score = brisque_metric(image_path)
            total_brisque_score += score.item()
            processed_count += 1
        except Exception as e:
            # 如果某张图片处理失败（例如文件损坏），则打印警告并跳过
            print(f"\nWARNING: Could not process file '{filename}'. Error: {e}")
            continue
    
    # --- 6. 计算并打印平均分 ---
    if processed_count > 0:
        average_score = total_brisque_score / processed_count
        print("\n" + "="*40)
        print(f"        BRISQUE Evaluation Summary")
        print("="*40)
        print(f"Evaluated {processed_count} / {len(image_filenames)} images successfully.")
        print(f"Average BRISQUE Score: {average_score:.4f}")
        print("(Note: Lower score means better image quality)")
        print("="*40)
    else:
        print("No images were successfully processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the average BRISQUE score for all images in a folder.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing the images to evaluate.")
    
    args = parser.parse_args()
    main(args)
