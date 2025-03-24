# # # #test_distillation.py

from omegaconf import OmegaConf
import torch
from ldm.util import instantiate_from_config
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from fcdiffusion.dataset import TestDataset
# from fcdiffusion_distill_samearch import FCDiffusionDistill
torch.cuda.set_device(0)

import os
os.environ['CURL_CA_BUNDLE'] = ''

def load_student_model_from_config(config, ckpt_file_path, device=torch.device("cuda"), verbose=True):
    print(f"Loading model from {ckpt_file_path}")
    pl_sd = torch.load(ckpt_file_path, map_location="cpu")
    # if "global_step" in pl_sd:
    #     print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    # student_sd = {k.replace("model.", ""): v for k, v in sd.items() if  k.startswith("model.") }
    student_sd = {k[6:]: v for k, v in ckpt['state_dict'].items()  if  k.startswith("model.") }
    # with open("student_sd.txt", 'a') as f:
    #     f.write(f"student_sd:{student_sd}\n")

    model = instantiate_from_config(config.model)

    
        
    # 加载学生参数
    m, u = model.load_state_dict(student_sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
        # f.write("missing keys:")
        # f.write(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
        # f.write("unexpected keys:")
        # f.write(u)
    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model


# Load configuration and checkpoint
yaml_file_path = 'configs/student_model_config.yaml'
# ckpt_file_path =  "lightning_logs/fcdiffusion_low_pass_checkpoint/epoch=2-step=55999-v1.ckpt"
ckpt_file_path = 'lightning_logs/distiallation_low_pass_checkpoint/epoch=3-step=141999.ckpt'
# yaml_file_path = "configs/teacher_model_config.yaml"
# ckpt_file_path = '/home/apulis-dev/userdata/FCDiffusion_code/lightning_logs/fcdiffusion_mid_pass_checkpoint/epoch=11-step=241999.ckpt'
ckpt = torch.load(ckpt_file_path, map_location="cpu")  # 加载 checkpoint


# 提取学生模型的权重
# student_sd = {k: v for k, v in ckpt['state_dict'].items()  }
# student_sd = {k[6:]: v for k, v in ckpt['state_dict'].items()  if  k.startswith("model.") }
# # 将输出保存到文件
# output_file_path = '0224_invert_student_model_parameters_from_ckpt1.txt'  # 输出文件路径
# with open(output_file_path, 'w') as f:
#     f.write("222Student model parameters in checkpoint:\n")
#     for key in student_sd.keys():
#         f.write(f"{key}\n")

config = OmegaConf.load(yaml_file_path)
device = torch.device("cuda")
model = load_student_model_from_config(config, ckpt_file_path, device)

# Ensure model is in eval mode
model.eval()

# # Debugging: Print model structure
# output_file_path = 'student_model_arch_dis33.txt'  # 输出文件路径
# with open(output_file_path, 'w') as f:
#     # for key in student_sd.keys():
#     f.write(f"Model architecture_dis2: {model}")

# Set test image path and target prompt
import os

def is_image_file(filename):
    """Check if the file is an image file."""
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg']
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)

def is_text_file(filename):
    """Check if the file is a text file."""
    return filename.lower().endswith('.txt')

def traverse_images_and_texts(directory):
    """Traverse all image and text files in a directory."""
    image_files = []
    text_contents = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if is_image_file(file):
                image_files.append(os.path.join(root, file))
            elif is_text_file(file):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    text_contents.append((content))
    return image_files, text_contents

# Set test image path and target prompt
directory_path = r'D:\paper\FCDiffusion_code-main\datasets\test'  # Replace with your directory path
image_files, text_contents = traverse_images_and_texts(directory_path)

# Output directory
repath = r"D:\paper\FCDiffusion_code-main\datasets\test_dis"
if not os.path.exists(repath):
    os.makedirs(repath)

# Loop through images and texts
for image_file, text_content in zip(image_files, text_contents):
    test_img_path, target_prompt = image_file, text_content
    _, reconstruction_img_path = os.path.split(test_img_path)
    reconstruction_img_path = os.path.join(repath, f"e_{reconstruction_img_path}")

    dataset = TestDataset(test_img_path, target_prompt, res_num=1)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=True)

    for step, batch in enumerate(dataloader):
        # Debugging: Print batch shape and sample input values
        # print(f"Batch shape: {batch['image'].shape}")
        # print(f"Input text: {target_prompt}")

        log = model.log_images(batch, ddim_steps=36)

        sample = log["samples"].squeeze()
        sample = sample.permute(1, 2, 0)
        sample = torch.clamp(sample, -1, 1)
        # Image.fromarray(((sample.cpu().numpy() + 1) * 127.5).astype(np.uint8)).show()
        Image.fromarray(((sample.cpu().numpy() + 1) * 127.5).astype(np.uint8)).save(reconstruction_img_path)
        
        # Debugging: Check output values
        # print(f"Log output: {log}")
        # reconstruction = log['reconstruction'].squeeze()
        # # print(f"Reconstruction min/max: {reconstruction.min()}, {reconstruction.max()}")
        # reconstruction = reconstruction.permute(1, 2, 0)
        # # Normalize and save the image
        # reconstruction = torch.clamp(reconstruction, -1, 1)
        # reconstruction_img = Image.fromarray(((reconstruction.cpu().numpy() + 1) * 127.5).astype(np.uint8))
        # # reconstruction_img.show()  # Display the image for debugging
        # reconstruction_img.save(reconstruction_img_path)

        # # Debugging: Check sample
        # sample = log["samples"].squeeze()
        # sample = sample.permute(1, 2, 0)
        # # print(f"Sample shape min/max:{sample.shape} {sample.min()}, {sample.max()}")
        # sample = torch.clamp(sample, -1, 1)
        # Image.fromarray(((sample.cpu().numpy() + 1) * 127.5).astype(np.uint8)).save(reconstruction_img_path)

