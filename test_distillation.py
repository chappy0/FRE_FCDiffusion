# # # #test_distillation.py

from omegaconf import OmegaConf
import torch
from ldm.util import instantiate_from_config
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from fcdiffusion.dataset import TestDataset
torch.cuda.set_device(0)

import os
os.environ['CURL_CA_BUNDLE'] = ''

class StudentModelCheckpoint:
    pass

def load_student_model_from_config(config, ckpt_file_path, device=torch.device("cuda"), verbose=True):
    print(f"Loading model from {ckpt_file_path}")
    pl_sd = torch.load(ckpt_file_path, map_location="cpu")

    sd = pl_sd["state_dict"]


    student_sd = {k[6:]: v for k, v in sd.items()  if  k.startswith("model.") }


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
ckpt_file_path = 'path/to/yourname'




config = OmegaConf.load(yaml_file_path)
device = torch.device("cuda")
model = load_student_model_from_config(config, ckpt_file_path, device)

# Ensure model is in eval mode
model.eval()




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
directory_path = 'path/to/yourname' # Replace with your directory path
image_files, text_contents = traverse_images_and_texts(directory_path)

# Output directory
repath = 'path/to/yourname'
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
        
