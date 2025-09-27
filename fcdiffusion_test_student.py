import torch
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import os
from collections import OrderedDict

from fcdiffusion.dataset import TestDataset
from ldm.util import instantiate_from_config


yaml_file_path = "configs/student_model_config.yaml"



ckpt_file_path = '/home/apulis-dev/userdata/FRE_FCDiffusion/lightning_logs/decoupled_distill_low_pass/epoch=22-step=40249-val_loss=3.1322.ckpt' #'/home/apulis-dev/userdata/FRE_FCDiffusion/lightning_logs/decoupled_distill_low_pass/epoch=1-step=1461-val_loss=11.0918.ckpt'  #'/home/apulis-dev/userdata/FRE_FCDiffusion/lightning_logs/decoupled_distill_low_pass/epoch=27-step=48999-val_loss=14.7525.ckpt' 


# # 3. setup the interference
# TEST_IMAGE_PATH = 'datasets/test_sub_200'  # replace the path to your source images


def load_student_model_from_distill_ckpt(config, ckpt_path, device=torch.device("cuda"), verbose=True):
    """
    Specifically designed for loading the student model from checkpoints saved by the DecoupledDistiller.
    It automatically handles the "student_model." prefix.
    """
    print(f"Loading distilled student model from: {ckpt_path}")
    
    # 1. load checkpoint
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    
    # get state_dict
    full_state_dict = pl_sd["state_dict"]
    
    # 2. remove prefix
    student_state_dict = OrderedDict()
    for k, v in full_state_dict.items():
        if k.startswith("student_model."):
            # 去除 "student_model." 前缀
            new_key = k[len("student_model."):]
            student_state_dict[new_key] = v
    
    if not student_state_dict:
        raise KeyError("Could not find weights with 'student_model.' prefix in the checkpoint. Please check the checkpoint file.")

    # 3. Instantiate and load the student model
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(student_state_dict, strict=False)
    
    if len(m) > 0 and verbose:
        print("Missing keys in student model:")
        print(m)
    if len(u) > 0 and verbose:
        print("Unexpected keys in student model:")
        print(u)
        
    model.to(device)
    model.eval()
    print("Student model loaded successfully.")
    return model


# create mode
config = OmegaConf.load(yaml_file_path)
device = torch.device("cuda")
model = load_student_model_from_distill_ckpt(config, ckpt_file_path, device)


model.eval()

# set test image path and target prompt
import os  
  
def is_image_file(filename):  
    """Determine if the file is a image file"""  
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg']  
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)  
  
def is_text_file(filename):  
    """Determine if the file is a text file"""  
    return filename.lower().endswith('.txt')  
  
def traverse_images_and_texts(directory):  
    """Traverse the image files"""  
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



# example
directory_path =   '/home/apulis-dev/userdata/DGM/datasets/test_quality' #'../DGM/datasets/test_sub_200'  '../DGM/datasets/test_sub_600'  # replace it to your own path 
image_files, text_contents = traverse_images_and_texts(directory_path)  

test_res_num = 1


repath = "datasets/test_low_dkd_qua_0927"  #nfe5    #reconstruction_EXA" 
if not os.path.exists(repath):
    os.makedirs(repath)
for image_file,text_content in zip(image_files,text_contents):  
    test_img_path,target_prompt = image_file,text_content
    _,reconstruction_img =  os.path.split(test_img_path) 
    reconstruction_img = repath + "/e_" + reconstruction_img 

    dataset = TestDataset(test_img_path, target_prompt, test_res_num)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=16, shuffle=True)
    
        
    for step, batch in enumerate(dataloader):
        log = model.log_images(batch, ddim_steps=50)  #, explicit_timesteps_file_path = 'custom_timesteps.txt'
        if step == 0:
            reconstruction = log['reconstruction'].squeeze()
            reconstruction = reconstruction.permute(1, 2, 0)
            reconstruction = torch.clamp(reconstruction, -1, 1)
            Image.fromarray(((reconstruction.cpu().numpy() + 1) * 127.5).astype(np.uint8)).save(reconstruction_img)


        # save image
        sample = log['samples'].squeeze()
        sample = sample.permute(1, 2, 0)
        sample = torch.clamp(sample, -1, 1)
        # Image.fromarray(((sample.cpu().numpy() + 1) * 127.5).astype(np.uint8)).show()
        Image.fromarray(((sample.cpu().numpy() + 1) * 127.5).astype(np.uint8)).save(reconstruction_img)



