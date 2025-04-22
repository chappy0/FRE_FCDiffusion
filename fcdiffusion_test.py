# from omegaconf import OmegaConf
# import torch
# from ldm.util import instantiate_from_config
# from PIL import Image
# import numpy as np
# from torch.utils.data import DataLoader
# from fcdiffusion.dataset import TestDataset
# torch.cuda.set_device(0)

# import os
# os.environ['CURL_CA_BUNDLE'] = ''


# def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
#     print(f"Loading model from {ckpt}")
#     pl_sd = torch.load(ckpt, map_location="cpu")
#     if "global_step" in pl_sd:
#         print(f"Global Step: {pl_sd['global_step']}")
#     sd = pl_sd["state_dict"]
#     model = instantiate_from_config(config.model)
#     m, u = model.load_state_dict(sd, strict=False)
#     if len(m) > 0 and verbose:
#         print("missing keys:")
#         print(m)
#     if len(u) > 0 and verbose:
#         print("unexpected keys:")
#         print(u)
#     if device == torch.device("cuda"):
#         model.cuda()
#     elif device == torch.device("cpu"):
#         model.cpu()
#         model.cond_stage_model.device = "cpu"
#     else:
#         raise ValueError(f"Incorrect device name. Received: {device}")
#     model.eval()
#     return model


# # set model configuration file
# yaml_file_path = "configs/model_config.yaml"

# # set the checkpoint path in the lightning_logs dir (the loaded model should be consistent
# # with the "control_mode" parameter in the yaml config file)
# ckpt_file_path = "lightning_logs/fcdiffusion_mid_pass_checkpoint/epoch=2-step=999.ckpt"

# # create mode
# config = OmegaConf.load(yaml_file_path)
# device = torch.device("cuda")
# model = load_model_from_config(config, ckpt_file_path, device)
# assert model.control_mode in ckpt_file_path.split('/')[1], \
#     'the checkpoint model is not consistent with the config file in control mode'
# model.eval()

# # set test image path and target prompt
# test_img_path = 'test_img.jpg'   # the path of the test image to be translated
# target_prompt = 'photo of a music room'  # the target text prompt for image-to-image translation
# test_res_num = 4


# dataset = TestDataset(test_img_path, target_prompt, test_res_num)
# dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False)
# for step, batch in enumerate(dataloader):
#     log = model.log_images(batch, ddim_steps=50)
#     if step == 0:
#         reconstruction = log['reconstruction'].squeeze()
#         reconstruction = reconstruction.permute(1, 2, 0)
#         reconstruction = torch.clamp(reconstruction, -1, 1)
#         Image.fromarray(((reconstruction.cpu().numpy() + 1) * 127.5).astype(np.uint8)).show()
#     sample = log["samples"].squeeze()
#     sample = sample.permute(1, 2, 0)
#     sample = torch.clamp(sample, -1, 1)
#     Image.fromarray(((sample.cpu().numpy() + 1) * 127.5).astype(np.uint8)).show()
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


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model


# set model configuration file
yaml_file_path = "configs/model_config.yaml"

# set the checkpoint path in the lightning_logs dir (the loaded model should be consistent
# with the "control_mode" parameter in the yaml config file)
# ckpt_file_path = "lightning_logs/fcdiffusion_mid_pass_checkpoint/epoch=0-step=9999.ckpt"
# ckpt_file_path = "lightning_logs/fcdiffusion_low_pass_checkpoint/epoch=3-step=12999.ckpt"
# ckpt_file_path = "lightning_logs/fcdiffusion_mini_pass_checkpoint/epoch=10-step=34999.ckpt"
# ckpt_file_path = r"D:\paper\FCDiffusion_code-main\lightning_logs\fcdiffusion_mid_pass_checkpoint\epoch=7-step=17999.ckpt"
# ckpt_file_path = r"D:\paper\FCDiffusion_code-main\lightning_logs\fcdiffusion_mid_pass_checkpoint\epoch=11-step=241999.ckpt"
ckpt_file_path = r'D:\paper\FRE_FCD\lightning_logs_SA\fcdiffusion_high_pass_checkpoint\epoch=7-step=9999.ckpt'     
# ckpt_file_path = r"D:\paper\FRE_FCD\lightning_logs\mini\epoch=1-step=2999.ckpt"
# ckpt_file_path = "lightning_logs_SA/fcdiffusion_mid_pass_checkpoint\epoch=0-step=1999.ckpt"
# ckpt_file_path = r"D:\paper\FCDiffusion_code-main\lightning_logs\fcdiffusion_high_pass_checkpoint\epoch=3-step=12999.ckpt"
# create mode
config = OmegaConf.load(yaml_file_path)
device = torch.device("cuda")
model = load_model_from_config(config, ckpt_file_path, device)
 # 通过 model 的类来查看继承关系

# print(f"jichengguanxi:{StudentFCDiffusion.mro()}") 

# assert model.control_mode in ckpt_file_path.split('\\')[-2], \
#     'the checkpoint model is not consistent with the config file in control mode'
model.eval()

# set test image path and target prompt
import os  
  
def is_image_file(filename):  
    """判断文件是否为图片文件"""  
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg']  
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)  
  
def is_text_file(filename):  
    """判断文件是否为文本文件"""  
    return filename.lower().endswith('.txt')  
  
def traverse_images_and_texts(directory):  
    """遍历文件夹中的所有图片文件和文本文件"""  
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
  
# 使用示例  
directory_path =r'D:\paper\FCDiffusion_code-main\datasets\test'   # 替换为你的文件夹路径  
image_files, text_contents = traverse_images_and_texts(directory_path)  
# image_files, text_contents = ["/home/apulis-dev/userdata/FCDiffusion_code/datasets/test_sub/image_1059.jpg"],["Picture The sky, Clouds, The city, Machine, The building, Art, The airfield, The plane, Mafia, Definitive …"]
test_res_num = 1

# 打印所有图片文件的路径  
# print("Image files:") 
repath = r"D:\paper\FCDiffusion_code-main\datasets\test_low_interpre"    #reconstruction_EXA" 
if not os.path.exists(repath):
    os.makedirs(repath)
for image_file,text_content in zip(image_files,text_contents):  
#     print(image_file)  
    test_img_path,target_prompt = image_file,text_content
    _,reconstruction_img =  os.path.split(test_img_path) 
    reconstruction_img = repath + "/e_" + reconstruction_img 
    # if  os.path.exists(reconstruction_img):
    #     continue

    
    #print(f"img:{test_img_path},target_prompt:{target_prompt}")
    dataset = TestDataset(test_img_path, target_prompt, test_res_num)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=True)
    
        
    for step, batch in enumerate(dataloader):
        # teacher_noise_loss, loss_dict = model.shared_step(batch)
        # print(f"Teacher自身预测误差: {teacher_noise_loss}")
        log = model.log_images(batch, ddim_steps=50)
        if step == 0:
            reconstruction = log['reconstruction'].squeeze()
            reconstruction = reconstruction.permute(1, 2, 0)
            reconstruction = torch.clamp(reconstruction, -1, 1)
            # Image.fromarray(((reconstruction.cpu().numpy() + 1) * 127.5).astype(np.uint8)).show()
            Image.fromarray(((reconstruction.cpu().numpy() + 1) * 127.5).astype(np.uint8)).save(reconstruction_img)
        sample = log["samples"].squeeze()
        sample = sample.permute(1, 2, 0)
        sample = torch.clamp(sample, -1, 1)
        # Image.fromarray(((sample.cpu().numpy() + 1) * 127.5).astype(np.uint8)).show()
        Image.fromarray(((sample.cpu().numpy() + 1) * 127.5).astype(np.uint8)).save(reconstruction_img)


  
# # 打印所有文本文件的路径  
# print("\nText files:")  
# for text_file in text_files:  
#     print(text_file)

# test_img_path = r'D:\paper\FCDiffusion_code-main\FCDiffusion_code-main\datasets\laion_aesthetics_6.5\test\122210364649593628.jpg'   #'test_img.jpg'   # the path of the test image to be translated
# target_prompt = 'photo of a music room'  # the target text prompt for image-to-image translation
# test_res_num = 4




