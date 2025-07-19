import torch
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import os
from collections import OrderedDict

# 确保这些import路径与您的项目结构一致
from fcdiffusion.dataset import TestDataset
from ldm.util import instantiate_from_config

# --- 配置区 ---
# 1. 学生模型的配置文件 (指定了GMEA架构)
yaml_file_path = "configs/student_model_config.yaml"

# 2. 蒸馏训练后保存的Checkpoint路径
#    请将其指向您通过DecoupledDistiller训练保存的.ckpt文件
# STUDENT_CKPT_PATH = "lightning_logs/decoupled_distill_low_pass/your_trained_student_checkpoint.ckpt"
ckpt_file_path ='/home/apulis-dev/userdata/FCDiffusion_code/lightning_logs/decoupled_distill_low_pass/epoch=38-step=68249-val_loss=12.1872.ckpt'  #mini


# # 3. 推理设置
# TEST_IMAGE_PATH = 'datasets/test_sub_200'  # 要翻译的源图片路径


# ----------------

def load_student_model_from_distill_ckpt(config, ckpt_path, device=torch.device("cuda"), verbose=True):
    """
    专门用于从蒸馏器(DecoupledDistiller)保存的checkpoint中加载学生模型。
    它会自动处理"student_model."前缀。
    """
    print(f"Loading distilled student model from: {ckpt_path}")
    
    # 1. 加载完整的checkpoint
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    
    # 获取包含所有权重的 state_dict
    full_state_dict = pl_sd["state_dict"]
    
    # 2. 创建一个新的state_dict，只包含学生模型的权重，并去除前缀
    student_state_dict = OrderedDict()
    for k, v in full_state_dict.items():
        if k.startswith("student_model."):
            # 去除 "student_model." 前缀
            new_key = k[len("student_model."):]
            student_state_dict[new_key] = v
    
    if not student_state_dict:
        raise KeyError("Could not find weights with 'student_model.' prefix in the checkpoint. Please check the checkpoint file.")

    # 3. 实例化并加载学生模型
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
directory_path ='datasets/test_quality'  # 替换为你的文件夹路径  
image_files, text_contents = traverse_images_and_texts(directory_path)  
# image_files, text_contents = ["/home/apulis-dev/userdata/FCDiffusion_code_EA/datasets/test_data/test_FID/image_242.jpg",
# '/home/apulis-dev/userdata/FCDiffusion_code_EA/datasets/test_data/test_FID/image_243.jpg',
# '/home/apulis-dev/userdata/FCDiffusion_code_EA/datasets/test_data/test_FID/image_2970.jpg',
# '/home/apulis-dev/userdata/FCDiffusion_code_EA/datasets/test_data/test_FID/image_2299.jpg',
# '/home/apulis-dev/userdata/FCDiffusion_code_EA/datasets/test_data/test_FID/image_2469.jpg',
# '/home/apulis-dev/userdata/FCDiffusion_code_EA/datasets/test_data/test_FID/image_106.jpg',
# '/home/apulis-dev/userdata/FCDiffusion_code_EA/datasets/test_data/test_FID/image_175.jpg'],['oil', 'colorful','Cloud,blue sky', 'colorful','colorful','Spring Festive']
test_res_num = 1

# 打印所有图片文件的路径  
# print("Image files:") 
repath = "datasets/test_low_dkd_perf"  #nfe5    #reconstruction_EXA" 
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
    dataloader = DataLoader(dataset, num_workers=0, batch_size=16, shuffle=True)
    
        
    for step, batch in enumerate(dataloader):
        log = model.log_images(batch, ddim_steps=50)  #, explicit_timesteps_file_path = 'custom_timesteps.txt'
        if step == 0:
            reconstruction = log['reconstruction'].squeeze()
            reconstruction = reconstruction.permute(1, 2, 0)
            reconstruction = torch.clamp(reconstruction, -1, 1)
            Image.fromarray(((reconstruction.cpu().numpy() + 1) * 127.5).astype(np.uint8)).save(reconstruction_img)


        # 保存样本图像
        sample = log['samples'].squeeze()
        sample = sample.permute(1, 2, 0)
        sample = torch.clamp(sample, -1, 1)
        # Image.fromarray(((sample.cpu().numpy() + 1) * 127.5).astype(np.uint8)).show()
        Image.fromarray(((sample.cpu().numpy() + 1) * 127.5).astype(np.uint8)).save(reconstruction_img)

# def main():
#     """主推理函数"""
#     # 设置运行设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.cuda.set_device(0)

#     # 加载学生模型配置
#     config = OmegaConf.load(STUDENT_CONFIG_PATH)
    
#     # 使用新的加载函数加载学生模型
#     student_model = load_student_model_from_distill_ckpt(config, STUDENT_CKPT_PATH, device)
    
#     # 准备推理数据
#     # TestDataset需要一个数字作为第三个参数，这里设为1
#     dataset = TestDataset(TEST_IMAGE_PATH, TARGET_PROMPT, 1)
#     dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False)

#     print(f"\nRunning inference on '{TEST_IMAGE_PATH}'")
#     print(f"With prompt: '{TARGET_PROMPT}'")

#     for batch in dataloader:
#         # 将数据移动到正确的设备
#         for k, v in batch.items():
#             if isinstance(v, torch.Tensor):
#                 batch[k] = v.to(device)

#         # 执行推理并获取结果
#         # 在这里可以测量推理时间和显存占用
#         with torch.no_grad():
#             log = student_model.log_images(batch, ddim_steps=DDIM_STEPS)

#         # 处理并保存生成的图像
#         sample = log["samples"].squeeze(0) # 去掉batch维度
#         sample = torch.clamp(sample, -1, 1)
#         sample = (sample + 1.0) / 2.0  # 从[-1, 1]转换到[0, 1]
#         sample = sample.permute(1, 2, 0).cpu().numpy()
#         sample = (sample * 255).astype(np.uint8)
        
#         img = Image.fromarray(sample)
#         img.save(OUTPUT_FILENAME)
#         print(f"Inference complete. Image saved as '{OUTPUT_FILENAME}'")
#         # img.show() # 如果在本地环境，可以取消注释以直接显示图片

# if __name__ == "__main__":
#     # 确保CURL证书路径设置正确，以防模型下载需要
#     os.environ['CURL_CA_BUNDLE'] = ''
#     main()

