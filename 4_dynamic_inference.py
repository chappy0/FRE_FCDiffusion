# 文件名: 4_dynamic_inference.py (最终决定版)
import torch
import torch.nn as nn
from PIL import Image
from torchvision.utils import save_image
import os
from tqdm import tqdm
import re
from typing import List, Tuple

# --- 核心模型与工具从FCDiffusion项目中导入 ---
from fcdiffusion.model import create_model, load_state_dict
from fcdiffusion.dataset import TestDataset
from torch.utils.data import DataLoader
from ldm.modules.diffusionmodules.openaimodel import ResBlock
from ldm.modules.attention import SpatialTransformer
# 【重要】导入我们自定义的门控模块
from fcdiffusion.dynamic_modules import GatedResBlock, GatedSpatialTransformer

# ----------------------------------------------------
# 1. 配置区域
# ----------------------------------------------------
MODEL_CONFIG_PATH: str = 'configs/model_config.yaml'

# 【核心设定】为本次批量推理手动指定一个固定的Hint处理模式
USER_SPECIFIED_HINT_MODE: str = "high_pass"
BASE_EXPERT_MODEL_PATH: str = r'D:\paper\FRE_FCD\lightning_logs_SA\fcdiffusion_high_pass_checkpoint\epoch=10-step=13999.ckpt'    #'D:\paper\FRE_FCD\lightning_logs_SA\fcdiffusion_low_pass_checkpoint\epoch=2-step=7999.ckpt'

GATING_NETWORK_CHECKPOINT_PATH: str = 'models/FCDiffusion_dynamic_best.ckpt'
INPUT_DATA_DIR: str = r'D:\paper\FCDiffusion_code-main\datasets\test_compare'
OUTPUT_DIR: str = f"dynamic_outputs_{USER_SPECIFIED_HINT_MODE}_mode_FINAL"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------
# 2. 关键辅助函数 (最终决定版)
# ----------------------------------------------------
def load_dynamic_model_with_expert_weights(model_config_path: str, expert_ckpt_path: str, gating_ckpt_path: str) -> nn.Module:
    """
    【最终决定版】权重加载函数，采用最鲁棒的模式匹配逻辑。
    """
    print("--> 1. 创建动态模型结构...")
    dynamic_model = create_model(model_config_path).cpu()
    dynamic_state_dict = dynamic_model.state_dict()

    print(f"--> 2. 加载专家模型的“身体”权重: {os.path.basename(expert_ckpt_path)}")
    expert_state_dict = load_state_dict(expert_ckpt_path, location='cpu')

    print("--> 3. 【鲁棒注入】“身体”权重...")
    final_body_state_dict = {}
    loaded_count = 0
    
    for expert_key, expert_value in expert_state_dict.items():
        # 跳过我们手动训练的gating_network
        if 'gating_network' in expert_key:
            continue
            
        # 尝试将expert_key翻译成所有可能的dynamic_key并检查是否存在
        
        # 模式1: 直接匹配 (用于Downsample, initial conv, zero_convs等)
        key_direct = expert_key
        
        # 模式2: ResBlock的键名翻译
        key_resblock = re.sub(r'(\.input_blocks\.\d+\.0)\.', r'\1.original_resblock.', expert_key)
        key_resblock = re.sub(r'(\.middle_block\.[02])\.', r'\1.original_resblock.', key_resblock)

        # 模式3: Transformer的键名翻译
        key_transformer = re.sub(r'(\.input_blocks\.\d+\.1)\.', r'\1.original_transformer.', expert_key)
        key_transformer = re.sub(r'(\.middle_block\.1)\.', r'\1.original_transformer.', key_transformer)

        if key_resblock in dynamic_state_dict and key_resblock != expert_key:
            final_body_state_dict[key_resblock] = expert_value
            loaded_count += 1
        elif key_transformer in dynamic_state_dict and key_transformer != expert_key:
            final_body_state_dict[key_transformer] = expert_value
            loaded_count += 1
        elif key_direct in dynamic_state_dict:
            final_body_state_dict[key_direct] = expert_value
            loaded_count += 1
        else:
            # 只有所有模式都匹配不上时，才打印警告
            print(f"    [警告] 权重 '{expert_key}' 无法匹配任何已知模式，已被忽略。")

    print(f"    共 {loaded_count} 个专家权重参数被识别并准备注入。")
    missing_keys, unexpected_keys = dynamic_model.load_state_dict(final_body_state_dict, strict=False)
    
    if unexpected_keys:
        print(f"    [警告] 加载时发现 {len(unexpected_keys)} 个意外的权重。")

    print("    “身体”权重注入成功。")

    print(f"--> 4. 加载“决策大脑”权重: {os.path.basename(gating_ckpt_path)}")
    gating_state_dict = torch.load(gating_ckpt_path, map_location='cpu')
    dynamic_model.control_model.gating_network.load_state_dict(gating_state_dict)
    print("    “决策大脑”权重加载成功。")

    print("\n动态自适应模型已完全准备就绪！\n")
    return dynamic_model


def load_image_prompt_pairs(directory: str) -> List[Tuple[str, str]]:
    """遍历文件夹，加载图文对"""
    pairs = []
    print(f"从 '{directory}' 加载图文对...")
    if not os.path.exists(directory):
        print(f"错误: 找不到输入文件夹 '{directory}'")
        return pairs
    filenames = sorted(os.listdir(directory))
    for filename in filenames:
        base, ext = os.path.splitext(filename)
        if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_path = os.path.join(directory, filename)
            text_path = os.path.join(directory, base + '.txt')
            if os.path.exists(text_path):
                with open(text_path, 'r', encoding='utf-8') as f: prompt = f.read().strip()
                if prompt: pairs.append((image_path, prompt))
    print(f"成功找到 {len(pairs)} 个图文对。")
    return pairs

# ----------------------------------------------------
# 3. 主推理逻辑
# ----------------------------------------------------
def main():
    print(f"使用设备: {DEVICE}")

    model = load_dynamic_model_with_expert_weights(
        MODEL_CONFIG_PATH,
        BASE_EXPERT_MODEL_PATH,
        GATING_NETWORK_CHECKPOINT_PATH
    )
    model = model.to(DEVICE)
    model.eval()
    
    image_prompt_pairs = load_image_prompt_pairs(INPUT_DATA_DIR)
    if not image_prompt_pairs: return

    for i, (image_path, prompt) in enumerate(tqdm(image_prompt_pairs, desc=f"批量推理 ({USER_SPECIFIED_HINT_MODE} mode)")):
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n{'='*20}\n正在处理: {base_filename} | 提示: '{prompt}'")
        
        model.control_mode = USER_SPECIFIED_HINT_MODE
        print(f"用户指定Hint模式 -> 使用 '{model.control_mode}' 模式处理hint。")

        dataset = TestDataset(image_path, prompt, 1)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        for batch in dataloader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor): batch[key] = batch[key].to(DEVICE)
            
            with torch.no_grad():
                log_dict = model.log_images(batch, N=1, ddim_steps=50, unconditional_guidance_scale=9.0)
                generated_image = log_dict['samples']
                
                generated_image = (generated_image.clamp(-1.0, 1.0) + 1.0) / 2.0
                output_filename = f"{base_filename}_output.png"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                save_image(generated_image, output_path)
                print(f"图像已保存到: {output_path}")
            break

if __name__ == "__main__":
    main()