# # # # 文件名: 1_offline_analysis.py (最终版)
# # # # 作用: 为动态剪枝门控网络准备训练数据

# # # import torch
# # # import torch.nn as nn
# # # from torch.utils.data import DataLoader
# # # from tqdm import tqdm
# # # import os
# # # from typing import List, Dict, Any, Tuple

# # # # --- 核心模型与工具从FCDiffusion项目中导入 ---
# # # from fcdiffusion.model import create_model, load_state_dict
# # # from fcdiffusion.dataset import TrainDataset
# # # from ldm.modules.diffusionmodules.openaimodel import ResBlock

# # # # --- 从我们自己创建的文件中导入度量计算器 ---
# # # from my_metrics import ClipSimilarity, DinoVitSimilarity


# # # # ----------------------------------------------------
# # # # 1. 配置区域
# # # # ----------------------------------------------------
# # # RESUME_PATH: str = 'models/FCDiffusion_ini.ckpt'
# # # CONFIG_PATH: str = 'configs/model_config.yaml'
# # # BATCH_SIZE: int = 1
# # # NUM_SAMPLES_PER_TASK: int = 100
# # # OUTPUT_DATASET_PATH: str = "importance_dataset.pt"
# # # DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # TASK_TO_CONTROL_MODE: Dict[str, str] = {
# # #     "style_translation": "high_pass",
# # #     "style_creation": "mini_pass",
# # #     "semantic_manipulation": "low_pass",
# # #     "scene_translation": "mid_pass" 
# # # }

# # # # ----------------------------------------------------
# # # # 2. 任务专属损失函数定义 (新版)
# # # # ----------------------------------------------------

# # # def get_task_specific_loss(
# # #     task_name: str, 
# # #     generated_img_tensor: torch.Tensor, 
# # #     source_img_tensor: torch.Tensor, 
# # #     text_prompt: str,
# # #     clip_calculator: ClipSimilarity,
# # #     dino_calculator: DinoVitSimilarity
# # # ) -> torch.Tensor:
# # #     """根据任务名称，使用传入的计算器计算损失"""

# # #     # FCDiffusion的decode_first_stage输出范围是[-1, 1], 需要转为[0, 1]
# # #     generated_img_tensor = (generated_img_tensor.clamp(-1, 1) + 1) / 2.0
# # #     source_img_tensor = (source_img_tensor.clamp(-1, 1) + 1) / 2.0

# # #     if task_name in ["style_translation", "semantic_manipulation", "scene_translation"]:
# # #         clip_sim = clip_calculator.calculate(generated_img_tensor, text_prompt)
# # #         dino_sim = dino_calculator.calculate(generated_img_tensor, source_img_tensor)
# # #         loss = - (0.5 * clip_sim + 0.5 * dino_sim)
# # #         return torch.tensor(loss, device=DEVICE)

# # #     elif task_name == "style_creation":
# # #         clip_sim = clip_calculator.calculate(generated_img_tensor, text_prompt)
# # #         dino_sim = dino_calculator.calculate(generated_img_tensor, source_img_tensor)
# # #         dino_dist = 1.0 - dino_sim
# # #         loss = - (0.5 * clip_sim + 0.5 * dino_dist)
# # #         return torch.tensor(loss, device=DEVICE)
# # #     else:
# # #         raise ValueError(f"未知的任务名称: {task_name}")


# # # # ----------------------------------------------------
# # # # 3. 主分析逻辑
# # # # ----------------------------------------------------

# # # def main():
# # #     print(f"使用的设备: {DEVICE}")

# # #     # --- 一次性加载所有模型 ---
# # #     print("加载 FCDiffusion 模型...")
# # #     model = create_model(CONFIG_PATH).cpu()
# # #     model.load_state_dict(load_state_dict(RESUME_PATH, location='cpu'))
# # #     model = model.to(DEVICE)
# # #     model.eval()

# # #     print("加载度量计算器模型...")
# # #     clip_calculator = ClipSimilarity(device=DEVICE)
# # #     dino_calculator = DinoVitSimilarity(device=DEVICE)

# # #     print("加载 LAION 数据集...")
# # #     dataset = TrainDataset('datasets/training_data.json',cache_size=1000)
# # #     dataloader = DataLoader(dataset, num_workers=0, batch_size=BATCH_SIZE, shuffle=True)
    
# # #     prunable_modules: List[Tuple[str, nn.Module]] = []
# # #     for name, module in model.control_model.named_modules():
# # #         if isinstance(module, ResBlock):
# # #             prunable_modules.append((name, module))
# # #     if not prunable_modules:
# # #         raise ValueError("在 control_model 中未找到任何 ResBlock 模块。")
# # #     print(f"在 FreqControlNet 中成功识别到 {len(prunable_modules)} 个可分析的 ResBlock 模块。")

# # #     final_dataset: List[Dict[str, torch.Tensor]] = []
# # #     for task_name, control_mode in TASK_TO_CONTROL_MODE.items():
# # #         print(f"\n{'='*20}\n开始分析任务: {task_name} (使用 control_mode: {control_mode})\n{'='*20}")
        
# # #         samples_processed = 0
# # #         pbar = tqdm(dataloader, total=NUM_SAMPLES_PER_TASK, desc=f"任务: {task_name}")
# # #         for batch in pbar:
# # #             if samples_processed >= NUM_SAMPLES_PER_TASK:
# # #                 break

# # #             for key in batch:
# # #                 if isinstance(batch[key], torch.Tensor):
# # #                     batch[key] = batch[key].to(DEVICE)

# # #             try:
# # #                 model.control_mode = control_mode
# # #                 z, c = model.get_input(batch, model.first_stage_key)
# # #                 text_embedding = c['c_crossattn'][0]
# # #                 text_prompt_str = batch['txt'][0]
                
# # #                 activations: Dict[str, torch.Tensor] = {}
# # #                 hooks: List[Any] = []
# # #                 for name, module in prunable_modules:
# # #                     def get_activation(name):
# # #                         def hook(model, input, output):
# # #                             activations[name] = output
# # #                         return hook
# # #                     hooks.append(module.register_forward_hook(get_activation(name)))
                
# # #                 with torch.no_grad():
# # #                     log_dict = model.log_images(batch, N=BATCH_SIZE, ddim_steps=50, unconditional_guidance_scale=9.0)
# # #                 generated_sample_decoded = log_dict['samples'].detach()
                
# # #                 for act in activations.values():
# # #                     act.requires_grad_(True)
                
# # #                 source_img_decoded = model.decode_first_stage(z).detach()
                
# # #                 loss = get_task_specific_loss(
# # #                     task_name, 
# # #                     generated_sample_decoded.squeeze(0), 
# # #                     source_img_decoded.squeeze(0), 
# # #                     text_prompt_str,
# # #                     clip_calculator,
# # #                     dino_calculator
# # #                 )

# # #                 if not activations:
# # #                     print("Warning: No activations were captured.")
# # #                     continue

# # #                 activation_tensors = [activations[name] for name, _ in prunable_modules]
# # #                 gradients = torch.autograd.grad(loss, activation_tensors)

# # #                 importance_vector = [torch.mean(torch.abs(grad)).item() for grad in gradients]
# # #                 max_val = max(importance_vector) if importance_vector else 1.0
# # #                 importance_vector_normalized = torch.tensor([v / (max_val + 1e-8) for v in importance_vector], device='cpu')
                
# # #                 final_dataset.append({
# # #                     "text_embedding": text_embedding.detach().cpu(),
# # #                     "importance_vector": importance_vector_normalized
# # #                 })
                
# # #                 samples_processed += BATCH_SIZE
# # #                 pbar.set_postfix({"已处理": f"{samples_processed}/{NUM_SAMPLES_PER_TASK}", "Loss": f"{loss.item():.4f}"})

# # #             except Exception as e:
# # #                 import traceback
# # #                 print(f"处理批次时发生错误: {e}")
# # #                 traceback.print_exc()
# # #             finally:
# # #                 for hook in hooks:
# # #                     hook.remove()

# # #     print(f"\n分析完成。总共生成 {len(final_dataset)} 条数据。")
# # #     print(f"正在保存数据集到: {OUTPUT_DATASET_PATH}...")
# # #     torch.save(final_dataset, OUTPUT_DATASET_PATH)
# # #     print("成功保存！第一阶段完成。")

# # # if __name__ == "__main__":
# # #     main()


# # # 文件名: 1_offline_analysis.py (修正版)
# # # 作用: 为动态剪枝门控网络准备训练数据
# # # 修正: 移除了 no_grad 上下文，并创建了允许梯度计算的生成函数

# # import torch
# # import torch.nn as nn
# # from torch.utils.data import DataLoader
# # from tqdm import tqdm
# # import os
# # from typing import List, Dict, Any, Tuple

# # # --- 核心模型与工具从FCDiffusion项目中导入 ---
# # from fcdiffusion.model import create_model, load_state_dict
# # from fcdiffusion.dataset import TrainDataset
# # from ldm.modules.attention import SpatialTransformer
# # from ldm.modules.diffusionmodules.openaimodel import ResBlock
# # from ldm.models.diffusion.ddim import DDIMSampler # 导入DDIM采样器

# # # --- 从您自己的文件中导入已实现的度量函数 ---
# # from my_metrics import ClipSimilarity, DinoVitSimilarity


# # # ----------------------------------------------------
# # # 1. 配置区域 (与之前相同)
# # # ----------------------------------------------------
# # RESUME_PATH: str = 'models/FCDiffusion_ini.ckpt'
# # CONFIG_PATH: str = 'configs/model_config.yaml'
# # BATCH_SIZE: int = 1
# # NUM_SAMPLES_PER_TASK: int = 100
# # OUTPUT_DATASET_PATH: str = "importance_dataset.pt"
# # DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # TASK_TO_CONTROL_MODE: Dict[str, str] = {
# #     "style_translation": "high_pass",
# #     "style_creation": "mini_pass",
# #     "semantic_manipulation": "low_pass",
# #     "scene_translation": "mid_pass" 
# # }

# # # ----------------------------------------------------
# # # 2. 任务专属损失函数定义 (与之前相同)
# # # ----------------------------------------------------

# # # def get_task_specific_loss(
# # #     task_name: str, 
# # #     generated_img_tensor: torch.Tensor, 
# # #     source_img_tensor: torch.Tensor, 
# # #     text_prompt: str,
# # #     clip_calculator: ClipSimilarity,
# # #     dino_calculator: DinoVitSimilarity
# # # ) -> torch.Tensor:
# # #     generated_img_tensor = (generated_img_tensor.clamp(-1, 1) + 1) / 2.0
# # #     source_img_tensor = (source_img_tensor.clamp(-1, 1) + 1) / 2.0

# # #     if task_name in ["style_translation", "semantic_manipulation", "scene_translation"]:
# # #         clip_sim = clip_calculator.calculate(generated_img_tensor, text_prompt)
# # #         dino_sim = dino_calculator.calculate(generated_img_tensor, source_img_tensor)
# # #         loss = - (0.5 * clip_sim + 0.5 * dino_sim)
# # #         return torch.tensor(loss, device=DEVICE)

# # #     elif task_name == "style_creation":
# # #         clip_sim = clip_calculator.calculate(generated_img_tensor, text_prompt)
# # #         dino_sim = dino_calculator.calculate(generated_img_tensor, source_img_tensor)
# # #         dino_dist = 1.0 - dino_sim
# # #         loss = - (0.5 * clip_sim + 0.5 * dino_dist)
# # #         return torch.tensor(loss, device=DEVICE)
# # #     else:
# # #         raise ValueError(f"未知的任务名称: {task_name}")


# # # ----------------------------------------------------
# # # 2. 任务专属损失函数定义 (修正版)
# # # ----------------------------------------------------

# # def get_task_specific_loss(
# #     task_name: str, 
# #     generated_img_tensor: torch.Tensor, 
# #     source_img_tensor: torch.Tensor, 
# #     text_prompt: str,
# #     clip_calculator: ClipSimilarity,
# #     dino_calculator: DinoVitSimilarity
# # ) -> torch.Tensor:
# #     generated_img_tensor = (generated_img_tensor.clamp(-1, 1) + 1) / 2.0
# #     source_img_tensor = (source_img_tensor.clamp(-1, 1) + 1) / 2.0

# #     if task_name in ["style_translation", "semantic_manipulation", "scene_translation"]:
# #         clip_sim = clip_calculator.calculate(generated_img_tensor, text_prompt)
# #         dino_sim = dino_calculator.calculate(generated_img_tensor, source_img_tensor)
# #         # 【修正】直接返回计算出的loss张量，保留计算图
# #         loss = - (0.5 * clip_sim + 0.5 * dino_sim)
# #         return loss

# #     elif task_name == "style_creation":
# #         clip_sim = clip_calculator.calculate(generated_img_tensor, text_prompt)
# #         dino_sim = dino_calculator.calculate(generated_img_tensor, source_img_tensor)
# #         dino_dist = 1.0 - dino_sim
# #         # 【修正】直接返回计算出的loss张量，保留计算图
# #         loss = - (0.5 * clip_sim + 0.5 * dino_dist)
# #         return loss
# #     else:
# #         raise ValueError(f"未知的任务名称: {task_name}")
# # # ----------------------------------------------------
# # # 3. 【新增】梯度友好的图像生成函数
# # # ----------------------------------------------------
# # def generate_with_grad(model, cond, batch_size, ddim_steps=50, ddim_eta=0.0, guidance_scale=9.0):
# #     """
# #     此函数模拟 model.log_images 中的核心采样逻辑，但允许梯度计算。
# #     """
# #     # 实例化采样器
# #     ddim_sampler = DDIMSampler(model)
    
# #     # 获取图像尺寸信息
# #     b, c, h, w = cond["c_concat"][0].shape
# #     shape = (model.channels, h, w)
    
# #     # 获取无条件提示，用于CFG (Classifier-Free Guidance)
# #     uc_cross = model.get_unconditional_conditioning(batch_size)
# #     uc_cat = cond["c_concat"][0]
# #     uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}

# #     # 执行采样
# #     # 这是关键：这个调用没有在 no_grad() 上下文中，因此会构建计算图
# #     samples, _ = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, eta=ddim_eta,
# #                                      unconditional_guidance_scale=guidance_scale,
# #                                      unconditional_conditioning=uc_full)
    
# #     # 将隐空间样本解码为图像
# #     decoded_samples = model.decode_first_stage(samples)
# #     return decoded_samples


# # # ----------------------------------------------------
# # # 4. 主分析逻辑 (修正版)
# # # ----------------------------------------------------
# # def main():
# #     print(f"使用的设备: {DEVICE}")

# #     print("加载 FCDiffusion 模型...")
# #     model = create_model(CONFIG_PATH).cpu()
# #     model.load_state_dict(load_state_dict(RESUME_PATH, location='cpu'))
# #     model = model.to(DEVICE)
# #     model.eval()

        
# #     # 手动解冻 control_model 的参数，以确保激活值能够追踪梯度。
# #     # 很多预训练模型在加载时默认会冻结所有权重。
# #     print("Unfreezing control_model parameters to enable gradient tracking...")
# #     for param in model.control_model.parameters():
# #         param.requires_grad = True



# #     # 2. 解冻主 U-Net 的输出模块 (这是控制信号注入的地方)
# #     for param in model.model.diffusion_model.output_blocks.parameters():
# #         param.requires_grad = True

# #     print("加载度量计算器模型...")
# #     clip_calculator = ClipSimilarity(device=DEVICE)
# #     dino_calculator = DinoVitSimilarity(device=DEVICE)

# #     print("加载 LAION 数据集...")
# #     # dataset = TrainDataset()
# #     dataset = TrainDataset('datasets/training_data.json',cache_size=1000)
# #     dataloader = DataLoader(dataset, num_workers=0, batch_size=BATCH_SIZE, shuffle=True)
    
# #     prunable_modules: List[Tuple[str, nn.Module]] = []
# #     for name, module in model.control_model.named_modules():
# #         if isinstance(module,(ResBlock, SpatialTransformer)):
# #             prunable_modules.append((name, module))
# #     print(f"在 FreqControlNet 中成功识别到 {len(prunable_modules)} 个可分析的 ResBlock 模块。")

# #     final_dataset: List[Dict[str, torch.Tensor]] = []
# #     for task_name, control_mode in TASK_TO_CONTROL_MODE.items():
# #         print(f"\n{'='*20}\n开始分析任务: {task_name} (使用 control_mode: {control_mode})\n{'='*20}")
        
# #         samples_processed = 0
# #         pbar = tqdm(dataloader, total=NUM_SAMPLES_PER_TASK, desc=f"任务: {task_name}")
# #         for batch in pbar:
# #             if samples_processed >= NUM_SAMPLES_PER_TASK:
# #                 break

# #             for key in batch:
# #                 if isinstance(batch[key], torch.Tensor):
# #                     batch[key] = batch[key].to(DEVICE)

# #             # 【重要】移除 with torch.no_grad() 上下文
# #             try:
# #                 model.control_mode = control_mode
# #                 z, c = model.get_input(batch, model.first_stage_key)
# #                 text_embedding = c['c_crossattn'][0]
# #                 text_prompt_str = batch['txt'][0]
                
# #                 activations: Dict[str, torch.Tensor] = {}
# #                 hooks: List[Any] = []
# #                 for name, module in prunable_modules:
# #                     def get_activation(name):
# #                         def hook(model, input, output):
# #                             activations[name] = output
# #                         return hook
# #                     hooks.append(module.register_forward_hook(get_activation(name)))
                
# #                 # # 【重要】调用我们新的梯度友好生成函数
# #                 # generated_sample_decoded = generate_with_grad(model, c, BATCH_SIZE)

# #                 # # 移除hooks
# #                 # for hook in hooks:
# #                 #     hook.remove()

# #                 # for act in activations.values():
# #                 #     act.requires_grad_(True)
                
# #                 # source_img_decoded = model.decode_first_stage(z).detach()
                
# #                 # loss = get_task_specific_loss(
# #                 #     task_name, 
# #                 #     generated_sample_decoded.squeeze(0), 
# #                 #     source_img_decoded.squeeze(0), 
# #                 #     text_prompt_str,
# #                 #     clip_calculator,
# #                 #     dino_calculator
# #                 # )

# #                 # if not activations:
# #                 #     print("Warning: No activations were captured.")
# #                 #     continue

# #                 # activation_tensors = [activations[name] for name, _ in prunable_modules]
                
# #                 # # 【重要】现在这个求导应该可以正常工作了
# #                 # gradients = torch.autograd.grad(loss, activation_tensors, allow_unused=True)

# #                 generated_sample_decoded = generate_with_grad(model, c, BATCH_SIZE)

# #                 # 移除hooks
# #                 for hook in hooks:
# #                     hook.remove()

# #                 # 【修正】移除下面这个循环。
# #                 # 因为activations中的张量在生成时已经是计算图的一部分，
# #                 # 无需也无法在事后手动设置 requires_grad。
# #                 # for act in activations.values():
# #                 #     act.requires_grad_(True)
                
# #                 source_img_decoded = model.decode_first_stage(z).detach()
                
# #                 # 现在 get_task_specific_loss 会返回一个带有计算图的loss
# #                 loss = get_task_specific_loss(
# #                     task_name, 
# #                     generated_sample_decoded.squeeze(0), # 保持和原来一致，但注意维度
# #                     source_img_decoded.squeeze(0), 
# #                     text_prompt_str,
# #                     clip_calculator,
# #                     dino_calculator
# #                 )

# #                 # if not activations:
# #                 #     print("Warning: No activations were captured.")
# #                 #     continue

# #                 # # 从字典中按顺序获取张量，以匹配 prunable_modules 的顺序
# #                 # activation_tensors = [activations[name] for name, _ in prunable_modules]
                
# #                 # gradients = torch.autograd.grad(loss, activation_tensors, allow_unused=True)
# #                             # --- 健全性检查代码 ---
# #                 # print("\n--- Sanity Check ---")
# #                 # if not activations:
# #                 #     print("!! 警告: 没有捕获到任何激活张量。")
# #                 # else:
# #                 #     # 检查我们想要微分的张量中，是否至少有一个 requires_grad
# #                 #     any_requires_grad = any(t.requires_grad for t in activations if t is not None)
# #                 #     if any_requires_grad:
# #                 #         print("✅ 成功: 至少有一个激活张量成功设置了 requires_grad=True。")
# #                 #     else:
# #                 #         print("❌ 失败: 所有捕获到的激活张量都设置了 requires_grad=False。")
# #                 #         print("   这就是导致下一行代码报错的直接原因。请检查解冻逻辑是否正确执行。")
# #                 # print("--- End Sanity Check ---\n")
                
# #                 # # 【重要】现在这个求导应该可以正常工作了
# #                 # gradients = torch.autograd.grad(loss, activations, allow_unused=True)

# #                 # importance_vector = []
# #                 # for grad in gradients:

# #                 # ... (在 main 函数的 for batch in pbar: 循环内部)

# #                 if not activations:
# #                     print("Warning: No activations were captured.")
# #                     continue

# #                 # ↓↓↓ 这里就是 activation_tensors 的定义之处 ↓↓↓
# #                 activation_tensors = [activations[name] for name, _ in prunable_modules]
                
# #                 # 【重要】现在这个求导应该可以正常工作了
# #                 gradients = torch.autograd.grad(loss, activation_tensors, allow_unused=True)

# #                 importance_vector = []
# #                 for grad in gradients:

# #                     if grad is not None:
# #                         importance_vector.append(torch.mean(torch.abs(grad)).item())
# #                     else:
# #                         importance_vector.append(0.0) # 如果某个激活值与loss无关，其梯度为None

# #                 max_val = max(importance_vector) if importance_vector else 1.0
# #                 importance_vector_normalized = torch.tensor([v / (max_val + 1e-8) for v in importance_vector], device='cpu')
                
# #                 final_dataset.append({
# #                     "text_embedding": text_embedding.detach().cpu(),
# #                     "importance_vector": importance_vector_normalized
# #                 })
                
# #                 samples_processed += BATCH_SIZE
# #                 pbar.set_postfix({"已处理": f"{samples_processed}/{NUM_SAMPLES_PER_TASK}", "Loss": f"{loss.item():.4f}"})

# #             except Exception as e:
# #                 import traceback
# #                 print(f"处理批次时发生错误: {e}")
# #                 traceback.print_exc()

# #     print(f"\n分析完成。总共生成 {len(final_dataset)} 条数据。")
# #     print(f"正在保存数据集到: {OUTPUT_DATASET_PATH}...")
# #     torch.save(final_dataset, OUTPUT_DATASET_PATH)
# #     print("成功保存！第一阶段完成。")

# # if __name__ == "__main__":
# #     main()


# # 文件名: 1_offline_analysis.py (最终完整版)
# # 作用: 为动态剪枝门控网络准备训练数据

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import os
# from typing import List, Dict, Any, Tuple

# # --- 核心模型与工具从FCDiffusion项目中导入 ---
# from fcdiffusion.model import create_model, load_state_dict
# from fcdiffusion.dataset import TrainDataset
# from ldm.modules.attention import SpatialTransformer
# from ldm.modules.diffusionmodules.openaimodel import ResBlock
# from ldm.models.diffusion.ddim import DDIMSampler # 导入DDIM采样器

# # --- 从您自己的文件中导入已实现的度量函数 (请确保 my_metrics.py 是我们修正后的版本) ---
# from my_metrics import ClipSimilarity, DinoVitSimilarity


# # ----------------------------------------------------
# # 1. 配置区域
# # ----------------------------------------------------
# RESUME_PATH: str = 'models/FCDiffusion_ini.ckpt'
# CONFIG_PATH: str = 'configs/model_config.yaml'
# BATCH_SIZE: int = 1
# NUM_SAMPLES_PER_TASK: int = 100
# OUTPUT_DATASET_PATH: str = "importance_dataset.pt"
# DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DDIM_STEPS: int = 50 # DDIM步数，定义在这里方便后续使用

# TASK_TO_CONTROL_MODE: Dict[str, str] = {
#     "style_translation": "high_pass",
#     "style_creation": "mini_pass",
#     "semantic_manipulation": "low_pass",
#     "scene_translation": "mid_pass" 
# }

# # ----------------------------------------------------
# # 2. 任务专属损失函数定义 (可微版)
# # ----------------------------------------------------
# def get_task_specific_loss(
#     task_name: str, 
#     generated_img_tensor: torch.Tensor, 
#     source_img_tensor: torch.Tensor, 
#     text_prompt: str,
#     clip_calculator: ClipSimilarity,
#     dino_calculator: DinoVitSimilarity
# ) -> torch.Tensor:
#     generated_img_tensor = (generated_img_tensor.clamp(-1, 1) + 1) / 2.0
#     source_img_tensor = (source_img_tensor.clamp(-1, 1) + 1) / 2.0

#     if task_name in ["style_translation", "semantic_manipulation", "scene_translation"]:
#         clip_sim = clip_calculator.calculate(generated_img_tensor, text_prompt)
#         dino_sim = dino_calculator.calculate(generated_img_tensor, source_img_tensor)
#         loss = - (0.5 * clip_sim + 0.5 * dino_sim)
#         return loss

#     elif task_name == "style_creation":
#         clip_sim = clip_calculator.calculate(generated_img_tensor, text_prompt)
#         dino_sim = dino_calculator.calculate(generated_img_tensor, source_img_tensor)
#         dino_dist = 1.0 - dino_sim
#         loss = - (0.5 * clip_sim + 0.5 * dino_dist)
#         return loss
#     else:
#         raise ValueError(f"未知的任务名称: {task_name}")

# # ----------------------------------------------------
# # 3. 梯度友好的图像生成函数
# # ----------------------------------------------------
# def generate_with_grad(model, cond, batch_size, ddim_steps=DDIM_STEPS, ddim_eta=0.0, guidance_scale=9.0):
#     """
#     此函数模拟 model.log_images 中的核心采样逻辑，但允许梯度计算。
#     """
#     ddim_sampler = DDIMSampler(model)
#     b, c, h, w = cond["c_concat"][0].shape
#     shape = (model.channels, h, w)
    
#     uc_cross = model.get_unconditional_conditioning(batch_size)
#     uc_cat = cond["c_concat"][0]
#     uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}

#     samples, _ = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, eta=ddim_eta,
#                                      unconditional_guidance_scale=guidance_scale,
#                                      unconditional_conditioning=uc_full)
    
#     decoded_samples = model.decode_first_stage(samples)
#     return decoded_samples


# # ----------------------------------------------------
# # 4. 主分析逻辑
# # ----------------------------------------------------
# def main():
#     print(f"使用的设备: {DEVICE}")

#     print("加载 FCDiffusion 模型...")
#     model = create_model(CONFIG_PATH).cpu()
#     model.load_state_dict(load_state_dict(RESUME_PATH, location='cpu'))
#     model = model.to(DEVICE)
#     model.eval()

#     print("Unfreezing model parts required for gradient analysis...")
#     # 1. 解冻整个 control_model
#     for param in model.control_model.parameters():
#         param.requires_grad = True
#     # 2. 解冻主 U-Net 的输出模块 (这是控制信号注入的地方)
#     for param in model.model.diffusion_model.output_blocks.parameters():
#         param.requires_grad = True

#     # 【最终修正】将 prunable_modules 的定义移到此处，方便后续初始化字典
#     prunable_modules: List[Tuple[str, nn.Module]] = []
#     for name, module in model.control_model.named_modules():
#         if isinstance(module,(ResBlock, SpatialTransformer)):
#             prunable_modules.append((name, module))
#     print(f"在 FreqControlNet 中成功识别到 {len(prunable_modules)} 个可分析的模块。")

#     print("加载度量计算器模型...")
#     clip_calculator = ClipSimilarity(device=DEVICE)
#     dino_calculator = DinoVitSimilarity(device=DEVICE)

#     print("加载 LAION 数据集...")
#     dataset = TrainDataset('datasets/training_data.json', cache_size=1000)
#     dataloader = DataLoader(dataset, num_workers=0, batch_size=BATCH_SIZE, shuffle=True)
    
#     final_dataset: List[Dict[str, torch.Tensor]] = []
#     for task_name, control_mode in TASK_TO_CONTROL_MODE.items():
#         print(f"\n{'='*20}\n开始分析任务: {task_name} (使用 control_mode: {control_mode})\n{'='*20}")
        
#         samples_processed = 0
#         pbar = tqdm(dataloader, total=NUM_SAMPLES_PER_TASK, desc=f"任务: {task_name}")
#         for batch in pbar:
#             if samples_processed >= NUM_SAMPLES_PER_TASK:
#                 break

#             for key in batch:
#                 if isinstance(batch[key], torch.Tensor):
#                     batch[key] = batch[key].to(DEVICE)
            
#             try:
#                 model.control_mode = control_mode
#                 z, c = model.get_input(batch, model.first_stage_key)
#                 text_embedding = c['c_crossattn'][0]
#                 text_prompt_str = batch['txt'][0]
                
#                 # --- 【最终修正 - 第1步：初始化用于累积的字典】 ---
#                 activations: Dict[str, List[torch.Tensor]] = {name: [] for name, _ in prunable_modules}
#                 hooks: List[Any] = []

#                 # --- 【最终修正 - 第2步：修改钩子函数以累积激活值】 ---
#                 for name, module in prunable_modules:
#                     def get_activation(name):
#                         def hook(model, input, output):
#                             activations[name].append(output) # 使用 .append() 而非覆盖
#                         return hook
#                     hooks.append(module.register_forward_hook(get_activation(name)))
                
#                 generated_sample_decoded = generate_with_grad(model, c, BATCH_SIZE)

#                 for hook in hooks:
#                     hook.remove()

#                 source_img_decoded = model.decode_first_stage(z).detach()
                
#                 loss = get_task_specific_loss(
#                     task_name, 
#                     generated_sample_decoded.squeeze(0),
#                     source_img_decoded.squeeze(0), 
#                     text_prompt_str,
#                     clip_calculator,
#                     dino_calculator
#                 )

#                 if not any(activations.values()):
#                     print("Warning: No activations were captured.")
#                     continue

#                 # --- 【最终修正 - 第3步：只选择有条件的激活值】 ---
#                 # CFG 会让每个钩子触发 2 * DDIM_STEPS 次。
#                 # 偶数索引 (0, 2, 4...) 来自有条件的传播，是我们需要的。
#                 conditional_activations = []
#                 for name, _ in prunable_modules:
#                     conditional_activations.extend(activations[name][0::2])
                
#                 activation_tensors = conditional_activations

#                 # 使用只包含“干净”张量的列表进行梯度计算
#                 gradients = torch.autograd.grad(loss, activation_tensors, allow_unused=True)

#                 importance_vector = []
#                 grad_idx = 0
#                 for name, _ in prunable_modules:
#                     # 每个模块在每个有条件步骤都会产生一个梯度
#                     num_activations_for_module = len(activations[name][0::2])
                    
#                     module_grads = gradients[grad_idx : grad_idx + num_activations_for_module]
                    
#                     # 计算该模块在所有时间步上的平均重要性
#                     module_importance = 0.0
#                     valid_grads = 0
#                     for grad in module_grads:
#                         if grad is not None:
#                             module_importance += torch.mean(torch.abs(grad)).item()
#                             valid_grads += 1
                    
#                     if valid_grads > 0:
#                         importance_vector.append(module_importance / valid_grads)
#                     else:
#                         importance_vector.append(0.0)
                        
#                     grad_idx += num_activations_for_module

#                 max_val = max(importance_vector) if importance_vector else 1.0
#                 importance_vector_normalized = torch.tensor([v / (max_val + 1e-8) for v in importance_vector], device='cpu')
                
#                 final_dataset.append({
#                     "text_embedding": text_embedding.detach().cpu(),
#                     "importance_vector": importance_vector_normalized
#                 })
                
#                 samples_processed += BATCH_SIZE
#                 pbar.set_postfix({"已处理": f"{samples_processed}/{NUM_SAMPLES_PER_TASK}", "Loss": f"{loss.item():.4f}"})

#             except Exception as e:
#                 import traceback
#                 print(f"处理批次时发生错误: {e}")
#                 traceback.print_exc()

#     print(f"\n分析完成。总共生成 {len(final_dataset)} 条数据。")
#     print(f"正在保存数据集到: {OUTPUT_DATASET_PATH}...")
#     torch.save(final_dataset, OUTPUT_DATASET_PATH)
#     print("成功保存！第一阶段完成。")

# if __name__ == "__main__":
#     main()


# 文件名: 1_offline_analysis.py (最终决战版)
# 作用: 为动态剪枝门控网络准备训练数据

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from typing import List, Dict, Any, Tuple
import numpy as np

# --- 核心模型与工具从FCDiffusion项目中导入 ---
from fcdiffusion.model import create_model, load_state_dict
from fcdiffusion.dataset import TrainDataset
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import ResBlock
from ldm.models.diffusion.ddim import DDIMSampler

# --- 从您自己的文件中导入已实现的度量函数 ---
from my_metrics import ClipSimilarity, DinoVitSimilarity
import gc


# ----------------------------------------------------
# 1. 配置区域
# ----------------------------------------------------
RESUME_PATH: str = 'models/FCDiffusion_ini.ckpt'
CONFIG_PATH: str = 'configs/model_config.yaml'
BATCH_SIZE: int = 1
NUM_SAMPLES_PER_TASK: int = 100
OUTPUT_DATASET_PATH: str = "importance_dataset.pt"
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DDIM_STEPS: int = 50

TASK_TO_CONTROL_MODE: Dict[str, str] = {
    "style_translation": "high_pass",
    "style_creation": "mini_pass",
    "semantic_manipulation": "low_pass",
    "scene_translation": "mid_pass" 
}

# ----------------------------------------------------
# 2. 任务专属损失函数定义 (可微版)
# ----------------------------------------------------
def get_task_specific_loss(
    task_name: str, 
    generated_img_tensor: torch.Tensor, 
    source_img_tensor: torch.Tensor, 
    text_prompt: str,
    clip_calculator: ClipSimilarity,
    dino_calculator: DinoVitSimilarity
) -> torch.Tensor:
    generated_img_tensor = (generated_img_tensor.clamp(-1, 1) + 1) / 2.0
    source_img_tensor = (source_img_tensor.clamp(-1, 1) + 1) / 2.0

    if task_name in ["style_translation", "semantic_manipulation", "scene_translation"]:
        clip_sim = clip_calculator.calculate(generated_img_tensor, text_prompt)
        dino_sim = dino_calculator.calculate(generated_img_tensor, source_img_tensor)
        loss = - (0.5 * clip_sim + 0.5 * dino_sim)
        return loss

    elif task_name == "style_creation":
        clip_sim = clip_calculator.calculate(generated_img_tensor, text_prompt)
        dino_sim = dino_calculator.calculate(generated_img_tensor, source_img_tensor)
        dino_dist = 1.0 - dino_sim
        loss = - (0.5 * clip_sim + 0.5 * dino_dist)
        return loss
    else:
        raise ValueError(f"未知的任务名称: {task_name}")

# ----------------------------------------------------
# 3. 【最终修正】手动实现的可微图像生成函数
# ----------------------------------------------------
# def generate_with_grad_manual_loop(model, cond, batch_size, ddim_steps=DDIM_STEPS, guidance_scale=9.0):
#     """
#     手动实现DDIM采样循环，以确保端到端的可微性，避免内置Sampler的黑箱问题。
#     """
#     # 初始化一个临时的DDIM Sampler，仅用于获取采样步长和alpha等参数
#     ddim_sampler = DDIMSampler(model)
#     ddim_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0.0, verbose=False)
    
#     b, c, h, w = cond["c_concat"][0].shape
#     shape = (model.channels, h, w)
#     device = model.betas.device
    
#     # 获取无条件提示
#     uc_cross = model.get_unconditional_conditioning(batch_size)
#     uc_cat = cond["c_concat"][0]
#     uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}

#     # 从随机噪声开始
#     img = torch.randn(shape, device=device)
    
#     # DDIM采样时间步
#     time_range = np.flip(ddim_sampler.ddim_timesteps)
#     total_steps = ddim_sampler.ddim_timesteps.shape[0]

#     iterator = tqdm(time_range, desc='Manual Diff. Sampler', total=total_steps, leave=False)

#     for i, step in enumerate(iterator):
#         index = total_steps - i - 1
#         ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

#         # 手动执行分类器无关引导 (CFG)
#         # 1. 有条件预测
#         e_t_cond = model.apply_model(img, ts, cond)
#         # 2. 无条件预测
#         e_t_uncond = model.apply_model(img, ts, uc_full)
#         # 3. 组合预测
#         e_t = e_t_uncond + guidance_scale * (e_t_cond - e_t_uncond)

#         # 从DDIM Sampler的p_sample_ddim方法中提取的核心采样逻辑
#         alphas = ddim_sampler.ddim_alphas
#         alphas_prev = ddim_sampler.ddim_alphas_prev
#         sqrt_one_minus_alphas = ddim_sampler.ddim_sqrt_one_minus_alphas
#         sigmas = ddim_sampler.ddim_sigmas

#         a_t = torch.full((batch_size, 1, 1, 1), alphas[index], device=device)
#         a_prev = torch.full((batch_size, 1, 1, 1), alphas_prev[index], device=device)
#         sigma_t = torch.full((batch_size, 1, 1, 1), sigmas[index], device=device)
#         sqrt_one_minus_at = torch.full((batch_size, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

#         pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()
#         dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
#         noise = sigma_t * torch.randn_like(img) * (0.0 if ddim_sampler.ddim_eta == 0.0 else 1.0)
        
#         img = a_prev.sqrt() * pred_x0 + dir_xt + noise

#     decoded_samples = model.decode_first_stage(img)
#     return decoded_samples


# 文件名: 1_offline_analysis.py
# ... (其他代码保持不变) ...

# ----------------------------------------------------
# 3. 【最终修正】手动实现的可微图像生成函数
# ----------------------------------------------------

# 文件名: 1_offline_analysis.py
# ... (其他代码保持不变) ...

# ----------------------------------------------------
# 3. 【最终修正】手动实现的可微图像生成函数
# ----------------------------------------------------
def generate_with_grad_manual_loop(model, cond, batch_size, ddim_steps=DDIM_STEPS, guidance_scale=9.0):
    """
    手动实现DDIM采样循环，以确保端到端的可微性，避免内置Sampler的黑箱问题。
    """
    # 初始化一个临时的DDIM Sampler，仅用于获取采样步长和alpha等参数
    # 【修正】将 ddim_eta 传入 make_schedule
    ddim_sampler = DDIMSampler(model)
    ddim_eta = 0.0 # DDIM's default eta
    ddim_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    
    b, c, h, w = cond["c_concat"][0].shape
    shape = (batch_size, model.channels, h, w)
    device = model.betas.device
    
    uc_cross = model.get_unconditional_conditioning(batch_size)
    uc_cat = cond["c_concat"][0]
    uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}

    img = torch.randn(shape, device=device)
    
    time_range = np.flip(ddim_sampler.ddim_timesteps)
    total_steps = ddim_sampler.ddim_timesteps.shape[0]

    iterator = tqdm(time_range, desc='Manual Diff. Sampler', total=total_steps, leave=False)

    for i, step in enumerate(iterator):
        index = total_steps - i - 1
        ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

        e_t_cond = model.apply_model(img, ts, cond)
        e_t_uncond = model.apply_model(img, ts, uc_full)
        e_t = e_t_uncond + guidance_scale * (e_t_cond - e_t_uncond)

        alphas = ddim_sampler.ddim_alphas
        alphas_prev = ddim_sampler.ddim_alphas_prev
        sqrt_one_minus_alphas = ddim_sampler.ddim_sqrt_one_minus_alphas
        sigmas = ddim_sampler.ddim_sigmas

        a_t = torch.full((batch_size, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((batch_size, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((batch_size, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((batch_size, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        
        # --- 【最终决战版 - Bug修复】 ---
        # 移除多余且错误的对 ddim_eta 的判断，sigma_t 本身已经包含了 eta 的影响
        noise = sigma_t * torch.randn_like(img)
        
        img = a_prev.sqrt() * pred_x0 + dir_xt + noise

    decoded_samples = model.decode_first_stage(img)
    return decoded_samples

# ... (文件的其余部分保持不变) ...
# def generate_with_grad_manual_loop(model, cond, batch_size, ddim_steps=DDIM_STEPS, guidance_scale=9.0):
#     """
#     手动实现DDIM采样循环，以确保端到端的可微性，避免内置Sampler的黑箱问题。
#     """
#     ddim_sampler = DDIMSampler(model)
#     ddim_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0.0, verbose=False)
    
#     b, c, h, w = cond["c_concat"][0].shape
    
#     # --- 【最终决战版 - 关键修正】 ---
#     # shape 必须包含 batch_size 维度，应该是 4D 的
#     shape = (batch_size, model.channels, h, w)
    
#     device = model.betas.device
    
#     uc_cross = model.get_unconditional_conditioning(batch_size)
#     uc_cat = cond["c_concat"][0]
#     uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}

#     # 从随机噪声开始 (现在 img 将是正确的 4D 张量)
#     img = torch.randn(shape, device=device)
    
#     time_range = np.flip(ddim_sampler.ddim_timesteps)
#     total_steps = ddim_sampler.ddim_timesteps.shape[0]

#     iterator = tqdm(time_range, desc='Manual Diff. Sampler', total=total_steps, leave=False)

#     for i, step in enumerate(iterator):
#         index = total_steps - i - 1
#         ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

#         e_t_cond = model.apply_model(img, ts, cond)
#         e_t_uncond = model.apply_model(img, ts, uc_full)
#         e_t = e_t_uncond + guidance_scale * (e_t_cond - e_t_uncond)

#         alphas = ddim_sampler.ddim_alphas
#         alphas_prev = ddim_sampler.ddim_alphas_prev
#         sqrt_one_minus_alphas = ddim_sampler.ddim_sqrt_one_minus_alphas
#         sigmas = ddim_sampler.ddim_sigmas

#         a_t = torch.full((batch_size, 1, 1, 1), alphas[index], device=device)
#         a_prev = torch.full((batch_size, 1, 1, 1), alphas_prev[index], device=device)
#         sigma_t = torch.full((batch_size, 1, 1, 1), sigmas[index], device=device)
#         sqrt_one_minus_at = torch.full((batch_size, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

#         pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()
#         dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
#         noise = sigma_t * torch.randn_like(img) * (0.0 if ddim_sampler.ddim_eta == 0.0 else 1.0)
        
#         img = a_prev.sqrt() * pred_x0 + dir_xt + noise

#     decoded_samples = model.decode_first_stage(img)
#     return decoded_samples



# ----------------------------------------------------
# 4. 主分析逻辑
# ----------------------------------------------------
# def main():
#     print(f"使用的设备: {DEVICE}")

#     print("加载 FCDiffusion 模型...")
#     model = create_model(CONFIG_PATH).cpu()
#     model.load_state_dict(load_state_dict(RESUME_PATH, location='cpu'))
#     model = model.to(DEVICE)
#     model.eval()

#     print("Unfreezing model parts for gradient analysis...")
#     for param in model.control_model.parameters():
#         param.requires_grad = True
#     for param in model.model.diffusion_model.output_blocks.parameters():
#         param.requires_grad = True

#     prunable_modules: List[Tuple[str, nn.Module]] = []
#     for name, module in model.control_model.named_modules():
#         if isinstance(module,(ResBlock, SpatialTransformer)):
#             prunable_modules.append((name, module))
#     print(f"在 FreqControlNet 中成功识别到 {len(prunable_modules)} 个可分析的模块。")

#     print("加载度量计算器模型...")
#     clip_calculator = ClipSimilarity(device=DEVICE)
#     dino_calculator = DinoVitSimilarity(device=DEVICE)

#     print("加载 LAION 数据集...")
#     dataset = TrainDataset('datasets/training_data.json', cache_size=1000)
#     dataloader = DataLoader(dataset, num_workers=0, batch_size=BATCH_SIZE, shuffle=True)
    
#     final_dataset: List[Dict[str, torch.Tensor]] = []
#     for task_name, control_mode in TASK_TO_CONTROL_MODE.items():
#         print(f"\n{'='*20}\n开始分析任务: {task_name} (使用 control_mode: {control_mode})\n{'='*20}")
        
#         samples_processed = 0
#         pbar = tqdm(dataloader, total=NUM_SAMPLES_PER_TASK, desc=f"任务: {task_name}")
#         for batch in pbar:
#             if samples_processed >= NUM_SAMPLES_PER_TASK:
#                 break

#             for key in batch:
#                 if isinstance(batch[key], torch.Tensor):
#                     batch[key] = batch[key].to(DEVICE)
            
#             try:
#                 model.control_mode = control_mode
#                 z, c = model.get_input(batch, model.first_stage_key)
#                 text_embedding = c['c_crossattn'][0]
#                 text_prompt_str = batch['txt'][0]
                
#                 activations: Dict[str, List[torch.Tensor]] = {name: [] for name, _ in prunable_modules}
#                 hooks: List[Any] = []

#                 for name, module in prunable_modules:
#                     def get_activation(name):
#                         def hook(model, input, output):
#                             activations[name].append(output)
#                         return hook
#                     hooks.append(module.register_forward_hook(get_activation(name)))
                
#                 # 【最终修正】调用我们手动实现的可微生成函数
#                 generated_sample_decoded = generate_with_grad_manual_loop(model, c, BATCH_SIZE)

#                 for hook in hooks:
#                     hook.remove()

#                 source_img_decoded = model.decode_first_stage(z).detach()
#                 loss = get_task_specific_loss(
#                     task_name, 
#                     generated_sample_decoded.squeeze(0),
#                     source_img_decoded.squeeze(0), 
#                     text_prompt_str,
#                     clip_calculator,
#                     dino_calculator
#                 )

#                 if not any(activations.values()):
#                     print("Warning: No activations were captured.")
#                     continue

#                 conditional_activations = []
#                 for name, _ in prunable_modules:
#                     # 在手动实现的循环中，CFG的两次调用紧挨着发生
#                     # 所以偶数索引(0, 2, 4...)对应有条件，奇数(1, 3, 5...)对应无条件
#                     conditional_activations.extend(activations[name][0::2])
                
#                 activation_tensors = conditional_activations

#                 gradients = torch.autograd.grad(loss, activation_tensors, allow_unused=True)

#                 importance_vector = []
#                 grad_idx = 0
#                 for name, _ in prunable_modules:
#                     num_activations_for_module = len(activations[name][0::2])
#                     module_grads = gradients[grad_idx : grad_idx + num_activations_for_module]
                    
#                     module_importance = 0.0
#                     valid_grads = 0
#                     for grad in module_grads:
#                         if grad is not None:
#                             module_importance += torch.mean(torch.abs(grad)).item()
#                             valid_grads += 1
                    
#                     if valid_grads > 0:
#                         importance_vector.append(module_importance / valid_grads)
#                     else:
#                         importance_vector.append(0.0)
#                     grad_idx += num_activations_for_module

#                 max_val = max(importance_vector) if importance_vector else 1.0
#                 importance_vector_normalized = torch.tensor([v / (max_val + 1e-8) for v in importance_vector], device='cpu')
                
#                 final_dataset.append({
#                     "text_embedding": text_embedding.detach().cpu(),
#                     "importance_vector": importance_vector_normalized
#                 })
                
#                 samples_processed += BATCH_SIZE
#                 pbar.set_postfix({"已处理": f"{samples_processed}/{NUM_SAMPLES_PER_TASK}", "Loss": f"{loss.item():.4f}"})

#             except Exception as e:
#                 import traceback
#                 print(f"处理批次时发生错误: {e}")
#                 traceback.print_exc()

#     print(f"\n分析完成。总共生成 {len(final_dataset)} 条数据。")
#     print(f"正在保存数据集到: {OUTPUT_DATASET_PATH}...")
#     torch.save(final_dataset, OUTPUT_DATASET_PATH)
#     print("成功保存！第一阶段完成。")


def main():
    print(f"使用的设备: {DEVICE}")

    print("加载 FCDiffusion 模型...")
    model = create_model(CONFIG_PATH).cpu()
    model.load_state_dict(load_state_dict(RESUME_PATH, location='cpu'))
    model = model.to(DEVICE)
    model.eval()

    print("Unfreezing model parts for gradient analysis...")
    for param in model.control_model.parameters():
        param.requires_grad = True
    for param in model.model.diffusion_model.output_blocks.parameters():
        param.requires_grad = True

    prunable_modules: List[Tuple[str, nn.Module]] = []
    for name, module in model.control_model.named_modules():
        if isinstance(module,(ResBlock, SpatialTransformer)):
            prunable_modules.append((name, module))
    print(f"在 FreqControlNet 中成功识别到 {len(prunable_modules)} 个可分析的模块。")

    print("加载度量计算器模型...")
    clip_calculator = ClipSimilarity(device=DEVICE)
    dino_calculator = DinoVitSimilarity(device=DEVICE)

    print("加载 LAION 数据集...")
    dataset = TrainDataset('datasets/training_data.json', cache_size=1000)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=BATCH_SIZE, shuffle=True)
    
    final_dataset: List[Dict[str, torch.Tensor]] = []
    for task_name, control_mode in TASK_TO_CONTROL_MODE.items():
        print(f"\n{'='*20}\n开始分析任务: {task_name} (使用 control_mode: {control_mode})\n{'='*20}")
        
        samples_processed = 0
        pbar = tqdm(dataloader, total=NUM_SAMPLES_PER_TASK, desc=f"任务: {task_name}")
        for batch in pbar:
            if samples_processed >= NUM_SAMPLES_PER_TASK:
                break

            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(DEVICE)
            
            try:
                model.control_mode = control_mode
                z, c = model.get_input(batch, model.first_stage_key)
                text_embedding = c['c_crossattn'][0]
                text_prompt_str = batch['txt'][0]
                
                activations: Dict[str, List[torch.Tensor]] = {name: [] for name, _ in prunable_modules}
                hooks: List[Any] = []

                for name, module in prunable_modules:
                    def get_activation(name):
                        def hook(model, input, output):
                            activations[name].append(output)
                        return hook
                    hooks.append(module.register_forward_hook(get_activation(name)))
                
                generated_sample_decoded = generate_with_grad_manual_loop(model, c, BATCH_SIZE)

                for hook in hooks:
                    hook.remove()

                source_img_decoded = model.decode_first_stage(z).detach()
                loss = get_task_specific_loss(
                    task_name, 
                    generated_sample_decoded.squeeze(0),
                    source_img_decoded.squeeze(0), 
                    text_prompt_str,
                    clip_calculator,
                    dino_calculator
                )

                if not any(activations.values()):
                    print("Warning: No activations were captured.")
                    continue

                conditional_activations = []
                for name, _ in prunable_modules:
                    conditional_activations.extend(activations[name][0::2])
                
                activation_tensors = conditional_activations

                gradients = torch.autograd.grad(loss, activation_tensors, allow_unused=True)

                importance_vector = []
                grad_idx = 0
                for name, _ in prunable_modules:
                    num_activations_for_module = len(activations[name][0::2])
                    module_grads = gradients[grad_idx : grad_idx + num_activations_for_module]
                    
                    module_importance = 0.0
                    valid_grads = 0
                    for grad in module_grads:
                        if grad is not None:
                            module_importance += torch.mean(torch.abs(grad)).item()
                            valid_grads += 1
                    
                    if valid_grads > 0:
                        importance_vector.append(module_importance / valid_grads)
                    else:
                        importance_vector.append(0.0)
                    grad_idx += num_activations_for_module

                max_val = max(importance_vector) if importance_vector else 1.0
                importance_vector_normalized = torch.tensor([v / (max_val + 1e-8) for v in importance_vector], device='cpu')
                
                final_dataset.append({
                    "text_embedding": text_embedding.detach().cpu(),
                    "importance_vector": importance_vector_normalized
                })
                
                samples_processed += BATCH_SIZE
                pbar.set_postfix({"已处理": f"{samples_processed}/{NUM_SAMPLES_PER_TASK}", "Loss": f"{loss.item():.4f}"})

            except Exception as e:
                import traceback
                print(f"处理批次时发生错误: {e}")
                traceback.print_exc()
            
            finally:
                # 【内存优化】无论成功与否，在每次循环后都执行清理
                # 删除本轮循环中占用大量显存的变量
                if 'generated_sample_decoded' in locals():
                    del generated_sample_decoded
                if 'loss' in locals():
                    del loss
                if 'conditional_activations' in locals():
                    del conditional_activations
                if 'activation_tensors' in locals():
                    del activation_tensors
                if 'gradients' in locals():
                    del gradients
                
                # 强制PyTorch清空CUDA缓存
                torch.cuda.empty_cache()
                # 强制Python进行垃圾回收
                gc.collect()

    print(f"\n分析完成。总共生成 {len(final_dataset)} 条数据。")
    print(f"正在保存数据集到: {OUTPUT_DATASET_PATH}...")
    torch.save(final_dataset, OUTPUT_DATASET_PATH)
    print("成功保存！第一阶段完成。")

if __name__ == "__main__":
    main()