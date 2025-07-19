# import torch
# import torch.nn.functional as F
# import pytorch_lightning as pl
# from omegaconf import OmegaConf
# import random
# import torch.nn as nn
# import os
# import numpy as np
# from PIL import Image
# from torch.utils.data import DataLoader, Dataset
# from pytorch_lightning.callbacks import ModelCheckpoint

# # --- 依赖项 ---
# # 确保这些import路径与您的项目结构一致
# # 下面的 fcdiffusion.py 和 openaimodel.py 就是您提供的两个文件
# from fcdiffusion.fcdiffusion import FCDiffusion, FreqControlNet, ControlledUnetModel
# from fcdiffusion.dataset import TrainDataset
# from fcdiffusion.logger import DistillationImageLogger
# from ldm.util import instantiate_from_config
# from fcdiffusion.model import load_state_dict


# # ----------------------------------------------------------------------------------
# # 1. 辅助函数 (add_hook 和 get_activation) - (无变化)
# # ----------------------------------------------------------------------------------
# def get_activation(mem, name):
#     def get_output_hook(module, input, output):
#         mem[name] = output
#     return get_output_hook

# def add_hook(net, mem, mapping_layers):
#     for n, m in net.named_modules():
#         if n in mapping_layers:
#             m.register_forward_hook(get_activation(mem, n))

# # ----------------------------------------------------------------------------------
# # 2. 核心蒸馏器：DecoupledDistiller (全新重写)
# # ----------------------------------------------------------------------------------
# class DecoupledDistiller(pl.LightningModule):
#     """
#     Implements Decoupled Distillation for Dual-Network Controllable Models.
#     This module distills knowledge by decoupling the problem into:
#     1. Distilling the main generation U-Net (ControlledUnetModel) via block-wise matching.
#     2. Distilling the control network (FreqControlNet) via output matching.
#     """
#     def __init__(
#         self,
#         teacher_config_path: str,
#         teacher_ckpt_path: str,
#         student_config_path: str,
#         student_ckpt_path: str,
#         distill_mode: str,
#         learning_rate: float = 1e-5,
#         lambda_sd: float = 1.0,
#         lambda_kd_unet: float = 1.0,
#         lambda_kd_control: float = 1.0,
#     ):
#         super().__init__()
#         self.save_hyperparameters()
#         self.automatic_optimization = True

#         # --- 模型加载 ---
#         print("Loading Student Model...")
#         student_cfg = OmegaConf.load(self.hparams.student_config_path)
#         self.student_model = self.load_model_from_config(student_cfg, self.hparams.student_ckpt_path)
#         self.student_model.control_mode = self.hparams.distill_mode

#         print("Loading Teacher Model...")
#         teacher_cfg = OmegaConf.load(self.hparams.teacher_config_path)
#         self.teacher_model = self.load_model_from_config(teacher_cfg, self.hparams.teacher_ckpt_path)
#         self.teacher_model.control_mode = self.hparams.distill_mode
#         self.teacher_model.requires_grad_(False)
#         self.teacher_model.eval()

#         # 冻结学生模型的非训练部分
#         if hasattr(self.student_model, 'first_stage_model'):
#             self.student_model.first_stage_model.requires_grad_(False)
#         if hasattr(self.student_model, 'cond_stage_model'):
#             self.student_model.cond_stage_model.requires_grad_(False)

#         # --- 设置蒸馏所需的Hooks ---
#         self._setup_distillation_hooks()

#     def _setup_distillation_hooks(self):
#         """Defines layers for the main U-Net block-wise distillation."""
#         self.acts_tea_unet, self.acts_stu_unet = {}, {}

#         # 使用正确的层级路径来定义U-Net模块化蒸馏的目标
#         # 根据 openaimodel.py, UNetModel有12个input_blocks, 1个middle_block, 12个output_blocks
#         self.unet_block_layers = (
#             [f'model.diffusion_model.input_blocks.{i}' for i in range(12)] +
#             ['model.diffusion_model.middle_block'] +
#             [f'model.diffusion_model.output_blocks.{i}' for i in range(12)]
#         )

#         print("Attaching hooks for main U-Net block-wise distillation...")
#         add_hook(self.teacher_model, self.acts_tea_unet, self.unet_block_layers)
#         add_hook(self.student_model, self.acts_stu_unet, self.unet_block_layers)

#     def training_step(self, batch, batch_idx):
#         # 1. 准备共享输入 (z_noisy, t, hint, cond_txt)
#         with torch.no_grad():
#             z0, c_dict = self.teacher_model.get_input(batch, self.teacher_model.first_stage_key)
#             hint = torch.cat(c_dict['c_concat'], 1)
#             cond_txt = torch.cat(c_dict['c_crossattn'], 1)

#         t = torch.randint(0, self.teacher_model.num_timesteps, (z0.shape[0],), device=self.device).long()
#         noise = torch.randn_like(z0)
#         z_noisy = self.teacher_model.q_sample(x_start=z0, t=t, noise=noise)

#         # 2. 解耦蒸馏：分别计算教师和学生的控制信号和最终输出
#         # -- 教师网络 --
#         with torch.no_grad():
#             # 获取教师的控制信号
#             control_teacher = self.teacher_model.control_model(x=z_noisy, hint=hint, timesteps=t, context=cond_txt)
#             # 获取教师的最终输出 (同时会触发hook)
#             eps_teacher = self.teacher_model.model.diffusion_model(x=z_noisy, timesteps=t, context=cond_txt, control=list(control_teacher))

#         # -- 学生网络 --
#         # 获取学生的控制信号
#         control_student = self.student_model.control_model(x=z_noisy, hint=hint, timesteps=t, context=cond_txt)
#         # 获取学生的最终输出 (同时会触发hook)
#         eps_student = self.student_model.model.diffusion_model(x=z_noisy, timesteps=t, context=cond_txt, control=list(control_student))

#         # 3. 计算三部分损失
#         # 损失1: 基础降噪损失
#         loss_sd = F.mse_loss(eps_student.float(), noise.float())

#         # 损失2: 控制网络蒸馏损失 (直接对比control_model的输出)
#         loss_kd_control = 0.0
#         for c_s, c_t in zip(control_student, control_teacher):
#             # c_s 和 c_t 都是包含两个张量的列表 [add, mul]
#             loss_kd_control += F.mse_loss(c_s[0].float(), c_t[0].float()) # 对比 add
#             loss_kd_control += F.mse_loss(c_s[1].float(), c_t[1].float()) # 对比 mul
#         loss_kd_control /= len(control_student) #取平均

#         # 损失3: 主U-Net模块化蒸馏损失 (对比hook捕获的中间块输出)
#         losses_kd_unet = []
#         for key in self.unet_block_layers:
#             a_tea = self.acts_tea_unet[key]
#             a_stu = self.acts_stu_unet[key]
#             if isinstance(a_tea, tuple): a_tea = a_tea[0]
#             if isinstance(a_stu, tuple): a_stu = a_stu[0]
#             losses_kd_unet.append(F.mse_loss(a_stu.float(), a_tea.float()))
#         loss_kd_unet = torch.stack(losses_kd_unet).mean()
        
#         # --- 合并总损失 ---
#         total_loss = (self.hparams.lambda_sd * loss_sd +
#                       self.hparams.lambda_kd_unet * loss_kd_unet +
#                       self.hparams.lambda_kd_control * loss_kd_control)


#         loss_threshold = 36.0 

#         # 检查总loss是否超过了我们设定的合理上限
#         if total_loss > loss_threshold:
#             print(f"!!! Training Deviation Detected at step {self.global_step} !!!")
#             print(f"Total Loss spiked to: {total_loss.item():.4f} (Threshold was {loss_threshold})")
            
#             # 关键诊断信息：打印所有分项loss
#             print(f"  - loss_sd: {loss_sd.item():.4f}")
#             print(f"  - loss_kd_unet: {loss_kd_unet.item():.4f}")
#             print(f"  - loss_kd_control: {loss_kd_control.item():.4f}")
            
#             # 记录导致问题的批次信息
#             problematic_paths = batch.get('path', ['Path not available'])
#             print(f"  - Problematic batch paths: {problematic_paths}")
            
#             # 中断训练以便调试
#             raise ValueError(f"Training deviation detected: Loss spiked to {total_loss.item():.4f}")

#         # --- 日志记录 ---
#         self.log_dict({
#             "train_loss": total_loss,
#             "loss_sd": loss_sd.detach(),
#             "loss_kd_unet": loss_kd_unet.detach(),
#             "loss_kd_control": loss_kd_control.detach()
#         }, prog_bar=True, on_step=True, logger=True)

#         return total_loss


#     def configure_optimizers(self):
#         # 目标：我们希望训练 control_model 的所有参数，以及主U-Net的部分参数。
#         # 'sd_locked' 的原意是 "是否锁定 Stable Diffusion 的主干网络"。
#         # 我们在这里假设我们不锁定它（即 sd_locked = False），
#         # 从而让主U-Net的输出层也参与训练。

#         # 1. 首先添加 control_model 的所有参数
#         params_to_optimize = list(self.student_model.control_model.parameters())

#         # 2. 然后添加主U-Net (self.student_model.model.diffusion_model) 的输出部分参数
#         # 这个逻辑与 fcdiffusion.py 中 `if not self.sd_locked:` 后的逻辑完全一致
#         params_to_optimize += list(self.student_model.model.diffusion_model.output_blocks.parameters())
#         params_to_optimize += list(self.student_model.model.diffusion_model.out.parameters())
        
#         print(f"Total number of trainable parameters: {sum(p.numel() for p in params_to_optimize)}")

#         optimizer = torch.optim.AdamW(params_to_optimize, lr=self.hparams.learning_rate)
#         # 学习率调度器可以保持不变
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_steps)
        
#         return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

#     @torch.no_grad()
#     def log_images(self, batch, N=5, *args, **kwargs):
#         """
#         这个方法将日志记录的请求直接委托给内部的学生模型。
#         学生模型的 control_mode 已经在初始化时被正确设置。
#         """
#         return self.student_model.log_images(batch, N=N, *args, **kwargs)

#     @staticmethod
#     def load_model_from_config(config, ckpt, device="cuda", verbose=False):
#         print(f"Loading model from {ckpt}...")
#         model = instantiate_from_config(config.model)
#         model.load_state_dict(load_state_dict(ckpt, location="cpu"), strict=False)
#         model = model.to(device)
#         return model


# # 将这个方法添加到您的 DecoupledDistiller class 内部

#     def validation_step(self, batch, batch_idx):
#         # 验证逻辑与训练逻辑非常相似，但我们不需要计算梯度
#         # PyTorch Lightning会自动处理 torch.no_grad()
        
#         # 1. 准备共享输入
#         z0, c_dict = self.teacher_model.get_input(batch, self.teacher_model.first_stage_key)
#         hint = torch.cat(c_dict['c_concat'], 1)
#         cond_txt = torch.cat(c_dict['c_crossattn'], 1)

#         t = torch.randint(0, self.teacher_model.num_timesteps, (z0.shape[0],), device=self.device).long()
#         noise = torch.randn_like(z0)
#         z_noisy = self.teacher_model.q_sample(x_start=z0, t=t, noise=noise)

#         # 2. 解耦蒸馏
#         control_teacher = self.teacher_model.control_model(x=z_noisy, hint=hint, timesteps=t, context=cond_txt)
#         eps_teacher = self.teacher_model.model.diffusion_model(x=z_noisy, timesteps=t, context=cond_txt, control=list(control_teacher))
        
#         control_student = self.student_model.control_model(x=z_noisy, hint=hint, timesteps=t, context=cond_txt)
#         eps_student = self.student_model.model.diffusion_model(x=z_noisy, timesteps=t, context=cond_txt, control=list(control_student))

#         # 3. 计算损失 (与training_step完全相同)
#         loss_sd = F.mse_loss(eps_student.float(), noise.float())
        
#         loss_kd_control = 0.0
#         for c_s, c_t in zip(control_student, control_teacher):
#             loss_kd_control += F.mse_loss(c_s[0].float(), c_t[0].float())
#             loss_kd_control += F.mse_loss(c_s[1].float(), c_t[1].float())
#         loss_kd_control /= len(control_student)
        
#         # 模块化蒸馏损失的计算在验证时也可以简化，或者保持原样以进行精确对比
#         # 为了简单，我们这里只用最终输出的loss作为验证指标
#         # total_val_loss = F.mse_loss(eps_student.float(), eps_teacher.float())
        
#         # 或者，为了和train_loss保持一致，我们计算同样的总损失
#         losses_kd_unet = []
#         for key in self.unet_block_layers:
#             a_tea = self.acts_tea_unet.get(key)
#             a_stu = self.acts_stu_unet.get(key)
#             if a_tea is not None and a_stu is not None:
#                 if isinstance(a_tea, tuple): a_tea = a_tea[0]
#                 if isinstance(a_stu, tuple): a_stu = a_stu[0]
#                 losses_kd_unet.append(F.mse_loss(a_stu.float(), a_tea.float()))
        
#         loss_kd_unet = torch.stack(losses_kd_unet).mean() if losses_kd_unet else torch.tensor(0.0, device=self.device)

#         total_val_loss = (self.hparams.lambda_sd * loss_sd +
#                           self.hparams.lambda_kd_unet * loss_kd_unet +
#                           self.hparams.lambda_kd_control * loss_kd_control)
                          
#         # 4. 记录验证损失
#         self.log("val_loss", total_val_loss, prog_bar=True, on_epoch=True, sync_dist=True)
#         return total_val_loss
# # ----------------------------------------------------------------------------------
# # 3. 数据集与主程序
# # ----------------------------------------------------------------------------------
# class ValidationDataset(Dataset):
#     def __init__(self, data_list):
#         super().__init__()
#         self.data = data_list
#         if len(self.data) > 0:
#             print(f"Initialized validation dataset with {len(self.data)} samples.")
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         try:
#             source = Image.open(item['image_path']).convert("RGB").resize((512, 512))
#             source = np.array(source).astype(np.uint8)
#         except Exception as e:
#             print(f"ERROR: Could not load image {item['image_path']}: {e}")
#             return None
#         return dict(jpg=source, txt=item['prompt'], path=item['image_path'])

# def traverse_images_and_texts(directory):
#     data_pairs = []
#     try:
#         image_basenames = {os.path.splitext(os.path.basename(f))[0]: os.path.join(directory, f) for f in os.listdir(directory) if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])}
#         text_basenames = {os.path.splitext(os.path.basename(f))[0]: os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.txt')}
#     except FileNotFoundError:
#         print(f"WARNING: Validation directory not found at: {directory}")
#         return []
#     common_keys = sorted(list(image_basenames.keys() & text_basenames.keys()))
#     if not common_keys:
#         print(f"WARNING: No matching image-text pairs found in {directory}.")
#         return []
#     for key in common_keys:
#         with open(text_basenames[key], 'r', encoding='utf-8') as f:
#             prompt = f.read().strip()
#         data_pairs.append({"image_path": image_basenames[key], "prompt": prompt})
#     return data_pairs

# if __name__ == "__main__":
#     # --- 1. 配置 ---
#     teacher_config_path = 'configs/model_config.yaml'
#     student_config_path = 'configs/student_model_config.yaml'
    
#     # 以 low_pass 任务为例进行单任务蒸馏
#     DISTILL_MODE = "low_pass"
#     TEACHER_CKPT_PATH = "/home/apulis-dev/userdata/FCDiffusion_code/lightning_logs_SA/fcdiffusion_low_pass_checkpoint/epoch=3-step=75999.ckpt"
#     STUDENT_INITIAL_CKPT_PATH = './models/FCDiffusion_ini_stu.ckpt'

#     # --- 2. 实例化新的解耦蒸馏器 ---
#     model = DecoupledDistiller(
#         teacher_config_path=teacher_config_path,
#         teacher_ckpt_path=TEACHER_CKPT_PATH,
#         student_config_path=student_config_path,
#         student_ckpt_path=STUDENT_INITIAL_CKPT_PATH,
#         distill_mode=DISTILL_MODE,
#         learning_rate=1e-6,  #2e-6
#         lambda_sd= 8,        #100.0,   10         # 基础降噪损失权重
#         lambda_kd_unet=1.0,       # 主网络模块化蒸馏损失权重
#         lambda_kd_control= 2,    # 控制网络蒸馏损失权重 (可以适当调高，因为控制是关键)
#     )
    
#     # --- 3. 数据加载 ---
#     train_dataset = TrainDataset('/home/apulis-dev/userdata/FCDiffusion_code/datasets/training_data.json', cache_size=100)
#     train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=16, shuffle=True)
    
#     validation_folder_path = '/home/apulis-dev/userdata/FCDiffusion_code/datasets/test_sub_600'
#     val_data_list = traverse_images_and_texts(validation_folder_path)
#     val_dataloader = None
#     if val_data_list:
#         val_dataset = ValidationDataset(val_data_list[:200])
#         def collate_fn(batch):
#             batch = list(filter(lambda x: x is not None, batch))
#             return torch.utils.data.dataloader.default_collate(batch) if batch else None
#         val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=4, shuffle=False, collate_fn=collate_fn)

#     # --- 4. 回调 ---
#     checkpoint_callback = ModelCheckpoint(
#         dirpath=f'lightning_logs/decoupled_distill_{DISTILL_MODE}',
#         filename='{epoch}-{step}-{train_loss:.4f}',
#         # every_n_train_steps=2000, #5000
#         save_top_k=5,
#         monitor= "val_loss",    #"train_loss",
#         mode="min"
#     )
#     image_logger = DistillationImageLogger(batch_frequency=2000, max_images=4)

#     BEST_CKPT_PATH = "/home/apulis-dev/userdata/FCDiffusion_code/lightning_logs/decoupled_distill_low_pass/epoch=24-step=59999-train_loss=3.3137.ckpt"

#     # --- 5. 训练器 ---
#     trainer = pl.Trainer(
#         accelerator="gpu",
#         devices=1,
#         max_steps=150000,
#         callbacks=[checkpoint_callback, image_logger],
#         precision=32,
#         log_every_n_steps=50,
#         # val_check_interval=2000 if val_dataloader else 0.0,
#         val_check_interval=1.0,
#         gradient_clip_val=1.0,
#         resume_from_checkpoint=BEST_CKPT_PATH 
#     )
    
#     print(f"Starting Decoupled Distillation for mode: {DISTILL_MODE}...")
#     trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


#version of copy init weight

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import OmegaConf
import random
import torch.nn as nn
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint

# --- 依赖项 ---
# 确保这些import路径与您的项目结构一致
from fcdiffusion.fcdiffusion import FCDiffusion, FreqControlNet, ControlledUnetModel
from fcdiffusion.dataset import TrainDataset  # 假设 smart_resize_and_crop 在 dataset.py 中
from fcdiffusion.logger import DistillationImageLogger
from ldm.util import instantiate_from_config
from fcdiffusion.model import load_state_dict


# ----------------------------------------------------------------------------------
# 1. 辅助函数
# ----------------------------------------------------------------------------------
def get_activation(mem, name):
    def get_output_hook(module, input, output):
        mem[name] = output
    return get_output_hook

def add_hook(net, mem, mapping_layers):
    for n, m in net.named_modules():
        if n in mapping_layers:
            m.register_forward_hook(get_activation(mem, n))

# ----------------------------------------------------------------------------------
# 2. 核心蒸馏器：DecoupledDistiller
# ----------------------------------------------------------------------------------
class DecoupledDistiller(pl.LightningModule):
    def __init__(
        self,
        teacher_config_path: str,
        teacher_ckpt_path: str,
        student_config_path: str,
        student_ckpt_path: str,
        distill_mode: str,
        learning_rate: float,
        # --- 新增的、用于两阶段训练的参数 ---
        stage_switch_step: int,      # 定义第二阶段开始的步数
        lambda_sd_stage1: float,     # 第一阶段 loss_sd 的权重
        lambda_kd_unet_stage1: float,
        lambda_kd_control_stage1: float,
        lambda_sd_stage2: float,     # 第二阶段 loss_sd 的权重
        lambda_kd_unet_stage2: float,
        lambda_kd_control_stage2: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = True

        # --- 模型加载 ---
        print("Loading Student Model...")
        student_cfg = OmegaConf.load(self.hparams.student_config_path)
        self.student_model = self.load_model_from_config(student_cfg, self.hparams.student_ckpt_path)
        self.student_model.control_mode = self.hparams.distill_mode

        print("Loading Teacher Model...")
        teacher_cfg = OmegaConf.load(self.hparams.teacher_config_path)
        self.teacher_model = self.load_model_from_config(teacher_cfg, self.hparams.teacher_ckpt_path)
        self.teacher_model.control_mode = self.hparams.distill_mode
        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()

        # --- 自动进行权重迁移 ---
        self._migrate_compatible_weights()

        # --- 冻结与Hooks ---
        if hasattr(self.student_model, 'first_stage_model'):
            self.student_model.first_stage_model.requires_grad_(False)
        if hasattr(self.student_model, 'cond_stage_model'):
            self.student_model.cond_stage_model.requires_grad_(False)
        self._setup_distillation_hooks()

    def _migrate_compatible_weights(self):
        print("\n--- Starting Automatic Weight Migration ---")
        teacher_sd = self.teacher_model.state_dict()
        student_sd = self.student_model.state_dict()
        migrated_weights = 0
        non_migrated_weights = 0
        for key in student_sd.keys():
            if key in teacher_sd and teacher_sd[key].shape == student_sd[key].shape:
                student_sd[key] = teacher_sd[key]
                migrated_weights += 1
            else:
                non_migrated_weights += 1
        self.student_model.load_state_dict(student_sd)
        print(f"Migration Complete: {migrated_weights} tensors migrated, {non_migrated_weights} tensors remain (mostly new GMEA layers).")
        print("--- End of Weight Migration ---\n")

    def _setup_distillation_hooks(self):
        self.acts_tea_unet, self.acts_stu_unet = {}, {}
        self.unet_block_layers = (
            [f'model.diffusion_model.input_blocks.{i}' for i in range(12)] +
            ['model.diffusion_model.middle_block'] +
            [f'model.diffusion_model.output_blocks.{i}' for i in range(12)]
        )
        add_hook(self.teacher_model, self.acts_tea_unet, self.unet_block_layers)
        add_hook(self.student_model, self.acts_stu_unet, self.unet_block_layers)

    def _calculate_losses(self, batch, lambda_sd, lambda_kd_unet, lambda_kd_control):
        # --- CRITICAL CHANGE: Lambdas are now passed as arguments ---
        with torch.no_grad():
            z0, c_dict = self.teacher_model.get_input(batch, self.teacher_model.first_stage_key)
            hint = torch.cat(c_dict['c_concat'], 1)
            cond_txt = torch.cat(c_dict['c_crossattn'], 1)
        t = torch.randint(0, self.teacher_model.num_timesteps, (z0.shape[0],), device=self.device).long()
        noise = torch.randn_like(z0)
        z_noisy = self.teacher_model.q_sample(x_start=z0, t=t, noise=noise)

        with torch.no_grad():
            control_teacher = self.teacher_model.control_model(x=z_noisy, hint=hint, timesteps=t, context=cond_txt)
            # Hooks are triggered here for teacher
            self.teacher_model.model.diffusion_model(x=z_noisy, timesteps=t, context=cond_txt, control=list(control_teacher))
        
        control_student = self.student_model.control_model(x=z_noisy, hint=hint, timesteps=t, context=cond_txt)
        # Hooks are triggered here for student
        eps_student = self.student_model.model.diffusion_model(x=z_noisy, timesteps=t, context=cond_txt, control=list(control_student))

        loss_sd = F.mse_loss(eps_student.float(), noise.float())
        
        loss_kd_control = 0.0
        if len(control_student) > 0:
            for c_s, c_t in zip(control_student, control_teacher):
                loss_kd_control += F.mse_loss(c_s[0].float(), c_t[0].float())
                loss_kd_control += F.mse_loss(c_s[1].float(), c_t[1].float())
            loss_kd_control /= len(control_student)

        losses_kd_unet = []
        for key in self.unet_block_layers:
            a_tea = self.acts_tea_unet.get(key)
            a_stu = self.acts_stu_unet.get(key)
            if a_tea is not None and a_stu is not None:
                if isinstance(a_tea, tuple): a_tea = a_tea[0]
                if isinstance(a_stu, tuple): a_stu = a_stu[0]
                losses_kd_unet.append(F.mse_loss(a_stu.float(), a_tea.float()))
        
        loss_kd_unet = torch.stack(losses_kd_unet).mean() if losses_kd_unet else torch.tensor(0.0, device=self.device)

        total_loss = (lambda_sd * loss_sd +
                      lambda_kd_unet * loss_kd_unet +
                      lambda_kd_control * loss_kd_control)
        
        return total_loss, loss_sd, loss_kd_unet, loss_kd_control
        
        return total_loss, loss_sd, loss_kd_unet, loss_kd_control, batch

    def training_step(self, batch, batch_idx):
        # Determine current lambdas based on training stage
        if self.global_step < self.hparams.stage_switch_step:
            lambdas = (self.hparams.lambda_sd_stage1, self.hparams.lambda_kd_unet_stage1, self.hparams.lambda_kd_control_stage1)
        else:
            lambdas = (self.hparams.lambda_sd_stage2, self.hparams.lambda_kd_unet_stage2, self.hparams.lambda_kd_control_stage2)
        
        # Pass lambdas to the calculation function
        total_loss, loss_sd, loss_kd_unet, loss_kd_control = self._calculate_losses(batch, *lambdas)

        self.log_dict({
            "train_loss": total_loss,
            "loss_sd": loss_sd.detach(),
            "loss_kd_unet": loss_kd_unet.detach(),
            "loss_kd_control": loss_kd_control.detach(),
            "current_lambda_sd": lambdas[0]
        }, prog_bar=True, on_step=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        # --- CRITICAL CHANGE: Use stage 1 weights for validation ---
        # This is because validation can happen at any point. Using stage 1 weights provides a consistent benchmark.
        # The sanity check at the beginning (global_step=0) will now work correctly.
        lambdas = (self.hparams.lambda_sd_stage1, self.hparams.lambda_kd_unet_stage1, self.hparams.lambda_kd_control_stage1)
        
        total_val_loss, _, _, _ = self._calculate_losses(batch, *lambdas)
        
        self.log("val_loss", total_val_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return total_val_loss

    def configure_optimizers(self):
        params_to_optimize = list(self.student_model.control_model.parameters())
        params_to_optimize += list(self.student_model.model.diffusion_model.output_blocks.parameters())
        params_to_optimize += list(self.student_model.model.diffusion_model.out.parameters())
        print(f"Total number of trainable parameters: {sum(p.numel() for p in params_to_optimize)}")
        optimizer = torch.optim.AdamW(params_to_optimize, lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    @torch.no_grad()
    def log_images(self, batch, N=5, *args, **kwargs):
        return self.student_model.log_images(batch, N=N, *args, **kwargs)

    @staticmethod
    def load_model_from_config(config, ckpt, device="cuda", verbose=False):
        print(f"Loading model from {ckpt}...")
        model = instantiate_from_config(config.model)
        model.load_state_dict(load_state_dict(ckpt, location="cpu"), strict=False)
        model = model.to(device)
        return model

# ----------------------------------------------------------------------------------
# 3. 数据集与主程序
# ----------------------------------------------------------------------------------
class ValidationDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        if len(self.data) > 0:
            print(f"Initialized validation dataset with {len(self.data)} samples.")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            # 使用我们之前修正的智能裁剪函数
            # source = smart_resize_and_crop(item['image_path']) 
            source = Image.open(item['image_path']).convert("RGB").resize((512, 512))
            source = np.array(source).astype(np.uint8)
        except Exception as e:
            print(f"ERROR: Could not load image {item['image_path']}: {e}")
            return None
        return dict(jpg=source, txt=item['prompt'], path=item['image_path'])

def traverse_images_and_texts(directory):
    data_pairs = []
    try:
        image_basenames = {os.path.splitext(os.path.basename(f))[0]: os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        text_basenames = {os.path.splitext(os.path.basename(f))[0]: os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.txt')}
    except FileNotFoundError:
        print(f"WARNING: Validation directory not found at: {directory}")
        return []
    common_keys = sorted(list(image_basenames.keys() & text_basenames.keys()))
    if not common_keys:
        print(f"WARNING: No matching image-text pairs found in {directory}.")
        return []
    for key in common_keys:
        with open(text_basenames[key], 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        data_pairs.append({"image_path": image_basenames[key], "prompt": prompt})
    return data_pairs

if __name__ == "__main__":
    # --- 1. 配置 ---
    teacher_config_path = 'configs/model_config.yaml'
    student_config_path = 'configs/student_model_config.yaml'
    
    DISTILL_MODE = "low_pass"
    TEACHER_CKPT_PATH = "/home/apulis-dev/userdata/FCDiffusion_code/lightning_logs_SA/fcdiffusion_low_pass_checkpoint/epoch=3-step=75999.ckpt"
    STUDENT_BASE_CKPT_PATH = './models/FCDiffusion_ini_stu_M128.ckpt'

    # --- 2. 实例化新的解耦蒸馏器 ---
    model = DecoupledDistiller(
        teacher_config_path=teacher_config_path,
        teacher_ckpt_path=TEACHER_CKPT_PATH,
        student_config_path=student_config_path,
        student_ckpt_path=STUDENT_BASE_CKPT_PATH,
        distill_mode=DISTILL_MODE,
        learning_rate=1e-6,
        
        # --- 设置两阶段的权重和切换点 ---
        stage_switch_step=300000,           # 在30000步时从第一阶段切换到第二阶段
        
        # 第一阶段权重：侧重模仿 (平衡初始的加权后损失)
        lambda_sd_stage1=6,
        lambda_kd_unet_stage1=0.8,
        lambda_kd_control_stage1=7,
        
        # 第二阶段权重：侧重自身质量 (大幅提高loss_sd权重)
        lambda_sd_stage2=30,
        lambda_kd_unet_stage2=0.1,         # 也可以适当降低，因为此时模仿已基本完成
        lambda_kd_control_stage2=0.2,
    )
    
    # --- 3. 数据加载 ---

    train_dataset = TrainDataset('/home/apulis-dev/userdata/FCDiffusion_code/datasets/training_data.json', cache_size=100)
    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=16, shuffle=True)
    
    validation_folder_path = '/home/apulis-dev/userdata/FCDiffusion_code/datasets/test_sub_600'
    val_data_list = traverse_images_and_texts(validation_folder_path)
    val_dataloader = None
    if val_data_list:
        val_dataset = ValidationDataset(val_data_list[:200])
        def collate_fn(batch):
            batch = list(filter(lambda x: x is not None, batch))
            return torch.utils.data.dataloader.default_collate(batch) if batch else None
        val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # --- 4. 回调 ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'lightning_logs/decoupled_distill_{DISTILL_MODE}',
        filename='{epoch}-{step}-{val_loss:.4f}',
        save_top_k=5,
        monitor="val_loss",
        mode="min"
    )
    image_logger = DistillationImageLogger(batch_frequency=2000, max_images=4)

    # --- 5. 训练器 ---
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=150000,
        callbacks=[checkpoint_callback, image_logger],
        precision=32,
        log_every_n_steps=50,
        val_check_interval=1.0, # 在每个epoch结束时进行验证
        gradient_clip_val=1.0,
        resume_from_checkpoint='/home/apulis-dev/userdata/FCDiffusion_code/lightning_logs/decoupled_distill_low_pass/epoch=29-step=52499-val_loss=12.3401.ckpt' # 如果需要断点续训，请指定路径
    )
    
    print(f"Starting Decoupled Distillation for mode: {DISTILL_MODE}...")
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)