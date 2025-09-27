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

from fcdiffusion.fcdiffusion import FCDiffusion, FreqControlNet, ControlledUnetModel
from fcdiffusion.dataset import TrainDataset  
from fcdiffusion.logger import DistillationImageLogger
from ldm.util import instantiate_from_config
from fcdiffusion.model import load_state_dict
from ldm.modules.attention import DynamicHybridAttention

def get_activation(mem, name):
    def get_output_hook(module, input, output):
        mem[name] = output
    return get_output_hook

def add_hook(net, mem, mapping_layers):
    for n, m in net.named_modules():
        if n in mapping_layers:
            m.register_forward_hook(get_activation(mem, n))

def get_attn_map_hook(mem, name):
    def get_output_hook(module, input, output):
        mem[name] = getattr(module, 'attn_map', None)
    return get_output_hook

# ----------------------------------------------------------------------------------
# 2.DecoupledDistiller
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

        stage_switch_step: int,      
        lambda_sd_stage1: float,     
        lambda_kd_unet_stage1: float,
        lambda_kd_control_stage1: float,
        lambda_kd_fcnet_stage1: float,
        lambda_sd_stage2: float,     
        lambda_kd_unet_stage2: float,
        lambda_kd_control_stage2: float,
        lambda_kd_fcnet_stage2: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = True

        # --- load model---
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

        # --- migrate state ---
        self._migrate_compatible_weights()

        # --- freeze and Hooks ---
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
        print("Setting up distillation hooks...")
        self.acts_tea_unet, self.acts_stu_unet = {}, {}
        self.acts_tea_fcnet, self.acts_stu_fcnet = {}, {}


        from ldm.modules.attention import CrossAttention, DynamicHybridAttention, MultiHeadExternalAttention


        self.unet_block_layers = (
            [f'model.diffusion_model.input_blocks.{i}' for i in range(12)] +
            ['model.diffusion_model.middle_block'] +
            [f'model.diffusion_model.output_blocks.{i}' for i in range(12)]
        )

        self.fcnet_block_layers = (
            [f'control_model.input_blocks.{i}' for i in range(12)] +
            ['control_model.middle_block']

        )
        add_hook(self.teacher_model, self.acts_tea_unet, self.unet_block_layers)
        add_hook(self.teacher_model, self.acts_tea_fcnet, self.fcnet_block_layers)
        add_hook(self.student_model, self.acts_stu_unet, self.unet_block_layers)
        add_hook(self.student_model, self.acts_stu_fcnet, self.fcnet_block_layers)


        for key in self.unet_block_layers:
            try:

                teacher_block = self.teacher_model.get_submodule(key)
                for n, m in teacher_block.named_modules():
                    if isinstance(m, CrossAttention):

                        hook_key = f'{key}_sa_attn'
                        m.register_forward_hook(get_attn_map_hook(self.acts_tea_unet, hook_key))
                        print(f"Hooked Teacher SA map for key: {hook_key}")

                student_block = self.student_model.get_submodule(key)
                for n, m in student_block.named_modules():
                    if isinstance(m, DynamicHybridAttention):
                        # Hook DHA模块内部的 ea_attn
                        hook_key = f'{key}_ea_attn'
                        m.ea_attn.register_forward_hook(get_attn_map_hook(self.acts_stu_unet, hook_key))
                        print(f"Hooked Student EA map for key: {hook_key}")

            except AttributeError:

                print(f"Warning: Submodule for key '{key}' not found, skipping hook.")
                continue


    def _calculate_losses(self, batch, lambda_sd, lambda_kd_unet, lambda_kd_control, lambda_kd_fcnet):
        with torch.no_grad():
            z0, c_dict = self.teacher_model.get_input(batch, self.teacher_model.first_stage_key)
            hint = torch.cat(c_dict['c_concat'], 1)
            cond_txt = torch.cat(c_dict['c_crossattn'], 1)
        t = torch.randint(0, self.teacher_model.num_timesteps, (z0.shape[0],), device=self.device).long()
        noise = torch.randn_like(z0)
        z_noisy = self.teacher_model.q_sample(x_start=z0, t=t, noise=noise)


        self.acts_tea_unet.clear(); self.acts_tea_fcnet.clear()
        with torch.no_grad():
            control_teacher = self.teacher_model.control_model(x=z_noisy, hint=hint, timesteps=t, context=cond_txt)
            _ = self.teacher_model.model.diffusion_model(x=z_noisy, timesteps=t, context=cond_txt, control=list(control_teacher))

        self.acts_stu_unet.clear(); self.acts_stu_fcnet.clear()
        control_student = self.student_model.control_model(x=z_noisy, hint=hint, timesteps=t, context=cond_txt)
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
            if a_tea is None or a_stu is None: continue
            if isinstance(a_tea, tuple): a_tea = a_tea[0]
            if isinstance(a_stu, tuple): a_stu = a_stu[0]
            

            losses_kd_unet.append(F.mse_loss(a_stu.float(), a_tea.float()))


        loss_kd_unet = torch.stack(losses_kd_unet).mean() if losses_kd_unet else torch.tensor(0.0, device=self.device)


        loss_attn_kl = torch.tensor(0.0, device=self.device)
        for key in self.unet_block_layers:
            sa_attn = self.acts_tea_unet.get(key+'_sa_attn')
            ea_attn = self.acts_stu_unet.get(key+'_ea_attn')
            # print(f"[DEBUG] sa_attn={sa_attn is not None}, ea_attn={ea_attn is not None}, key={key}")
            if sa_attn is None or ea_attn is None: continue
            loss_attn_kl += self._attn_kl_loss(sa_attn, ea_attn)


        losses_kd_fcnet = []
        for key in self.fcnet_block_layers:
            a_tea = self.acts_tea_fcnet.get(key)
            a_stu = self.acts_stu_fcnet.get(key)
            if a_tea is not None and a_stu is not None:
                if isinstance(a_tea, tuple): a_tea = a_tea[0]
                if isinstance(a_stu, tuple): a_stu = a_stu[0]
                losses_kd_fcnet.append(F.mse_loss(a_stu, a_tea))
        loss_kd_fcnet = torch.stack(losses_kd_fcnet).mean() if losses_kd_fcnet else torch.tensor(0.0, device=self.device)

        total_loss = (lambda_sd * loss_sd +
                      lambda_kd_unet * loss_kd_unet +
                      lambda_kd_control * loss_kd_control +
                      lambda_kd_fcnet * loss_kd_fcnet +
                      lambda_attn_kl * loss_attn_kl)

        return total_loss, loss_sd, loss_kd_unet, loss_kd_control, loss_kd_fcnet, loss_attn_kl



    def _attn_kl_loss(self, sa_attn, ea_attn, T=4.0):
        if sa_attn is None or ea_attn is None:
            return torch.tensor(0.0, device=self.device)

        if sa_attn.dim() == 3:
            B, H, _, _ = ea_attn.shape
            sa_attn = sa_attn.view(B, H, sa_attn.shape[1], sa_attn.shape[2])

        sa_attn_resized = sa_attn
        ea_attn_resized = F.adaptive_avg_pool2d(ea_attn, (sa_attn_resized.shape[-2], sa_attn_resized.shape[-1]))
        
        log_p = F.log_softmax(ea_attn_resized / T, dim=-1)
        q     = F.softmax(sa_attn_resized / T, dim=-1)
        
        return F.kl_div(log_p, q, reduction='batchmean') * (T * T)

    def _get_alpha_for_key(self, key):

        for name, module in self.student_model.named_modules():

            if key in name and isinstance(module, DynamicHybridAttention): 

                alpha = getattr(module, 'global_alpha_cache', None)
                if alpha is not None:
                    return alpha.view(1, 1, 1) 

        return torch.tensor(1.0, device=self.device).view(1, 1, 1)
        
    def training_step(self, batch, batch_idx):
        # Determine current lambdas based on training stage
        if self.global_step < self.hparams.stage_switch_step:
            lambdas = (self.hparams.lambda_sd_stage1, self.hparams.lambda_kd_unet_stage1, self.hparams.lambda_kd_control_stage1, self.hparams.lambda_kd_fcnet_stage1)
        else:
            lambdas = (self.hparams.lambda_sd_stage2, self.hparams.lambda_kd_unet_stage2, self.hparams.lambda_kd_control_stage2, self.hparams.lambda_kd_fcnet_stage2)


        total_loss, loss_sd, loss_kd_unet, loss_kd_control, loss_kd_fcnet, loss_attn_kl = self._calculate_losses(batch, *lambdas)

        self.log_dict({
            "train_loss": total_loss,
            "loss_sd": loss_sd.detach(),
            "loss_kd_unet": loss_kd_unet.detach(),
            "loss_kd_control": loss_kd_control.detach(),
            "loss_kd_fcnet": loss_kd_fcnet.detach(),
            "loss_attn_kl": loss_attn_kl.detach(), # <-- Add this line
        }, prog_bar=True, on_step=True, logger=True)       
        # total_loss, loss_sd, loss_kd_unet, loss_kd_control, loss_kd_fcnet = self._calculate_losses(batch, *lambdas)

        # self.log_dict({
        #     "train_loss": total_loss,
        #     "loss_sd": loss_sd.detach(),
        #     "loss_kd_unet": loss_kd_unet.detach(),
        #     "loss_kd_control": loss_kd_control.detach(),
        #     "loss_kd_fcnet": loss_kd_fcnet.detach(),
        # }, prog_bar=True, on_step=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        # Use stage 1 weights for validation ---
        lambdas = (self.hparams.lambda_sd_stage1, self.hparams.lambda_kd_unet_stage1, self.hparams.lambda_kd_control_stage1, self.hparams.lambda_kd_fcnet_stage1)
        
        # total_val_loss, _, _, _, _ = self._calculate_losses(batch, *lambdas)
        total_val_loss, _, _, _, _, _ = self._calculate_losses(batch, *lambdas)
        
        self.log("val_loss", total_val_loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return total_val_loss

    # def configure_optimizers(self):
    #     params_to_optimize = list(self.student_model.control_model.parameters())
    #     params_to_optimize += list(self.student_model.model.diffusion_model.output_blocks.parameters())
    #     params_to_optimize += list(self.student_model.model.diffusion_model.out.parameters())
    #     print(f"Total number of trainable parameters: {sum(p.numel() for p in params_to_optimize)}")
    #     optimizer = torch.optim.AdamW(params_to_optimize, lr=self.hparams.learning_rate)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_steps)
    #     return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def configure_optimizers(self):
        params_to_optimize = []

        params_to_optimize.extend(list(self.student_model.control_model.parameters()))

        params_to_optimize.extend(list(self.student_model.model.diffusion_model.output_blocks.parameters()))
        params_to_optimize.extend(list(self.student_model.model.diffusion_model.out.parameters()))


        for module in self.student_model.model.diffusion_model.input_blocks.modules():
            if isinstance(module, DynamicHybridAttention): 
                params_to_optimize.extend(list(module.parameters()))
                
        for module in self.student_model.model.diffusion_model.middle_block.modules():
            if isinstance(module, DynamicHybridAttention): 
                params_to_optimize.extend(list(module.parameters()))

        total_params = sum(p.numel() for p in params_to_optimize)
        print(f"Total number of trainable parameters: {total_params}")

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
# 3. dataset and main function
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
    # --- 1. Setup ---
    teacher_config_path = 'configs/model_config.yaml'
    student_config_path = 'configs/student_model_config.yaml'
    
    DISTILL_MODE = "low_pass"
    TEACHER_CKPT_PATH = "you/path/to/your/teacher/model"
    STUDENT_BASE_CKPT_PATH = "you/path/to/your/student/model"

    # --- 2. Instantiate a new Decoupled Distiller.
    model = DecoupledDistiller(
        teacher_config_path=teacher_config_path,
        teacher_ckpt_path=TEACHER_CKPT_PATH,
        student_config_path=student_config_path,
        student_ckpt_path=STUDENT_BASE_CKPT_PATH,
        distill_mode=DISTILL_MODE,
        learning_rate=1e-6,
        
        # --- Set the weights and transition points for the two phases 
        # (adjust these values based on your own hardware and data conditions).---
        stage_switch_step=30000,           
        
        # Weights for the first phase: focus on imitation (balance the weighted losses initially).
        # lambda_sd_stage1=6,
        # lambda_kd_unet_stage1=0.8,
        # lambda_kd_control_stage1=7,
        # lambda_kd_fcnet_stage1=0.5,
        
        lambda_sd_stage1=5,
        lambda_kd_unet_stage1=0.04,
        lambda_kd_fcnet_stage1=0.03,
        lambda_kd_control_stage1=0.63,
        # Weights for the second phase: focus on self-quality (significantly increase the weight of loss_sd).
        lambda_sd_stage2=8,
        lambda_kd_unet_stage2=0.04,         
        lambda_kd_control_stage2=0.03,
        lambda_kd_fcnet_stage2=0.63,
    )


    train_dataset = TrainDataset('../DGM/datasets/training_data.json', cache_size=100)
    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=16, shuffle=True)
    
    validation_folder_path = '../DGM/datasets/test_sub_200'
    val_data_list = traverse_images_and_texts(validation_folder_path)
    val_dataloader = None
    if val_data_list:
        val_dataset = ValidationDataset(val_data_list[:200])
        def collate_fn(batch):
            batch = list(filter(lambda x: x is not None, batch))
            return torch.utils.data.dataloader.default_collate(batch) if batch else None
        val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=16, shuffle=False, collate_fn=collate_fn)


    checkpoint_callback = ModelCheckpoint(
        dirpath=f'lightning_logs/decoupled_distill_{DISTILL_MODE}',
        filename='{epoch}-{step}-{val_loss:.4f}',
        save_top_k=5,
        monitor="val_loss",
        mode="min"
    )
    image_logger = DistillationImageLogger(batch_frequency=2000, max_images=4)


    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=150000,
        callbacks=[checkpoint_callback, image_logger],
        precision=32,
        log_every_n_steps=50,
        val_check_interval=1.0, 
        gradient_clip_val=1.0,
    )
    
    print(f"Starting Decoupled Distillation for mode: {DISTILL_MODE}...")
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
