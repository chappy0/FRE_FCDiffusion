import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config, log_txt_as_img
from fcdiffusion.model import load_state_dict

class FCDiffusionDistill(pl.LightningModule):
    def __init__(
        self,
        teacher_config: str,
        teacher_ckpt: dict,   # 字典格式： {"low_pass": "path/to/teacher_low.ckpt", "mini_pass": ..., ...}
        student_config: str,
        student_ckpt: str,
        log_every_n_steps: int = 1000,
        lambda_dict=None,   # 各频段损失权重字典，默认各频段权重相等
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.log_every_n_steps = log_every_n_steps
        # 如果没有提供lambda_dict，则默认均分权重（总和为1）
        if lambda_dict is None:
            self.lambda_dict = {"low_pass": 0.25, "mini_pass": 0.25, "mid_pass": 0.25, "high_pass": 0.25}
        else:
            self.lambda_dict = lambda_dict

        # 加载教师模型，每个教师模型在其配置文件中已固化 control_mode
        teacher_cfg = OmegaConf.load(teacher_config)
        self.teacher_models = {}
        for mode, ckpt_path in teacher_ckpt.items():
            teacher_model = self.load_model_from_config(
                config=teacher_cfg,
                ckpt=ckpt_path
            )
            teacher_model.requires_grad_(False)
            teacher_model.eval()
            # 此处教师模型内部的 get_input 会依据自身配置中的 control_mode 进行频域过滤
            self.teacher_models[mode] = teacher_model

        # 加载学生模型（单一模型）
        student_cfg = OmegaConf.load(student_config)
        self.model = instantiate_from_config(student_cfg.model)
        self.model.load_state_dict(load_state_dict(student_ckpt, location='cpu'))
        # 冻结 first_stage 与 cond_stage 模块
        self.model.first_stage_model.requires_grad_(False)
        self.model.cond_stage_model.requires_grad_(False)
        # 学生模型假设支持传入 return_eps=True，用于预测噪声

        # 学习率等超参数可以在后续 configure_optimizers 中设置
        self.learning_rate = 1e-5

    @staticmethod
    def noise_loss_normalized(student_noise, teacher_noise):
        student_noise_norm = FCDiffusionDistill.normalize_noise(student_noise)
        teacher_noise_norm = FCDiffusionDistill.normalize_noise(teacher_noise)
        return torch.mean((student_noise_norm - teacher_noise_norm) ** 2)

    @staticmethod
    def normalize_noise(noise):
        norm = torch.sqrt(torch.sum(noise ** 2, dim=[1, 2, 3], keepdim=True) + 1e-8)
        return noise / norm

    def training_step(self, batch, batch_idx):
        # 分别使用教师和学生模型获取输入
        # 注意：教师模型内部的 get_input 根据各自的 control_mode 返回对应的控制信号
        x_teacher, c_teacher = None, None
        x_student, c_student = None, None

        # 假设教师和学生 get_input 接口一致，这里用任一教师模型获取输入（数据一致）
        teacher_model_sample = next(iter(self.teacher_models.values()))
        x_teacher, c_teacher = teacher_model_sample.get_input(batch, teacher_model_sample.first_stage_key)
        x_student, c_student = self.model.get_input(batch, self.model.first_stage_key)
    
        total_loss = 0.0
        loss_dict = {}
        # 对于每个教师模型（各个频段），计算对应的蒸馏损失
        for mode, teacher_model in self.teacher_models.items():
            with torch.no_grad():
                # 这里教师模型内部根据配置中的 control_mode 进行处理，不需要额外传参
                teacher_eps = teacher_model(x_teacher, c_teacher, return_eps=True)
            student_eps = self.model(x_student, c_student, return_eps=True, control= None)  # 学生模型也需支持对齐接口
            loss_mode = self.noise_loss_normalized(student_eps, teacher_eps)
            weighted_loss = self.lambda_dict[mode] * loss_mode
            total_loss += weighted_loss
            loss_dict[mode] = loss_mode.item()
    
        # 记录日志
        self.log("training/total_loss", total_loss, on_step=True, prog_bar=True)
        for mode in self.teacher_models.keys():
            self.log(f"training/{mode}_loss", loss_dict[mode], on_step=True)
    
        if batch_idx % self.log_every_n_steps == 0:
            print(f"Step {self.trainer.global_step}: " + ", ".join([f"{mode}_loss={loss_dict[mode]:.4f}" for mode in self.teacher_models.keys()]))
    
        return total_loss

    @staticmethod
    def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=True):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("Missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("Unexpected keys:")
            print(u)
        model.to(device)
        model.eval()
        return model

    def configure_optimizers(self):
        # 仅收集学生模型中 diffusion_model 和 control_model 中需要训练的参数
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and ('diffusion_model' in name or 'control_model' in name):
                trainable_params.append(param)
        optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=500,
                T_mult=1,
                eta_min=1e-6
            ),
            'interval': 'step'
        }
        return [optimizer], [scheduler]



# 训练配置示例
if __name__ == "__main__":
    # 教师模型 checkpoint 字典，每个频段对应一个 checkpoint 路径
    teacher_ckpt = {
        "low_pass": r"D:\paper\FRE_FCD\lightning_logs\low\epoch=5-step=17999.ckpt",
        "mini_pass": r"D:\paper\FRE_FCD\lightning_logs\mini\epoch=1-step=2999.ckpt",
        "mid_pass": r"D:\paper\FCDiffusion_code-main\lightning_logs\fcdiffusion_mid_pass_checkpoint\epoch=11-step=241999.ckpt",
        "high_pass": r"D:\paper\FCDiffusion_code-main\lightning_logs\fcdiffusion_high_pass_checkpoint\epoch=3-step=12999.ckpt"
    }
    
    student_config = 'configs/student_model_config.yaml'
    student_ckpt  = './models/FCDiffusion_ini_stu.ckpt'
    teacher_config = 'configs/model_config.yaml'
    
    model = FCDiffusionDistill(
        teacher_config=teacher_config,
        teacher_ckpt=teacher_ckpt,
        student_config=student_config,
        student_ckpt=student_ckpt
    )
    model.learning_rate = 1e-5
    
    from fcdiffusion.dataset import TrainDataset
    from torch.utils.data import DataLoader
    dataset = TrainDataset('/home/apulis-dev/userdata/FCDiffusion_code/datasets/training_data.json', cache_size=1000)
    dataloader = DataLoader(dataset, num_workers=2, batch_size=1, shuffle=True)
    
    from pytorch_lightning.callbacks import ModelCheckpoint

    import os
    
    class StudentModelCheckpoint(ModelCheckpoint):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    
        def _save_checkpoint(self, trainer, filepath):
            # 只保存学生模型的权重
            student_state_dict = trainer.model.model.state_dict()
            torch.save(student_state_dict, filepath)
    
    checkpoint_path = 'distillation_checkpoint'
    val_every_n_train_steps = 1000
    # 使用自定义的 StudentModelCheckpoint
    val_checkpoint = StudentModelCheckpoint(
        dirpath='lightning_logs/' + checkpoint_path,
        every_n_train_steps=val_every_n_train_steps,
        save_top_k=-1
    )
    
    # checkpoint_path = 'distillation_control_mode_checkpoint'
    # val_checkpoint = ModelCheckpoint(dirpath='lightning_logs/' + checkpoint_path,
    #                                  every_n_train_steps=1000, save_top_k=-1)
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=100,
        gradient_clip_val=1.0,
        log_every_n_steps=1000,
        callbacks=[val_checkpoint],
        precision=16
    )
    
    trainer.fit(model, dataloader)
