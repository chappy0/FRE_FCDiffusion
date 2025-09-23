import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only


class ImageLogger(Callback):
    def __init__(self, root_path='./', batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.root_path = root_path
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(self.root_path, 'image_log', split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c, h, w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "gs-{:06}_e-{:06}_b-{:06}-{}.png".format(global_step, current_epoch, batch_idx, k)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    @rank_zero_only
    def log_local_eval(self, split, images):
        root = os.path.join(self.root_path, "image_log", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}.png".format(k)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="training"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            if split.startswith('train'):
                self.log_local(split, images, pl_module.global_step, pl_module.current_epoch, batch_idx)
            elif split.startswith('test'):
                self.log_local_eval(split, images)
            else:
                raise Exception("'split' must be either 'training' or 'testing'")

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="training")


# -------------------------------------------------
# 新增的、为我们多任务模型定制的ImageLogger
# -------------------------------------------------
class MultiTaskImageLogger(ImageLogger):
    """
    一个特殊的ImageLogger，用于在每个记录点，
    为所有四个频域任务生成并保存一次图像。
    """
    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        # 检查频率等条件
        if not self.check_frequency(batch_idx):
            return
        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            # 核心改动：循环所有任务模式，并为每种模式生成图像
            for mode in pl_module.task_modes:
                print(f"  Logging images for task: {mode}")
                
                # 在调用log_images前，先设置好模型的模式
                pl_module.student_model.control_mode = mode
                
                # 调用模型自身的log_images方法
                images = pl_module.student_model.log_images(
                    batch, N=self.max_images, **self.log_images_kwargs
                )
                
                # 保存图像 (复用您原来的log_local方法)
                # 我们在文件名中加入mode来区分
                for k in images:
                    grid = torchvision.utils.make_grid(images[k], nrow=4)
                    if self.rescale:
                        grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1
                    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    # grid = grid.numpy()
                    grid = grid.cpu().numpy()
                    grid = (grid * 255).astype(np.uint8)
                    
                    # 修改文件名以包含任务模式
                    filename = "gs-{:06}_e-{:06}_b-{:06}_mode-{}_{}.png".format(
                        pl_module.global_step, pl_module.current_epoch, batch_idx, mode, k
                    )
                    
                    root = os.path.join(self.root_path, 'image_log', split)
                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    Image.fromarray(grid).save(path)

        if is_train:
            pl_module.train()

    def check_frequency(self, check_idx):
        # 简化频率检查，使其在每个log_every_n_steps时都触发
        return (check_idx + 1) % self.batch_freq == 0


# -------------------------------------------------
# 修正后的、为我们单任务蒸馏定制的ImageLogger
# -------------------------------------------------
class DistillationImageLogger(ImageLogger):
    """
    一个适配单任务蒸馏的ImageLogger。
    它不再循环多个任务，而是直接为当前正在蒸馏的模式生成图像。
    """
    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        # 检查频率等条件 (从基类继承或保持不变)
        if not self.check_frequency(batch_idx):
            return
        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            # 【核心修正 1】: 不再需要 for 循环
            # 我们直接让 pl_module (即 DecoupledDistiller) 调用它自己的 log_images 方法。
            # DecoupledDistiller 的 control_mode 在初始化时已经设定好了。
            
            # 为了让这个调用生效，我们需要在 DecoupledDistiller 中添加一个 log_images 方法
            if not hasattr(pl_module, "log_images"):
                print("ERROR: The training module (DecoupledDistiller) must have a 'log_images' method.")
                return

            print(f"  Logging images for current distillation mode...")
            images = pl_module.log_images(batch, N=self.max_images, **self.log_images_kwargs)

            # 【核心修正 2】: 获取当前模式名用于保存文件
            current_mode = pl_module.hparams.distill_mode

            # 保存图像 (复用您原来的log_local方法，但文件名稍作修改)
            for k in images:
                if isinstance(images[k], torch.Tensor):
                    grid = torchvision.utils.make_grid(images[k], nrow=4)
                    if self.rescale:
                        grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1
                    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    grid = grid.cpu().numpy()
                    grid = (grid * 255).astype(np.uint8)
                    
                    # 修改文件名以包含当前的任务模式
                    filename = "gs-{:06}_e-{:06}_b-{:06}_mode-{}_{}.png".format(
                        pl_module.global_step, pl_module.current_epoch, batch_idx, current_mode, k
                    )
                    
                    root = os.path.join(self.root_path, 'image_log', split)
                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    Image.fromarray(grid).save(path)

        if is_train:
            pl_module.train()