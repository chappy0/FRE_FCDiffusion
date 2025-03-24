
# fcdiffusion_distill.py
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from fcdiffusion.dataset import TrainDataset
from fcdiffusion.logger import ImageLogger
from fcdiffusion.model import create_model, load_state_dict
import torch.nn.functional as F
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.models.diffusion.ddpm import LatentDiffusion
from omegaconf import OmegaConf
from fcdiffusion.fcdiffusion import ControlledUnetModel, FreqControlNet
from ldm.util import instantiate_from_config
from tools.dct_util import dct_2d, high_pass, idct_2d, low_pass, low_pass_and_shuffle
from pytorch_lightning.callbacks import ModelCheckpoint
from ldm.util import log_txt_as_img
from piq import LPIPS
from ldm.models.diffusion.ddim import DDIMSampler
import torch.nn as nn
import clip
# from fcdiffusion_reduce_model import calculate_layer_influences, filter_layers_by_influence, update_model_with_filtered_layers


class FCDiffusionDistill(pl.LightningModule):
    def __init__(
        self,
        teacher_config: str,
        teacher_ckpt: str,
        student_config: str,
        student_ckpt: str,
        # control_stage_config: OmegaConf,
        # first_stage_config: OmegaConf,
        # cond_stage_config: OmegaConf,
        # unet_config: OmegaConf,
        # timesteps: int = 1000,
        # cond_stage_key: str = 'txt',
        # first_stage_key: str = 'jpg',
        # control_scales: list = [1.0]*12,  # 匹配控制网络层数
        log_every_n_steps: int = 500,
        *args, lambda_kl=0.1, lambda_perceptual=0.1, lambda_loss1=0.5,only_mid_control=False,channels=4, **kwargs
    ):
        #初始化父类
        super().__init__(
            # first_stage_config=first_stage_config,
            # cond_stage_config=cond_stage_config,
            # unet_config=unet_config,
            # timesteps=timesteps,
            # cond_stage_key=cond_stage_key,
            *args, **kwargs
        )
        
        self.lambda_kl = lambda_kl
        self.lambda_perceptual = lambda_perceptual
        self.lambda_loss1 =lambda_loss1
        self.lambda_loss2 =1- lambda_loss1
        self.lpips = LPIPS().eval() 
        # self.register_buffer('sigma_data', torch.tensor(0.5))  # EDM参数
        self.only_mid_control = only_mid_control
        self.channels= channels
        self.log_every_n_steps = log_every_n_steps
        
        # 加载教师配置
        teacher_cfg = OmegaConf.load(teacher_config)
        # self.control_mode = teacher_cfg.model.params.control_mode
        # print(f"control_mode:{self.control_mode}")
        # 初始化控制网络
        # self.control_model = instantiate_from_config(control_stage_config)
        # self.control_scales = control_scales
        

        # 正确加载教师模型 --------------------------------------------------
        # 1. 加载完整配置
        teacher_config = OmegaConf.load(teacher_config)
        
        # 2. 使用与验证代码相同的加载函数
        self.teacher = self.load_model_from_config(
            config=teacher_config,
            ckpt=teacher_ckpt,
            # device=self.device  # 确保设备一致性
        )
        
        # 3. 冻结所有参数
        self.teacher.requires_grad_(False)
        self.teacher.eval()  # 必须设置为评估模式
        
        student_config = OmegaConf.load(student_config)
        # self.model= self.load_model_from_config(
        #     config=student_config,
        #     ckpt=student_ckpt,
        #     # device=self.device  # 确保设备一致性
        # )
        # config = OmegaConf.load(student_config)  #unet_config
        self.model = instantiate_from_config(student_config.model)
        # self.model = create_model('configs/student_model_config.yaml').cpu()
        self.model.load_state_dict(load_state_dict(student_ckpt, location='cpu'))
        # self.model.load_state_dict(torch.load(student_ckpt, map_location="cpu")["state_dict"], strict=False)
        # print(f"self.model:{self.model}")
        # 冻结 first_stage_model 和 cond_stage_model
        self.model.first_stage_model.requires_grad_(False)
        self.model.cond_stage_model.requires_grad_(False)
        

        # # 确保 diffusion_model 和 control_model 可以更新
        # self.model.diffusion_model.requires_grad_(True)
        # self.model.control_model.requires_grad_(True)
        # # 只加载学生模型的 diffusion_model 部分权重
        # student_state_dict = torch.load(student_ckpt, map_location="cpu")["state_dict"]
        # student_diffusion_state_dict = {k: v for k, v in student_state_dict.items() if k.startswith('model.diffusion_model')}
        # student_control_state_dict = {k: v for k, v in student_state_dict.items() if k.startswith('control_model')}
        # student_first_state_dict = {k: v for k, v in student_state_dict.items() if k.startswith('first_stage_model')}
        # student_cond_state_dict = {k: v for k, v in student_state_dict.items() if k.startswith('cond_stage_model')}
        # self.model.diffusion_model.load_state_dict(student_diffusion_state_dict, strict=False)
        # self.control_model.load_state_dict(student_control_state_dict, strict=False)
        # self.first_stage_model.load_state_dict(student_first_state_dict, strict=False)
        # self.cond_stage_model.load_state_dict(student_cond_state_dict, strict=False)
        # self.model.train()
        # self.model.requires_grad_(True) 
        # self.configure_model()
        # # 学生模型初始化 --------------------------------------------------
        # self.model = ...  # 学生模型定义
        
        # 验证教师完整性 -------------------------------------------------
        # self._validate_teacher()
        
        # 计算影响力并筛选层
        # layers_influences = calculate_layer_influences(model, alpha=2.0, beta=1.0)
        # selected_layers = filter_layers_by_influence(layers_influences, threshold=0.1)

        # # 更新模型结构
        # self.model=update_model_with_filtered_layers(model, selected_layers)





    
    # @torch.no_grad()
    # def get_input(self, batch, k, bs=None, *args, **kwargs):
    #     z0, c = super().get_input(batch, 'jpg', *args, **kwargs)
    #     z0_dct = dct_2d(z0, norm='ortho')
    #     if self.control_mode == 'low_pass':
    #         z0_dct_filter = low_pass(z0_dct, 30)      # the threshold value can be adjusted
    #     elif self.control_mode == 'mini_pass':
    #         z0_dct_filter = low_pass_and_shuffle(z0_dct, 10)   # the threshold value can be adjusted
    #     elif self.control_mode == 'mid_pass':
    #         z0_dct_filter = high_pass(low_pass(z0_dct, 40), 20)  # the threshold value can be adjusted
    #     elif self.control_mode == 'high_pass':
    #         z0_dct_filter = high_pass(z0_dct, 50)   # the threshold value can be adjusted
    #     control = idct_2d(z0_dct_filter, norm='ortho')
    #     if bs is not None:
    #         control = control[:bs]
    #     return z0, dict(c_crossattn=[c], c_concat=[control])
        
    def configure_model(self):
        """参数初始化配置"""
        # 方案1：部分继承教师参数
        self._partial_init_from_teacher()
        
        # 方案2：Xavier初始化（备用）
        self._xavier_init()
        
        # # 冻结不需要训练的参数
        # self._freeze_params()

    def _partial_init_from_teacher(self):
        """部分继承教师模型参数"""
        for s_name, s_param in self.model.named_parameters():
            if s_name in self.teacher.model.state_dict():
                t_param = self.teacher.model.state_dict()[s_name]
                if s_param.shape == t_param.shape:
                    s_param.data.copy_(t_param.data * 0.8)  # 混合初始化
                    print(f"Init {s_name} from teacher")

    def _xavier_init(self):
        """Xavier初始化未匹配的参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.dim() > 1:
                if 'conv' in name or 'linear' in name:
                    torch.nn.init.xavier_uniform_(param)
                    # print(f"Xavier init: {name}")

    def _freeze_params(self):
        """冻结控制网络外的参数"""
        for name, param in self.named_parameters():
            if 'control_model' not in name:
                param.requires_grad = False
        print("冻结非控制网络参数")


    @staticmethod
    def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=True):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            # print(f"Global Step: {pl_sd['global_step']}")
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
        model.to(device)
        model.eval()
        return model



    @torch.no_grad()
    def get_intermediates(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            # print(f"uc_full,unconditional_guidance_scale:{uc_full,unconditional_guidance_scale}")
            samples, intermediates = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,verbose=False,progbar=False,**kwargs
                                             )


        return samples,intermediates

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        # print(f"self.model:{self.model}")
        
        if self.model == self.teacher.model:
            # print("now is teacher")
            self.control_scales = [1.0]*13
            control_model = self.teacher.control_model
        else:
            #print("now is student")
            self.control_scales = [1.0]*13   #[1.0]*9
            control_model = self.control_model
        diffusion_model = self.model.diffusion_model  # ControlledUnetModel

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)

        else:
            control = control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [[c[0] * scale, c[1] * scale] for c, scale in zip(control, self.control_scales)]
            # print("dis Control_add shape2:", [c[0].shape for c in control])  # 打印 control_add 的形状
            # print("dis Control_mul shape2:", [c[1].shape for c in control])  # 打印 control_mul 的形状
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        return eps

    def training_step(self, batch, batch_idx):
        # 准备输入
        # z0, cond = self.get_input(batch, self.first_stage_key)
        
        # t = torch.randint(0, self.num_timesteps, (z0.shape[0],), device=self.device)
        
        # noise = torch.randn_like(z0)
        # x_noisy = self.q_sample(x_start=z0, t=t, noise=noise)
        # 编码文本条件
        # with torch.no_grad():
        #     text_emb = self.get_learned_conditioning(cond['c_crossattn'])

        
        # original_model = self.model
        # self.model = self.teacher.model
        # 教师推理
        with torch.no_grad():
            #去噪输出
            # teacher_out = self.teacher.apply_model(
            #     x_noisy, t, cond
            #     # {'c_crossattn': [text_emb], 'c_concat': cond['c_concat']}
            # )
            
            # teacher
            # teacher_noise_loss = F.mse_loss(teacher_out, noise)
            # 
            # teacher_noise_loss = super().p_losses(z0, cond, t)
            # print(f"Teacher自身预测误差: {teacher_noise_loss}")
            x, c = self.teacher.get_input(batch, 'jpg')       
            teacher_eps = self.teacher(x,c,return_eps=True)

            teacher_noise_loss, _ = self.teacher.shared_step(batch)
            # print(f"Teacher自身预测误差: {teacher_noise_loss}")
            # # teacher_out,teacher_inters = self.teacher.log_images(batch,for_distillation=True,dis_inter=True)
            # teacher_out,teacher_inters = self.teacher.log_images(batch,for_distillation=True)
            # # print(f"teacher_inters:{teacher_inters}")
            
        
        # self.model = original_model
        # student_noise_loss = super().p_losses(z0, cond, t)
        student_noise_loss, student_loss_dict = self.model.shared_step(batch)
        # student_out,student_inters = self.get_intermediates(batch)
        # student_out,student_inters = self.get_intermediates(batch,dis_inter=True)
        # print(f"student_out:{student_out.shape}")
        # print("teacher done")
        x, c = self.model.get_input(batch, 'jpg')    
        student_eps = self.model(x,c,return_eps=True)

        # l1_loss = F.l1_loss(student_eps, teacher_eps, reduction='none')

        # 对所有维度的损失进行平均
        kl_loss = self.noise_loss_normalized(student_eps,teacher_eps)

        # #
        # # time_steps = list(range(len(teacher_inters)))  # 时间步
        # # total_steps = len(time_steps)
        # # student_dim = student_inters['x_inter'][0].shape[1]
        # # teacher_dim = teacher_inters['x_inter'][0].shape[1]
        # # spatial_dims = (student_inters['x_inter'][0].shape[2],student_inters['x_inter'][0].shape[3])
        # # # 初始化LatentProjector
        # # latent_projector = LatentProjector(student_dim=student_dim, teacher_dim=teacher_dim, use_conv=True, scale_factor=0.13025,spatial_dims=spatial_dims)

        # # # 计算蒸馏损失
        # # dis_loss = self.distillation_loss(teacher_inters, student_inters, time_steps, latent_projector, total_steps)

        # # # 计算去噪重建
        # # # z0_teacher = super().predict_start_from_noise(x_noisy, t, teacher_out)
        # # # z0_student = super().predict_start_from_noise(x_noisy, t, student_out)
        
        # # # # 解码图像
        # # # with torch.no_grad():
        # # #     real_img = self.decode_first_stage(z0)
        # # #     teacher_img = self.decode_first_stage(z0_teacher)
        # # # student_img = self.decode_first_stage(z0_student)
        
        # # # # 计算感知损失
        # # # percep_term = (
        # # #     self.lpips((real_img+1)/2, (teacher_img+1)/2) +
        # # #     self.lpips((real_img+1)/2, (student_img+1)/2) +
        # # #     self.lpips((teacher_img+1)/2, (student_img+1)/2)
        # # # ) / 3.0
        
        # # kl_loss = F.mse_loss(student_noise_loss[0], teacher_noise_loss[0])
        # # # 综合损失计算
        # # # print(f"student_noise_loss,percep_term,total_loss:{student_noise_loss,percep_term}")
        # # total_loss = student_noise_loss[0] + self.lambda_perceptual * percep_term
        # # # self.lambda_kl * kl_loss 
        
        # # print(f"student_noise_loss[0],percep_term,total_loss:{student_noise_loss,percep_term,total_loss}")
        # # # t_weights = self.get_t_weights(t).view(-1,1,1,1)
        # # # orig_loss = (F.mse_loss(student_out, noise, reduction='none') * t_weights).mean()
        
        # # total_loss = 0.382*student_noise_loss[0] +0.618*dis_loss
        # # print(f"student_noise_loss,kl_loss,total_loss:{student_noise_loss[0],dis_loss,total_loss}")
        # # 在计算损失时使用归一化的潜变量
        # kl_loss_steps = self.dis_step_loss(teacher_inters,student_inters)
        
        # # lambda_loss1,lambda_loss2=0.618,0.382
        current_step = self.trainer.global_step
        if current_step % 500==0 :
            self.lambda_loss1 = 0.7
        #     self.lambda_loss2 = 1.0-self.lambda_loss1
        # # 动态阈值
        # # 动态调整阈值或权重
        # # current_step = self.trainer.global_step
        # # threshold = 0.5 - 0.05 * (current_step // 100)  # 每1000个迭代减少0.05
        # # threshold = max(threshold, 0.2)  # 最小阈值为0.2

        # # self.lambda_loss1, self.lambda_loss2 = 0.618, 0.382
        # # if student_noise_loss<=0.2:
        # #     self.lambda_loss1, self.lambda_loss2 = 0.77, 0.23
        
        # # if student_noise_loss <= threshold:
        # #     self.lambda_loss1, self.lambda_loss2 = 0.618, 0.382
        # # else:
        # #     self.lambda_loss1, self.lambda_loss2 = 0.382, 0.618

        # # total_loss = lambda_loss1*student_noise_loss +lambda_loss2*kl_loss
        # # print(f"蒸馏损失：student_noise_loss,kl_loss,total_loss:{student_noise_loss,kl_loss,total_loss}")

        # total_loss = self.lambda_loss1*student_noise_loss +self.lambda_loss2*kl_loss_steps

        total_loss = self.lambda_loss1*student_noise_loss + (1-self.lambda_loss1)*kl_loss
        
        loss_dict = {}
        log_prefix = 'training' if self.training else 'val'
        # # 更新损失字典
        # loss_dict.update({f'{log_prefix}/s_noise_loss': student_noise_loss})
        # loss_dict.update({f'{log_prefix}/t_noise_loss': teacher_noise_loss})
        # loss_dict.update({f'{log_prefix}/kl_loss': kl_loss})
        # loss_dict.update({f'{log_prefix}/total_loss': total_loss})

        # # 每隔一定频率输出日志
        # if batch_idx % self.log_every_n_steps == 0:
        #     self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        #     print(f"蒸馏损失：student_noise_loss,kl_loss,total_loss,lambda_loss1,lambda_loss2:{student_noise_loss,kl_loss,total_loss,self.lambda_loss1}")

            
        # 通过 self.log 实时记录各项指标到进度条上
        self.log(f'{log_prefix}/s_noise_loss', student_noise_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log(f'{log_prefix}/t_noise_loss', teacher_noise_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log(f'{log_prefix}/kl_loss', kl_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log(f'{log_prefix}/total_loss', total_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log(f'{log_prefix}/lambda_loss1', self.lambda_loss1, on_step=True, on_epoch=False, prog_bar=True)
        return total_loss 
       

    @staticmethod
    def normalize_noise(noise):
        """
        对噪声进行 L2 归一化。
        :param noise: 噪声张量，形状为 [batch_size, channels, height, width]
        :return: 归一化后的噪声
        """
        norm = torch.sqrt(torch.sum(noise ** 2, dim=[1, 2, 3], keepdim=True) + 1e-8)
        return noise / norm
    
    @staticmethod
    def noise_loss_normalized(student_noise, teacher_noise):
        """
        计算归一化后的学生模型和教师模型预测噪声之间的均方误差。
        :param student_noise: 学生模型预测的噪声
        :param teacher_noise: 教师模型预测的噪声
        :return: 归一化后的噪声损失
        """
        student_noise_norm = FCDiffusionDistill.normalize_noise(student_noise)
        teacher_noise_norm = FCDiffusionDistill.normalize_noise(teacher_noise)
        return torch.mean((student_noise_norm - teacher_noise_norm) ** 2)

    # 在validation_step中添加
    def validation_step(self, batch, batch_idx):
        # 计算梯度范数
        norms = [p.grad.norm().item() for p in self.parameters() if p.grad is not None]
        self.log("grad_norm", np.median(norms), prog_bar=True)
        
        # 参数更新量监控
        with torch.no_grad():
            param_change = torch.cat([(p - p_old).abs().flatten() 
                                    for p, p_old in zip(self.parameters(), self.last_params)])
            self.log("param_update", param_change.mean().item())
        self.last_params = [p.clone() for p in self.parameters()]

    # @staticmethod
    # def calculate_feature_loss(teacher_steps,student_steps):
    #     # 适合同结构压缩
    #     # loss = sum(F.l1_loss(s, t) for s, t in zip(student_steps, teacher_steps)) / len(teacher_steps)

        
    #     # 初始化投影器（假设潜变量是空间结构）
    #     self.projection = LatentProjector(
    #         student_dim=512,
    #         teacher_dim=1024,
    #         spatial_dims=(32, 32),
    #         use_conv=True
    #     )
        
    #     # CLIP模型冻结
    #     self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    #     self.clip_model.requires_grad_(False)
        

    #     # 分层蒸馏损失
    #     total_loss = 0
    #     for t in range(50):  # 关键时间步
    #         z_s = student_inters['x_inter'][t]
    #         z_t = teacher_inters['x_inter'][t]
            
    #         # 投影学生潜变量
    #         z_s_proj = self.projection(z_s)
            
    #         # 动态加权损失
    #         weight = 1 - (t / 50)  # 总步数50
    #         total_loss += weight * F.smooth_l1_loss(z_s_proj, z_t)
            
    #     # CLIP语义一致性
    #     clip_tea = self.clip_model.encode_image(teacher_z)
    #     clip_stu = self.clip_model.encode_image(self.projection(student_z))
    #     total_loss += 0.3 * (1 - F.cosine_similarity(clip_tea, clip_stu).mean())
        
    #     return total_loss
    
    def normalize_latents(self, latents):
        """将潜变量归一化到 [-1, 1] 范围内"""
        return latents / latents.abs().max()




    
    # def dis_step_loss(self,teacher_intermediates,student_intermediates,ddim_steps=50,verbose=False):
    #     total_loss = 0
        
    #     steps = len(teacher_intermediates['x_inter'])
    #     print(f"teacher_intermediates:{steps}")
    #     for t in range(steps):
    #         teacher_x_t = teacher_intermediates['x_inter'][t]
    #         student_x_t = student_intermediates['x_inter'][t]
    #         teacher_x_t_normalized = self.normalize_latents(teacher_x_t)
    #         student_x_t_normalized = self.normalize_latents(student_x_t)
    #         loss = F.mse_loss(student_x_t_normalized, teacher_x_t_normalized)
    #         # 使用 L2 损失对齐潜变量
    #         # loss = F.mse_loss(student_x_t, teacher_x_t)
    #         total_loss += loss

    #         if verbose:
    #             print(f"Time step {t}: Loss = {loss.item()}")

    #     return total_loss / steps

    def dis_step_loss(self, teacher_inters, student_inters):
        total_loss = 0
        # 获取实际通道数
        in_channels = teacher_inters['x_inter'][0].size(1) 
        # 动态创建注意力层
        attn_layer = ChannelAttention(in_channels).to(self.device)
        
        for t in range(len(teacher_inters['x_inter'])):
            tea_feat = self.normalize_latents(teacher_inters['x_inter'][t])
            stu_feat = self.normalize_latents(student_inters['x_inter'][t])
            
            attn_map = attn_layer(tea_feat)
            
            loss = F.mse_loss(
                stu_feat * attn_map, 
                tea_feat * attn_map,
                reduction='mean'  # 改用平均损失
            )
            total_loss += loss * (0.9 ** t)
            
        return total_loss / len(teacher_inters['x_inter'])
    @staticmethod
    def distillation_loss(teacher_z, student_z, time_steps, latent_projector, total_steps):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load("ViT-B/32", device=device)

        total_loss = 0
        for t in time_steps:
            # 投影学生潜变量到教师空间
            
            proj_z = latent_projector(student_z['x_inter'][t])
          
            # 动态加权损失
            weight = 1 - (t / total_steps)  # 早期时间步权重更大
            total_loss += weight * F.smooth_l1_loss(proj_z, teacher_z[t])
      
        # CLIP空间语义一致性
        clip_loss = 1 - torch.cosine_similarity(
            clip_model.encode_image(teacher_z[-1]),
            clip_model.encode_image(latent_projector(student_z[-1]))
        )
      
        return total_loss + 0.3 * clip_loss
    

    

    # 改进的KL损失计算
    @staticmethod
    def kl_loss(student, teacher):
            # 通道维度独立计算
            mu_s = student.mean(dim=[1,2,3], keepdim=True)
            mu_t = teacher.mean(dim=[1,2,3], keepdim=True)
            
            # 使用余弦相似度对齐分布
            cos_sim = F.cosine_similarity(
                student.flatten(1), 
                teacher.flatten(1), 
                dim=1
            ).mean()
            
            # 带温度系数的KL散度
            temp = 0.1
            kl = 0.5 * (
                (student - teacher/temp).pow(2).sum(dim=[1,2,3]) 
                - (1 - temp**2) * student.pow(2).sum(dim=[1,2,3])
            ) / (temp**2 + 1e-8)
            
            return kl.mean() - 0.5 * cos_sim  # 结合相似度优化



    @staticmethod
    # 根据时间步动态调整损失权重
    def get_t_weights(t, max_steps=1000):
        # 早期时间步赋予更高权重
        return 1.0 - (t.float() / max_steps)  
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, N):

        return self.get_learned_conditioning([""] * N)
    
    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):

        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h, w)
        # print(f"dis shape:{shape}")
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True,  unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log['file_path'] = log_txt_as_img((512, 512), batch['path'], size=20)
        log["reconstruction"] = self.decode_first_stage(z)
        log["prompt"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=20)

        if unconditional_guidance_scale > 1.0:
            # print("fcd unconditional_guidance_scale")
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,**kwargs
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log["samples"] = x_samples_cfg

        return log

    # def configure_optimizers(self):
    #     lr = self.learning_rate
    #     params = list(self.control_model.parameters())

    #     opt = torch.optim.AdamW(params, lr=lr)
    #     return opt
    # 在FCDiffusionDistill类中添加
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=500,   # 每500步重启周期
                T_mult=1,
                eta_min=1e-6  # 最小学习率
            ),
            'interval': 'step'
        }
        return [optimizer], [scheduler]

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, min_reduction=4):
        super().__init__()
        # 动态计算最大可用reduction
        self.reduction = max(min_reduction, in_channels//4)  # 保证至少能整除
        self.reduction = min(self.reduction, in_channels)    # 不能超过输入通道数
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels//self.reduction),
            nn.ReLU(),
            nn.Linear(in_channels//self.reduction, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)  # 此时维度保证合法
        return x * y

class LatentProjector(nn.Module):
    def __init__(self, student_dim: int = 512, teacher_dim: int = 1024, use_conv: bool = True, scale_factor=0.13025, spatial_dims: tuple = (32, 32)):
        super().__init__()
        self.use_conv = use_conv
        self.scale_factor = scale_factor
        
        if use_conv:
            # 根据spatial_dims定义卷积层
            C_in, H_in, W_in = student_dim, spatial_dims[0], spatial_dims[1]
            C_out, H_out, W_out = teacher_dim, spatial_dims[0], spatial_dims[1]
            # print(f"C_out:{C_out}")
            self.proj = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=1, num_channels=C_out),
                nn.GELU(),
                nn.Conv2d(C_out, C_out, kernel_size=1)
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(student_dim, teacher_dim),
                nn.GELU(),
                nn.Linear(teacher_dim, teacher_dim)
            )

        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, z_s: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            z = self.proj(z_s)
            return z * self.scale_factor
        else:
            if z_s.dim() == 4:
                z_s = z_s.flatten(start_dim=1)
            return self.proj(z_s) * self.scale_factor



# 训练配置
if __name__ == "__main__":
    # 初始化配置

    student_config = 'configs/student_model_config.yaml'
    student_ckpt  =   './models/FCDiffusion_ini_mid_8_64_for_dis.ckpt'      #  #'models/FCDiffusion_ini_EA.ckpt'
    # student_ckpt = "/home/apulis-dev/userdata/FCDiffusion_code_EA/lightning_logs/fcdiffusion_mid_pass_checkpoint/epoch=3-step=79999-v1.ckpt"
    # teacher_ckpt = '/home/apulis-dev/userdata/FCDiffusion_code/lightning_logs_SA/fcdiffusion_low_pass_checkpoint/epoch=2-step=53999.ckpt'
    teacher_ckpt =  'lightning_logs/fcdiffusion_mid_pass_checkpoint/epoch=11-step=241999.ckpt'
    # teacher_ckpt = '/home/apulis-dev/userdata/FCDiffusion_code/lightning_logs_SA/fcdiffusion_low_pass_checkpoint/epoch=1-step=3999.ckpt'
    # 创建模型
    model = FCDiffusionDistill(
        teacher_config='configs/model_config.yaml',
        teacher_ckpt=teacher_ckpt,
        student_config=student_config,
        student_ckpt=student_ckpt,
        # control_stage_config=OmegaConf.load('configs/control_stage_config.yaml'),
        # first_stage_config=OmegaConf.load('configs/first_stage_config.yaml'),
        # cond_stage_config=OmegaConf.load('configs/cond_stage_config.yaml'),
        # unet_config=OmegaConf.load('configs/unet_config.yaml'),
        # control_scales=[1.0]*13
    )
    model.learning_rate = 1e-5
    # 数据加载
    dataset = TrainDataset('/home/apulis-dev/userdata/FCDiffusion_code/datasets/training_data_new.json',cache_size=1000)
    dataloader = DataLoader(dataset,num_workers=2,batch_size=2, shuffle=True)
    mode = 'mid_pass'
    val_every_n_train_steps = 1000
    checkpoint_path = 'distiallation_' + mode + '_checkpoint'
    val_checkpoint = ModelCheckpoint(dirpath='lightning_logs/' + checkpoint_path,
                                 every_n_train_steps=val_every_n_train_steps, save_top_k=-1)
    # 训练器配置
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=100,
        gradient_clip_val=1.0,
        # callbacks=[ImageLogger(batch_frequency=500)],
        log_every_n_steps=1000,
        callbacks=[val_checkpoint],
        precision=16
    )
    
    # 开始训练
    trainer.fit(model, dataloader)

