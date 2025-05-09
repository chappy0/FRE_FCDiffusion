import torch
import torch as th
import torch.nn as nn
from tools.dct_util import DCTBasisCache, dct_2d, idct_2d, low_pass, high_pass, low_pass_and_shuffle
import os
import sys
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


# The Unet of the LDM, which is controlled by the FreqControlNet
class ControlledUnetModel(UNetModel):
    def __init__(self, *args, **kwargs):
        # 从 kwargs 中提取 args 和 config
        self.args = kwargs.pop('args', None)
        self.config = kwargs.pop('config', None)
        super().__init__(*args, **kwargs)


    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        # print(f"Control unet forward params{x.shape, timesteps.shape, context.shape, len(control), only_mid_control}")
        for key, value in kwargs.items():
            print(f"{key}: {value}")
        hs = []
        
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            # for idx, block in enumerate(self.input_blocks):
                # print(f"Layer {idx}:")
                # for sub_idx, sub_block in enumerate(block.children()):
                #     print(f"Sub-layer {sub_idx}: {sub_block}")
            for module in self.input_blocks:
                # print(f"module:{module.__class__.__name__}")
                # print(f"h,emb,context:{h.shape,emb.shape,context.shape}")
                
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)
        # print("hs value:", [value.shape for value in hs])
        if control is not None:
            
            [control_add, control_mul] = control.pop()
            # #print(f"before control_add, control_mul,h:{control_add.shape, control_mul.shape,h.shape}")
            h = (1 + control_mul) * h + control_add
            # #print(f"after control_add, control_mul,h:{control_add.shape, control_mul.shape,h.shape}")            
        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                [control_add, control_mul] = control.pop()
                # #print(f"output_block control_add, control_mul,h:{control_add.shape, control_mul.shape,h.shape}")
                h = torch.cat([h, (1 + control_mul) * hs.pop() + control_add], dim=1)
            h = module(h, emb, context)
            # #print(f"final control_add, control_mul,h:{control_add.shape, control_mul.shape,h.shape}")
        h = h.type(x.dtype)
        return self.out(h)





# The FCNet
class FreqControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 4),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            use_external_attention = False
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs_add = nn.ModuleList([self.make_zero_conv(model_channels)])
        self.zero_convs_mul = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, in_channels, 512, 3, padding=1),
            nn.SiLU()
        )
        self.hint_add = zero_module(conv_nd(dims, 512, model_channels, 1, padding=0))
        self.hint_mul = zero_module(conv_nd(dims, 512, model_channels, 1, padding=0))

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                # print(f"ch,time_embed_dim,dropout,mult,model_channels,dims,use_checkpoint,use_scale_shift_norm:{ch,time_embed_dim,dropout,mult,model_channels,dims,use_checkpoint,use_scale_shift_norm}")
                
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,use_external_attention=use_external_attention
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs_add.append(self.make_zero_conv(ch))
                self.zero_convs_mul.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs_add.append(self.make_zero_conv(ch))
                self.zero_convs_mul.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint,use_external_attention=use_external_attention
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out_add = self.make_zero_conv(ch)
        self.middle_block_out_mul = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)
        guided_hint_add = self.hint_add(guided_hint)
        guided_hint_mul = self.hint_mul(guided_hint)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv_add, zero_conv_mul in zip(self.input_blocks, self.zero_convs_add, self.zero_convs_mul):
            if guided_hint is not None:
                # print(f"h.shape:{h.shape}")
                h = module(h, emb, context)
                h = h * (1 + guided_hint_mul) + guided_hint_add
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append([zero_conv_add(h, emb, context), zero_conv_mul(h, emb, context)])

        h = self.middle_block(h, emb, context)
        outs.append([self.middle_block_out_add(h, emb, context), self.middle_block_out_mul(h, emb, context)])

        return outs


class FCDiffusion(LatentDiffusion):

    def __init__(self, control_stage_config, only_mid_control, control_mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #class_name = self.__class__.__name__
        #function_name = sys._getframe().f_code.co_name
        ##print(f"Executing {function_name} in class {class_name}")

        self.control_model = instantiate_from_config(control_stage_config)  # FreqControlNet
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        assert control_mode in ['mini_pass', 'low_pass', 'mid_pass', 'high_pass'], \
            'control_mode must be in the list of [\'mini_pass\', \'low_pass\', \'mid_pass\', \'high_pass\'], but get value {0}'.format(control_mode)
        self.control_mode = control_mode
        self.save_eps_path = "/home/apulis-dev/userdata/FCDiffusion_code/models"

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        #class_name = self.__class__.__name__
        #function_name = sys._getframe().f_code.co_name
        ##print(f"Executing {function_name} in class {class_name}")

        z0, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        device = self.device 
        z0 = z0.to(device)
        # c = {k: [t.to(device) for t in v] for k, v in c.items()}
        #print(f"fcd z0 shape: {z0.shape,bs}") 
        dct_cache = DCTBasisCache(max_cache_size=8)
        z0_dct = dct_2d(z0, norm='ortho',dct_cache=dct_cache)
        if self.control_mode == 'low_pass':
            z0_dct_filter = low_pass(z0_dct, 30)      # the threshold value can be adjusted
        elif self.control_mode == 'mini_pass':
            z0_dct_filter = low_pass_and_shuffle(z0_dct, 10)   # the threshold value can be adjusted
        elif self.control_mode == 'mid_pass':
            z0_dct_filter = high_pass(low_pass(z0_dct, 40), 20)  # the threshold value can be adjusted
        elif self.control_mode == 'high_pass':
            z0_dct_filter = high_pass(z0_dct, 50)   # the threshold value can be adjusted
        control = idct_2d(z0_dct_filter, norm='ortho')
        if bs is not None:
            control = control[:bs]
        return z0, dict(c_crossattn=[c], c_concat=[control])

    def save_model_structure(self):
        # 创建保存路径的目录
        os.makedirs(os.path.dirname(self.model_structure_path), exist_ok=True)

        # 打印模型结构并保存到文件
        with open(self.model_structure_path, "w") as f:
            print(self.model.diffusion_model, file=f)
        #print(f"Model structure saved to {self.model_structure_path}")

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        #class_name = self.__class__.__name__
        #function_name = sys._getframe().f_code.co_name
        ##print(f"Executing {function_name} in class {class_name}")
        # #print(f"model:{self.model.diffusion_model}")
        # self.model_structure_path = '/home/apulis-dev/userdata/FCDiffusion_code/models/modellayer.txt'  # 模型结构保存路径

        # 保存模型结构到文本文件
        # if self.model_structure_path:
        #     self.save_model_structure()


        # print("c_crossattn shape:", [c.shape for c in cond['c_crossattn']])
        # print("c_concat shape:", [c.shape for c in cond['c_concat']])
        # print("start fcd apply_model")
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model  # ControlledUnetModel

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        # #print(f"x_noisy shape:{x_noisy.shape}")
        # #print(f"t :{t}")
        # #print(f"cond_txt shape:{cond_txt.shape}")

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            # print("cond['c_concat'] is not None")
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            # #print(f"control length:{len(control)}")
            # print("Control_add shape2:", [c[0].shape for c in control])  # 打印 control_add 的形状
            # print("Control_mul shape2:", [c[1].shape for c in control])  # 打印 control_mul 的形状
            control = [[c[0] * scale, c[1] * scale] for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        
        # if self.save_eps_path:
        #     os.makedirs(self.save_eps_path, exist_ok=True)
        #     filename = os.path.join(self.save_eps_path, f"eps_t_{t[0].item()}.txt")
        #     np.savetxt(filename, eps.cpu().numpy().flatten(), fmt="%.6f")
        #     #print(f"Saved eps to {filename}")

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        #class_name = self.__class__.__name__
        #function_name = sys._getframe().f_code.co_name
        ##print(f"Executing {function_name} in class {class_name}")

        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,progbar=True,
                   **kwargs):
        #class_name = self.__class__.__name__
        #function_name = sys._getframe().f_code.co_name
        ##print(f"Executing {function_name} in class {class_name}")
        # print("fcd log_images")
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log['file_path'] = log_txt_as_img((512, 512), batch['path'], size=20)
        log["reconstruction"] = self.decode_first_stage(z)
        log["prompt"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=20)

        if plot_diffusion_rows:
            # get diffusion row
            print("fcd get diffusion row")
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            print("fcd get denoise row")
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            # print("fcd unconditional_guidance_scale")
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, intermediates = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,progbar=True,**kwargs
                                             )
            if 'for_distillation' in kwargs:
                return samples_cfg,intermediates

            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log["samples"] = x_samples_cfg

            return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        #class_name = self.__class__.__name__
        #function_name = sys._getframe().f_code.co_name
        ##print(f"Executing {function_name} in class {class_name}")
        # print("fcd sample_log")
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h, w)
        # print(f"progbar:{kwargs['progbar']}")
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        #class_name = self.__class__.__name__
        #function_name = sys._getframe().f_code.co_name
        ##print(f"Executing {function_name} in class {class_name}")

        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        #class_name = self.__class__.__name__
        #function_name = sys._getframe().f_code.co_name
        ##print(f"Executing {function_name} in class {class_name}")

        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
