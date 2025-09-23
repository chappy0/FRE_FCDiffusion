# # # # import torch
# # # # from diffusers import StableDiffusionPipeline
# # # # import argparse
# # # # import time
# # # # import numpy as np

# # # # # 导入prompt-to-prompt的核心工具
# # # # # 假设此脚本与prompt_to_prompt文件夹在同一目录下
# # # # # from prompt_to_prompt.ptp_utils import register_attention_control, AttentionStore
# # # # # from prompt_to_prompt.null_inversion import NullInversion

# # # # from ptp_utils import register_attention_control
# # # # from attention_control import AttentionStore
# # # # from null_inversion import NullInversion


# # # # def run_p2p_inference(pipeline, prompt, num_inference_steps=50, guidance_scale=7.5, generator=None):
# # # #     """
# # # #     执行一次完整的Prompt-to-Prompt编辑推理过程。
# # # #     这是一个简化的例子，执行 "replace" 操作。
# # # #     """
# # # #     # 编辑指令示例：将 "cat" 替换为 "dog"
# # # #     prompts = [prompt, prompt.replace("cat", "dog")]

# # # #     # 1. Null-Text Inversion (获取用于重构的 latents)
# # # #     # 这是P2P编辑的第一步，也需要计入总时间
# # # #     null_inversion = NullInversion(pipeline)
# # # #     (image_latents, image_gt, a, b) = null_inversion.invert(prompt, latents=None, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)

# # # #     # 2. 执行编辑
# # # #     controller = AttentionStore()
# # # #     register_attention_control(pipeline, controller) # 注册注意力控制器

# # # #     # 运行编辑推理
# # # #     images, _ = pipeline(prompts,
# # # #                          latents=image_latents,
# # # #                          num_inference_steps=num_inference_steps,
# # # #                          guidance_scale=guidance_scale,
# # # #                          generator=generator)

# # # #     return images


# # # # def main(args):
# # # #     # 1. 初始化模型
# # # #     print(f"Loading Stable Diffusion v1.5 model from: {args.model_path}")
# # # #     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# # # #     # 加载模型
# # # #     sd_pipeline = StableDiffusionPipeline.from_single_file(
# # # #         args.model_path,
# # # #         torch_dtype=torch.float16,
# # # #         safety_checker=None
# # # #     ).to(device)
# # # #     sd_pipeline.scheduler.set_timings(args.steps)
    
# # # #     # 使用固定的种子以保证可复现性
# # # #     generator = torch.Generator(device).manual_seed(1234)
# # # #     prompt = "A photo of a cat riding a skateboard"
# # # #     print(f"Using fixed prompt: '{prompt}'")
# # # #     print("-" * 50)

# # # #     # --- 显存测试 ---
# # # #     print("Running a single inference to measure peak GPU memory...")
# # # #     torch.cuda.empty_cache()
# # # #     torch.cuda.reset_peak_memory_stats()
    
# # # #     # 执行一次完整的推理
# # # #     _ = run_p2p_inference(sd_pipeline, prompt, num_inference_steps=args.steps, generator=generator)
    
# # # #     # 获取峰值显存
# # # #     peak_memory_bytes = torch.cuda.max_memory_allocated()
# # # #     peak_memory_gb = peak_memory_bytes / (1024 ** 3)
# # # #     print(f"Peak GPU Memory Allocated: {peak_memory_gb:.2f} GB")
# # # #     torch.cuda.empty_cache()
# # # #     print("-" * 50)

# # # #     # --- 速度测试 ---
# # # #     # 预热运行 (非常重要)
# # # #     print(f"Performing {args.warmup_runs} warm-up runs...")
# # # #     for _ in range(args.warmup_runs):
# # # #         _ = run_p2p_inference(sd_pipeline, prompt, num_inference_steps=args.steps, generator=generator)

# # # #     # 正式计时运行
# # # #     print(f"Performing {args.timed_runs} timed runs...")
# # # #     timings = []
# # # #     for i in range(args.timed_runs):
# # # #         torch.cuda.synchronize() # 等待GPU完成所有先前工作
# # # #         start_time = time.time()
        
# # # #         _ = run_p2p_inference(sd_pipeline, prompt, num_inference_steps=args.steps, generator=generator)
        
# # # #         torch.cuda.synchronize() # 确保推理已在GPU上完成
# # # #         end_time = time.time()
        
# # # #         elapsed = end_time - start_time
# # # #         timings.append(elapsed)
# # # #         print(f"Run {i+1}/{args.timed_runs}: {elapsed:.4f} seconds")

# # # #     # 计算并打印结果
# # # #     timings_np = np.array(timings)
# # # #     print("-" * 50)
# # # #     print("Benchmark Results:")
# # # #     print(f"  Average Inference Time: {timings_np.mean():.4f} seconds")
# # # #     print(f"  Standard Deviation:   {timings_np.std():.4f} seconds")
# # # #     print(f"  Median Inference Time:  {np.median(timings_np):.4f} seconds")
# # # #     print("-" * 50)


# # # # if __name__ == "__main__":
# # # #     parser = argparse.ArgumentParser(description="Benchmark script for Prompt-to-Prompt.")
    
# # # #     # 将此处的默认路径修改为您离线服务器上SD v1.5模型的实际路径
# # # #     parser.add_argument(
# # # #         "--model_path",
# # # #         type=str,
# # # #         default="/home/apulis-dev/userdata/stablediffusion/models/v2-1_512-ema-pruned.ckpt", # <--- 请务必修改这里
# # # #         help="Path to the Stable Diffusion v1.5 model checkpoint file (.ckpt or .safetensors)."
# # # #     )
# # # #     parser.add_argument(
# # # #         "--steps",
# # # #         type=int,
# # # #         default=50,
# # # #         help="Number of inference steps."
# # # #     )
# # # #     parser.add_argument(
# # # #         "--warmup_runs",
# # # #         type=int,
# # # #         default=3,
# # # #         help="Number of warm-up runs before timing."
# # # #     )
# # # #     parser.add_argument(
# # # #         "--timed_runs",
# # # #         type=int,
# # # #         default=20,
# # # #         help="Number of timed runs to average."
# # # #     )

# # # #     args = parser.parse_args()
# # # #     main(args)


# # # import torch
# # # from diffusers import StableDiffusionPipeline
# # # import argparse
# # # import time
# # # import numpy as np
# # # import abc
# # # from typing import Optional, Union, Tuple, List, Callable, Dict
# # # from tqdm import tqdm


# # # # ===============================================================================================
# # # # == 代码块 1: 从Notebook中提取的 Attention Control 定义
# # # # == 来源: prompt-to-prompt_stable.ipynb
# # # # ===============================================================================================
# # # class AttentionControl(abc.ABC):
# # #     def step_callback(self, x_t):
# # #         return x_t
    
# # #     def between_steps(self):
# # #         return
    
# # #     @property
# # #     def num_uncond_att_layers(self):
# # #         return 0
    
# # #     @abc.abstractmethod
# # #     def forward (self, attn, is_cross: bool, place_in_unet: str):
# # #         raise NotImplementedError

# # #     def __call__(self, attn, is_cross: bool, place_in_unet: str):
# # #         if self.cur_att_layer >= self.num_uncond_att_layers:
# # #             if self.low_resource:
# # #                 attn = self.forward(attn, is_cross, place_in_unet)
# # #             else:
# # #                 h = attn.shape[0]
# # #                 attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
# # #         self.cur_att_layer += 1
# # #         if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
# # #             self.cur_att_layer = 0
# # #             self.cur_step += 1
# # #             self.between_steps()
# # #         return attn
    
# # #     def reset(self):
# # #         self.cur_step = 0
# # #         self.cur_att_layer = 0

# # #     def __init__(self, low_resource=False):
# # #         self.low_resource = low_resource
# # #         self.cur_step = 0
# # #         self.num_att_layers = -1
# # #         self.cur_att_layer = 0

# # # class AttentionStore(AttentionControl):
# # #     @staticmethod
# # #     def get_empty_store():
# # #         return {"down_cross": [], "mid_cross": [], "up_cross": [],
# # #                 "down_self": [],  "mid_self": [],  "up_self": []}

# # #     def forward(self, attn, is_cross: bool, place_in_unet: str):
# # #         key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
# # #         if attn.shape[1] <= 32 ** 2:
# # #             self.step_store[key].append(attn)
# # #         return attn

# # #     def between_steps(self):
# # #         if len(self.attention_store) == 0:
# # #             self.attention_store = self.step_store
# # #         else:
# # #             for key in self.attention_store:
# # #                 for i in range(len(self.attention_store[key])):
# # #                     self.attention_store[key][i] += self.step_store[key][i]
# # #         self.step_store = self.get_empty_store()

# # #     def get_average_attention(self):
# # #         average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
# # #         return average_attention

# # #     def reset(self):
# # #         super(AttentionStore, self).reset()
# # #         self.step_store = self.get_empty_store()
# # #         self.attention_store = {}

# # #     def __init__(self, low_resource=False):
# # #         super(AttentionStore, self).__init__(low_resource)
# # #         self.step_store = self.get_empty_store()
# # #         self.attention_store = {}
        
# # # # ===============================================================================================
# # # # == 代码块 2: 从Notebook中提取的 register_attention_control 定义
# # # # == 来源: ptp_utils.py / Notebooks
# # # # ===============================================================================================
# # # def register_attention_control(model, controller):
# # #     def ca_forward(self, place_in_unet):
# # #         to_out = self.to_out
# # #         if type(to_out) is torch.nn.modules.container.ModuleList:
# # #             to_out = self.to_out[0]
# # #         else:
# # #             to_out = self.to_out

# # #         def forward(x, context=None, mask=None):
# # #             batch_size, sequence_length, dim = x.shape
# # #             h = self.heads
# # #             q = self.to_q(x)
# # #             is_cross = context is not None
# # #             context = context if is_cross else x
# # #             k = self.to_k(context)
# # #             v = self.to_v(context)
# # #             q = self.reshape_heads_to_batch_dim(q)
# # #             k = self.reshape_heads_to_batch_dim(k)
# # #             v = self.reshape_heads_to_batch_dim(v)

# # #             sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

# # #             if mask is not None:
# # #                 mask = mask.reshape(batch_size, -1)
# # #                 max_neg_value = -torch.finfo(sim.dtype).max
# # #                 mask = mask[:, None, :].repeat(h, 1, 1)
# # #                 sim.masked_fill_(~mask, max_neg_value)

# # #             attn = sim.softmax(dim=-1)
# # #             attn = controller(attn, is_cross, place_in_unet)
# # #             out = torch.einsum("b i j, b j d -> b i d", attn, v)
# # #             out = self.reshape_batch_dim_to_heads(out)
# # #             return to_out(out)

# # #         return forward

# # #     class DummyController:
# # #         def __call__(self, *args):
# # #             return args[0]
# # #         def __init__(self):
# # #             self.num_att_layers = 0

# # #     if controller is None:
# # #         controller = DummyController()

# # #     def register_recr(net_, count, place_in_unet):
# # #         if net_.__class__.__name__ == 'CrossAttention':
# # #             net_.forward = ca_forward(net_, place_in_unet)
# # #             return count + 1
# # #         elif hasattr(net_, 'children'):
# # #             for net__ in net_.children():
# # #                 count = register_recr(net__, count, place_in_unet)
# # #         return count

# # #     cross_att_count = 0
# # #     sub_nets = model.unet.named_children()
# # #     for net in sub_nets:
# # #         if "down" in net[0]:
# # #             cross_att_count += register_recr(net[1], 0, "down")
# # #         elif "up" in net[0]:
# # #             cross_att_count += register_recr(net[1], 0, "up")
# # #         elif "mid" in net[0]:
# # #             cross_att_count += register_recr(net[1], 0, "mid")

# # #     controller.num_att_layers = cross_att_count

# # # # ===============================================================================================
# # # # == 代码块 3: 从Notebook中提取的 NullInversion 定义
# # # # == 来源: null_text_w_ptp.ipynb
# # # # ===============================================================================================
# # # class NullInversion:
    
# # #     def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
# # #         prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
# # #         alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
# # #         alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
# # #         beta_prod_t = 1 - alpha_prod_t
# # #         pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
# # #         pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
# # #         prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
# # #         return prev_sample

# # #     def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
# # #         timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
# # #         alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
# # #         alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
# # #         beta_prod_t = 1 - alpha_prod_t
# # #         next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
# # #         next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
# # #         next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
# # #         return next_sample
    
# # #     def get_noise_pred_single(self, latents, t, context):
# # #         noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
# # #         return noise_pred

# # #     def get_noise_pred(self, latents, t, is_forward=True, context=None):
# # #         latents_input = torch.cat([latents] * 2)
# # #         if context is None:
# # #             context = self.context
# # #         guidance_scale = 1 if is_forward else self.guidance_scale
# # #         noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
# # #         noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
# # #         noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
# # #         if is_forward:
# # #             latents = self.next_step(noise_pred, t, latents)
# # #         else:
# # #             latents = self.prev_step(noise_pred, t, latents)
# # #         return latents

# # #     @torch.no_grad()
# # #     def latent2image(self, latents, return_type='np'):
# # #         latents = 1 / 0.18215 * latents.detach()
# # #         image = self.model.vae.decode(latents)['sample']
# # #         if return_type == 'np':
# # #             image = (image / 2 + 0.5).clamp(0, 1)
# # #             image = image.cpu().permute(0, 2, 3, 1).numpy()
# # #             image = (image * 255).astype(np.uint8)
# # #         return image

# # #     @torch.no_grad()
# # #     def image2latent(self, image):
# # #         with torch.no_grad():
# # #             if type(image) is Image:
# # #                 image = np.array(image)
# # #             if type(image) is torch.Tensor and image.dim() == 4:
# # #                 latents = image
# # #             else:
# # #                 image = torch.from_numpy(image).float() / 127.5 - 1
# # #                 image = image.permute(2, 0, 1).unsqueeze(0).to(self.model.device)
# # #                 latents = self.model.vae.encode(image)['latent_dist'].mean
# # #                 latents = latents * 0.18215
# # #         return latents

# # #     @torch.no_grad()
# # #     def init_prompt(self, prompt: str):
# # #         uncond_input = self.model.tokenizer(
# # #             [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
# # #             return_tensors="pt"
# # #         )
# # #         uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
# # #         text_input = self.model.tokenizer(
# # #             [prompt],
# # #             padding="max_length",
# # #             max_length=self.model.tokenizer.model_max_length,
# # #             truncation=True,
# # #             return_tensors="pt",
# # #         )
# # #         text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
# # #         self.context = torch.cat([uncond_embeddings, text_embeddings])
# # #         self.prompt = prompt

# # #     @torch.no_grad()
# # #     def ddim_loop(self, latent):
# # #         uncond_embeddings, cond_embeddings = self.context.chunk(2)
# # #         all_latent = [latent]
# # #         latent = latent.clone().detach()
# # #         for i in range(self.num_ddim_steps):
# # #             t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
# # #             noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
# # #             latent = self.next_step(noise_pred, t, latent)
# # #             all_latent.append(latent)
# # #         return all_latent

# # #     @property
# # #     def scheduler(self):
# # #         return self.model.scheduler

# # #     @torch.no_grad()
# # #     def ddim_inversion(self, image):
# # #         latent = self.image2latent(image)
# # #         image_rec = self.latent2image(latent)
# # #         ddim_latents = self.ddim_loop(latent)
# # #         return image_rec, ddim_latents

# # #     def invert(self, prompt: str, latents: torch.FloatTensor, num_inference_steps: int, guidance_scale: float = 7.5, generator: Optional[torch.Generator] = None, low_resource: bool = False):
# # #         self.guidance_scale = guidance_scale
# # #         self.num_ddim_steps = num_inference_steps
# # #         self.init_prompt(prompt)
# # #         register_attention_control(self.model, None)
# # #         uncond_embeddings, cond_embeddings = self.context.chunk(2)
# # #         uncond_embeddings_list = []
# # #         for i in range(self.num_ddim_steps):
# # #             uncond_embeddings_list.append(uncond_embeddings)

# # #         with torch.no_grad():
# # #             latents_ddim_inversion = latents.clone()
# # #             image_rec = None 
# # #             ddim_latents = [latents_ddim_inversion] * self.num_ddim_steps

# # #         return (latents, image_rec, ddim_latents, uncond_embeddings_list)
    
# # #     def __init__(self, model):
# # #         self.model = model
# # #         self.tokenizer = self.model.tokenizer
# # #         self.num_ddim_steps = 50
# # #         self.guidance_scale = 7.5
# # #         self.prompt = None
# # #         self.context = None

# # # # ===============================================================================================
# # # # == 代码块 4: 基准测试主逻辑
# # # # ===============================================================================================
# # # def run_p2p_inference(pipeline, prompt, num_inference_steps=50, guidance_scale=7.5, generator=None):
# # #     """
# # #     执行一次完整的Prompt-to-Prompt编辑推理过程。
# # #     这是一个简化的例子，执行 "replace" 操作。
# # #     """
# # #     prompts = [prompt, prompt.replace("cat", "dog")]
    
# # #     latents = torch.randn((1, pipeline.unet.in_channels, 512 // 8, 512 // 8), generator=generator, device=pipeline.device, dtype=torch.float16)

# # #     null_inversion = NullInversion(pipeline)
# # #     (image_latents, _, _, _) = null_inversion.invert(prompt, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)

# # #     controller = AttentionStore()
# # #     register_attention_control(pipeline, controller)

# # #     images, _ = pipeline(prompts,
# # #                          latents=image_latents,
# # #                          num_inference_steps=num_inference_steps,
# # #                          guidance_scale=guidance_scale,
# # #                          generator=generator)

# # #     return images


# # # def main(args):
# # #     print(f"Loading Stable Diffusion v1.5 model from: {args.model_path}")
# # #     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# # #     sd_pipeline = StableDiffusionPipeline.from_single_file(
# # #         args.model_path,
# # #         torch_dtype=torch.float16,
# # #         safety_checker=None
# # #     ).to(device)
# # #     sd_pipeline.scheduler.set_timings(args.steps)
    
# # #     generator = torch.Generator(device).manual_seed(1234)
# # #     prompt = "A photo of a cat riding a skateboard"
# # #     print(f"Using fixed prompt: '{prompt}'")
# # #     print("-" * 50)

# # #     print("Running a single inference to measure peak GPU memory...")
# # #     torch.cuda.empty_cache()
# # #     torch.cuda.reset_peak_memory_stats()
    
# # #     _ = run_p2p_inference(sd_pipeline, prompt, num_inference_steps=args.steps, generator=generator)
    
# # #     peak_memory_bytes = torch.cuda.max_memory_allocated()
# # #     peak_memory_gb = peak_memory_bytes / (1024 ** 3)
# # #     print(f"Peak GPU Memory Allocated: {peak_memory_gb:.2f} GB")
# # #     torch.cuda.empty_cache()
# # #     print("-" * 50)

# # #     print(f"Performing {args.warmup_runs} warm-up runs...")
# # #     for _ in tqdm(range(args.warmup_runs), desc="Warm-up"):
# # #         _ = run_p2p_inference(sd_pipeline, prompt, num_inference_steps=args.steps, generator=generator)

# # #     print(f"Performing {args.timed_runs} timed runs...")
# # #     timings = []
# # #     for _ in tqdm(range(args.timed_runs), desc="Benchmarking"):
# # #         torch.cuda.synchronize()
# # #         start_time = time.time()
        
# # #         _ = run_p2p_inference(sd_pipeline, prompt, num_inference_steps=args.steps, generator=generator)
        
# # #         torch.cuda.synchronize()
# # #         end_time = time.time()
        
# # #         elapsed = end_time - start_time
# # #         timings.append(elapsed)

# # #     timings_np = np.array(timings)
# # #     print("-" * 50)
# # #     print("Benchmark Results:")
# # #     print(f"  Average Inference Time: {timings_np.mean():.4f} seconds")
# # #     print(f"  Standard Deviation:   {timings_np.std():.4f} seconds")
# # #     print(f"  Median Inference Time:  {np.median(timings_np):.4f} seconds")
# # #     print("-" * 50)


# # # if __name__ == "__main__":
# # #     parser = argparse.ArgumentParser(description="All-in-one benchmark script for Prompt-to-Prompt.")
    
# # #     parser.add_argument(
# # #         "--model_path",
# # #         type=str,
# # #         default="/home/apulis-dev/userdata/stablediffusion/models/v2-1_512-ema-pruned.ckpt", 
# # #         help="Path to the Stable Diffusion v1.5 model checkpoint file (.ckpt or .safetensors)."
# # #     )
# # #     parser.add_argument(
# # #         "--steps",
# # #         type=int,
# # #         default=50,
# # #         help="Number of inference steps."
# # #     )
# # #     parser.add_argument(
# # #         "--warmup_runs",
# # #         type=int,
# # #         default=3,
# # #         help="Number of warm-up runs before timing."
# # #     )
# # #     parser.add_argument(
# # #         "--timed_runs",
# # #         type=int,
# # #         default=20,
# # #         help="Number of timed runs to average."
# # #     )

# # #     args = parser.parse_args()
# # #     main(args)

# # import torch
# # from diffusers import StableDiffusionPipeline
# # import argparse
# # import time
# # import numpy as np
# # import abc
# # from typing import Optional, Union, Tuple, List, Callable, Dict
# # from tqdm import tqdm

# # # ===============================================================================================
# # # == 代码块 1: 从Notebook中提取的 Attention Control 定义
# # # == 来源: prompt-to-prompt_stable.ipynb
# # # ===============================================================================================
# # class AttentionControl(abc.ABC):
# #     def step_callback(self, x_t):
# #         return x_t
    
# #     def between_steps(self):
# #         return
    
# #     @property
# #     def num_uncond_att_layers(self):
# #         return 0
    
# #     @abc.abstractmethod
# #     def forward (self, attn, is_cross: bool, place_in_unet: str):
# #         raise NotImplementedError

# #     def __call__(self, attn, is_cross: bool, place_in_unet: str):
# #         if self.cur_att_layer >= self.num_uncond_att_layers:
# #             if self.low_resource:
# #                 attn = self.forward(attn, is_cross, place_in_unet)
# #             else:
# #                 h = attn.shape[0]
# #                 attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
# #         self.cur_att_layer += 1
# #         if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
# #             self.cur_att_layer = 0
# #             self.cur_step += 1
# #             self.between_steps()
# #         return attn
    
# #     def reset(self):
# #         self.cur_step = 0
# #         self.cur_att_layer = 0

# #     def __init__(self, low_resource=False):
# #         self.low_resource = low_resource
# #         self.cur_step = 0
# #         self.num_att_layers = -1
# #         self.cur_att_layer = 0

# # class AttentionStore(AttentionControl):
# #     @staticmethod
# #     def get_empty_store():
# #         return {"down_cross": [], "mid_cross": [], "up_cross": [],
# #                 "down_self": [],  "mid_self": [],  "up_self": []}

# #     def forward(self, attn, is_cross: bool, place_in_unet: str):
# #         key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
# #         if attn.shape[1] <= 32 ** 2:
# #             self.step_store[key].append(attn)
# #         return attn

# #     def between_steps(self):
# #         if len(self.attention_store) == 0:
# #             self.attention_store = self.step_store
# #         else:
# #             for key in self.attention_store:
# #                 for i in range(len(self.attention_store[key])):
# #                     self.attention_store[key][i] += self.step_store[key][i]
# #         self.step_store = self.get_empty_store()

# #     def get_average_attention(self):
# #         average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
# #         return average_attention

# #     def reset(self):
# #         super(AttentionStore, self).reset()
# #         self.step_store = self.get_empty_store()
# #         self.attention_store = {}

# #     def __init__(self, low_resource=False):
# #         super(AttentionStore, self).__init__(low_resource)
# #         self.step_store = self.get_empty_store()
# #         self.attention_store = {}
        
# # # ===============================================================================================
# # # == 代码块 2: 从Notebook中提取的 register_attention_control 定义
# # # == 来源: ptp_utils.py / Notebooks
# # # ===============================================================================================
# # def register_attention_control(model, controller):
# #     def ca_forward(self, place_in_unet):
# #         to_out = self.to_out
# #         if type(to_out) is torch.nn.modules.container.ModuleList:
# #             to_out = self.to_out[0]
# #         else:
# #             to_out = self.to_out

# #         def forward(x, context=None, mask=None):
# #             batch_size, sequence_length, dim = x.shape
# #             h = self.heads
# #             q = self.to_q(x)
# #             is_cross = context is not None
# #             context = context if is_cross else x
# #             k = self.to_k(context)
# #             v = self.to_v(context)
# #             q = self.reshape_heads_to_batch_dim(q)
# #             k = self.reshape_heads_to_batch_dim(k)
# #             v = self.reshape_heads_to_batch_dim(v)

# #             sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

# #             if mask is not None:
# #                 mask = mask.reshape(batch_size, -1)
# #                 max_neg_value = -torch.finfo(sim.dtype).max
# #                 mask = mask[:, None, :].repeat(h, 1, 1)
# #                 sim.masked_fill_(~mask, max_neg_value)

# #             attn = sim.softmax(dim=-1)
# #             attn = controller(attn, is_cross, place_in_unet)
# #             out = torch.einsum("b i j, b j d -> b i d", attn, v)
# #             out = self.reshape_batch_dim_to_heads(out)
# #             return to_out(out)

# #         return forward

# #     class DummyController:
# #         def __call__(self, *args):
# #             return args[0]
# #         def __init__(self):
# #             self.num_att_layers = 0

# #     if controller is None:
# #         controller = DummyController()

# #     def register_recr(net_, count, place_in_unet):
# #         if net_.__class__.__name__ == 'CrossAttention':
# #             net_.forward = ca_forward(net_, place_in_unet)
# #             return count + 1
# #         elif hasattr(net_, 'children'):
# #             for net__ in net_.children():
# #                 count = register_recr(net__, count, place_in_unet)
# #         return count

# #     cross_att_count = 0
# #     sub_nets = model.unet.named_children()
# #     for net in sub_nets:
# #         if "down" in net[0]:
# #             cross_att_count += register_recr(net[1], 0, "down")
# #         elif "up" in net[0]:
# #             cross_att_count += register_recr(net[1], 0, "up")
# #         elif "mid" in net[0]:
# #             cross_att_count += register_recr(net[1], 0, "mid")

# #     controller.num_att_layers = cross_att_count

# # # ===============================================================================================
# # # == 代码块 3: 从Notebook中提取的 NullInversion 定义
# # # == 来源: null_text_w_ptp.ipynb
# # # ===============================================================================================
# # class NullInversion:
    
# #     def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
# #         prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
# #         alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
# #         alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
# #         beta_prod_t = 1 - alpha_prod_t
# #         pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
# #         pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
# #         prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
# #         return prev_sample

# #     def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
# #         timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
# #         alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
# #         alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
# #         beta_prod_t = 1 - alpha_prod_t
# #         next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
# #         next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
# #         next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
# #         return next_sample
    
# #     def get_noise_pred_single(self, latents, t, context):
# #         noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
# #         return noise_pred

# #     def get_noise_pred(self, latents, t, is_forward=True, context=None):
# #         latents_input = torch.cat([latents] * 2)
# #         if context is None:
# #             context = self.context
# #         guidance_scale = 1 if is_forward else self.guidance_scale
# #         noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
# #         noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
# #         noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
# #         if is_forward:
# #             latents = self.next_step(noise_pred, t, latents)
# #         else:
# #             latents = self.prev_step(noise_pred, t, latents)
# #         return latents

# #     @torch.no_grad()
# #     def latent2image(self, latents, return_type='np'):
# #         latents = 1 / 0.18215 * latents.detach()
# #         image = self.model.vae.decode(latents)['sample']
# #         if return_type == 'np':
# #             image = (image / 2 + 0.5).clamp(0, 1)
# #             image = image.cpu().permute(0, 2, 3, 1).numpy()
# #             image = (image * 255).astype(np.uint8)
# #         return image

# #     @torch.no_grad()
# #     def image2latent(self, image):
# #         with torch.no_grad():
# #             if type(image) is Image:
# #                 image = np.array(image)
# #             if type(image) is torch.Tensor and image.dim() == 4:
# #                 latents = image
# #             else:
# #                 image = torch.from_numpy(image).float() / 127.5 - 1
# #                 image = image.permute(2, 0, 1).unsqueeze(0).to(self.model.device)
# #                 latents = self.model.vae.encode(image)['latent_dist'].mean
# #                 latents = latents * 0.18215
# #         return latents

# #     @torch.no_grad()
# #     def init_prompt(self, prompt: str):
# #         uncond_input = self.model.tokenizer(
# #             [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
# #             return_tensors="pt"
# #         )
# #         uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
# #         text_input = self.model.tokenizer(
# #             [prompt],
# #             padding="max_length",
# #             max_length=self.model.tokenizer.model_max_length,
# #             truncation=True,
# #             return_tensors="pt",
# #         )
# #         text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
# #         self.context = torch.cat([uncond_embeddings, text_embeddings])
# #         self.prompt = prompt

# #     @torch.no_grad()
# #     def ddim_loop(self, latent):
# #         uncond_embeddings, cond_embeddings = self.context.chunk(2)
# #         all_latent = [latent]
# #         latent = latent.clone().detach()
# #         for i in range(self.num_ddim_steps):
# #             t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
# #             noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
# #             latent = self.next_step(noise_pred, t, latent)
# #             all_latent.append(latent)
# #         return all_latent

# #     @property
# #     def scheduler(self):
# #         return self.model.scheduler

# #     @torch.no_grad()
# #     def ddim_inversion(self, image):
# #         latent = self.image2latent(image)
# #         image_rec = self.latent2image(latent)
# #         ddim_latents = self.ddim_loop(latent)
# #         return image_rec, ddim_latents

# #     def invert(self, prompt: str, latents: torch.FloatTensor, num_inference_steps: int, guidance_scale: float = 7.5, generator: Optional[torch.Generator] = None, low_resource: bool = False):
# #         self.guidance_scale = guidance_scale
# #         self.num_ddim_steps = num_inference_steps
# #         self.init_prompt(prompt)
# #         register_attention_control(self.model, None)
# #         uncond_embeddings, cond_embeddings = self.context.chunk(2)
# #         uncond_embeddings_list = []
# #         for i in range(self.num_ddim_steps):
# #             uncond_embeddings_list.append(uncond_embeddings)

# #         with torch.no_grad():
# #             latents_ddim_inversion = latents.clone()
# #             image_rec = None 
# #             ddim_latents = [latents_ddim_inversion] * self.num_ddim_steps

# #         return (latents, image_rec, ddim_latents, uncond_embeddings_list)
    
# #     def __init__(self, model):
# #         self.model = model
# #         self.tokenizer = self.model.tokenizer
# #         self.num_ddim_steps = 50
# #         self.guidance_scale = 7.5
# #         self.prompt = None
# #         self.context = None

# # # ===============================================================================================
# # # == 代码块 4: 基准测试主逻辑
# # # ===============================================================================================
# # def run_p2p_inference(pipeline, prompt, num_inference_steps=50, guidance_scale=7.5, generator=None):
# #     """
# #     执行一次完整的Prompt-to-Prompt编辑推理过程。
# #     这是一个简化的例子，执行 "replace" 操作。
# #     """
# #     prompts = [prompt, prompt.replace("cat", "dog")]
    
# #     latents = torch.randn((1, pipeline.unet.in_channels, 512 // 8, 512 // 8), generator=generator, device=pipeline.device, dtype=torch.float16)

# #     null_inversion = NullInversion(pipeline)
# #     (image_latents, _, _, _) = null_inversion.invert(prompt, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)

# #     controller = AttentionStore()
# #     register_attention_control(pipeline, controller)

# #     images, _ = pipeline(prompts,
# #                          latents=image_latents,
# #                          num_inference_steps=num_inference_steps,
# #                          guidance_scale=guidance_scale,
# #                          generator=generator)

# #     return images


# # def main(args):
# #     print(f"Loading Stable Diffusion model from: {args.model_path}")
# #     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# #     # [关键修改] 使用 original_config_file 参数进行纯离线加载
# #     sd_pipeline = StableDiffusionPipeline.from_single_file(
# #         args.model_path,
# #         original_config_file=args.config_path, # <-- 使用本地配置文件
# #         torch_dtype=torch.float16,
# #         safety_checker=None
# #     ).to(device)
# #     sd_pipeline.scheduler.set_timings(args.steps)
    
# #     generator = torch.Generator(device).manual_seed(1234)
# #     prompt = "A photo of a cat riding a skateboard"
# #     print(f"Using fixed prompt: '{prompt}'")
# #     print("-" * 50)

# #     print("Running a single inference to measure peak GPU memory...")
# #     torch.cuda.empty_cache()
# #     torch.cuda.reset_peak_memory_stats()
    
# #     _ = run_p2p_inference(sd_pipeline, prompt, num_inference_steps=args.steps, generator=generator)
    
# #     peak_memory_bytes = torch.cuda.max_memory_allocated()
# #     peak_memory_gb = peak_memory_bytes / (1024 ** 3)
# #     print(f"Peak GPU Memory Allocated: {peak_memory_gb:.2f} GB")
# #     torch.cuda.empty_cache()
# #     print("-" * 50)

# #     print(f"Performing {args.warmup_runs} warm-up runs...")
# #     for _ in tqdm(range(args.warmup_runs), desc="Warm-up"):
# #         _ = run_p2p_inference(sd_pipeline, prompt, num_inference_steps=args.steps, generator=generator)

# #     print(f"Performing {args.timed_runs} timed runs...")
# #     timings = []
# #     for _ in tqdm(range(args.timed_runs), desc="Benchmarking"):
# #         torch.cuda.synchronize()
# #         start_time = time.time()
        
# #         _ = run_p2p_inference(sd_pipeline, prompt, num_inference_steps=args.steps, generator=generator)
        
# #         torch.cuda.synchronize()
# #         end_time = time.time()
        
# #         elapsed = end_time - start_time
# #         timings.append(elapsed)

# #     timings_np = np.array(timings)
# #     print("-" * 50)
# #     print("Benchmark Results:")
# #     print(f"  Average Inference Time: {timings_np.mean():.4f} seconds")
# #     print(f"  Standard Deviation:   {timings_np.std():.4f} seconds")
# #     print(f"  Median Inference Time:  {np.median(timings_np):.4f} seconds")
# #     print("-" * 50)


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description="All-in-one, offline-ready benchmark script for Prompt-to-Prompt.")
    
# #     # [新] 增加config_path参数
# #     parser.add_argument(
# #         "--config_path",
# #         type=str,
# #         default="./v2-inference.yaml", # <-- 默认指向当前目录的配置文件
# #         help="Path to the local model config YAML file."
# #     )
# #     parser.add_argument(
# #         "--model_path",
# #         type=str,
# #         default="/home/apulis-dev/userdata/stablediffusion/models/v2-1_512-ema-pruned.ckpt", # <-- 您的模型路径
# #         help="Path to the Stable Diffusion model checkpoint file (.ckpt or .safetensors)."
# #     )
# #     parser.add_argument(
# #         "--steps",
# #         type=int,
# #         default=50,
# #         help="Number of inference steps."
# #     )
# #     parser.add_argument(
# #         "--warmup_runs",
# #         type=int,
# #         default=3,
# #         help="Number of warm-up runs before timing."
# #     )
# #     parser.add_argument(
# #         "--timed_runs",
# #         type=int,
# #         default=20,
# #         help="Number of timed runs to average."
# #     )

# #     args = parser.parse_args()
# #     main(args)

# import torch
# from diffusers import AutoencoderKL, UNet2DConditionModel
# from transformers import CLIPTextModel, CLIPTokenizer
# import argparse
# import time
# import numpy as np
# import abc
# from typing import Optional, Union, Tuple, List, Callable, Dict
# from tqdm import tqdm
# from PIL import Image
# import os

# # 关键：从您自己的项目中导入DDIMSampler
# # 确保 ddim.py 和此脚本在同一个目录下
# from ldm.models.diffusion.ddim import DDIMSampler

# from safetensors.torch import load_file as load_safetensors

# class AttentionControl(abc.ABC):
#     def step_callback(self, x_t): return x_t
#     def between_steps(self): return
#     @property
#     def num_uncond_att_layers(self): return 0
#     @abc.abstractmethod
#     def forward (self, attn, is_cross: bool, place_in_unet: str): raise NotImplementedError
#     def __call__(self, attn, is_cross: bool, place_in_unet: str):
#         if self.cur_att_layer >= self.num_uncond_att_layers:
#             h = attn.shape[0]
#             attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
#         self.cur_att_layer += 1
#         if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
#             self.cur_att_layer = 0
#             self.cur_step += 1
#             self.between_steps()
#         return attn
#     def reset(self): self.cur_step = 0; self.cur_att_layer = 0
#     def __init__(self): self.cur_step = 0; self.num_att_layers = -1; self.cur_att_layer = 0

# class AttentionStore(AttentionControl):
#     @staticmethod
#     def get_empty_store():
#         return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [],  "mid_self": [],  "up_self": []}
#     def forward(self, attn, is_cross: bool, place_in_unet: str):
#         key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
#         if attn.shape[1] <= 32 ** 2: self.step_store[key].append(attn)
#         return attn
#     def between_steps(self):
#         if len(self.attention_store) == 0: self.attention_store = self.step_store
#         else:
#             for key in self.attention_store:
#                 for i in range(len(self.attention_store[key])): self.attention_store[key][i] += self.step_store[key][i]
#         self.step_store = self.get_empty_store()
#     def reset(self):
#         super(AttentionStore, self).reset()
#         self.step_store = self.get_empty_store()
#         self.attention_store = {}
#     def __init__(self):
#         super(AttentionStore, self).__init__()
#         self.step_store = self.get_empty_store()
#         self.attention_store = {}

# # ===============================================================================================
# # == 代码块 2: register_attention_control 定义 (无变化)
# # ===============================================================================================
# def register_attention_control(model, controller):
#     def ca_forward(self, place_in_unet):
#         to_out = self.to_out[0] if type(self.to_out) is torch.nn.modules.container.ModuleList else self.to_out
#         def forward(x, context=None, mask=None):
#             h = self.heads
#             q = self.to_q(x)
#             context = context if context is not None else x
#             k = self.to_k(context)
#             v = self.to_v(context)
#             q, k, v = map(lambda t: t.view(t.shape[0], -1, h, t.shape[-1] // h).transpose(1, 2), (q, k, v))
#             sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
#             attn = sim.softmax(dim=-1)
#             attn = controller(attn, context is not None, place_in_unet)
#             out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
#             out = out.transpose(1, 2).reshape(x.shape[0], -1, h * (x.shape[-1] // h))
#             return to_out(out)
#         return forward

#     class DummyController:
#         def __call__(self, *args): return args[0]
#         def __init__(self): self.num_att_layers = 0

#     if controller is None: controller = DummyController()
#     def register_recr(net_, count, place_in_unet):
#         if net_.__class__.__name__ == 'CrossAttention':
#             net_.forward = ca_forward(net_, place_in_unet)
#             return count + 1
#         elif hasattr(net_, 'children'):
#             for net__ in net_.children(): count = register_recr(net__, count, place_in_unet)
#         return count
#     cross_att_count = 0
#     sub_nets = model.unet.named_children()
#     for net in sub_nets:
#         if "down" in net[0]: cross_att_count += register_recr(net[1], 0, "down")
#         elif "up" in net[0]: cross_att_count += register_recr(net[1], 0, "up")
#         elif "mid" in net[0]: cross_att_count += register_recr(net[1], 0, "mid")
#     controller.num_att_layers = cross_att_count
    
# # ===============================================================================================
# # == 代码块 3: 手动实现的推理逻辑 (适配您的DDIMSampler)
# # ===============================================================================================
# @torch.no_grad()
# def p2p_ddim_sampling(model, sampler, controller, prompts, steps, guidance_scale, generator, shape):
#     uncond_input = model.tokenizer([""], padding="max_length", max_length=model.tokenizer.model_max_length, return_tensors="pt")
#     uncond_embeddings = model.cond_stage_model(uncond_input.input_ids.to(model.device))[0]
#     text_input = model.tokenizer(prompts, padding="max_length", max_length=model.tokenizer.model_max_length, return_tensors="pt")
#     text_embeddings = model.cond_stage_model(text_input.input_ids.to(model.device))[0]
#     latents = torch.randn(shape, generator=generator, device=model.device, dtype=torch.float16)
#     sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0.0, verbose=False)
#     timesteps = sampler.ddim_timesteps
#     time_range = np.flip(timesteps)
#     total_steps = timesteps.shape[0]
#     iterator = tqdm(time_range, desc="DDIM Sampling", total=total_steps, leave=False)
#     for i, step in enumerate(iterator):
#         index = total_steps - i - 1
#         ts = torch.full((shape[0],), step, device=model.device, dtype=torch.long)
#         controller.reset()
#         register_attention_control(model.model.diffusion_model, controller)
#         latents, _ = sampler.p_sample_ddim(x=latents, c=text_embeddings, t=ts, index=index, 
#                                            unconditional_guidance_scale=guidance_scale, 
#                                            unconditional_conditioning=uncond_embeddings)
#     images = model.first_stage_model.decode(latents / 0.18215).sample
#     images = (images / 2 + 0.5).clamp(0, 1)
#     images = images.cpu().permute(0, 2, 3, 1).numpy()
#     images = (images * 255).astype(np.uint8)
#     return images

# def load_sd_model_from_local_project(project_dir, device, torch_dtype=torch.float16):
#     """
#     [关键修改] 这个函数现在会优先加载.safetensors文件
#     """
#     print(f"Loading model components from local project directory: {project_dir}")
    
#     # --- 加载各个组件 ---
#     # 1. Tokenizer (只有配置)
#     tokenizer = CLIPTokenizer.from_pretrained(os.path.join(project_dir, "tokenizer"))

#         # ---------- text_encoder ----------
#     text_encoder = CLIPTextModel._from_config(
#         CLIPTextConfig.from_pretrained(os.path.join(project_dir, "text_encoder"))
#     ).to(device, dtype=torch_dtype)
#     te_st = load_safetensors(os.path.join(project_dir, "text_encoder", "model.fp16.safetensors"))
#     text_encoder.load_state_dict(te_st, strict=False)

#     # ---------- vae ----------
#     vae = AutoencoderKL._from_config(
#         AutoencoderKL.config.from_pretrained(os.path.join(project_dir, "vae"))
#     ).to(device, dtype=torch_dtype)
#     vae_st = load_safetensors(os.path.join(project_dir, "vae", "diffusion_pytorch_model.fp16.safetensors"))
#     vae.load_state_dict(vae_st, strict=False)

#     # ---------- unet ----------
#     unet = UNet2DConditionModel._from_config(
#         UNet2DConditionModel.config.from_pretrained(os.path.join(project_dir, "unet"))
#     ).to(device, dtype=torch_dtype)
#     unet_st = load_safetensors(os.path.join(project_dir, "unet", "diffusion_pytorch_model.fp16.safetensors"))
#     unet.load_state_dict(unet_st, strict=False)
    
#     # # 2. Text Encoder
#     # text_encoder = CLIPTextModel.from_pretrained(os.path.join(project_dir, "text_encoder"), torch_dtype=torch_dtype).to(device)
    
#     # # 3. VAE
#     # vae = AutoencoderKL.from_pretrained(os.path.join(project_dir, "vae"), torch_dtype=torch_dtype).to(device)

#     # # 4. UNet
#     # unet = UNet2DConditionModel.from_pretrained(os.path.join(project_dir, "unet"), torch_dtype=torch_dtype).to(device)
    
#     # # --- 手动加载权重 ---
#     # # diffusers默认在每个子文件夹寻找 diffusion_pytorch_model.bin 或 model.safetensors
#     # # 我们将手动加载，使其更明确
    
#     # def load_weights(model, subfolder_name):
#     #     weight_path_safetensors = os.path.join(project_dir, subfolder_name, 'diffusion_pytorch_model.safetensors')
#     #     weight_path_bin = os.path.join(project_dir, subfolder_name, 'diffusion_pytorch_model.bin')
        
#     #     if os.path.exists(weight_path_safetensors):
#     #         print(f"Found and loading weights from: {weight_path_safetensors}")
#     #         state_dict = load_safetensors(weight_path_safetensors)
#     #         model.load_state_dict(state_dict)
#     #     elif os.path.exists(weight_path_bin):
#     #         print(f"Found and loading weights from: {weight_path_bin}")
#     #         # from_pretrained 已经处理了.bin的加载，我们无需额外操作
#     #         pass
#     #     else:
#     #         print(f"Warning: No weight file found for {subfolder_name}. Model might be randomly initialized.")

#     # load_weights(text_encoder, "text_encoder")
#     # load_weights(vae, "vae")
#     # load_weights(unet, "unet")
    
#     # --- 组装成FCDiffusion兼容的结构 ---
#     model = torch.nn.Module()
#     model.first_stage_model = vae
#     model.cond_stage_model = text_encoder
#     model.model = torch.nn.Module()
#     model.model.diffusion_model = unet
#     model.tokenizer = tokenizer
#     model.device = device
#     model.num_timesteps = 1000
#     model.betas = torch.linspace(0.00085**0.5, 0.0120**0.5, 1000, dtype=torch.float32) ** 2
#     model.alphas_cumprod = torch.cumprod(1. - model.betas, dim=0)
    
#     return model

# # ===============================================================================================
# # == 代码块 4: 基准测试主逻辑
# # ===============================================================================================
# def main(args):
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#     model_components = load_sd_model_from_local_project(args.model_dir, device, torch_dtype=torch.float16)
#     sampler = DDIMSampler(model_components, device=device)

#     generator = torch.Generator(device).manual_seed(1234)
#     prompt = "A photo of a cat riding a skateboard"
#     prompts_for_p2p = [prompt, prompt.replace("cat", "dog")]
    
#     shape = (1, model_components.model.diffusion_model.config.in_channels, 512 // 8, 512 // 8)
    
#     print(f"Using fixed prompt: '{prompt}' for editing.")
#     print("-" * 50)

#     # --- 显存测试 ---
#     print("Running a single inference to measure peak GPU memory...")
#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats()
    
#     controller = AttentionStore()
#     _ = p2p_ddim_sampling(model_components, sampler, controller, prompts_for_p2p, args.steps, 7.5, generator, shape)
    
#     peak_memory_bytes = torch.cuda.max_memory_allocated()
#     peak_memory_gb = peak_memory_bytes / (1024 ** 3)
#     print(f"Peak GPU Memory Allocated: {peak_memory_gb:.2f} GB")
#     torch.cuda.empty_cache()
#     print("-" * 50)

#     # --- 速度测试 ---
#     print(f"Performing {args.warmup_runs} warm-up runs...")
#     for _ in tqdm(range(args.warmup_runs), desc="Warm-up"):
#         _ = p2p_ddim_sampling(model_components, sampler, controller, prompts_for_p2p, args.steps, 7.5, generator, shape)

#     print(f"Performing {args.timed_runs} timed runs...")
#     timings = []
#     for _ in tqdm(range(args.timed_runs), desc="Benchmarking"):
#         torch.cuda.synchronize()
#         start_time = time.time()
        
#         _ = p2p_ddim_sampling(model_components, sampler, controller, prompts_for_p2p, args.steps, 7.5, generator, shape)
        
#         torch.cuda.synchronize()
#         end_time = time.time()
        
#         elapsed = end_time - start_time
#         timings.append(elapsed)

#     timings_np = np.array(timings)
#     print("-" * 50)
#     print("Benchmark Results:")
#     print(f"  Average Inference Time: {timings_np.mean():.4f} seconds")
#     print(f"  Standard Deviation:   {timings_np.std():.4f} seconds")
#     print(f"  Median Inference Time:  {np.median(timings_np):.4f} seconds")
#     print("-" * 50)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="All-in-one benchmark script for P2P using custom DDIMSampler.")
    
#     parser.add_argument(
#         "--model_dir", 
#         type=str, 
#         default = '/home/apulis-dev/userdata/stablediffusion/stable-diffusion-v1-5',
#         help="Path to the local Stable Diffusion v1.5 project directory containing vae, unet, etc. subfolders."
#     )
#     parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
#     parser.add_argument("--warmup_runs", type=int, default=3, help="Number of warm-up runs before timing.")
#     parser.add_argument("--timed_runs", type=int, default=20, help="Number of timed runs to average.")

#     args = parser.parse_args()
#     main(args)

# # # ===============================================================================================
# # # == 代码块 1: 从Notebook中提取的 Attention Control 定义 (无变化)
# # # ===============================================================================================
# # class AttentionControl(abc.ABC):
# #     def step_callback(self, x_t):
# #         return x_t
    
# #     def between_steps(self):
# #         return
    
# #     @property
# #     def num_uncond_att_layers(self):
# #         return 0
    
# #     @abc.abstractmethod
# #     def forward (self, attn, is_cross: bool, place_in_unet: str):
# #         raise NotImplementedError

# #     def __call__(self, attn, is_cross: bool, place_in_unet: str):
# #         if self.cur_att_layer >= self.num_uncond_att_layers:
# #             h = attn.shape[0]
# #             # In P2P, we apply control only to the conditional part
# #             attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
# #         self.cur_att_layer += 1
# #         if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
# #             self.cur_att_layer = 0
# #             self.cur_step += 1
# #             self.between_steps()
# #         return attn
    
# #     def reset(self):
# #         self.cur_step = 0
# #         self.cur_att_layer = 0

# #     def __init__(self):
# #         self.cur_step = 0
# #         self.num_att_layers = -1
# #         self.cur_att_layer = 0

# # class AttentionStore(AttentionControl):
# #     @staticmethod
# #     def get_empty_store():
# #         return {"down_cross": [], "mid_cross": [], "up_cross": [],
# #                 "down_self": [],  "mid_self": [],  "up_self": []}

# #     def forward(self, attn, is_cross: bool, place_in_unet: str):
# #         key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
# #         if attn.shape[1] <= 32 ** 2:
# #             self.step_store[key].append(attn)
# #         return attn

# #     def between_steps(self):
# #         if len(self.attention_store) == 0:
# #             self.attention_store = self.step_store
# #         else:
# #             for key in self.attention_store:
# #                 for i in range(len(self.attention_store[key])):
# #                     self.attention_store[key][i] += self.step_store[key][i]
# #         self.step_store = self.get_empty_store()

# #     def reset(self):
# #         super(AttentionStore, self).reset()
# #         self.step_store = self.get_empty_store()
# #         self.attention_store = {}

# #     def __init__(self):
# #         super(AttentionStore, self).__init__()
# #         self.step_store = self.get_empty_store()
# #         self.attention_store = {}
        
# # # ===============================================================================================
# # # == 代码块 2: register_attention_control 定义 (无变化)
# # # ===============================================================================================
# # def register_attention_control(model, controller):
# #     def ca_forward(self, place_in_unet):
# #         to_out = self.to_out
# #         if type(to_out) is torch.nn.modules.container.ModuleList:
# #             to_out = self.to_out[0]
# #         else:
# #             to_out = self.to_out

# #         def forward(x, context=None, mask=None):
# #             batch_size, sequence_length, dim = x.shape
# #             h = self.heads
# #             q = self.to_q(x)
# #             is_cross = context is not None
# #             context = context if is_cross else x
# #             k = self.to_k(context)
# #             v = self.to_v(context)
# #             q = self.reshape_heads_to_batch_dim(q)
# #             k = self.reshape_heads_to_batch_dim(k)
# #             v = self.reshape_heads_to_batch_dim(v)
# #             sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
# #             attn = sim.softmax(dim=-1)
# #             attn = controller(attn, is_cross, place_in_unet)
# #             out = torch.einsum("b i j, b j d -> b i d", attn, v)
# #             out = self.reshape_batch_dim_to_heads(out)
# #             return to_out(out)

# #         return forward

# #     class DummyController:
# #         def __call__(self, *args): return args[0]
# #         def __init__(self): self.num_att_layers = 0

# #     if controller is None: controller = DummyController()

# #     def register_recr(net_, count, place_in_unet):
# #         if net_.__class__.__name__ == 'CrossAttention':
# #             net_.forward = ca_forward(net_, place_in_unet)
# #             return count + 1
# #         elif hasattr(net_, 'children'):
# #             for net__ in net_.children():
# #                 count = register_recr(net__, count, place_in_unet)
# #         return count

# #     cross_att_count = 0
# #     # The UNet is now at model.unet, not model
# #     sub_nets = model.unet.named_children()
# #     for net in sub_nets:
# #         if "down" in net[0]:
# #             cross_att_count += register_recr(net[1], 0, "down")
# #         elif "up" in net[0]:
# #             cross_att_count += register_recr(net[1], 0, "up")
# #         elif "mid" in net[0]:
# #             cross_att_count += register_recr(net[1], 0, "mid")
# #     controller.num_att_layers = cross_att_count

# # # ===============================================================================================
# # # == 代码块 3: 手动实现的推理逻辑 (使用您的DDIMSampler)
# # # ===============================================================================================
# # @torch.no_grad()
# # def p2p_ddim_sampling(model, sampler, controller, prompts, steps, guidance_scale, generator, shape):
    
# #     # 1. 获取文本嵌入
# #     uncond_input = model.tokenizer([""], padding="max_length", max_length=model.tokenizer.model_max_length, return_tensors="pt")
# #     uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

# #     text_input = model.tokenizer(prompts, padding="max_length", max_length=model.tokenizer.model_max_length, return_tensors="pt")
# #     text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    
# #     # context for P2P: [uncond, source_cond, target_cond]
# #     context = torch.cat([uncond_embeddings, text_embeddings])
    
# #     # 2. 初始化 latents
# #     latents = torch.randn(shape, generator=generator, device=model.device, dtype=torch.float16)
    
# #     # 3. 设置 DDIM sampler
# #     sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0.0, verbose=False)
    
# #     # 4. 手动执行DDIM采样循环
# #     timesteps = sampler.ddim_timesteps
# #     time_range = np.flip(timesteps)
# #     total_steps = timesteps.shape[0]
    
# #     iterator = tqdm(time_range, desc="DDIM Sampling", total=total_steps, leave=False)
# #     for i, step in enumerate(iterator):
# #         index = total_steps - i - 1
# #         ts = torch.full((shape[0],), step, device=model.device, dtype=torch.long)
        
# #         # 扩展 latents 用于 classifier-free guidance
# #         # P2P a bit different, we have 2 prompts, so we need 3 inputs to unet
# #         # [latent_uncond, latent_source, latent_target]
# #         # For simplicity, we can pass a batch of 2 and handle CFG inside the loop
# #         latent_model_input = torch.cat([latents] * 2)
        
# #         # 预测噪声
# #         noise_pred = model.unet(latent_model_input, ts, encoder_hidden_states=context).sample
        
# #         # 执行 classifier-free guidance
# #         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
# #         noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
# #         # 应用Attention Control
# #         controller.cur_att_layer = 0
# #         noise_pred = controller(noise_pred, "cross", "")
        
# #         # DDIMSampler.p_sample_ddim needs specific arguments from your ddim.py
# #         latents, _ = sampler.p_sample_ddim(x=latents, c=text_embeddings, t=ts, index=index, 
# #                                            unconditional_guidance_scale=guidance_scale, 
# #                                            unconditional_conditioning=uncond_embeddings)

# #     # 5. 将 latents 解码为图像
# #     images = model.vae.decode(latents / 0.18215).sample
# #     images = (images / 2 + 0.5).clamp(0, 1)
# #     images = images.cpu().permute(0, 2, 3, 1).numpy()
# #     images = (images * 255).astype(np.uint8)
    
# #     return images


# # def load_sd_model_from_local_project(project_dir, device, torch_dtype=torch.float16):
# #     """
# #     从本地的、完整的Stable Diffusion v1.5项目文件夹中加载所有组件
# #     """
# #     print(f"Loading model components from local project directory: {project_dir}")
    
# #     # 从本地文件夹加载配置和模型
# #     vae = AutoencoderKL.from_pretrained(os.path.join(project_dir, "vae"), torch_dtype=torch_dtype).to(device)
# #     text_encoder = CLIPTextModel.from_pretrained(os.path.join(project_dir, "text_encoder"), torch_dtype=torch_dtype).to(device)
# #     unet = UNet2DConditionModel.from_pretrained(os.path.join(project_dir, "unet"), torch_dtype=torch_dtype).to(device)
# #     tokenizer = CLIPTokenizer.from_pretrained(os.path.join(project_dir, "tokenizer"))
    
# #     # 将所有组件打包成一个类似您的FCDiffusion模型的对象结构
# #     model = torch.nn.Module()
# #     model.first_stage_model = vae
# #     model.cond_stage_model = text_encoder
# #     # 您的DDIM Sampler似乎是直接把unet作为model.model传入的
# #     model.model = torch.nn.Module()
# #     model.model.diffusion_model = unet
# #     model.tokenizer = tokenizer
# #     model.device = device
    
# #     # 为DDIMSampler提供所需的参数
# #     model.num_timesteps = 1000
# #     model.betas = torch.linspace(0.00085**0.5, 0.0120**0.5, 1000, dtype=torch.float32) ** 2
# #     model.alphas_cumprod = torch.cumprod(1. - model.betas, dim=0)

# #     # 在您的代码中，DDIMSampler直接接收整个FCDiffusion模型
# #     # 我们这里创建一个简化的模型对象来模拟
# #     class SimpleModelWrapper:
# #         def __init__(self, unet, device):
# #             self.model = torch.nn.Module()
# #             self.model.diffusion_model = unet
# #             self.device = device
# #         def apply(self, *args, **kwargs):
# #             # DDIMSampler a.
# #             pass
    
# #     return model

# # # ===============================================================================================
# # # == 代码块 4: 基准测试主逻辑
# # # ===============================================================================================
# # def main(args):
# #     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# #     # [关键] 从本地项目文件夹加载模型
# #     model_components = load_sd_model_from_local_project(args.model_dir, device, torch_dtype=torch.float16)
    
# #     # [关键] 初始化您的DDIMSampler
# #     sampler = DDIMSampler(model_components, device=device)

# #     generator = torch.Generator(device).manual_seed(1234)
# #     prompt = "A photo of a cat riding a skateboard"
# #     prompts_for_p2p = [prompt, prompt.replace("cat", "dog")]
    
# #     shape = (1, model_components.model.diffusion_model.config.in_channels, 512 // 8, 512 // 8)
    
# #     print(f"Using fixed prompt: '{prompt}' for editing.")
# #     print("-" * 50)

# #     # --- 显存测试 ---
# #     print("Running a single inference to measure peak GPU memory...")
# #     torch.cuda.empty_cache()
# #     torch.cuda.reset_peak_memory_stats()
    
# #     controller = AttentionStore()
# #     register_attention_control(model_components.model.diffusion_model, controller)
# #     _ = p2p_ddim_sampling(model_components, sampler, controller, prompts_for_p2p, args.steps, 7.5, generator, shape)
    
# #     peak_memory_bytes = torch.cuda.max_memory_allocated()
# #     peak_memory_gb = peak_memory_bytes / (1024 ** 3)
# #     print(f"Peak GPU Memory Allocated: {peak_memory_gb:.2f} GB")
# #     torch.cuda.empty_cache()
# #     print("-" * 50)

# #     # --- 速度测试 ---
# #     print(f"Performing {args.warmup_runs} warm-up runs...")
# #     for _ in tqdm(range(args.warmup_runs), desc="Warm-up"):
# #         controller.reset()
# #         _ = p2p_ddim_sampling(model_components, sampler, controller, prompts_for_p2p, args.steps, 7.5, generator, shape)

# #     print(f"Performing {args.timed_runs} timed runs...")
# #     timings = []
# #     for _ in tqdm(range(args.timed_runs), desc="Benchmarking"):
# #         controller.reset()
# #         torch.cuda.synchronize()
# #         start_time = time.time()
        
# #         _ = p2p_ddim_sampling(model_components, sampler, controller, prompts_for_p2p, args.steps, 7.5, generator, shape)
        
# #         torch.cuda.synchronize()
# #         end_time = time.time()
        
# #         elapsed = end_time - start_time
# #         timings.append(elapsed)

# #     timings_np = np.array(timings)
# #     print("-" * 50)
# #     print("Benchmark Results:")
# #     print(f"  Average Inference Time: {timings_np.mean():.4f} seconds")
# #     print(f"  Standard Deviation:   {timings_np.std():.4f} seconds")
# #     print(f"  Median Inference Time:  {np.median(timings_np):.4f} seconds")
# #     print("-" * 50)

# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description="All-in-one benchmark script for P2P using custom DDIMSampler.")
    
# #     parser.add_argument(
# #         "--model_dir", 
# #         type=str, 
# #         default = '/home/apulis-dev/userdata/stablediffusion/stable-diffusion-v1-5',
# #         help="Path to the local Stable Diffusion v1.5 project directory containing vae, unet, etc. subfolders."
# #     )
# #     parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
# #     parser.add_argument("--warmup_runs", type=int, default=3, help="Number of warm-up runs before timing.")
# #     parser.add_argument("--timed_runs", type=int, default=20, help="Number of timed runs to average.")

# #     args = parser.parse_args()
# #     main(args)


import sys
import os
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import argparse
import time
import numpy as np
import abc
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
from PIL import Image
from safetensors.torch import load_file as load_safetensors

# ===============================================================================================
# # == [关键修复] 解决 ldm 库的相对路径导入问题
# # ===============================================================================================
# script_path = os.path.abspath(__file__)
# script_dir = os.path.dirname(script_path)
# parent_dir = os.path.dirname(script_dir)
# sys.path.insert(0, parent_dir)
# print(f"Added '{parent_dir}' to Python path to resolve LDM imports.")
# # ===============================================================================================

# [关键修改] 从您的标准 ldm 库中导入通用的 DDIMSampler
from ldm.models.diffusion.ddim import DDIMSampler

# ===============================================================================================
# == 代码块 1: Attention Control 定义 (无变化)
# ===============================================================================================
class AttentionControl(abc.ABC):
    def step_callback(self, x_t): return x_t
    def between_steps(self): return
    @property
    def num_uncond_att_layers(self): return 0
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str): raise NotImplementedError
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    def reset(self): self.cur_step = 0; self.cur_att_layer = 0
    def __init__(self): self.cur_step = 0; self.num_att_layers = -1; self.cur_att_layer = 0

class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [],  "mid_self": [],  "up_self": []}
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2: self.step_store[key].append(attn)
        return attn
    def between_steps(self):
        if len(self.attention_store) == 0: self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])): self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()
    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

# ===============================================================================================
# == 代码块 2: register_attention_control 定义 (无变化)
# ===============================================================================================
def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out[0] if type(self.to_out) is torch.nn.modules.container.ModuleList else self.to_out
        def forward(x, context=None, mask=None):
            h = self.heads
            q = self.to_q(x)
            context = context if context is not None else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: t.view(t.shape[0], -1, h, t.shape[-1] // h).transpose(1, 2), (q, k, v))
            sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
            attn = sim.softmax(dim=-1)
            attn = controller(attn, context is not None, place_in_unet)
            out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
            out = out.transpose(1, 2).reshape(x.shape[0], -1, h * (x.shape[-1] // h))
            return to_out(out)
        return forward

    class DummyController:
        def __call__(self, *args): return args[0]
        def __init__(self): self.num_att_layers = 0

    if controller is None: controller = DummyController()
    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children(): count = register_recr(net__, count, place_in_unet)
        return count
    cross_att_count = 0
    sub_nets = model.named_children() # Now model is the UNet
    for net in sub_nets:
        if "down" in net[0]: cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]: cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]: cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count
    
# ===============================================================================================
# == 代码块 3: 手动实现的推理逻辑 (使用标准 LDM DDIMSampler)
# ===============================================================================================
@torch.no_grad()
def p2p_ddim_sampling(model, sampler, controller, prompts, steps, guidance_scale, generator, shape):
    
    # 1. 获取文本嵌入
    uncond_input = model.tokenizer([""], padding="max_length", max_length=model.tokenizer.model_max_length, return_tensors="pt")
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    text_input = model.tokenizer(prompts, padding="max_length", max_length=model.tokenizer.model_max_length, return_tensors="pt")
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    
    # 2. 注册P2P的注意力控制器
    register_attention_control(model.unet, controller)

    # 3. 使用标准 LDM Sampler 的 sample 方法
    # 这个方法内部处理了采样循环
    samples, _ = sampler.sample(S=steps,
                                conditioning=text_embeddings,
                                batch_size=shape[0],
                                shape=shape[1:],
                                verbose=False,
                                unconditional_guidance_scale=guidance_scale,
                                unconditional_conditioning=uncond_embeddings,
                                eta=0.0,
                                x_T=torch.randn(shape, generator=generator, device=model.device, dtype=torch.float16))

    # 4. 将 latents 解码为图像
    images = model.vae.decode(samples / 0.18215).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).astype(np.uint8)
    
    return images




import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors

def load_sd_model_from_local_project(project_dir, device, torch_dtype=torch.float16):
    """
    [最终修正] 这个函数现在使用variant="fp16"和use_safetensors=True来正确加载fp16.safetensors文件，
    并创建一个完全兼容您的DDIMSampler的ModelWrapper。
    """
    print(f"Loading model components from local project directory: {project_dir}")
    
    # --- 从本地配置实例化并加载权重 ---
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(project_dir, "tokenizer"), local_files_only=True)
    
    text_encoder = CLIPTextModel.from_pretrained(
        os.path.join(project_dir, "text_encoder"), 
        torch_dtype=torch_dtype, 
        use_safetensors=True,
        variant="fp16",
        local_files_only=True
    ).to(device)
    
    vae = AutoencoderKL.from_pretrained(
        os.path.join(project_dir, "vae"), 
        torch_dtype=torch_dtype, 
        use_safetensors=True,
        variant="fp16",
        local_files_only=True
    ).to(device)

    unet = UNet2DConditionModel.from_pretrained(
        os.path.join(project_dir, "unet"), 
        torch_dtype=torch_dtype, 
        use_safetensors=True,
        variant="fp16",
        local_files_only=True
    ).to(device)
    
    # --- 组装成一个适配您的DDIM Sampler的结构 ---
    class ModelWrapper:
        def __init__(self):
            self.first_stage_model = vae
            self.cond_stage_model = text_encoder
            self.model = torch.nn.Module()
            self.model.diffusion_model = unet
            self.tokenizer = tokenizer
            self.device = device
            self.num_timesteps = 1000
            self.betas = torch.linspace(0.00085**0.5, 0.0120**0.5, 1000, dtype=torch.float32) ** 2
            self.alphas_cumprod = torch.cumprod(1. - self.betas, dim=0)
            
            # [关键修复] 添加DDIMSampler所需的 alphas_cumprod_prev 属性
            self.alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        def apply(self, *args, **kwargs):
            # This is needed because DDIMSampler calls model.apply(self.register_buffer)
            # which is a method of torch.nn.Module. We can make our wrapper inherit from it.
            pass

        # [关键修复] 添加DDIMSampler在采样时调用的 apply_model 方法
        def apply_model(self, x_noisy, t, cond, **kwargs):
            return self.model.diffusion_model(x_noisy, t, encoder_hidden_states=cond, **kwargs).sample
            
    wrapper = ModelWrapper()

    # 简单容器，用于p2p_ddim_sampling函数
    model_container = torch.nn.Module()
    model_container.vae = vae
    model_container.text_encoder = text_encoder
    model_container.unet = unet
    model_container.tokenizer = tokenizer
    model_container.device = device
    
    return model_container, wrapper



def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_components,sample_model_wrapper = load_sd_model_from_local_project(args.model_dir, device, torch_dtype=torch.float16)
    
    # [关键] 初始化标准的 LDM DDIMSampler
    sampler = DDIMSampler(sample_model_wrapper, device=device)

    generator = torch.Generator(device).manual_seed(1234)
    prompt = "A photo of a cat riding a skateboard"
    prompts_for_p2p = [prompt, prompt.replace("cat", "dog")]
    
    shape = (1, model_components.unet.config.in_channels, 512 // 8, 512 // 8)
    
    print(f"Using fixed prompt: '{prompt}' for editing.")
    print("-" * 50)

    # --- 显存测试 ---
    print("Running a single inference to measure peak GPU memory...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    controller = AttentionStore()
    _ = p2p_ddim_sampling(model_components, sampler, controller, prompts_for_p2p, args.steps, 7.5, generator, shape)
    
    peak_memory_bytes = torch.cuda.max_memory_allocated()
    peak_memory_gb = peak_memory_bytes / (1024 ** 3)
    print(f"Peak GPU Memory Allocated: {peak_memory_gb:.2f} GB")
    torch.cuda.empty_cache()
    print("-" * 50)

    # --- 速度测试 ---
    print(f"Performing {args.warmup_runs} warm-up runs...")
    for _ in tqdm(range(args.warmup_runs), desc="Warm-up"):
        _ = p2p_ddim_sampling(model_components, sampler, controller, prompts_for_p2p, args.steps, 7.5, generator, shape)

    print(f"Performing {args.timed_runs} timed runs...")
    timings = []
    for _ in tqdm(range(args.timed_runs), desc="Benchmarking"):
        torch.cuda.synchronize()
        start_time = time.time()
        
        _ = p2p_ddim_sampling(model_components, sampler, controller, prompts_for_p2p, args.steps, 7.5, generator, shape)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        elapsed = end_time - start_time
        timings.append(elapsed)

    timings_np = np.array(timings)
    print("-" * 50)
    print("Benchmark Results:")
    print(f"  Average Inference Time: {timings_np.mean():.4f} seconds")
    print(f"  Standard Deviation:   {timings_np.std():.4f} seconds")
    print(f"  Median Inference Time:  {np.median(timings_np):.4f} seconds")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="All-in-one benchmark script for P2P using a standard LDM DDIMSampler.")
    
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default= '/home/apulis-dev/userdata/stablediffusion/stable-diffusion-v1-5', 
        help="Path to the local Stable Diffusion v1.5 project directory containing vae, unet, etc. subfolders with .safetensors files."
    )
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--warmup_runs", type=int, default=3, help="Number of warm-up runs before timing.")
    parser.add_argument("--timed_runs", type=int, default=20, help="Number of timed runs to average.")

    args = parser.parse_args()
    main(args)