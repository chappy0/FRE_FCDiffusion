import argparse
import clip # 确保 clip 被导入
import concurrent.futures
import numpy as np
import os
import random
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import torch
import sys 
import time
import json
import logging
import ast
from PIL import Image
import torchvision.transforms as transforms
from contextlib import contextmanager, nullcontext

from tqdm import tqdm

# --- 假设的外部模块导入 (与上一版相同，请确保替换占位符) ---
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from search_ea2 import build_dataloader, load_model_from_config # 您的 DDIMSampler
# from ldm.models.diffusion.plms import PLMSSampler # 如果需要支持
# from ldm.models.diffusion.dpm_solver import DPMSolverSampler # 如果需要支持
# try:
#     from step_optim_z1 import StepOptim, NoiseScheduleVP 
# except ImportError:
#     logging.error("Failed to import StepOptim or NoiseScheduleVP.")
#     class NoiseScheduleVP: pass
#     class StepOptim: pass

# CLIP 和 DINO 的占位符 (与上一版相同)
if 'truncate_text' not in globals():
    def truncate_text(text, max_length=77):
        if isinstance(text, list): return [t[:max_length] for t in text]
        return text[:max_length]
_clip_device_placeholder = "cuda" if torch.cuda.is_available() else "cpu"
_clip_model_placeholder, _clip_preprocess_placeholder = None, None
try:
    if 'clip' in sys.modules:
        _clip_model_placeholder, _clip_preprocess_placeholder = clip.load("ViT-B/32", device=_clip_device_placeholder)
        if _clip_model_placeholder: _clip_model_placeholder.eval()
except Exception as e: logging.warning(f"Could not load CLIP model for placeholder: {e}")

def calculate_clip_similarity_batch(images_pil_list, texts_list):
    if _clip_model_placeholder is None: return [0.0] * len(texts_list if texts_list else [])
    similarities = []
    try:
        processed_images = torch.stack([_clip_preprocess_placeholder(img) for img in images_pil_list]).to(_clip_device_placeholder)
        text_tokens = clip.tokenize(truncate_text(texts_list, _clip_model_placeholder.context_length)).to(_clip_device_placeholder)
        with torch.no_grad():
            image_features = _clip_model_placeholder.encode_image(processed_images)
            text_features = _clip_model_placeholder.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity_scores = (image_features * text_features).sum(dim=-1)
            similarities = similarity_scores.cpu().tolist()
    except Exception as e:
        logging.error(f"Error in calculate_clip_similarity_batch placeholder: {e}")
        return [0.0] * len(texts_list)
    return similarities

if 'DinoVitExtractor' not in globals():
    class DinoVitExtractor:
        def __init__(self, model_name='dino_vits16', device=None):
            self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model, self.preprocess = None, None
            try:
                self.model = torch.hub.load('facebookresearch/dino:main', model_name, pretrained=True).to(self.device).eval()
                self.preprocess = transforms.Compose([
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224), transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
            except Exception as e: logging.warning(f"Placeholder DinoVitExtractor: Could not load model - {e}")
        def extract_features(self, pil_image):
            if not self.model: return torch.randn(1, 197, 384, device=self.device)
            if not isinstance(pil_image, Image.Image): pil_image = transforms.ToPILImage()(pil_image) if isinstance(pil_image, torch.Tensor) else pil_image
            img_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad(): return self.model.get_intermediate_layers(img_tensor, n=1)[0]

if 'compute_structure_similarity' not in globals():
    def compute_structure_similarity(images1_pil_list, images2_pil_list, dino_extractor):
        similarities = []
        for img1, img2 in zip(images1_pil_list, images2_pil_list):
            try:
                f1 = dino_extractor.extract_features(img1).mean(dim=1)
                f2 = dino_extractor.extract_features(img2).mean(dim=1)
                similarities.append(torch.cosine_similarity(f1, f2).item())
            except: similarities.append(0.0)
        return similarities

if 'TestDataset' not in globals():
    from torch.utils.data import Dataset
    class TestDataset(Dataset):
        def __init__(self, test_img_path, target_prompt, length=1, opt_H=512, opt_W=512):
            self.img_path = test_img_path; self.prompt = target_prompt; self.length = length
            self.opt_H=opt_H; self.opt_W=opt_W
            try: self.pil_image = Image.open(self.img_path).convert("RGB")
            except Exception as e:
                logging.error(f"TestDataset: Failed to load {self.img_path}: {e}")
                self.pil_image = Image.new("RGB", (self.opt_W, self.opt_H), "grey")
        def __len__(self): return self.length
        def __getitem__(self, idx):
            img_for_model = self.pil_image.resize((self.opt_W, self.opt_H))
            tensor_for_model = transforms.ToTensor()(img_for_model) * 2.0 - 1.0
            return {"jpg": tensor_for_model, "txt": self.prompt, "pil_image": self.pil_image}
# --- 占位符结束 ---

sys.setrecursionlimit(10000)
from sklearn.ensemble import RandomForestClassifier

class EvolutionSearcher(object):
    def __init__(self, opt, model, sampler: DDIMSampler, dataloader_info,
                 reference_schedule_indices: list, # 必须提供，且为索引列表
                 step_search_delta: int = 20):      # 每个索引的搜索偏移量
        
        self.opt = opt
        self.model = model.cpu() 
        self.sampler = sampler # 期望是 DDIMSampler 实例
        self.dataloader_info = dataloader_info
        self.eval_main_batch_size = opt.n_samples

        if not isinstance(reference_schedule_indices, list) or \
           not all(isinstance(x, int) for x in reference_schedule_indices):
            raise ValueError("reference_schedule_indices must be a list of integers.")
        
        self.reference_schedule_indices = sorted(list(set(reference_schedule_indices))) # 确保排序和唯一
        self.num_steps_to_optimize = len(self.reference_schedule_indices) # 搜索的步数固定为参考调度的步数
        
        logging.info(f"EvolutionSearcher initialized to fine-tune a schedule of {self.num_steps_to_optimize} steps.")
        logging.info(f"Reference schedule (first 10): {self.reference_schedule_indices[:10]}")

        self.step_search_delta = int(step_search_delta)
        logging.info(f"Search delta for each step index: +/- {self.step_search_delta}")

        self.max_epochs = opt.max_epochs
        self.select_num = opt.select_num        
        self.population_num = opt.population_num 
        self.m_prob = opt.m_prob                
        self.crossover_num = opt.crossover_num  
        self.mutation_num = opt.mutation_num    
        self.num_evaluation_samples = opt.num_sample 

        self.keep_top_k = {k: [] for k in [self.select_num, 50, self.population_num]}
        self.epoch = 0
        self.candidates_str_set = set() 
        self.vis_dict = {}   
        
        self.stop_counter = 0
        self.best_history = [] 
        self.early_stop_thresh = getattr(opt, 'early_stop_thresh', 0.0005) # 更小的阈值用于精细搜索
        self.early_stop_patience = getattr(opt, 'early_stop_patience', 10) # 可能需要更多耐心
        
        self.eval_workers = getattr(opt, 'eval_workers', 1) 
        self.eval_batch_size_cand = getattr(opt, 'eval_batch_size_cand', 4)

        if hasattr(self.sampler, 'ddpm_num_timesteps'):
            self.total_ddpm_timesteps = self.sampler.ddpm_num_timesteps
        else:
            logging.warning("DDIMSampler instance does not have ddpm_num_timesteps. Defaulting to 1000.")
            self.total_ddpm_timesteps = 1000
        
        self.min_valid_index = 0
        self.max_valid_index = self.total_ddpm_timesteps - 1

        # 评估指标模型加载 (与之前版本类似)
        self.eval_metrics_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dino_extractor = DinoVitExtractor(device=self.eval_metrics_device)
        self.clip_model, self.clip_preprocess = None, None
        if self.opt.use_clip:
            try:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.eval_metrics_device)
                if self.clip_model: self.clip_model.eval()
            except Exception as e:
                logging.error(f"Failed to load CLIP model: {e}")


    def _normalize_cand_str(self, cand_list): # 输入必须是列表
        return json.dumps(sorted(list(set(map(int, cand_list)))))

    def _ensure_candidate_validity_and_bounds(self, cand_list: list, ref_schedule_for_bounds: list) -> list:
        """确保候选列表有效（排序，唯一，长度，边界，单调性递减对于时间值索引）"""
        if len(cand_list) != self.num_steps_to_optimize:
            # 这是一个严重问题，不应该发生，因为我们固定了长度
            logging.warning(f"Candidate length {len(cand_list)} differs from target {self.num_steps_to_optimize}. This should not happen in fine-tuning mode.")
            # 尝试修复：如果太长则截断，太短则从参考调度补充（但这很复杂且可能引入偏差）
            # 最好的策略是确保生成和操作时长度不变。这里简单截断或报错。
            if len(cand_list) > self.num_steps_to_optimize:
                cand_list = cand_list[:self.num_steps_to_optimize]
            else: # 太短，放弃这个候选可能更好
                return [] # 返回空列表表示无效


        new_cand = []
        for i in range(self.num_steps_to_optimize):
            val = cand_list[i]
            # 基于参考调度对应位置的值来限定范围
            ref_val_for_bound = ref_schedule_for_bounds[i] # 使用传入的参考点
            
            lower_b = max(self.min_valid_index, ref_val_for_bound - self.step_search_delta)
            upper_b = min(self.max_valid_index, ref_val_for_bound + self.step_search_delta)
            
            # 如果父代的值已经超出了基于参考的范围，则以父代的值为中心小范围浮动
            # (或者严格限制在参考调度定义的范围内)
            # 为了简单且允许一些探索，我们让范围基于 ref_val_for_bound
            # 但修复时，如果 val 已经跑太远，可以拉回或以 val 为中心再约束
            
            current_val_lower_b = max(self.min_valid_index, val - self.step_search_delta // 2) # 更小的自身偏移
            current_val_upper_b = min(self.max_valid_index, val + self.step_search_delta // 2)

            final_lower_b = max(lower_b, current_val_lower_b)
            final_upper_b = min(upper_b, current_val_upper_b)
            
            # 确保下界不大于上界
            if final_lower_b > final_upper_b:
                final_lower_b, final_upper_b = min(lower_b, current_val_lower_b), max(upper_b, current_val_upper_b) # 放宽一点
                if final_lower_b > final_upper_b: # 仍然有问题，使用更宽的原始范围
                    final_lower_b = lower_b
                    final_upper_b = upper_b


            val_clipped = max(final_lower_b, min(val, final_upper_b))
            new_cand.append(int(round(val_clipped))) # 四舍五入到整数索引

        # 排序、去重，并处理长度问题（由于去重或极端裁剪）
        new_cand = sorted(list(set(new_cand)), reverse=False) # DDIMSampler 内部会 flip, 所以这里用升序索引

        # 如果去重后长度变了，需要修复到 self.num_steps_to_optimize
        # 这是一个复杂的问题，因为添加或删除点会改变其他点的相对意义
        # 简单策略：如果长度不符，认为此候选无效（在精细搜索中应尽量避免长度变化）
        if len(new_cand) != self.num_steps_to_optimize:
            logging.debug(f"Candidate length changed after validation/clipping from {self.num_steps_to_optimize} to {len(new_cand)}. Discarding. Original: {cand_list}, New: {new_cand}")
            return [] # 返回空列表表示无效

        # 确保单调性 (对于索引，应该是严格单调递增)
        for i in range(len(new_cand) - 1):
            if new_cand[i] >= new_cand[i+1]:
                logging.debug(f"Candidate failed monotonicity check after validation: {new_cand}. Discarding.")
                return [] # 返回空列表表示无效
        
        return new_cand


    def get_initial_population(self, population_size):
        logging.info(f'Generating initial population of {population_size} around reference schedule...')
        initial_population_str_list = []
        
        # 1. 添加精确的参考调度
        ref_str_norm = self._normalize_cand_str(self.reference_schedule_indices)
        if self.is_legal(ref_str_norm):
            initial_population_str_list.append(ref_str_norm)
            self.candidates_str_set.add(ref_str_norm)

        # 2. 生成参考调度的轻微变体
        attempts = 0
        max_attempts_factor = 20 * population_size
        
        while len(initial_population_str_list) < population_size and attempts < max_attempts_factor:
            attempts += 1
            
            # 对参考调度进行一次“轻微”的、全基因的随机扰动
            mutant_from_ref = list(self.reference_schedule_indices)
            for i in range(self.num_steps_to_optimize):
                ref_val = self.reference_schedule_indices[i]
                # 在参考值的小邻域内随机选择，而不是当前值的小邻域
                # 避免一开始就漂移太远
                delta = random.randint(-self.step_search_delta // 2, self.step_search_delta // 2) # 更小的初始扰动
                new_val = ref_val + delta
                
                # 确保在全局有效索引范围内
                new_val = max(self.min_valid_index, min(new_val, self.max_valid_index))
                mutant_from_ref[i] = new_val
            
            # 清理和验证这个新生成的候选
            # 使用 self.reference_schedule_indices 作为边界检查的参考
            processed_mutant_list = self._ensure_candidate_validity_and_bounds(mutant_from_ref, self.reference_schedule_indices)

            if not processed_mutant_list: continue # 无效候选

            cand_str_norm = self._normalize_cand_str(processed_mutant_list)
            if cand_str_norm not in self.candidates_str_set:
                if self.is_legal(cand_str_norm):
                    initial_population_str_list.append(cand_str_norm)
                    self.candidates_str_set.add(cand_str_norm)
        
        # 如果数量仍然不足，可以用更随机的方式填充（但仍基于范围）
        # (此部分可以省略，如果上面的扰动足够产生多样性)

        logging.info(f'Generated {len(initial_population_str_list)} initial candidates.')
        return initial_population_str_list


    def get_mutation_op(self, k_top_selection_key, num_to_mutate, mutation_probability):
        if not self.keep_top_k[k_top_selection_key]: return []
        logging.info(f'Mutation: attempting {num_to_mutate} mutants...')
        mutated_offspring_str_list = []
        attempts = 0; max_attempts_factor = 20

        while len(mutated_offspring_str_list) < num_to_mutate and attempts < num_to_mutate * max_attempts_factor:
            attempts += 1
            parent_cand_str = random.choice(self.keep_top_k[k_top_selection_key])
            try: parent_list = ast.literal_eval(parent_cand_str)
            except: logging.error(f"Mutation: Failed to eval parent: {parent_cand_str}"); continue

            if len(parent_list) != self.num_steps_to_optimize: # 长度必须固定
                logging.warning(f"Mutation: Parent {parent_cand_str} has incorrect length {len(parent_list)}. Skipping.")
                continue

            mutant_list = list(parent_list)
            num_actual_mutations = 0

            for i in range(self.num_steps_to_optimize):
                if np.random.random_sample() < mutation_probability:
                    original_val = mutant_list[i]
                    ref_val_for_bound = self.reference_schedule_indices[i] # 基于原始StepOptim结果定义范围中心

                    search_min = max(self.min_valid_index, ref_val_for_bound - self.step_search_delta)
                    search_max = min(self.max_valid_index, ref_val_for_bound + self.step_search_delta)
                    
                    # 从search_min到search_max随机选择一个新值
                    if search_min >= search_max: # 如果范围太小
                        new_val = ref_val_for_bound # 或 original_val
                    else:
                        new_val = random.randint(search_min, search_max)
                    
                    mutant_list[i] = new_val
                    if new_val != original_val: num_actual_mutations +=1
            
            if num_actual_mutations == 0 and self.num_steps_to_optimize > 0 : # 至少强制一个点变异
                idx_to_force = random.randrange(self.num_steps_to_optimize)
                original_val = mutant_list[idx_to_force]
                ref_val_for_bound = self.reference_schedule_indices[idx_to_force]
                search_min = max(self.min_valid_index, ref_val_for_bound - self.step_search_delta)
                search_max = min(self.max_valid_index, ref_val_for_bound + self.step_search_delta)
                if search_min < search_max: # 确保有范围可选
                    mutant_list[idx_to_force] = random.randint(search_min, search_max)
                elif search_min == search_max:
                     mutant_list[idx_to_force] = search_min


            # 清理和验证，使用 self.reference_schedule_indices 作为边界的“锚点”
            processed_mutant_list = self._ensure_candidate_validity_and_bounds(mutant_list, self.reference_schedule_indices)
            if not processed_mutant_list: continue

            mutant_str_norm = self._normalize_cand_str(processed_mutant_list)
            if mutant_str_norm not in self.candidates_str_set:
                 if self.is_legal(mutant_str_norm):
                    mutated_offspring_str_list.append(mutant_str_norm)
        
        logging.info('Generated {} actual new mutants.'.format(len(mutated_offspring_str_list)))
        return mutated_offspring_str_list

    def get_crossover_op(self, k_top_selection_key, num_to_crossover):
        if not self.keep_top_k[k_top_selection_key] or len(self.keep_top_k[k_top_selection_key]) < 2: return []
        logging.info(f'Crossover: attempting {num_to_crossover} offspring...')
        crossed_offspring_str_list = []
        attempts = 0; max_attempts_factor = 20

        while len(crossed_offspring_str_list) < num_to_crossover and attempts < num_to_crossover * max_attempts_factor:
            attempts += 1
            parent1_str, parent2_str = random.sample(self.keep_top_k[k_top_selection_key], 2)
            try:
                p1_list = ast.literal_eval(parent1_str)
                p2_list = ast.literal_eval(parent2_str)
            except: continue

            if len(p1_list) != self.num_steps_to_optimize or len(p2_list) != self.num_steps_to_optimize:
                logging.warning("Crossover: Parents have incorrect length. Skipping.")
                continue

            # 单点交叉
            child1_list, child2_list = list(p1_list), list(p2_list) # 副本
            if self.num_steps_to_optimize > 1:
                cx_point = random.randint(1, self.num_steps_to_optimize - 1)
                child1_list = p1_list[:cx_point] + p2_list[cx_point:]
                child2_list = p2_list[:cx_point] + p1_list[cx_point:]
            
            children_to_process = [child1_list]
            if child1_list != child2_list : children_to_process.append(child2_list)


            for child_l in children_to_process:
                if len(crossed_offspring_str_list) >= num_to_crossover: break
                # 清理和验证，使用 self.reference_schedule_indices 作为边界的“锚点”
                processed_child_list = self._ensure_candidate_validity_and_bounds(child_l, self.reference_schedule_indices)
                if not processed_child_list: continue
                
                child_str_norm = self._normalize_cand_str(processed_child_list)
                if child_str_norm not in self.candidates_str_set:
                    if self.is_legal(child_str_norm):
                        crossed_offspring_str_list.append(child_str_norm)
        
        logging.info('Generated {} actual new crossover offspring.'.format(len(crossed_offspring_str_list)))
        return crossed_offspring_str_list

    # get_cand_similarity 和 parallel_evaluate 方法与上一版类似，但需要适配这里的逻辑
    # 特别是 get_cand_similarity 中使用 self.num_steps_to_optimize
    # parallel_evaluate 应该在 self.candidates_str_set 更新后调用

    def get_cand_similarity(self, cand=None, opt_config=None, device_str='cuda'):
        # (与上一版基本一致，但确保 S=self.num_steps_to_optimize)
        t_start_eval = time.time()
        if cand is None: logging.error("get_cand_similarity: None candidate."); self.model.cpu(); return 0.0

        cand_eval = sorted(list(set(map(int, cand)))) 
        if len(cand_eval) != self.num_steps_to_optimize: # 严格长度检查
            logging.warning(f"Invalid candidate length for similarity: {len(cand_eval)} vs {self.num_steps_to_optimize}.")
            self.model.cpu(); return 0.0 

        eval_device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.model.to(eval_device)

        timesteps_to_save_for_ddim_file = np.array(cand_eval, dtype=np.int64)[::-1]
        sampler_timesteps_file_path = self.sampler.timesteps_file
        try:
            np.savetxt(sampler_timesteps_file_path, timesteps_to_save_for_ddim_file, fmt='%d')
            # sampler.sample 内部的 make_schedule 会从文件加载并重新计算所有参数
        except Exception as e:
            logging.error(f"Error preparing custom timesteps file: {e}"); self.model.cpu(); return 0.0

        # ... (图像生成和评估逻辑，与上一版 get_cand_similarity 相同，但 S 固定为 self.num_steps_to_optimize)
        dino_sim_list_epoch, clip_sim_list_epoch, num_generated_total = [], [], 0
        val_loader = self.dataloader_info.get('validation_loader')
        if val_loader is None: logging.error("Validation loader missing."); self.model.cpu(); return 0.0
        num_eval_batches_needed = (self.num_evaluation_samples + self.eval_main_batch_size - 1) // self.eval_main_batch_size
        
        cfg_scale = getattr(opt_config, 'scale', 7.5)
        ddim_eta_val = getattr(opt_config, 'ddim_eta', 0.0)
        latent_C = getattr(opt_config, 'C', 4)
        latent_H_base = getattr(opt_config, 'H', 512); latent_W_base = getattr(opt_config, 'W', 512)
        downsample_f = getattr(opt_config, 'f', 8)
        shape = [latent_C, latent_H_base // downsample_f, latent_W_base // downsample_f]

        with torch.no_grad():
            autocast_context = torch.autocast(device_type=eval_device.type, enabled=(opt_config.precision == "autocast" and eval_device.type == 'cuda'))
            with autocast_context:
                ema_scope_context = self.model.ema_scope if hasattr(self.model, 'ema_scope') and callable(self.model.ema_scope) else nullcontext
                with ema_scope_context():
                    for batch_idx, batch_content in enumerate(val_loader):
                        if num_generated_total >= self.num_evaluation_samples: break
                        prompts_in_batch = batch_content.get('txt'); original_pil_for_dino = batch_content.get('pil_image')
                        if not prompts_in_batch: continue
                        current_process_batch_size = len(prompts_in_batch)
                        if current_process_batch_size == 0: continue
                        
                        actual_prompts = prompts_in_batch[:current_process_batch_size]
                        actual_original_pil = original_pil_for_dino[:current_process_batch_size] if original_pil_for_dino else []
                        
                        uc = self.model.get_learned_conditioning(current_process_batch_size * [""]) if cfg_scale != 1.0 else None
                        c = self.model.get_learned_conditioning(actual_prompts)
                        
                        samples_latent, _ = self.sampler.sample(
                            S=self.num_steps_to_optimize, # 使用固定的步数
                            conditioning=c, batch_size=current_process_batch_size, shape=shape, verbose=False,
                            unconditional_guidance_scale=cfg_scale, unconditional_conditioning=uc, eta=ddim_eta_val)
                        
                        gen_images_tensors_01 = torch.clamp((self.model.decode_first_stage(samples_latent) + 1.0) / 2.0, 0.0, 1.0)
                        generated_pil_list = [transforms.ToPILImage()(img.cpu()) for img in gen_images_tensors_01]
                        num_generated_total += len(generated_pil_list)

                        if opt_config.use_clip and self.clip_model:
                            clip_sim_list_epoch.extend(calculate_clip_similarity_batch(generated_pil_list, actual_prompts))
                        if opt_config.use_dino and self.dino_extractor and self.dino_extractor.model and actual_original_pil:
                             if len(actual_original_pil) == len(generated_pil_list):
                                dino_sim_list_epoch.extend(compute_structure_similarity(generated_pil_list, actual_original_pil, self.dino_extractor))
                        if batch_idx >= num_eval_batches_needed - 1: break
        
        final_combined_similarity = 0.0
        avg_clip = np.mean(clip_sim_list_epoch[:self.num_evaluation_samples]) if opt_config.use_clip and clip_sim_list_epoch else 0.0
        avg_dino = np.mean(dino_sim_list_epoch[:self.num_evaluation_samples]) if opt_config.use_dino and dino_sim_list_epoch else 0.0
        
        cand_str_norm_key = self._normalize_cand_str(cand_eval)
        if cand_str_norm_key not in self.vis_dict: self.vis_dict[cand_str_norm_key] = {}
        if opt_config.use_clip: self.vis_dict[cand_str_norm_key]['clip_similarity'] = float(avg_clip)
        if opt_config.use_dino: self.vis_dict[cand_str_norm_key]['dino_similarity'] = float(avg_dino)

        if opt_config.use_clip and opt_config.use_dino and clip_sim_list_epoch and dino_sim_list_epoch:
            final_combined_similarity = (avg_clip + avg_dino) / 2.0
        elif opt_config.use_clip and clip_sim_list_epoch: final_combined_similarity = avg_clip
        elif opt_config.use_dino and dino_sim_list_epoch: final_combined_similarity = avg_dino
        
        self.vis_dict[cand_str_norm_key]['similarity'] = float(final_combined_similarity)
        self.vis_dict[cand_str_norm_key]['visited_eval'] = True

        self.model.cpu(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logging.info(f"Cand {cand_str_norm_key} (len {len(cand_eval)}), CLIP: {avg_clip:.4f}, DINO: {avg_dino:.4f}, Combined: {final_combined_similarity:.4f}. Eval time: {time.time()-t_start_eval:.2f}s")
        return float(final_combined_similarity)


    def parallel_evaluate(self, candidates_str_list_to_process):
        # (与上一版基本一致, 但确保 is_legal 和 vis_dict 键一致性)
        unique_candidates_to_eval_str = []
        for cand_str_raw in candidates_str_list_to_process:
            try:
                # 确保用于检查和添加到评估列表的键是规范化的
                cand_list = ast.literal_eval(cand_str_raw)
                normalized_cand_str = self._normalize_cand_str(cand_list)
            except: # 如果原始字符串无法解析，则它本身就是键（可能已损坏）
                normalized_cand_str = cand_str_raw 

            if normalized_cand_str not in self.vis_dict or not self.vis_dict[normalized_cand_str].get('visited_eval', False):
                if normalized_cand_str not in unique_candidates_to_eval_str:
                     unique_candidates_to_eval_str.append(normalized_cand_str) # 使用规范化字符串进行评估
        
        if not unique_candidates_to_eval_str:
            logging.info("All candidates in current batch already evaluated with scores.")
            return

        logging.info(f"Parallel evaluating {len(unique_candidates_to_eval_str)} unique candidates...")
        results_map = {}
        # ... (与上一版相同的并行或串行评估逻辑，调用 self.get_cand_similarity) ...
        # ... 确保传递给 get_cand_similarity 的是列表，而不是字符串 ...
        # Fallback to sequential for simplicity here, as parallel ProcessPool with PyTorch models is complex
        sequential_eval = True # or based on self.eval_workers <= 1 or some other condition
        if self.eval_workers > 1:
             logging.warning("ProcessPoolExecutor for DL model evaluation can be tricky. Running sequentially for stability in this example. For true parallelism, ensure models are correctly handled in worker processes.")
        
        # Sequential evaluation loop
        for cand_s_to_eval_norm in tqdm(unique_candidates_to_eval_str, desc="Evaluating Candidates"):
            try:
                # get_cand_similarity 期望的是一个list of int, 不是它的字符串表示
                cand_list_for_eval = json.loads(cand_s_to_eval_norm) # 从规范化JSON字符串转回列表
                similarity = self.get_cand_similarity(cand=cand_list_for_eval, opt_config=self.opt)
                results_map[cand_s_to_eval_norm] = similarity # 键是规范化字符串
            except Exception as e:
                logging.error(f"Error evaluating candidate {cand_s_to_eval_norm}: {e}")
                results_map[cand_s_to_eval_norm] = 0.0
        
        for cand_s_norm, sim in results_map.items(): # 键是规范化字符串
            if cand_s_norm not in self.vis_dict: self.vis_dict[cand_s_norm] = {}
            self.vis_dict[cand_s_norm]['similarity'] = sim
            self.vis_dict[cand_s_norm]['visited_eval'] = True
            logging.info('Evaluated cand: {}, similarity: {:.4f}'.format(cand_s_norm, sim))


    def update_top_k(self, current_population_cand_strs, *, k, key_func_cand_str_to_score, reverse=True):
        # (与上一版基本一致，确保使用规范化的键从 vis_dict 获取分数)
        combined_pool_str_norm = list(set(
            [self._normalize_cand_str(ast.literal_eval(cs)) for cs in current_population_cand_strs] +
            [self._normalize_cand_str(ast.literal_eval(cs_top)) for cs_top in self.keep_top_k[k]]
        ))

        valid_candidates_with_sim = []
        for cand_s_norm in combined_pool_str_norm:
            info = self.vis_dict.get(cand_s_norm)
            if info and 'similarity' in info and info.get('visited_eval', False):
                valid_candidates_with_sim.append((cand_s_norm, info['similarity']))
        
        valid_candidates_with_sim.sort(key=lambda x: x[1], reverse=reverse)
        self.keep_top_k[k] = [item[0] for item in valid_candidates_with_sim[:k]]


    def search(self):
        logging.info(f"Starting fine-tuning search for {self.num_steps_to_optimize}-step schedule.")
        logging.info(f"Population: {self.population_num}, Elites: {self.select_num}, Mutation: {self.mutation_num}, Crossover: {self.crossover_num}, Epochs: {self.max_epochs}")
        if not self.reference_schedule_indices:
            logging.error("Reference schedule from StepOptim is required for fine-tuning search. Aborting.")
            return

        self.candidates_str_set = set() 
        current_population_str_list = self.get_initial_population(self.population_num)
        self.candidates_str_set.update(current_population_str_list) # Add initial pop to global set

        if not current_population_str_list:
            logging.error("Initial population for fine-tuning is empty. Aborting."); return

        logging.info(f"Evaluating initial population of {len(current_population_str_list)} fine-tuning candidates...")
        self.parallel_evaluate(current_population_str_list)

        key_for_sort = lambda cand_s_norm: self.vis_dict.get(cand_s_norm, {}).get('similarity', -float('inf'))
        
        self.update_top_k(current_population_str_list, k=self.population_num, key_func_cand_str_to_score=key_for_sort)
        self.update_top_k(current_population_str_list, k=self.select_num, key_func_cand_str_to_score=key_for_sort)
        self.update_top_k(current_population_str_list, k=50, key_func_cand_str_to_score=key_for_sort)

        for epoch_num in range(1, self.max_epochs + 1):
            self.epoch = epoch_num
            logging.info(f"\n>>>>>> Fine-tuning Epoch {self.epoch}/{self.max_epochs} <<<<<<")

            parents_for_reproduction = self.keep_top_k[self.select_num]
            if not parents_for_reproduction:
                logging.warning(f"Epoch {self.epoch}: No parents for reproduction. Re-initializing population partially.")
                # Re-initialize a portion, keeping some of the current best from population_num
                keep_best_n = self.population_num // 3
                self.update_top_k(self.keep_top_k[self.population_num], k=keep_best_n, key_func_cand_str_to_score=key_for_sort)
                next_gen_candidates_str_list = list(self.keep_top_k[keep_best_n])
                self.candidates_str_set = set(next_gen_candidates_str_list) # Reset global set for this gen
                next_gen_candidates_str_list.extend(self.get_initial_population(self.population_num - keep_best_n)) # Add new perturbed from ref
            else:
                next_gen_candidates_str_list = list(parents_for_reproduction) # Elitism
                self.candidates_str_set = set(next_gen_candidates_str_list) # Start with elites

                mutants_str = self.get_mutation_op(self.select_num, self.mutation_num, self.m_prob)
                for m_s in mutants_str: 
                    if m_s not in self.candidates_str_set: next_gen_candidates_str_list.append(m_s); self.candidates_str_set.add(m_s)
                
                crossed_offspring_str = self.get_crossover_op(self.select_num, self.crossover_num)
                for c_s in crossed_offspring_str:
                    if c_s not in self.candidates_str_set: next_gen_candidates_str_list.append(c_s); self.candidates_str_set.add(c_s)
                
                num_random_needed = self.population_num - len(self.candidates_str_set)
                if num_random_needed > 0:
                    random_new_str = self.get_initial_population(num_random_needed) # Use initial pop logic for diversity
                    for r_s in random_new_str:
                        if r_s not in self.candidates_str_set: next_gen_candidates_str_list.append(r_s); self.candidates_str_set.add(r_s)
            
            current_population_str_list = list(self.candidates_str_set) # Final unique list for this epoch
            if len(current_population_str_list) > self.population_num: # Trim if overpopulated
                current_population_str_list.sort(key=key_for_sort, reverse=True)
                current_population_str_list = current_population_str_list[:self.population_num]
                self.candidates_str_set = set(current_population_str_list)


            logging.info(f"Epoch {self.epoch}: Evaluating population of {len(current_population_str_list)} candidates...")
            self.parallel_evaluate(current_population_str_list)

            self.update_top_k(current_population_str_list, k=self.population_num, key_func_cand_str_to_score=key_for_sort)
            self.update_top_k(current_population_str_list, k=self.select_num, key_func_cand_str_to_score=key_for_sort)
            self.update_top_k(current_population_str_list, k=50, key_func_cand_str_to_score=key_for_sort)

            if self.keep_top_k[self.select_num]:
                best_cand_str = self.keep_top_k[self.select_num][0]
                current_best_sim = self.vis_dict.get(best_cand_str, {}).get('similarity', -float('inf'))
                logging.info(f"Epoch {self.epoch}: Current best similarity = {current_best_sim:.4f} for {best_cand_str}")
                if self.check_early_stop(current_best_sim):
                    logging.info(f"Early stopping triggered at epoch {self.epoch}."); break
            else:
                logging.warning(f"Epoch {self.epoch}: Top selection is empty!"); break

            # Logging and Saving
            if self.epoch % 1 == 0 or self.epoch == self.max_epochs : # Log every epoch, save every 5 or last
                logging.info(f"--- Epoch {self.epoch} Top Results (up to 50) ---")
                for i, cand_s_log in enumerate(self.keep_top_k[50]):
                    if i >= 10 and self.epoch % 5 != 0:
                        if i == 10: logging.info("    (... remaining top 50 results hidden, full log every 5th epoch ...)")
                        break
                    metrics = self.vis_dict.get(cand_s_log, {}); sim_val = metrics.get('similarity', 0.0)
                    clip_val = metrics.get('clip_similarity', 0.0); dino_val = metrics.get('dino_similarity', 0.0)
                    try: cand_list_log = json.loads(cand_s_log); len_log = len(cand_list_log)
                    except: cand_list_log = cand_s_log; len_log = "N/A"
                    logging.info(f"  Rank {i+1:2d}: Score={sim_val:.4f} (C:{clip_val:.4f} D:{dino_val:.4f}) Len:{len_log} Steps: {str(cand_list_log)[:80]}{'...' if len(str(cand_list_log))>80 else ''}")

            if self.epoch % 5 == 0 or self.epoch == self.max_epochs :
                save_dir = os.path.join(self.opt.outdir, f'search_epoch_{self.epoch}')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'optimized_time_steps_epoch_{self.epoch}.txt')
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Top {len(self.keep_top_k[50])} candidates from epoch {self.epoch}\n")
                    for i, cand_s_f in enumerate(self.keep_top_k[50]):
                        # ... (similar saving logic as before)
                        metrics_f = self.vis_dict.get(cand_s_f, {})
                        # ...
                        f.write(f"Rank {i+1}\nCombined Similarity: {metrics_f.get('similarity',0.0):.4f}\n...\n")
                logging.info(f"Saved top results for epoch {self.epoch} to {save_path}")
        
        logging.info("Evolutionary fine-tuning search finished.")
        # ... (Final saving logic) ...


# (main function and argparser from your previous code, ensure to pass
#  initial_optimized_schedule_indices and step_search_delta to EvolutionSearcher)
def main():
    parser = argparse.ArgumentParser()
    # === Essential arguments for EvolutionSearcher ===
    parser.add_argument("--outdir", type=str, default="outputs/ea_finetune_timesteps", help="Dir to write results to")
    parser.add_argument("--config", type=str, required=True, help="Path to model config (e.g., v2-inference.yaml)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to COCO-style validation data (images and .txt captions with same names)")
    
    # Reference schedule from StepOptim (MUST be provided for this fine-tuning mode)
    parser.add_argument("--reference_schedule_json", type=str, required=True, help="Path to a JSON file containing the StepOptim schedule (list of indices) to fine-tune")
    parser.add_argument("--step_search_delta", type=int, default=20, help="Allowed deviation for each step index during mutation, relative to reference schedule's step.")

    # EA parameters
    parser.add_argument("--max_epochs", type=int, default=30, help="Max generations for EA")
    parser.add_argument("--population_num", type=int, default=20, help="Population size")
    parser.add_argument("--select_num", type=int, default=5, help="Number of elites selected")
    parser.add_argument("--mutation_num", type=int, default=8, help="Number of mutants to generate")
    parser.add_argument("--crossover_num", type=int, default=7, help="Number of crossovers to generate (ensure pop_size = select_num + mutation_num + crossover_num + randoms)")
    parser.add_argument("--m_prob", type=float, default=0.15, help="Probability of mutating each gene (timestep index)")
    
    # Evaluation parameters for get_cand_similarity
    parser.add_argument("--num_sample", type=int, default=64, help="Number of total images to generate for evaluating one candidate schedule")
    parser.add_argument("--n_samples", type=int, default=8, help="Batch size for dataloader and image generation within one candidate's evaluation")
    parser.add_argument("--use_clip", action='store_true', default=True) # Assuming default from your code
    parser.add_argument("--use_dino", action='store_true', default=True) # Assuming default

    # Sampler and model parameters (needed by get_cand_similarity and sampler init)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=7.5, help="CFG Scale")
    parser.add_argument("--C", type=int, default=4, help="Latent channels")
    parser.add_argument("--H", type=int, default=512, help="Image height for generation and TestDataset")
    parser.add_argument("--W", type=int, default=512, help="Image width for generation and TestDataset")
    parser.add_argument("--f", type=int, default=8, help="Downsampling factor")
    parser.add_argument("--precision", type=str, default="autocast", choices=["full", "autocast"])
    parser.add_argument("--seed", type=int, default=42)
    
    # opt.time_step_range is not strictly needed if num_steps is fixed by reference schedule
    # but EA ops might still use min/max_time_step_count if they try to change length.
    # For this focused search, length is fixed.
    # We'll set min/max_time_step_count based on reference schedule length.
    
    # For DPM solver (if used, though focused on DDIM)
    parser.add_argument("--dpm_solver", action='store_true', help="If DDIMSampler is actually DPM (not recommended here)")
    
    # Parallel evaluation (use with caution)
    parser.add_argument("--eval_workers", type=int, default=1, help="Num parallel workers for candidate eval (1 for sequential)")
    parser.add_argument("--eval_batch_size_cand", type=int, default=1, help="Num candidates per worker call (if parallel)")


    opt = parser.parse_args()
    seed_everything(opt.seed)

    os.makedirs(opt.outdir, exist_ok=True)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout),
                                 logging.FileHandler(os.path.join(opt.outdir, 'ea_finetune_log.txt'))],
                        level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')

    if not opt.reference_schedule_json or not os.path.exists(opt.reference_schedule_json):
        logging.error(f"Error: --reference_schedule_json path '{opt.reference_schedule_json}' is required and must exist for fine-tuning. This file should contain the list of timestep indices from StepOptim.")
        return

    try:
        with open(opt.reference_schedule_json, 'r') as f:
            initial_schedule_indices = json.load(f)
        if not (isinstance(initial_schedule_indices, list) and all(isinstance(x, int) for x in initial_schedule_indices)):
            logging.error("Invalid format in reference_schedule_json. Expected a list of integers."); return
        initial_schedule_indices = sorted(list(set(initial_schedule_indices))) # Normalize
    except Exception as e:
        logging.error(f"Error loading or parsing reference schedule from {opt.reference_schedule_json}: {e}"); return
    
    if not initial_schedule_indices:
        logging.error("Reference schedule is empty after loading. Aborting."); return

    # Override opt.time_step_range based on the length of the reference schedule
    # as this version of EA fine-tunes a fixed-length schedule.
    num_steps_from_ref = len(initial_schedule_indices)
    opt.time_step_range = f"{num_steps_from_ref}-{num_steps_from_ref}" # Fixed length
    logging.info(f"Search will optimize for a fixed number of steps: {num_steps_from_ref}, based on reference schedule.")


    # Create dummy data_dir if it doesn't exist (as in previous main)
    if not os.path.exists(opt.data_dir):
        os.makedirs(opt.data_dir, exist_ok=True); logging.info(f"Created dummy data_dir: {opt.data_dir}")
        try:
            Image.new("RGB", (opt.W, opt.H), "blue").save(os.path.join(opt.data_dir, "dummy_image.png"))
            with open(os.path.join(opt.data_dir, "dummy_image.txt"), "w") as f: f.write("dummy prompt")
        except: pass


    config_obj = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config_obj, f"{opt.ckpt}") 
    
    # We are focusing on DDIMSampler
    sampler = DDIMSampler(model) 
    # Ensure the sampler's timesteps_file is known and used by EvolutionSearcher.get_cand_similarity
    # DDIMSampler __init__ sets self.timesteps_file = 'custom_timesteps_search.txt'
    logging.info(f"DDIMSampler will use '{sampler.timesteps_file}' for custom timesteps during EA.")


    dataloader_info = build_dataloader(config_obj.model, opt)
    
    dpm_params_main = None # Not focusing on DPM here

    t_start_ea = time.time()
    searcher = EvolutionSearcher(opt=opt, model=model, sampler=sampler, 
                                 dataloader_info=dataloader_info,
                                 dpm_params=dpm_params_main, # Will be None
                                 initial_optimized_schedule_indices=initial_schedule_indices,
                                 step_search_delta=opt.step_search_delta)
    searcher.search()
    logging.info('Total evolutionary fine-tuning time = {:.2f} hours'.format((time.time() - t_start_ea) / 3600))

if __name__ == "__main__":
    main()