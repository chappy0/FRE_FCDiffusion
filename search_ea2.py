# import argparse
# import clip
# import numpy as np
# import os
# import random
# from omegaconf import OmegaConf
# from pytorch_lightning import seed_everything
# import torch
# import torch.nn as nn
# import sys
# import time

# import argparse
# import os
# from PIL import Image
# from tqdm import tqdm
# import torchvision.transforms as transforms
# import collections

# from calculate_CLIP import truncate_text
# from calculate_DINO_VIT import DinoVitExtractor

# sys.setrecursionlimit(10000)
# import functools

# import argparse
# import os
# from torch import autocast
# from contextlib import contextmanager, nullcontext
# import numpy as np
# import torch as th
# import torch.distributed as dist
# import torch.nn.functional as F
# from scipy import linalg
# from sklearn.ensemble import RandomForestClassifier

# from ldm.util import instantiate_from_config
# from ldm.models.diffusion.ddim import DDIMSampler
# from ldm.models.diffusion.plms import PLMSSampler
# from ldm.models.diffusion.dpm_solver import DPMSolverSampler
# import logging
# import ast

# class EvolutionSearcher(object):
#     def __init__(self, opt, model, time_step, ref_mu, ref_sigma, sampler, dataloader_info, batch_size, dpm_params=None):
#         self.opt = opt
#         self.model = model
#         self.sampler = sampler
#         self.time_step = time_step
#         self.dataloader_info = dataloader_info
#         self.batch_size = batch_size
#         self.min_time_step = 5  # 最小时间步数
#         self.max_time_step = 50  # 最大时间步数
#         ## EA hyperparameters
#         self.max_epochs = opt.max_epochs
#         self.select_num = opt.select_num
#         self.population_num = opt.population_num
#         self.m_prob = opt.m_prob
#         self.crossover_num = opt.crossover_num
#         self.mutation_num = opt.mutation_num
#         self.num_samples = opt.num_sample
#         self.ddim_discretize = "uniform"
#         ## tracking variable 
#         self.keep_top_k = {self.select_num: [], 50: []}
#         self.epoch = 0
#         self.candidates = []
#         self.vis_dict = {}

#         self.max_fid = opt.max_fid
        
#         self.RandomForestClassifier = RandomForestClassifier(n_estimators=40)
#         self.rf_features = []
#         self.rf_lebal = []

#         self.use_ddim_init_x = opt.use_ddim_init_x

#         self.dpm_params = dpm_params
    
#     # def update_top_k(self, candidates, *, k, key, reverse=True):
#     #     assert k in self.keep_top_k
#     #     logging.info('select ......')
#     #     t = self.keep_top_k[k]
#     #     t += candidates
#     #     t.sort(key=key, reverse=reverse)
#     #     self.keep_top_k[k] = t[:k]

#     def update_top_k(self, candidates, *, k, key, reverse=True):
#         assert k in self.keep_top_k
#         logging.info('select ......')
#         t = self.keep_top_k[k]
#         # 将候选人和他们的相似度组合在一起
#         candidates_with_sim = [(cand, self.vis_dict[cand]['similarity']) for cand in candidates if cand in self.vis_dict]
#         # 按相似度排序
#         candidates_with_sim.sort(key=lambda x: x[1], reverse=reverse)
#         # 只取排序后的候选人
#         sorted_candidates = [item[0] for item in candidates_with_sim]
#         # 更新 keep_top_k
#         self.keep_top_k[k] = sorted_candidates[:k]
    
#     # def is_legal_before_search(self, cand):
#     #     cand = eval(cand)
#     #     cand = sorted(cand)
#     #     cand = str(cand)
#     #     if cand not in self.vis_dict:
#     #         self.vis_dict[cand] = {}
#     #     info = self.vis_dict[cand]
#     #     if 'visited' in info:
#     #         logging.info('cand: {} has visited!'.format(cand))
#     #         return False
#     #     info['similarity'] = self.get_cand_similarity(opt=self.opt, cand=eval(cand))
#     #     logging.info('cand: {}, similarity: {}'.format(cand, info['similarity']))

#     #     info['visited'] = True
#     #     return True
    
#     # def is_legal(self, cand):
#     #     cand = eval(cand)
#     #     cand = sorted(cand)
#     #     cand = str(cand)
#     #     if cand not in self.vis_dict:
#     #         self.vis_dict[cand] = {}
#     #     info = self.vis_dict[cand]
#     #     if 'visited' in info:
#     #         logging.info('cand: {} has visited!'.format(cand))
#     #         return False
#     #     info['similarity'] = self.get_cand_similarity(opt=self.opt, cand=eval(cand))
#     #     logging.info('cand: {}, similarity: {}'.format(cand, info['similarity']))

#     #     info['visited'] = True
#     #     return True

#     def is_legal_before_search(self, cand):
#         cand = eval(cand)
#         cand = sorted(cand)
#         cand = str(cand)
#         if cand not in self.vis_dict:
#             self.vis_dict[cand] = {}
#         info = self.vis_dict[cand]
#         if 'visited' in info:
#             logging.info('cand: {} has visited!'.format(cand))
#             return False
#         info['similarity'] = self.get_cand_similarity(opt=self.opt, cand=eval(cand))
#         logging.info('cand: {}, similarity: {}'.format(cand, info['similarity']))

#         info['visited'] = True
#         return True

#     def is_legal(self, cand):
#         cand = eval(cand)
#         cand = sorted(cand)
#         cand = str(cand)
#         if cand not in self.vis_dict:
#             self.vis_dict[cand] = {}
#         info = self.vis_dict[cand]
#         if 'visited' in info:
#             logging.info('cand: {} has visited!'.format(cand))
#             return False
#         info['similarity'] = self.get_cand_similarity(opt=self.opt, cand=eval(cand))
#         logging.info('cand: {}, similarity: {}'.format(cand, info['similarity']))

#         info['visited'] = True
#         return True
    
#     def get_random_before_search(self, num):
#         logging.info('random select ........')
#         while len(self.candidates) < num:
#             if self.opt.dpm_solver:
#                 cand = self.sample_active_subnet_dpm()
#             else:
#                 cand = self.sample_active_subnet()
#             cand = sorted(cand)
#             cand = str(cand)
#             if not self.is_legal_before_search(cand):
#                 continue
#             self.candidates.append(cand)
#             logging.info('random {}/{}'.format(len(self.candidates), num))
#         logging.info('random_num = {}'.format(len(self.candidates)))
    
#     # def get_random(self, num):
#     #     logging.info('random select ........')
#     #     while len(self.candidates) < num:
#     #         if self.opt.dpm_solver:
#     #             cand = self.sample_active_subnet_dpm()
#     #         else:
#     #             cand = self.sample_active_subnet()
#     #         cand = sorted(cand)
#     #         cand = str(cand)
#     #         if not self.is_legal(cand):
#     #             continue
#     #         self.candidates.append(cand)
#     #         logging.info('random {}/{}'.format(len(self.candidates), num))
#     #     logging.info('random_num = {}'.format(len(self.candidates)))
    
#     def get_random(self, num):
#         logging.info('random select ........')
#         while len(self.candidates) < num:
#             if self.opt.dpm_solver:
#                 cand = self.sample_active_subnet_dpm()
#             else:
#                 cand = self.sample_active_subnet()
#             cand = sorted(cand)
#             cand = str(cand)
#             if not self.is_legal(cand):
#                 continue
#             self.candidates.append(cand)
#             # 确保存储到 keep_top_k 中的是元组
#             self.keep_top_k[self.population_num].append((cand, len(cand)))
#             logging.info('random {}/{}'.format(len(self.candidates), num))
#         logging.info('random_num = {}'.format(len(self.candidates)))
    
#     def get_cross(self, k, cross_num):
#         assert k in self.keep_top_k
#         logging.info('cross ......')
#         res = []
#         max_iters = cross_num * 10

#         def random_cross():
#             cand1 = random.choice(self.keep_top_k[k])
#             cand2 = random.choice(self.keep_top_k[k])

#             new_cand = []
#             cand1 = eval(cand1)
#             cand2 = eval(cand2)

#             for i in range(len(cand1)):
#                 if np.random.random_sample() < 0.5:
#                     new_cand.append(cand1[i])
#                 else:
#                     new_cand.append(cand2[i])

#             new_time_step_num = (len(new_cand) + len(cand2)) // 2  # 简单的取平均作为新的时间步数
#             new_cand = new_cand[:new_time_step_num]

#             return new_cand, new_time_step_num

#         while len(res) < cross_num and max_iters > 0:
#             max_iters -= 1
#             cand, time_step_num = random_cross()
#             cand = sorted(cand)
#             cand = str(cand)
#             if not self.is_legal(cand):
#                 continue
#             res.append((cand, time_step_num))
#             logging.info('cross {}/{}'.format(len(res), cross_num))

#         logging.info('cross_num = {}'.format(len(res)))
#         return res
    
#     # def get_mutation(self, k, mutation_num, m_prob):
#     #     assert k in self.keep_top_k
#     #     logging.info('mutation ......')
#     #     res = []
#     #     iter = 0
#     #     max_iters = mutation_num * 10

#     #     def random_func():
#     #         cand, time_step_num = random.choice(self.keep_top_k[k])
#     #         cand = eval(cand)

#     #         candidates = []
#     #         for i in range(self.sampler.ddpm_num_timesteps):
#     #             if i not in cand:
#     #                 candidates.append(i)

#     #         for i in range(len(cand)):
#     #             if np.random.random_sample() < m_prob:
#     #                 new_c = random.choice(candidates)
#     #                 new_index = candidates.index(new_c)
#     #                 del(candidates[new_index])
#     #                 cand[i] = new_c
#     #                 if len(candidates) == 0:  
#     #                     break

#     #         new_time_step_num = time_step_num + np.random.randint(-2, 3)  # 在原时间步数附近进行微调
#     #         new_time_step_num = max(self.min_time_step, min(new_time_step_num, self.max_time_step))
#     #         cand = cand[:new_time_step_num]

#     #         return cand, new_time_step_num

#     #     while len(res) < mutation_num and max_iters > 0:
#     #         max_iters -= 1
#     #         cand, time_step_num = random_func()
#     #         cand = sorted(cand)
#     #         cand = str(cand)
#     #         if not self.is_legal(cand):
#     #             continue
#     #         res.append((cand, time_step_num))
#     #         logging.info('mutation {}/{}'.format(len(res), mutation_num))

#     #     logging.info('mutation_num = {}'.format(len(res)))
#     #     return res
    
#     # def get_mutation(self, k, mutation_num, m_prob):
#     #     assert k in self.keep_top_k
#     #     logging.info('mutation ......')
#     #     res = []
#     #     iter = 0
#     #     max_iters = mutation_num * 10

#     #     def random_func():
#     #         # 确保从 keep_top_k[k] 中选择的是元组 (cand, time_step_num)
#     #         cand_tuple = random.choice(self.keep_top_k[k])
#     #         cand = eval(cand_tuple[0])  # 假设存储的是字符串形式的列表
#     #         time_step_num = cand_tuple[1]
#     #         candidates = []
#     #         for i in range(self.sampler.ddpm_num_timesteps):
#     #             if i not in cand:
#     #                 candidates.append(i)

#     #         for i in range(len(cand)):
#     #             if np.random.random_sample() < m_prob:
#     #                 new_c = random.choice(candidates)
#     #                 new_index = candidates.index(new_c)
#     #                 del(candidates[new_index])
#     #                 cand[i] = new_c
#     #                 if len(candidates) == 0:  
#     #                     break

#     #         new_time_step_num = time_step_num + np.random.randint(-2, 3)  # 在原时间步数附近进行微调
#     #         new_time_step_num = max(self.min_time_step, min(new_time_step_num, self.max_time_step))
#     #         cand = cand[:new_time_step_num]

#     #         return cand, new_time_step_num

#     #     while len(res) < mutation_num and max_iters > 0:
#     #         max_iters -= 1
#     #         cand, time_step_num = random_func()
#     #         cand = sorted(cand)
#     #         cand = str(cand)
#     #         if not self.is_legal(cand):
#     #             continue
#     #         res.append((cand, time_step_num))
#     #         logging.info('mutation {}/{}'.format(len(res), mutation_num))

#     #     logging.info('mutation_num = {}'.format(len(res)))
#     #     return res
    

#     # def get_mutation(self, k, mutation_num, m_prob):
#     #     assert k in self.keep_top_k
#     #     logging.info('mutation ......')
#     #     res = []
#     #     iter = 0
#     #     max_iters = mutation_num * 10

#     #     def random_func():
#     #         # 确保从 keep_top_k[k] 中选择的是元组 (cand, time_step_num)
#     #         cand_tuple = random.choice(self.keep_top_k[k])
#     #         try:
#     #             cand = ast.literal_eval(cand_tuple[0])  # 使用 ast.literal_eval 安全地解析字符串
#     #         except Exception as e:
#     #             logging.error(f"Error parsing cand_tuple[0]: {cand_tuple[0]}, Error: {e}")
#     #             return None, None  # 返回 None 表示解析失败
#     #         time_step_num = cand_tuple[1]

#     #         candidates = []
#     #         for i in range(self.sampler.ddpm_num_timesteps):
#     #             if i not in cand:
#     #                 candidates.append(i)

#     #         for i in range(len(cand)):
#     #             if np.random.random_sample() < m_prob:
#     #                 if not candidates:  # 避免除以零的错误
#     #                     break
#     #                 new_c = random.choice(candidates)
#     #                 new_index = candidates.index(new_c)
#     #                 del(candidates[new_index])
#     #                 cand[i] = new_c
#     #                 if len(candidates) == 0:  
#     #                     break

#     #         new_time_step_num = time_step_num + np.random.randint(-2, 3)  # 在原时间步数附近进行微调
#     #         new_time_step_num = max(self.min_time_step, min(new_time_step_num, self.max_time_step))
#     #         cand = cand[:new_time_step_num]

#     #         return cand, new_time_step_num

#     #     while len(res) < mutation_num and max_iters > 0:
#     #         max_iters -= 1
#     #         result = random_func()
#     #         if result is None:
#     #             continue  # 跳过解析失败的情况
#     #         cand, time_step_num = result
#     #         cand = sorted(cand)
#     #         cand = str(cand)
#     #         if not self.is_legal(cand):
#     #             continue
#     #         res.append((cand, time_step_num))
#     #         logging.info('mutation {}/{}'.format(len(res), mutation_num))

#     #     logging.info('mutation_num = {}'.format(len(res)))
#     #     return res

#     def get_mutation(self, k, mutation_num, m_prob):
#         assert k in self.keep_top_k
#         logging.info('mutation ......')
#         res = []
#         iter = 0
#         max_iters = mutation_num * 10

#         def random_func():
#             # 确保从 keep_top_k[k] 中选择的是有效元组
#             if not self.keep_top_k[k]:
#                 return None, None  # 如果为空，返回 None 表示无法生成
#             cand_tuple = random.choice(self.keep_top_k[k])
#             if cand_tuple is None:
#                 return None, None
#             try:
#                 cand = ast.literal_eval(cand_tuple[0])  # 使用 ast.literal_eval 安全地解析字符串
#             except Exception as e:
#                 logging.error(f"Error parsing cand_tuple[0]: {cand_tuple[0]}, Error: {e}")
#                 return None, None  # 解析失败返回 None
#             time_step_num = cand_tuple[1]

#             candidates = []
#             for i in range(self.sampler.ddpm_num_timesteps):
#                 if i not in cand:
#                     candidates.append(i)

#             for i in range(len(cand)):
#                 if np.random.random_sample() < m_prob:
#                     if not candidates:  # 避免除以零的错误
#                         break
#                     new_c = random.choice(candidates)
#                     new_index = candidates.index(new_c)
#                     del(candidates[new_index])
#                     cand[i] = new_c
#                     if len(candidates) == 0:  
#                         break

#             new_time_step_num = time_step_num + np.random.randint(-2, 3)  # 在原时间步数附近进行微调
#             new_time_step_num = max(self.min_time_step, min(new_time_step_num, self.max_time_step))
#             cand = cand[:new_time_step_num]

#             return cand, new_time_step_num

#         while len(res) < mutation_num and max_iters > 0:
#             max_iters -= 1
#             result = random_func()
#             if result is None:
#                 continue  # 跳过无效结果
#             cand, time_step_num = result
#             if cand is None:
#                 continue  # 跳过无效结果
#             cand = sorted(cand)
#             cand = str(cand)
#             if not self.is_legal(cand):
#                 continue
#             res.append((cand, time_step_num))
#             logging.info('mutation {}/{}'.format(len(res), mutation_num))

#         logging.info('mutation_num = {}'.format(len(res)))
#         return res
    
#     def get_mutation_dpm(self, k, mutation_num, m_prob):
#         assert k in self.keep_top_k
#         logging.info('mutation ......')
#         res = []
#         iter = 0
#         max_iters = mutation_num * 10

#         def random_func():
#             cand, time_step_num = random.choice(self.keep_top_k[k])
#             cand = eval(cand)

#             candidates = []
#             for i in self.dpm_params['full_timesteps']:
#                 if i not in cand:
#                     candidates.append(i)

#             for i in range(len(cand)):
#                 if np.random.random_sample() < m_prob:
#                     new_c = random.choice(candidates)
#                     new_index = candidates.index(new_c)
#                     del(candidates[new_index])
#                     cand[i] = new_c
#                     if len(candidates) == 0:  
#                         break

#             new_time_step_num = time_step_num + np.random.randint(-2, 3)  # 在原时间步数附近进行微调
#             new_time_step_num = max(self.min_time_step, min(new_time_step_num, self.max_time_step))
#             cand = cand[:new_time_step_num]

#             return cand, new_time_step_num

#         while len(res) < mutation_num and max_iters > 0:
#             max_iters -= 1
#             cand, time_step_num = random_func()
#             cand = sorted(cand)
#             cand = str(cand)
#             if not self.is_legal(cand):
#                 continue
#             res.append((cand, time_step_num))
#             logging.info('mutation {}/{}'.format(len(res), mutation_num))

#         logging.info('mutation_num = {}'.format(len(res)))
#         return res

#     def mutate_init_x_dpm(self, x0, mutation_num, m_prob):
#         logging.info('mutation x0 ......')
#         res = []
#         iter = 0
#         max_iters = mutation_num * 10

#         def random_func():
#             cand, time_step_num = x0
#             cand = eval(cand)

#             candidates = []
#             for i in self.dpm_params['full_timesteps']:
#                 if i not in cand:
#                     candidates.append(i)

#             for i in range(len(cand)):
#                 if np.random.random_sample() < m_prob:
#                     new_c = random.choice(candidates)
#                     new_index = candidates.index(new_c)
#                     del(candidates[new_index])
#                     cand[i] = new_c
#                     if len(candidates) == 0:  
#                         break

#             new_time_step_num = time_step_num + np.random.randint(-2, 3)  # 在原时间步数附近进行微调
#             new_time_step_num = max(self.min_time_step, min(new_time_step_num, self.max_time_step))
#             cand = cand[:new_time_step_num]

#             return cand, new_time_step_num

#         while len(res) < mutation_num and max_iters > 0:
#             max_iters -= 1
#             cand, time_step_num = random_func()
#             cand = sorted(cand)
#             cand = str(cand)
#             if not self.is_legal_before_search(cand):
#                 continue
#             res.append((cand, time_step_num))
#             logging.info('mutation x0 {}/{}'.format(len(res), mutation_num))

#         logging.info('mutation_num = {}'.format(len(res)))
#         return res

#     def sample_active_subnet(self):
#         original_num_steps = self.sampler.ddpm_num_timesteps
#         use_timestep = [i for i in range(original_num_steps)]
#         random.shuffle(use_timestep)
#         use_timestep = use_timestep[:self.time_step]
#         return use_timestep
    
#     def sample_active_subnet_dpm(self):
#         use_timestep = np.copy.deepcopy(self.dpm_params['full_timesteps'])
#         random.shuffle(use_timestep)
#         use_timestep = use_timestep[:self.time_step + 1]
#         return use_timestep
    
#     # def get_cand_similarity(self, cand=None, opt=None, device='cuda'):
#     #     t1 = time.time()
#     #     start_code = None
#     #     if opt.fixed_code:
#     #         start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

#     #     precision_scope = autocast if opt.precision == "autocast" else nullcontext

#     #     dino_extractor = DinoVitExtractor()

#     #     with torch.no_grad():
#     #         with precision_scope("cuda"):
#     #             with self.model.ema_scope():
#     #                 dino_similarity_list = []  
#     #                 clip_similarity_list = []  

#     #                 print(f"dataloader_info:{len(self.dataloader_info['validation_loader'])}")
#     #                 for itr, batch in enumerate(self.dataloader_info['validation_loader']):
#     #                     current_prompts = batch.get('txt', [])
#     #                     current_original_images = batch.get('jpg', [])
#     #                     print(f'current_prompts:{current_prompts}')

#     #                     np.savetxt('custom_timesteps.txt', cand[::-1], fmt='%d')
#     #                     logging.info(f'cand_step:{cand[::-1]}')
                        
#     #                     # Generate images using the model's log_images method
#     #                     log = self.model.log_images(batch, ddim_steps=len(cand))  # 使用当前时间步数
#     #                     generated_images = log["samples"]  

#     #                     # Preprocess generated images
#     #                     processed_generated_images = []
#     #                     for img in generated_images:
#     #                         img = img.squeeze()  
#     #                         img = img.permute(1, 2, 0)  
#     #                         img = torch.clamp(img, -1, 1)  
#     #                         pil_image = Image.fromarray(((img.cpu().numpy() + 1) * 127.5).astype(np.uint8))  
#     #                         processed_generated_images.append(pil_image)

#     #                     # Prepare original images as PIL
#     #                     original_images = []
#     #                     for img in current_original_images:
#     #                         if isinstance(img, torch.Tensor):
#     #                             img = img.cpu().numpy()
#     #                         if img.dtype != np.uint8:
#     #                             img = ((img + 1) * 127.5).astype(np.uint8)
#     #                         pil_image = Image.fromarray(img)
#     #                         original_images.append(pil_image)

#     #                         if len(dino_similarity_list) >= self.num_samples and len(clip_similarity_list) >= self.num_samples:
#     #                             break

#     #                     if opt.use_clip:
#     #                         logging.info("calculate clip")
#     #                         clip_sim_batch = calculate_clip_similarity_batch(processed_generated_images, current_prompts)
#     #                         clip_similarity_list.extend(clip_sim_batch)

#     #                     if opt.use_dino:
#     #                         logging.info("calculate dino")
#     #                         dino_sim_batch = compute_structure_similarity(processed_generated_images, original_images, dino_extractor=dino_extractor)
#     #                         dino_similarity_list.extend(dino_sim_batch)

#     #                 if not dino_similarity_list or not clip_similarity_list:
#     #                     logging.warning("No valid samples generated or no similarity metrics calculated.")
#     #                     return 0.0  

#     #                 if opt.use_clip and opt.use_dino:
#     #                     combined_similarity = (np.mean(clip_similarity_list) + np.mean(dino_similarity_list)) / 2
#     #                     logging.info(f'Combined Similarity: {combined_similarity:.4f}')
#     #                     self.vis_dict[tuple(cand)]['clip_similarity'] = np.mean(clip_similarity_list)
#     #                     self.vis_dict[tuple(cand)]['dino_similarity'] = np.mean(dino_similarity_list)
#     #                     return combined_similarity
#     #                 elif opt.use_clip:
#     #                     return np.mean(clip_similarity_list)
#     #                 elif opt.use_dino:
#     #                     return np.mean(dino_similarity_list)
#     #                 else:
#     #                     logging.error("At least one similarity metric must be enabled.")
#     #                     return 0.0
    
#     def get_cand_similarity(self, cand=None, opt=None, device='cuda'):
#         t1 = time.time()
#         start_code = None
#         if opt.fixed_code:
#             start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

#         precision_scope = autocast if opt.precision == "autocast" else nullcontext

#         dino_extractor = DinoVitExtractor()

#         with torch.no_grad():
#             with precision_scope("cuda"):
#                 with self.model.ema_scope():
#                     dino_similarity_list = []  
#                     clip_similarity_list = []  

#                     print(f"dataloader_info:{len(self.dataloader_info['validation_loader'])}")
#                     for itr, batch in enumerate(self.dataloader_info['validation_loader']):
#                         current_prompts = batch.get('txt', [])
#                         current_original_images = batch.get('jpg', [])
#                         print(f'current_prompts:{current_prompts}')

#                         np.savetxt('custom_timesteps.txt', cand[::-1], fmt='%d')
#                         logging.info(f'cand_step:{cand[::-1]}')
                        
#                         # Generate images using the model's log_images method
#                         log = self.model.log_images(batch, ddim_steps=len(cand))  # 使用当前时间步数
#                         generated_images = log["samples"]  

#                         # Preprocess generated images
#                         processed_generated_images = []
#                         for img in generated_images:
#                             img = img.squeeze()  
#                             img = img.permute(1, 2, 0)  
#                             img = torch.clamp(img, -1, 1)  
#                             pil_image = Image.fromarray(((img.cpu().numpy() + 1) * 127.5).astype(np.uint8))  
#                             processed_generated_images.append(pil_image)

#                         # Prepare original images as PIL
#                         original_images = []
#                         for img in current_original_images:
#                             if isinstance(img, torch.Tensor):
#                                 img = img.cpu().numpy()
#                             if img.dtype != np.uint8:
#                                 img = ((img + 1) * 127.5).astype(np.uint8)
#                             pil_image = Image.fromarray(img)
#                             original_images.append(pil_image)

#                             if len(dino_similarity_list) >= self.num_samples and len(clip_similarity_list) >= self.num_samples:
#                                 break

#                         if opt.use_clip:
#                             logging.info("calculate clip")
#                             clip_sim_batch = calculate_clip_similarity_batch(processed_generated_images, current_prompts)
#                             clip_similarity_list.extend(clip_sim_batch)

#                         if opt.use_dino:
#                             logging.info("calculate dino")
#                             dino_sim_batch = compute_structure_similarity(processed_generated_images, original_images, dino_extractor=dino_extractor)
#                             dino_similarity_list.extend(dino_sim_batch)

#                     if not dino_similarity_list or not clip_similarity_list:
#                         logging.warning("No valid samples generated or no similarity metrics calculated.")
#                         return 0.0  

#                     # 确保 self.vis_dict 中存在该 cand 的键
#                     cand_tuple = tuple(sorted(cand))  # 确保 cand 是元组并且排序
#                     if cand_tuple not in self.vis_dict:
#                         self.vis_dict[cand_tuple] = {}

#                     if opt.use_clip and opt.use_dino:
#                         combined_similarity = (np.mean(clip_similarity_list) + np.mean(dino_similarity_list)) / 2
#                         logging.info(f'Combined Similarity: {combined_similarity:.4f}')
#                         self.vis_dict[cand_tuple]['clip_similarity'] = np.mean(clip_similarity_list)
#                         self.vis_dict[cand_tuple]['dino_similarity'] = np.mean(dino_similarity_list)
#                         self.vis_dict[cand_tuple]['similarity'] = combined_similarity
#                         return combined_similarity
#                     elif opt.use_clip:
#                         clip_sim = np.mean(clip_similarity_list)
#                         self.vis_dict[cand_tuple]['clip_similarity'] = clip_sim
#                         self.vis_dict[cand_tuple]['similarity'] = clip_sim
#                         return clip_sim
#                     elif opt.use_dino:
#                         dino_sim = np.mean(dino_similarity_list)
#                         self.vis_dict[cand_tuple]['dino_similarity'] = dino_sim
#                         self.vis_dict[cand_tuple]['similarity'] = dino_sim
#                         return dino_sim
#                     else:
#                         logging.error("At least one similarity metric must be enabled.")
#                         return 0.0

#     def search(self):
#         logging.info('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
#             self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
#         if self.use_ddim_init_x is False:
#             self.get_random_before_search(self.population_num)

#         else:
#             if self.opt.dpm_solver:
#                 init_x = self.dpm_params['init_timesteps']
#             else:
#                 init_x = self.sampler.ddim_timesteps[:self.time_step]  
#                 print(f"init_x:{init_x}")
#             init_x = sorted(list(init_x))
#             self.is_legal_before_search(str(init_x))
#             self.candidates.append(str(init_x))
#             self.get_random_before_search(self.population_num // 2)
#             if self.opt.dpm_solver:
#                 res = self.mutate_init_x_dpm(x0=(str(init_x), self.time_step), mutation_num=self.population_num - self.population_num // 2 - 1, m_prob=0.1)
#             else:
#                 res = self.mutate_init_x(x0=(str(init_x), self.time_step), mutation_num=self.population_num - self.population_num // 2 - 1, m_prob=0.1)
#             self.candidates += [cand for cand, _ in res]
        
#         while self.epoch < self.max_epochs:
#             logging.info('epoch = {}'.format(self.epoch))
#             # 更新 top k 时只考虑 candidates 和对应的相似度
#             current_candidates_with_sim = [(cand, self.vis_dict[cand]['similarity']) for cand in self.candidates if cand in self.vis_dict and 'similarity' in self.vis_dict[cand]]
#             current_candidates, current_similarities = zip(*current_candidates_with_sim)
#             self.update_top_k(current_candidates, k=self.select_num, key=lambda x: current_similarities[current_candidates.index(x)])
#             self.update_top_k(current_candidates, k=50, key=lambda x: current_similarities[current_candidates.index(x)])

#             logging.info('epoch = {} : top {} result'.format(
#                 self.epoch, len(self.keep_top_k[50])))
#             for i, cand in enumerate(self.keep_top_k[50]):
#                 logging.info('No.{} {} similarity = {}'.format(
#                     i + 1, cand, self.vis_dict[cand]['similarity']))
            
#             if self.epoch + 1 == self.max_epochs:
#                 break
#             if self.opt.dpm_solver:
#                 mutation = self.get_mutation_dpm(self.select_num, self.mutation_num, self.m_prob)
#             else:
#                 mutation = self.get_mutation(self.select_num, self.mutation_num, self.m_prob)

#             self.candidates = [cand for cand, _ in mutation]

#             cross_cand = self.get_cross(self.select_num, self.crossover_num)
#             self.candidates += [cand for cand, _ in cross_cand]

#             self.get_random(self.population_num)

#             self.epoch += 1
            
#             if self.epoch % 5 == 0:  
#                 save_path = os.path.join(self.opt.outdir, f'search_epoch_{self.epoch}')
#                 os.makedirs(save_path, exist_ok=True)

#                 model_config = {
#                     'model_class': self.model.__class__.__name__,
#                     'state_dict': self.model.state_dict(),
#                     'time_step': self.time_step,
#                     'dpm_params': self.dpm_params
#                 }
#                 torch.save(model_config, os.path.join(save_path, 'model_config.pth'))

#                 top_k = self.keep_top_k.get(50, self.candidates)  
#                 with open(os.path.join(save_path, 'optimized_time_steps.txt'), 'w') as f:
#                     for i, cand in enumerate(top_k):
#                         clip_sim = self.vis_dict[cand].get('clip_similarity', 0.0)
#                         dino_sim = self.vis_dict[cand].get('dino_similarity', 0.0)
#                         # f.write(f'No.{i + 1} {cand} CLIP Similarity = {clip_sim:.4f}, DINO Similarity = {dino_sim:.4f}\n')
#                         f.write(f'No.{i + 1} {cand} CLIP Similarity = {clip_sim:.4f}, DINO Similarity = {dino_sim:.4f}, Combined Similarity = {combined_sim:.4f}\n')
#                         f.write(f'Time Steps: {eval(cand)}\n\n')
#                         clip_sim = self.vis_dict[cand].get('clip_similarity', 0.0)




#         save_path = os.path.join(self.opt.outdir, 'search_final')
#         os.makedirs(save_path, exist_ok=True)

#         model_config = {
#             'model_class': self.model.__class__.__name__,
#             'state_dict': self.model.state_dict(),
#             'time_step': self.time_step,
#             'dpm_params': self.dpm_params
#         }
#         torch.save(model_config, os.path.join(save_path, 'model_config.pth'))

#         top_k = self.keep_top_k.get(50, self.candidates)  
#         with open(os.path.join(save_path, 'optimized_time_steps.txt'), 'w') as f:
#             for i, cand in enumerate(top_k):
#                 clip_sim = self.vis_dict[cand].get('clip_similarity', 0.0)
#                 dino_sim = self.vis_dict[cand].get('dino_similarity', 0.0)
#                 f.write(f'No.{i + 1} {cand} CLIP Similarity = {clip_sim:.4f}, DINO Similarity = {dino_sim:.4f}\n')
#                 f.write(f'Time Steps: {eval(cand)}\n\n')

# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--prompt",
#         type=str,
#         nargs="?",
#         default="a painting of a virus monster playing guitar",
#         help="the prompt to render"
#     )
#     parser.add_argument(
#         "--outdir",
#         type=str,
#         nargs="?",
#         help="dir to write results to",
#         default="outputs/txt2img-samples"
#     )
#     parser.add_argument(
#         "--skip_grid",
#         action='store_true',
#         help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
#     )
#     parser.add_argument(
#         "--skip_save",
#         action='store_true',
#         help="do not save individual samples. For speed measurements.",
#     )
#     parser.add_argument(
#         "--ddim_steps",
#         type=int,
#         default=50,
#         help="number of ddim sampling steps",
#     )
#     parser.add_argument(
#         "--plms",
#         action='store_true',
#         help="use plms sampling",
#     )
#     parser.add_argument(
#         "--dpm_solver",
#         action='store_true',
#         help="use dpm_solver sampling",
#     )
#     parser.add_argument(
#         "--laion400m",
#         action='store_true',
#         help="uses the LAION400M model",
#     )
#     parser.add_argument(
#         "--fixed_code",
#         action='store_true',
#         help="if enabled, uses the same starting code across samples ",
#     )
#     parser.add_argument(
#         "--ddim_eta",
#         type=float,
#         default=0.0,
#         help="ddim eta (eta=0.0 corresponds to deterministic sampling",
#     )
#     parser.add_argument(
#         "--n_iter",
#         type=int,
#         default=2,
#         help="sample this often",
#     )
#     parser.add_argument(
#         "--H",
#         type=int,
#         default=512,
#         help="image height, in pixel space",
#     )
#     parser.add_argument(
#         "--W",
#         type=int,
#         default=512,
#         help="image width, in pixel space",
#     )
#     parser.add_argument(
#         "--C",
#         type=int,
#         default=4,
#         help="latent channels",
#     )
#     parser.add_argument(
#         "--f",
#         type=int,
#         default=8,
#         help="downsampling factor",
#     )
#     parser.add_argument(
#         "--n_samples",
#         type=int,
#         default=3,
#         help="how many samples to produce for each given prompt. A.k.a. batch size",
#     )
#     parser.add_argument(
#         "--n_rows",
#         type=int,
#         default=0,
#         help="rows in the grid (default: n_samples)",
#     )
#     parser.add_argument(
#         "--scale",
#         type=float,
#         default=7.5,
#         help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
#     )
#     parser.add_argument(
#         "--from_file",
#         type=str,
#         help="if specified, load prompts from this file",
#     )
#     parser.add_argument(
#         "--config",
#         type=str,
#         default="configs/stable-diffusion/v1-inference.yaml",
#         help="path to config which constructs model",
#     )
#     parser.add_argument(
#         "--ckpt",
#         type=str,
#         default="models/ldm/stable-diffusion-v1/model.ckpt",
#         help="path to checkpoint of model",
#     )
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="the seed (for reproducible sampling)",
#     )
#     parser.add_argument(
#         "--precision",
#         type=str,
#         help="evaluate at this precision",
#         choices=["full", "autocast"],
#         default="autocast"
#     )
#     parser.add_argument(
#         "--data_dir",
#         type=str,
#         default="",
#         help="path to data",
#     )
#     parser.add_argument(
#         "--num_sample",
#         type=int,
#         default=4,
#         help="samples num",
#     )
#     parser.add_argument(
#         "--cal_fid",
#         type=bool,
#         default=False,
#     )
#     parser.add_argument(
#         "--max_epochs",
#         type=int,
#         default=10,
#     )
#     parser.add_argument(
#         "--select_num",
#         type=int,
#         default=10,
#     )
#     parser.add_argument(
#         "--population_num",
#         type=int,
#         default=50,
#     )
#     parser.add_argument(
#         "--m_prob",
#         type=float,
#         default=0.1,
#     )
#     parser.add_argument(
#         "--crossover_num",
#         type=int,
#         default=25,
#     )
#     parser.add_argument(
#         "--mutation_num",
#         type=int,
#         default=25,
#     )
#     parser.add_argument(
#         "--max_fid",
#         type=float,
#         default=3.,
#     )
#     parser.add_argument(
#         "--thres",
#         type=float,
#         default=0.2,
#     )
#     parser.add_argument(
#         "--ref_mu",
#         type=str,
#         default='',
#     )
#     parser.add_argument(
#         "--ref_sigma",
#         type=str,
#         default='',
#     )
#     parser.add_argument(
#         "--time_step",
#         type=int,
#         default=50,
#     )
#     parser.add_argument(
#         "--use_ddim_init_x",
#         type=bool,
#         default=False,
#     )
#     parser.add_argument(
#         "--use_clip",
#         action='store_true',
#         default=True,
#         help="Use CLIP similarity metric",
#     )
#     parser.add_argument(
#         "--use_dino",
#         action='store_true',
#         default=True,
#         help="Use DINO-ViT similarity metric",
#     )
#     opt = parser.parse_args()

#     if opt.laion400m:
#         print("Falling back to LAION 400M model...")
#         opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
#         opt.ckpt = "models/ldm/text2img-large/model.ckpt"
#         opt.outdir = "outputs/txt2img-samples-laion400m"

#     seed_everything(opt.seed)

#     config = OmegaConf.load(f"{opt.config}")
#     model = load_model_from_config(config, f"{opt.ckpt}")  

#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     model = model.to(device)

#     if opt.dpm_solver:
#         sampler = DPMSolverSampler(model)  
#     elif opt.plms:
#         sampler = PLMSSampler(model)
#     else:
#         sampler = DDIMSampler(model)

#     dataloader_info = build_dataloader(config, opt)

#     outpath = opt.outdir

#     os.makedirs(outpath, exist_ok=True)
#     log_format = '%(asctime)s %(message)s'
#     logging.basicConfig(stream=sys.stdout, level=logging.INFO,
#         format=log_format, datefmt='%m/%d %I:%M:%S %p')
#     fh = logging.FileHandler(os.path.join(outpath, 'log.txt'))
#     fh.setFormatter(logging.Formatter(log_format))
#     logging.getLogger().addHandler(fh)

#     batch_size = opt.n_samples

#     if opt.dpm_solver:
#         tmp_sampler = DPMSolverSampler(model)
#         from ldm.models.diffusion.dpm_solver.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
#         ns = NoiseScheduleVP('discrete', alphas_cumprod=tmp_sampler.alphas_cumprod)
#         dpm_solver = DPM_Solver(None, ns, predict_x0=True, thresholding=False)
#         skip_type = "time_uniform"
#         t_0 = 1. / dpm_solver.noise_schedule.total_N  
#         t_T = dpm_solver.noise_schedule.T  
#         full_timesteps = dpm_solver.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=1000, device='cpu')
#         init_timesteps = dpm_solver.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=opt.time_step, device='cpu')
#         dpm_params = dict()
#         full_timesteps = list(full_timesteps)
#         dpm_params['full_timesteps'] = [full_timesteps[i].item() for i in range(len(full_timesteps))]
#         init_timesteps = list(init_timesteps)
#         dpm_params['init_timesteps'] = [init_timesteps[i].item() for i in range(len(init_timesteps))]
#     else:
#         dpm_params = None

#     t = time.time()
#     searcher = EvolutionSearcher(opt=opt, model=model, time_step=opt.time_step, ref_mu=opt.ref_mu, ref_sigma=opt.ref_sigma, sampler=sampler, dataloader_info=dataloader_info, batch_size=batch_size, dpm_params=dpm_params)
#     searcher.search()
#     logging.info('total searching time = {:.2f} hours'.format((time.time() - t) / 3600))

# import os  

# def is_image_file(filename):  
#     IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg']  
#     return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)  

# def is_text_file(filename):  
#     return filename.lower().endswith('.txt')  

# from torch.utils.data import DataLoader
# from fcdiffusion.dataset import TestDataset

# def build_dataloader(config, opt):
#     def traverse_images_and_texts(directory):  
#         image_files = []  
#         text_contents = []  
#         for root, dirs, files in os.walk(directory):  
#             for file in files:  
#                 if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.txt')):  
#                     file_path = os.path.join(root, file)  
#                     if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  
#                         image_files.append(file_path)  
#                     elif file.lower().endswith('.txt'):  
#                         with open(file_path, 'r', encoding='utf-8') as f:  
#                             text_contents.append(f.read())  
#         return image_files, text_contents  

#     image_files, text_contents = traverse_images_and_texts(opt.data_dir)
#     print(f"Found {len(image_files)} image files and {len(text_contents)} text files.")  

#     all_datasets = []

#     for image_file, text_content in zip(image_files, text_contents):  
#         test_img_path, target_prompt = image_file, text_content
#         dataset = TestDataset(test_img_path, target_prompt, 1)
#         all_datasets.append(dataset)

#     if not all_datasets:
#         raise ValueError("No valid test data found in directory: {}".format(opt.data_dir))
    
#     combined_dataset = torch.utils.data.ConcatDataset(all_datasets)
#     validation_loader = DataLoader(
#         combined_dataset,
#         batch_size=opt.n_samples,
#         shuffle=True,
#         num_workers=4
#     )
#     print(f"Combined dataset length: {len(combined_dataset)}")  
#     print(f"Validation loader batch size: {validation_loader.batch_size}")  

#     return {'validation_loader': validation_loader}

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
# def load_images_and_texts(processed_images, text_contents):
#     image_text_pairs = []
#     for image, text_content in zip(processed_images, text_contents):
#         try:
#             processed_image = preprocess(image) if hasattr(preprocess, '__call__') else image
#             try:
#                 text = truncate_text(text_content)  
#                 image_text_pairs.append((processed_image, text))
#             except Exception as e:
#                 print(f"Error truncating text: {e}")
#                 continue

#         except Exception as e:
#             print(f"Error processing pair (image, text): {e}")
#             continue  
#     return image_text_pairs

# def calculate_clip_similarity_batch(images, texts):
#     image_text_pairs = load_images_and_texts(images, texts)
#     cosine_similarities = []
#     with torch.no_grad():
#         for img, text in image_text_pairs:
#             img = img.unsqueeze(0).to(device)  
#             image_features = model.encode_image(img)  
#             image_features /= image_features.norm(dim=-1, keepdim=True)  

#             text_tokens = clip.tokenize([text]).to(device)
#             text_features = model.encode_text(text_tokens)  
#             text_features /= text_features.norm(dim=-1, keepdim=True)  

#             similarity = (image_features @ text_features.T).item()  
#             cosine_similarities.append(similarity)
    
#     return cosine_similarities

# def compute_structure_similarity(images1, images2, dino_extractor):
#     similarities = []
#     for image1, image2 in zip(images1, images2):
#         features1 = dino_extractor.extract_features(image1)
#         features2 = dino_extractor.extract_features(image2)
#         features1 = features1.mean(dim=1)  
#         features2 = features2.mean(dim=1)  
#         similarity = torch.cosine_similarity(features1, features2)
#         similarities.append(similarity)
#     return similarity  

# def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
#     print(f"Loading model from {ckpt}")
#     pl_sd = torch.load(ckpt, map_location="cpu")
#     if "global_step" in pl_sd:
#         print(f"Global Step: {pl_sd['global_step']}")
#     sd = pl_sd["state_dict"]
#     model = instantiate_from_config(config.model)
#     m, u = model.load_state_dict(sd, strict=False)
#     if len(m) > 0 and verbose:
#         print("missing keys:")
#         print(m)
#     if len(u) > 0 and verbose:
#         print("unexpected keys:")
#         print(u)
#     if device == torch.device("cuda"):
#         model.cuda()
#     elif device == torch.device("cpu"):
#         model.cpu()
#         model.cond_stage_model.device = "cpu"
#     else:
#         raise ValueError(f"Incorrect device name. Received: {device}")
#     model.eval()
#     return model

# if __name__ == "__main__":
#     main()

import argparse
import clip
import concurrent
import numpy as np
import os
import random
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import torch
import torch.nn as nn
import sys
import time
import json

import argparse
import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import collections

from calculate_CLIP import truncate_text
from calculate_DINO_VIT import DinoVitExtractor

sys.setrecursionlimit(10000)
import functools

import argparse
import os
from torch import autocast
from contextlib import contextmanager, nullcontext
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from scipy import linalg
from sklearn.ensemble import RandomForestClassifier

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
import logging
import ast

class EvolutionSearcher(object):
    def __init__(self, opt, model, time_step_range, ref_mu, ref_sigma, sampler, dataloader_info, batch_size, dpm_params=None):
        self.opt = opt
        self.model = model
        self.sampler = sampler
        self.dataloader_info = dataloader_info
        self.batch_size = batch_size
        time_step_range = opt.time_step_range.split("-")
        self.min_time_step = int(time_step_range[0])
        self.max_time_step = int(time_step_range[1])
        ## EA hyperparameters
        self.max_epochs = opt.max_epochs
        self.select_num = opt.select_num
        self.population_num = opt.population_num
        self.m_prob = opt.m_prob
        self.crossover_num = opt.crossover_num
        self.mutation_num = opt.mutation_num
        self.num_samples = opt.num_sample
        self.ddim_discretize = "uniform"
        ## tracking variable 
        self.keep_top_k = {
            self.select_num: [], 
            50: [], 
            self.population_num: []
        }
        self.epoch = 0
        self.candidates = []
        self.vis_dict = {}
        self.max_fid = opt.max_fid
        self.RandomForestClassifier = RandomForestClassifier(n_estimators=40)
        self.rf_features = []
        self.rf_lebal = []
        self.use_ddim_init_x = opt.use_ddim_init_x
        self.dpm_params = dpm_params
        # 早停机制参数
        self.stop_counter = 0
        self.best_history = []
        self.early_stop_thresh = 0.01  
        self.early_stop_patience = 3  
        self.eval_workers = 4  
        self.eval_batch_size = 8  

    def parallel_evaluate(self, candidates):
        def evaluate_batch(cand_batch):
            return [(cand, self.get_cand_similarity(cand=eval(cand))) 
                   for cand in cand_batch]
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.eval_workers) as executor:
            batches = [candidates[i:i+self.eval_batch_size] 
                      for i in range(0, len(candidates), self.eval_batch_size)]
            
            futures = [executor.submit(evaluate_batch, batch) for batch in batches]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        
        for cand, sim in results:
            self.vis_dict[cand]['similarity'] = sim

    def check_early_stop(self, current_best):
        if len(self.best_history) >= self.early_stop_patience:
            improvements = [
                (current_best - self.best_history[i-1])/self.best_history[i-1]
                for i in range(1, len(self.best_history))
            ]
            avg_improve = sum(improvements[-self.early_stop_patience:])/len(improvements)
            
            if avg_improve < self.early_stop_thresh:
                self.stop_counter += 1
                if self.stop_counter >= self.early_stop_patience:
                    return True
        return False

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        t = self.keep_top_k[k]
        candidates_with_sim = [(cand, self.vis_dict[cand]['similarity']) for cand in candidates if cand in self.vis_dict]
        candidates_with_sim.sort(key=lambda x: x[1], reverse=reverse)
        sorted_candidates = [item[0] for item in candidates_with_sim]
        self.keep_top_k[k] = sorted_candidates[:k]
    
    def is_legal_before_search(self, cand):
        cand = eval(cand)
        cand = sorted(cand)
        cand = str(cand)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logging.info('cand: {} has visited!'.format(cand))
            return False
        info['similarity'] = self.get_cand_similarity(opt=self.opt, cand=eval(cand))
        logging.info('cand: {}, similarity: {}'.format(cand, info['similarity']))
        info['visited'] = True
        return True

    def is_legal(self, cand):
        cand = eval(cand)
        cand = sorted(cand)
        cand = str(cand)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logging.info('cand: {} has visited!'.format(cand))
            return False
        info['similarity'] = self.get_cand_similarity(opt=self.opt, cand=eval(cand))
        logging.info('cand: {}, similarity: {}'.format(cand, info['similarity']))
        info['visited'] = True
        return True

    def get_random(self, num):
        logging.info('random select ........')
        while len(self.candidates) < num:
            if self.opt.dpm_solver:
                cand = self.sample_active_subnet_dpm()
            else:
                cand = self.sample_active_subnet()
            time_step_num = random.randint(self.min_time_step, self.max_time_step)
            cand = cand[:time_step_num]
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            self.keep_top_k[self.population_num].append((cand, len(cand)))
            logging.info('random {}/{}'.format(len(self.candidates), num))
        logging.info('random_num = {}'.format(len(self.candidates)))
    

    def get_random_before_search(self, num):
        logging.info('random select ........')
        while len(self.candidates) < num:
            if self.opt.dpm_solver:
                cand = self.sample_active_subnet_dpm()
            else:
                cand = self.sample_active_subnet()
            time_step_num = random.randint(self.min_time_step, self.max_time_step)
            cand = cand[:time_step_num]
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            self.candidates.append(cand)
            logging.info('random {}/{}'.format(len(self.candidates), num))
        logging.info('random_num = {}'.format(len(self.candidates)))
    

    
    def get_cross(self, k, cross_num):
        assert k in self.keep_top_k
        logging.info('cross ......')
        res = []
        max_iters = cross_num * 10

        def random_cross():
            cand1 = random.choice(self.keep_top_k[k])
            cand2 = random.choice(self.keep_top_k[k])

            new_cand = []
            cand1 = eval(cand1)
            cand2 = eval(cand2)

            max_len = max(len(cand1), len(cand2))
            cand1 += [cand1[-1]] * (max_len - len(cand1))
            cand2 += [cand2[-1]] * (max_len - len(cand2))

            for i in range(max_len):
                if np.random.random_sample() < 0.5:
                    new_cand.append(cand1[i])
                else:
                    new_cand.append(cand2[i])

            new_time_step_num = len(new_cand) + np.random.randint(-2, 3)
            new_time_step_num = max(self.min_time_step, min(new_time_step_num, self.max_time_step))
            new_cand = new_cand[:new_time_step_num]

            return new_cand, new_time_step_num

        while len(res) < cross_num and max_iters > 0:
            max_iters -= 1
            cand, time_step_num = random_cross()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append((cand, time_step_num))
            logging.info('cross {}/{}'.format(len(res), cross_num))

        logging.info('cross_num = {}'.format(len(res)))
        return res

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            if not self.keep_top_k[k]:
                return None, None
            cand_tuple = random.choice(self.keep_top_k[k])
            if cand_tuple is None:
                return None, None
            try:
                cand = ast.literal_eval(cand_tuple[0])
            except Exception as e:
                logging.error(f"Error parsing cand_tuple[0]: {cand_tuple[0]}, Error: {e}")
                return None, None
            time_step_num = cand_tuple[1]

            candidates = []
            for i in range(self.sampler.ddpm_num_timesteps):
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    if not candidates:
                        break
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:
                        break

            new_time_step_num = time_step_num + np.random.randint(-2, 3)
            new_time_step_num = max(self.min_time_step, min(new_time_step_num, self.max_time_step))
            cand = cand[:new_time_step_num]

            return cand, new_time_step_num

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            result = random_func()
            if result is None:
                continue
            cand, time_step_num = result
            if cand is None:
                continue
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append((cand, time_step_num))
            logging.info('mutation {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res
    
    def get_mutation_dpm(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand, time_step_num = random.choice(self.keep_top_k[k])
            cand = eval(cand)

            candidates = []
            for i in self.dpm_params['full_timesteps']:
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  
                        break

            new_time_step_num = time_step_num + np.random.randint(-2, 3)  
            new_time_step_num = max(self.min_time_step, min(new_time_step_num, self.max_time_step))
            cand = cand[:new_time_step_num]

            return cand, new_time_step_num

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand, time_step_num = random_func()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append((cand, time_step_num))
            logging.info('mutation {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def mutate_init_x_dpm(self, x0, mutation_num, m_prob):
        logging.info('mutation x0 ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand, time_step_num = x0
            cand = eval(cand)

            candidates = []
            for i in self.dpm_params['full_timesteps']:
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  
                        break

            new_time_step_num = time_step_num + np.random.randint(-2, 3)  
            new_time_step_num = max(self.min_time_step, min(new_time_step_num, self.max_time_step))
            cand = cand[:new_time_step_num]

            return cand, new_time_step_num

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand, time_step_num = random_func()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            res.append((cand, time_step_num))
            logging.info('mutation x0 {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res



    
    def sample_active_subnet(self):
        time_step_num = random.randint(self.min_time_step, self.max_time_step)
        original_num_steps = self.sampler.ddpm_num_timesteps
        use_timestep = [i for i in range(original_num_steps)]
        random.shuffle(use_timestep)
        use_timestep = use_timestep[:time_step_num]
        return use_timestep
    
    def sample_active_subnet_dpm(self):
        use_timestep = np.copy.deepcopy(self.dpm_params['full_timesteps'])
        random.shuffle(use_timestep)
        use_timestep = use_timestep[:self.time_step + 1]
        return use_timestep
    
    def get_cand_similarity(self, cand=None, opt=None, device='cuda'):
        t1 = time.time()
        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

        precision_scope = autocast if opt.precision == "autocast" else nullcontext

        dino_extractor = DinoVitExtractor()

        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    dino_similarity_list = []  
                    clip_similarity_list = []  

                    for itr, batch in enumerate(self.dataloader_info['validation_loader']):
                        current_prompts = batch.get('txt', [])
                        current_original_images = batch.get('jpg', [])

                        # np.savetxt('custom_timesteps.txt', cand[::-1], fmt='%d')
                        logging.info(f'cand_step:{cand[::-1]}')
                        
                        log = self.model.log_images(batch, ddim_steps=len(cand))  
                        generated_images = log["samples"]  

                        processed_generated_images = []
                        for img in generated_images:
                            img = img.squeeze()  
                            img = img.permute(1, 2, 0)  
                            img = torch.clamp(img, -1, 1)  
                            pil_image = Image.fromarray(((img.cpu().numpy() + 1) * 127.5).astype(np.uint8))  
                            processed_generated_images.append(pil_image)

                        original_images = []
                        for img in current_original_images:
                            if isinstance(img, torch.Tensor):
                                img = img.cpu().numpy()
                            if img.dtype != np.uint8:
                                img = ((img + 1) * 127.5).astype(np.uint8)
                            pil_image = Image.fromarray(img)
                            original_images.append(pil_image)

                            if len(dino_similarity_list) >= self.num_samples and len(clip_similarity_list) >= self.num_samples:
                                break

                        if opt.use_clip:
                            clip_sim_batch = calculate_clip_similarity_batch(processed_generated_images, current_prompts)
                            clip_similarity_list.extend(clip_sim_batch)

                        if opt.use_dino:
                            dino_sim_batch = compute_structure_similarity(processed_generated_images, original_images, dino_extractor=dino_extractor)
                            dino_similarity_list.extend(dino_sim_batch)

                    if not dino_similarity_list or not clip_similarity_list:
                        logging.warning("No valid samples generated or no similarity metrics calculated.")
                        return 0.0  

                    cand_tuple = tuple(sorted(cand))  
                    if cand_tuple not in self.vis_dict:
                        self.vis_dict[cand_tuple] = {}

                    if opt.use_clip and opt.use_dino:
                        combined_similarity = (np.mean(clip_similarity_list) + np.mean(dino_similarity_list)) / 2
                        logging.info(f'Combined Similarity: {combined_similarity:.4f}')
                        self.vis_dict[cand_tuple]['clip_similarity'] = np.mean(clip_similarity_list)
                        self.vis_dict[cand_tuple]['dino_similarity'] = np.mean(dino_similarity_list)
                        self.vis_dict[cand_tuple]['similarity'] = combined_similarity
                        return combined_similarity
                    elif opt.use_clip:
                        clip_sim = np.mean(clip_similarity_list)
                        self.vis_dict[cand_tuple]['clip_similarity'] = clip_sim
                        self.vis_dict[cand_tuple]['similarity'] = clip_sim
                        return clip_sim
                    elif opt.use_dino:
                        dino_sim = np.mean(dino_similarity_list)
                        self.vis_dict[cand_tuple]['dino_similarity'] = dino_sim
                        self.vis_dict[cand_tuple]['similarity'] = dino_sim
                        return dino_sim
                    else:
                        logging.error("At least one similarity metric must be enabled.")
                        return 0.0

    def search(self):
        logging.info('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        
        # 初始阶段，基于time_step_range生成不同步数的初始时间步值
        time_step_range = list(range(self.min_time_step, self.max_time_step + 1))
        step_num_candidates = []  # 用于存储不同步数的候选
        
        for step_num in time_step_range:
 
                
            init_timesteps = self.sampler.make_schedule(step_num,overwrite=True,return_flag=True)
            

            # custom_timesteps = np.loadtxt('custom_timesteps.txt', dtype=int)
            # init_timesteps = np.array(custom_timesteps[::-1])
            # init_timesteps_str = str(init_timesteps) # 去除多余的空格和换行符
            init_timesteps_list = init_timesteps.tolist()
            # 将列表转换为字符串
            init_timesteps_str = json.dumps(init_timesteps_list)
            print(init_timesteps_str)
            
            # 检查是否已经评估过这个时间步配置
            if not self.is_legal_before_search(init_timesteps_str):
                continue
            
            self.candidates.append(init_timesteps_str)
            step_num_candidates.append(step_num)  # 添加步数到候选列表
        
        # 确保候选集大小符合要求
        while len(self.candidates) < self.population_num:
            step_num = random.choice(time_step_range)

            init_timesteps = self.sampler.make_schedule(step_num,overwrite=True,return_flag=True)

            # custom_timesteps = np.loadtxt('custom_timesteps.txt', dtype=int)
            # init_timesteps = np.array(custom_timesteps[::-1])
            # init_timesteps_str = str(init_timesteps)
            # .replace('\n', '').replace(' ', '')
            init_timesteps_list = init_timesteps.tolist()
            # 将列表转换为字符串
            init_timesteps_str = json.dumps(init_timesteps_list)
            print(init_timesteps_str)
            
            if not self.is_legal_before_search(init_timesteps_str):
                continue
            
            self.candidates.append(init_timesteps_str)
            step_num_candidates.append(step_num)
        
        # 筛选出初始的较优时间步数
        logging.info('Initial screening ......')
        self.parallel_evaluate(self.candidates)
        current_candidates_with_sim = [(cand, self.vis_dict[cand]['similarity']) for cand in self.candidates if cand in self.vis_dict and 'similarity' in self.vis_dict[cand]]
        current_candidates, current_similarities = zip(*current_candidates_with_sim)
        self.update_top_k(current_candidates, k=self.select_num, key=lambda x: current_similarities[current_candidates.index(x)])
        top_step_nums = [step_num_candidates[current_candidates.index(cand)] for cand in self.keep_top_k[self.select_num]]
        
        # 选择相似度最高的3个步数作为下一步的搜索重点
        if len(top_step_nums) >= 3:
            selected_top_step_nums = sorted(top_step_nums[:3])
        else:
            selected_top_step_nums = sorted(top_step_nums)
        
        logging.info('Selected top step numbers for next stage: {}'.format(selected_top_step_nums))
        
        # 第二阶段，对选择出的较优步数进行更细致的搜索
        logging.info('Second stage search ......')
        self.epoch = 0
        while self.epoch < self.max_epochs:
            logging.info('epoch = {}'.format(self.epoch))
            
            # 基于选定的步数生成新的时间步配置
            for step_num in selected_top_step_nums:
                if self.opt.dpm_solver:
                    tmp_sampler = DPMSolverSampler(self.model)
                    from ldm.models.diffusion.dpm_solver.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
                    ns = NoiseScheduleVP('discrete', alphas_cumprod=tmp_sampler.alphas_cumprod)
                    dpm_solver = DPM_Solver(None, ns, predict_x0=True, thresholding=False)
                    skip_type = "time_uniform"
                    t_0 = 1. / dpm_solver.noise_schedule.total_N  
                    t_T = dpm_solver.noise_schedule.T  
                    init_timesteps = dpm_solver.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=step_num, device='cpu')
                    init_timesteps = [init_timesteps[i].item() for i in range(len(init_timesteps))]
                else:
                    init_timesteps = self.sampler.make_schedule(step_num)
                
                init_timesteps = sorted(init_timesteps)
                np.savetxt('custom_timesteps.txt', init_timesteps, fmt='%d')
                custom_timesteps = np.loadtxt('custom_timesteps.txt', dtype=int)
                init_timesteps = np.array(custom_timesteps[::-1])
                init_timesteps_str = str(init_timesteps).replace('\n', '').replace(' ', '')
                
                if not self.is_legal(init_timesteps_str):
                    continue
                
                self.candidates.append(init_timesteps_str)
            
            # 填充候选集到population_num的大小
            self.get_random(self.population_num)
            
            # 并行评估当前种群
            self.parallel_evaluate(self.candidates)
            
            # 更新 top k
            current_candidates_with_sim = [(cand, self.vis_dict[cand]['similarity']) for cand in self.candidates if cand in self.vis_dict and 'similarity' in self.vis_dict[cand]]
            current_candidates, current_similarities = zip(*current_candidates_with_sim)
            self.update_top_k(current_candidates, k=self.select_num, key=lambda x: current_similarities[current_candidates.index(x)])
            self.update_top_k(current_candidates, k=50, key=lambda x: current_similarities[current_candidates.index(x)])
            
            # 早停检查
            current_best = max([self.vis_dict[cand]['similarity'] for cand in self.keep_top_k[self.select_num] if cand in self.vis_dict and 'similarity' in self.vis_dict[cand]])
            self.best_history.append(current_best)
            
            if len(self.best_history) >= 2:
                if self.check_early_stop(current_best):
                    logging.info(f"Early stopping at epoch {self.epoch}")
                    break
            
            # 日志记录
            logging.info('epoch = {} : top {} result'.format(self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                clip_sim = self.vis_dict[cand].get('clip_similarity', 0.0)
                dino_sim = self.vis_dict[cand].get('dino_similarity', 0.0)
                combined_sim = self.vis_dict[cand].get('similarity', 0.0)
                logging.info(f'No.{i + 1} {cand} CLIP Similarity = {clip_sim:.4f}, DINO Similarity = {dino_sim:.4f}, Combined Similarity = {combined_sim:.4f}')
                logging.info(f'Time Steps: {eval(cand)}\n')
            
            self.epoch += 1
            
            # 每隔5个epoch保存一次中间结果
            if self.epoch % 5 == 0:  
                save_path = os.path.join(self.opt.outdir, f'search_epoch_{self.epoch}')
                os.makedirs(save_path, exist_ok=True)
                top_k = self.keep_top_k.get(50, self.candidates)  
                with open(os.path.join(save_path, 'optimized_time_steps.txt'), 'w') as f:
                    for i, cand in enumerate(top_k):
                        clip_sim = self.vis_dict[cand].get('clip_similarity', 0.0)
                        dino_sim = self.vis_dict[cand].get('dino_similarity', 0.0)
                        combined_sim = self.vis_dict[cand].get('similarity', 0.0)
                        f.write(f'No.{i + 1} {cand} CLIP Similarity = {clip_sim:.4f}, DINO Similarity = {dino_sim:.4f}, Combined Similarity = {combined_sim:.4f}\n')
                        f.write(f'Time Steps: {eval(cand)}\n\n')
        
        # 最后保存一次结果
        save_path = os.path.join(self.opt.outdir, 'search_final')
        os.makedirs(save_path, exist_ok=True)
        top_k = self.keep_top_k.get(50, self.candidates)  
        with open(os.path.join(save_path, 'optimized_time_steps.txt'), 'w') as f:
            for i, cand in enumerate(top_k):
                clip_sim = self.vis_dict[cand].get('clip_similarity', 0.0)
                dino_sim = self.vis_dict[cand].get('dino_similarity', 0.0)
                combined_sim = self.vis_dict[cand].get('similarity', 0.0)
                f.write(f'No.{i + 1} {cand} CLIP Similarity = {clip_sim:.4f}, DINO Similarity = {dino_sim:.4f}, Combined Similarity = {combined_sim:.4f}\n')
                f.write(f'Time Steps: {eval(cand)}\n\n')



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from_file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="path to data",
    )
    parser.add_argument(
        "--num_sample",
        type=int,
        default=4,
        help="samples num",
    )
    parser.add_argument(
        "--cal_fid",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--select_num",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--population_num",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--m_prob",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--crossover_num",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--mutation_num",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--max_fid",
        type=float,
        default=3.,
    )
    parser.add_argument(
        "--thres",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--ref_mu",
        type=str,
        default='',
    )
    parser.add_argument(
        "--ref_sigma",
        type=str,
        default='',
    )
    # parser.add_argument(
    #     "--time_step",
    #     type=int,
    #     default=50,
    # )
    parser.add_argument(
    "--time_step_range",
    type=str,
    default="30-50",
    help="range of time steps to search, format: min-max",
    )
    parser.add_argument(
        "--use_ddim_init_x",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--use_clip",
        action='store_true',
        default=True,
        help="Use CLIP similarity metric",
    )
    parser.add_argument(
        "--use_dino",
        action='store_true',
        default=True,
        help="Use DINO-ViT similarity metric",
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")  

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # if opt.dpm_solver:
    #     sampler = DPMSolverSampler(model)  
    # elif opt.plms:
    #     sampler = PLMSSampler(model)
    # else:
    sampler = DDIMSampler(model)

    dataloader_info = build_dataloader(config, opt)

    outpath = opt.outdir

    os.makedirs(outpath, exist_ok=True)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(outpath, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    batch_size = opt.n_samples


    dpm_params = None

    t = time.time()
    searcher = EvolutionSearcher(opt=opt, model=model, time_step_range=opt.time_step_range, ref_mu=opt.ref_mu, ref_sigma=opt.ref_sigma, sampler=sampler, dataloader_info=dataloader_info, batch_size=batch_size, dpm_params=dpm_params)
    searcher.search()
    logging.info('total searching time = {:.2f} hours'.format((time.time() - t) / 3600))

import os  

def is_image_file(filename):  
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg']  
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)  

def is_text_file(filename):  
    return filename.lower().endswith('.txt')  

from torch.utils.data import DataLoader
from fcdiffusion.dataset import TestDataset

def build_dataloader(config, opt):
    def traverse_images_and_texts(directory):  
        image_files = []  
        text_contents = []  
        for root, dirs, files in os.walk(directory):  
            for file in files:  
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.txt')):  
                    file_path = os.path.join(root, file)  
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  
                        image_files.append(file_path)  
                    elif file.lower().endswith('.txt'):  
                        with open(file_path, 'r', encoding='utf-8') as f:  
                            text_contents.append(f.read())  
        return image_files, text_contents  

    image_files, text_contents = traverse_images_and_texts(opt.data_dir)
    print(f"Found {len(image_files)} image files and {len(text_contents)} text files.")  

    all_datasets = []

    for image_file, text_content in zip(image_files, text_contents):  
        test_img_path, target_prompt = image_file, text_content
        dataset = TestDataset(test_img_path, target_prompt, 1)
        all_datasets.append(dataset)

    if not all_datasets:
        raise ValueError("No valid test data found in directory: {}".format(opt.data_dir))
    
    combined_dataset = torch.utils.data.ConcatDataset(all_datasets)
    validation_loader = DataLoader(
        combined_dataset,
        batch_size=opt.n_samples,
        shuffle=True,
        num_workers=4
    )
    print(f"Combined dataset length: {len(combined_dataset)}")  
    print(f"Validation loader batch size: {validation_loader.batch_size}")  

    return {'validation_loader': validation_loader}

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
def load_images_and_texts(processed_images, text_contents):
    image_text_pairs = []
    for image, text_content in zip(processed_images, text_contents):
        try:
            processed_image = preprocess(image) if hasattr(preprocess, '__call__') else image
            try:
                text = truncate_text(text_content)  
                image_text_pairs.append((processed_image, text))
            except Exception as e:
                print(f"Error truncating text: {e}")
                continue

        except Exception as e:
            print(f"Error processing pair (image, text): {e}")
            continue  
    return image_text_pairs

def calculate_clip_similarity_batch(images, texts):
    image_text_pairs = load_images_and_texts(images, texts)
    cosine_similarities = []
    with torch.no_grad():
        for img, text in image_text_pairs:
            img = img.unsqueeze(0).to(device)  
            image_features = model.encode_image(img)  
            image_features /= image_features.norm(dim=-1, keepdim=True)  

            text_tokens = clip.tokenize([text]).to(device)
            text_features = model.encode_text(text_tokens)  
            text_features /= text_features.norm(dim=-1, keepdim=True)  

            similarity = (image_features @ text_features.T).item()  
            cosine_similarities.append(similarity)
    
    return cosine_similarities

def compute_structure_similarity(images1, images2, dino_extractor):
    similarities = []
    for image1, image2 in zip(images1, images2):
        features1 = dino_extractor.extract_features(image1)
        features2 = dino_extractor.extract_features(image2)
        features1 = features1.mean(dim=1)  
        features2 = features2.mean(dim=1)  
        similarity = torch.cosine_similarity(features1, features2)
        similarities.append(similarity)
    return similarity  

def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    sd = {k[6:]: v for k, v in sd.items()  if  k.startswith("model.") }
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
    model.eval()
    return model

class StudentModelCheckpoint:
    pass

if __name__ == "__main__":
    main()