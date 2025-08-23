
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from fcdiffusion.dataset import TestDataset
from ldm.util import instantiate_from_config
import numpy as np
import os

# 导入你自定义的工具函数（需确保 fcdiffusion_reduce_model.py 中有对应实现）
from fcdiffusion_reduce_model import calculate_layer_influences, filter_layers_by_influence, read_text_from_file, parse_text
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from fcdiffusion.logger import ImageLogger
############################################
# 1. 模型加载与环境准备
############################################
def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 0 and verbose:
        print("missing keys:", missing)
    if len(unexpected) > 0 and verbose:
        print("unexpected keys:", unexpected)
    model.to(device)
    model.eval()
    return model, sd

def is_image_file(filename):
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg']
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)

def is_text_file(filename):
    return filename.lower().endswith('.txt')

def traverse_images_and_texts(directory):
    image_files = []
    text_contents = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if is_image_file(file):
                image_files.append(os.path.join(root, file))
            elif is_text_file(file):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    text_contents.append(content)
    return image_files, text_contents

############################################
# 2. 激活值采集
############################################
activation_values = {}
handles = []

def activation_hook(module, input, output, name):
    """记录模块的激活值并打印统计信息"""
    key = f"{name}"
    # 处理 output，确保其为 NumPy 数组
    if isinstance(output, tuple):  # 如果输出是元组，取第一个元素
        output = output[0]
    if isinstance(output, list):  # 如果输出是嵌套列表，取第一个子列表的第一个元素
        output = output[0][0] if isinstance(output[0], list) else output[0]
    
    # 如果 output 是张量，转换为 NumPy 数组
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    
    # 确保 output 是 NumPy 数组
    if not isinstance(output, np.ndarray):
        raise ValueError(f"Activation value for {key} is not a NumPy array: {type(output)}")

    # 存储激活值
    activation_values[key] = output

    # 打印激活值统计信息
    # print(f"激活值统计（{key}）：形状={output.shape}, 均值={np.mean(output):.4f}, 方差={np.std(output):.4f}")
    with open("activation_stats_low.txt", "a",encoding='utf-8') as f:
        f.write(f"激活值统计:{name},形状:{output.shape},均值:{np.mean(output):.4f},方差:{np.std(output):.4f}\n")

def create_hook(name):
    return lambda module, input, output: activation_hook(module, input, output, name)

def register_hooks(model):
    # 仅对 model.diffusion_model 内的模块注册钩子（主要关注卷积和残差块）
    if os.path.exists("activation_stats.txt"):
        os.remove("activation_stats.txt")
    # diffusion_model = model.model.diffusion_model
    
    for name, module in model.named_modules():
        # 根据实际情况，可以过滤出卷积层、ResBlock 等
        # if "conv" in name.lower() or "resblock" in name.lower():
            hook_name = f"{name}_{type(module).__name__}"
            # hook_name = f"{name}"
            # print(f"hook_name:{hook_name}")
            handle = module.register_forward_hook(create_hook(hook_name))
            handles.append(handle)

############################################
# 3. 构建剪枝后的 UNet 模块（仅剪枝 diffusion_model）
############################################


# def create_pruned_diffusion_model(original_diffusion_model, layers_to_keep):
#     """
#     根据激活统计信息筛选后，构造一个新的 UNet（diffusion_model），
#     仅保留名称在 layers_to_keep 列表中的层。
#     注意：layers_to_keep 中的名称为 "model.diffusion_model.xxx" 格式，
#     所以这里需要剥离前缀以匹配 diffusion_model 内部的子模块名称。
#     """
#     class PrunedDiffusionModel(nn.Module):
#         def __init__(self, original_diffusion_model, layers_to_keep):
#             super().__init__()
#             self.pruned_layers = nn.ModuleList()
#             self.layers_to_keep = layers_to_keep
#             self.build_pruned_model(original_diffusion_model)

#         def build_pruned_model(self, original_module):
#             for name, module in original_module.named_children():
#                 full_name = f"model.diffusion_model.{name}_{type(module).__name__}"
#                 # 检查当前模块是否在待保留列表中
#                 print(f"full_name:{full_name}")
#                 if any(layer.startswith(name) for layer in self.layers_to_keep):
#                     # 如果当前模块是叶子节点（即直接匹配），则添加到剪枝模型中

#                     if full_name in self.layers_to_keep:
#                         print("in keep")
#                         self.pruned_layers.append(module)
#                     else:
#                         # 如果当前模块不是叶子节点，则递归处理其子模块
#                         new_module = PrunedDiffusionModel(module, self.layers_to_keep)
#                         new_module.build_pruned_model(module)
#                         if len(new_module.pruned_layers) > 0:
#                             print("in keep children")
#                             self.pruned_layers.append(new_module)

#         def forward(self, x, c=None):
#             for layer in self.pruned_layers:
#                 x = layer(x)
#             return x

#     pruned_diffusion_model = PrunedDiffusionModel(original_diffusion_model, layers_to_keep)
#     return pruned_diffusion_model


# def create_pruned_diffusion_model(original_diffusion_model, layers_to_keep):
#     class PrunedDiffusionModel(nn.Module):
#         def __init__(self, parent_module, parent_path=""):
#             super().__init__()
#             self.layers = nn.ModuleDict()
#             self._build_pruned_structure(parent_module, parent_path)

#         def _build_pruned_structure(self, module, current_path):
#             """递归构建修剪后的模型结构"""
#             for name, child_module in module.named_children():
#                 # 构建完整路径（兼容两种格式）
#                 full_path = f"{current_path}.{name}" if current_path else name
#                 formatted_path_v1 = f"model.diffusion_model.{full_path}_{type(child_module).__name__}"
#                 # formatted_path_v2 = f"{full_path}_{type(child_module).__name__}"
#                 print(f"formatted_path_v1:{formatted_path_v1}")
#                 # 检查是否在保留列表中
#                 if any(layer in formatted_path_v1 for layer in layers_to_keep):
#                     print(f"child_module:{child_module}")
#                     self.layers[full_path] = child_module
#                     print(f"✅ 保留层: {full_path} (匹配到 {formatted_path_v1} )")
#                 else:
#                     # 递归处理子模块
#                     sub_module = PrunedDiffusionModel(child_module, full_path)
#                     if len(sub_module.layers) > 0:
#                         self.layers[full_path] = sub_module
#                         print(f"🔍 保留子树: {full_path}")

#         def forward(self, x, c=None):
#             for name, layer in self.layers.items():
#                 if isinstance(layer, PrunedDiffusionModel):
#                     x = layer(x, c)
#                 else:
#                     x = layer(x)
#             return x

#     # 初始化并打印最终结构
#     pruned_model = PrunedDiffusionModel(original_diffusion_model)
#     print("\n最终剪枝结构：")
#     for name, module in pruned_model.layers.items():
#         print(f"└─ {name} ({type(module).__name__})")
#     return pruned_model

# def create_pruned_diffusion_model(original_diffusion_model, layers_to_remove):
#     class PrunedDiffusionModel(nn.Module):
#         def __init__(self, parent_module, parent_path=""):
#             super().__init__()
#             self.layers = nn.ModuleDict()
#             self._build_pruned_structure(parent_module, parent_path.split("/"))

#         def _build_pruned_structure(self, module, path_parts):
#             """递归构建修剪后的模型结构"""
#             for name, child_module in module.named_children():
#                 # 构建当前层级路径
#                 current_path = "/".join(path_parts + [name])
#                 formatted_path = f"{current_path}_{type(child_module).__name__}"
#                 formatted_path = formatted_path[1::]
#                 print(f"formatted_path:{formatted_path}")
#                 # 检查是否在保留列表中
#                 if formatted_path in layers_to_keep:
#                     # 逐层创建子模块
#                     current = self
#                     for part in path_parts + [name]:
#                         if not hasattr(current, part):
#                             setattr(current, part, nn.Module())
#                         current = getattr(current, part)
#                     current.add_module(name, child_module)
#                     print(f"✅ 保留层: {current_path}")
#                 else:
#                     # 递归处理子模块
#                     sub_module = PrunedDiffusionModel(child_module, current_path)
#                     if len(sub_module.layers) > 0:
#                         self.layers[name] = sub_module

#         def forward(self, x, c=None):
#             for name, layer in self.layers.items():
#                 x = layer(x)
#             return x

#     return PrunedDiffusionModel(original_diffusion_model)
# def create_pruned_diffusion_model(original_diffusion_model, layers_to_keep):
#     class PrunedDiffusionModel(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.pruned_layers = nn.ModuleDict()
#             self.layers_to_keep = layers_to_keep
            
#             # 递归遍历所有子模块
#             for name, module in original_diffusion_model.named_modules():
#                 # 移除可能的父模块前缀（如 "model.diffusion_model."）
#                 clean_name = name.replace("model.diffusion_model.", "")
#                 if clean_name in self.layers_to_keep:
#                     # 动态构建模块路径
#                     parts = clean_name.split('.')
#                     current = self
#                     for part in parts[:-1]:
#                         if not hasattr(current, part):
#                             setattr(current, part, nn.Module())
#                         current = getattr(current, part)
#                     # 添加模块
#                     setattr(current, parts[-1], module)
#                     self.pruned_layers[clean_name] = module
            
#             print(f"剪枝后保留的层: {list(self.pruned_layers.keys())}")
#             if not self.pruned_layers:
#                 raise ValueError("剪枝后无有效层，请检查层筛选阈值或名称格式！")

#         def forward(self, x, c=None):
#             # 按原始顺序执行保留的层
#             for name in self.layers_to_keep:
#                 module = self.pruned_layers.get(name)
#                 if module is not None:
#                     x = module(x)
#             return x
    
#     return PrunedDiffusionModel()

# def create_pruned_diffusion_model(original_diffusion_model, layers_to_remove):
#     class PrunedDiffusionModel(nn.Module):
#         def __init__(self, parent_module, parent_path=""):
#             super().__init__()
#             self.layers = nn.ModuleDict()
#             self._build_pruned_structure(parent_module, parent_path.split("."))

#         def _build_pruned_structure(self, module, path_parts):
#             """递归构建修剪后的模型结构（移除模式）"""
#             for name, child_module in module.named_children():
#                 # 构建当前模块的完整路径
#                 current_path = ".".join(path_parts + [name])
#                 formatted_path = f"model.diffusion_model{current_path}_{type(child_module).__name__}"
                
#                 # 检查是否在移除列表中
#                 if formatted_path in layers_to_remove:
#                     print(f"🗑️ 移除层: {formatted_path}")
#                     continue  # 跳过当前模块及其所有子模块
                
#                 # 创建当前层级结构
#                 current = self
#                 for part in path_parts + [name]:
#                     if not hasattr(current, part):
#                         setattr(current, part, nn.Module())
#                     current = getattr(current, part)
#                 print(f"current:{current}")
                
#                 # 递归处理子模块
#                 if len(list(child_module.children())) > 0:
#                     sub_module = PrunedDiffusionModel(child_module, current_path)
#                     if len(sub_module.layers) > 0:
#                         current.add_module(name, sub_module)
#                 else:

#                     current.add_module(name, child_module)
#                     print(f"✅ 保留层: {formatted_path,name,child_module}")

#         def forward(self, x, c=None):
#             print(f"layers:{self.layers.items()}")
#             for name, layer in self.layers.items():
#                 x = layer(x)
#             return x

#     return PrunedDiffusionModel(original_diffusion_model)


# def create_pruned_diffusion_model(original_diffusion_model, layers_to_remove):
#     class PrunedDiffusionModel(nn.Module):
#         def __init__(self, parent_module, parent_path=""):
#             super().__init__()
#             self.layers = nn.ModuleDict()
#             self._build_pruned_structure(parent_module, parent_path.split("."))

#         def _build_pruned_structure(self, module, path_parts):
#             """优化后的递归构建逻辑"""
#             for name, child_module in module.named_children():
#                 # 构建完整路径（无类型后缀）
#                 current_path = ".".join(path_parts + [name])
#                 formatted_path = f"model.diffusion_model{current_path}"
                
#                 # 检查是否需要移除
#                 if formatted_path in layers_to_remove:
#                     print(f"🗑️ 移除层: {formatted_path}")
#                     continue
                
#                 # 创建层级结构
#                 current = self
#                 for part in path_parts + [name]:
#                     if not hasattr(current, part):
#                         # 动态添加模块容器
#                         container = nn.ModuleDict() if part.isdigit() else nn.Module()
#                         setattr(current, part, container)
#                     current = getattr(current, part)
                
#                 # 添加叶子节点
#                 if len(list(child_module.children())) == 0:
#                     if isinstance(current, nn.ModuleDict):
#                         current[name] = child_module
#                     else:
#                         setattr(current, name, child_module)
#                     print(f"✅ 保留层: {formatted_path}")
#                 else:
#                     # 递归处理子模块
#                     sub_module = PrunedDiffusionModel(child_module, current_path)
#                     if len(sub_module.layers) > 0:
#                         if isinstance(current, nn.ModuleDict):
#                             current[name] = sub_module
#                         else:
#                             setattr(current, name, sub_module)

#         def forward(self, x, c=None):
#             for name, layer in self.layers.items():
#                 x = layer(x)
#             return x

#     return PrunedDiffusionModel(original_diffusion_model)

# 在 create_pruned_diffusion_model 中添加调试信息和路径匹配修正
def create_pruned_diffusion_model(original_diffusion_model, layers_to_remove):
    class PrunedDiffusionModel(nn.Module):
        def __init__(self, parent_module, parent_path=""):
            super().__init__()
            self.layers = nn.ModuleDict()
            self._build_pruned_structure(parent_module, parent_path.split("."))

        def _build_pruned_structure(self, module, path_parts):
            """优化后的递归构建逻辑"""
            for name, child_module in module.named_children():
                # 构建完整路径（无类型后缀）
                current_path = ".".join(path_parts + [name])
                formatted_path = f"model.diffusion_model{current_path}"
                
                # 检查是否需要移除
                if formatted_path in layers_to_remove:
                    print(f"🗑️ 移除层: {formatted_path}")
                    continue
                
                # 创建层级结构
                current = self
                for part in path_parts + [name]:
                    if not hasattr(current, part):
                        # 动态添加模块容器
                        container = nn.ModuleDict() if part.isdigit() else nn.Module()
                        setattr(current, part, container)
                    current = getattr(current, part)
                
                # 添加叶子节点
                if len(list(child_module.children())) == 0:
                    if isinstance(current, nn.ModuleDict):
                        current[name] = child_module
                    else:
                        setattr(current, name, child_module)
                    print(f"✅ 保留层: {formatted_path}")
                else:
                    # 递归处理子模块
                    sub_module = PrunedDiffusionModel(child_module, current_path)
                    if len(sub_module.layers) > 0:
                        if isinstance(current, nn.ModuleDict):
                            current[name] = sub_module
                        else:
                            setattr(current, name, sub_module)

        def forward(self, x, c=None):
            for name, layer in self.layers.items():
                x = layer(x)
            return x

    return PrunedDiffusionModel(original_diffusion_model)


# def remove_layers_from_model(model, layers_to_remove):
#     """
#     直接在模型上移除指定的层
#     :param model: 原始模型
#     :param layers_to_remove: 需要移除的层路径列表（格式为 "model.diffusion_model.xxx"）
#     """
#     for name, module in model.named_modules():
#         # 移除前缀并处理路径格式
#         # clean_name = name.replace("model.diffusion_model.", "")
#         # print(f'layers_to_remove:{layers_to_remove}')
#         clean_name = name
#         print(f"clean_name:{clean_name}")
#         if clean_name in layers_to_remove:
#             # 获取父模块和层名称
#             parent_name, layer_name = os.path.split(name)
#             parent_module = model.get_submodule(parent_name)
#             print(f"parent_module:{parent_module,layer_name}")
#             # 检查父模块是否为 ModuleDict 或 ModuleList
#             if isinstance(parent_module, nn.ModuleDict):
#                 print('del module')
#                 del parent_module[layer_name]
#             elif isinstance(parent_module, nn.ModuleList):
#                 # 需要处理索引转换，比较复杂
#                 print('need to handle')
#                 pass
#             else:
#                 print('del common module')
#                 # 对于普通模块，直接删除属性
#                 delattr(parent_module, layer_name)
#             print(f"🗑️ 已移除层: {name}")


# def remove_layers_from_model(model, layers_to_remove):
#     """
#     直接在模型上移除指定的层
#     :param model: 原始模型
#     :param layers_to_remove: 需要移除的层路径列表（格式为 "model.diffusion_model.xxx"）
#     """
#     for target_path in layers_to_remove:
#         # 移除前缀并处理路径格式
#         clean_path = target_path.replace("model.diffusion_model.", "")
#         path_parts = clean_path.split('.')
        
#         # 遍历到目标模块的父模块
#         current_module = model
#         for part in path_parts[:-1]:
#             if hasattr(current_module, part):
#                 current_module = getattr(current_module, part)
#             else:
#                 print(f"⚠️ 跳过不存在的路径: {'.'.join(path_parts[:i+1])}")
#                 break
        
#         # 获取目标层名称
#         layer_name = path_parts[-1]
        
#         # 检查目标层是否存在
#         if hasattr(current_module, layer_name):
#             # 删除目标层
#             delattr(current_module, layer_name)
#             print(f"🗑️ 已移除层: {target_path}")
#         else:
#             print(f"⚠️ 层不存在，跳过: {target_path}")

def remove_layers_from_model(model, layers_to_remove):
    """
    移除指定层，并递归清理空父模块
    """
    removed_paths = set()  # 记录已移除的路径

    for target_path in layers_to_remove:
        # 处理路径格式
        clean_path = target_path.replace("model.diffusion_model.", "")
        path_parts = clean_path.split('.')
        
        # 遍历到目标模块的父模块
        parent_module = model
        for part in path_parts[:-1]:
            if hasattr(parent_module, part):
                parent_module = getattr(parent_module, part)
            else:
                break
        
        # 尝试删除目标层
        layer_name = path_parts[-1]
        if hasattr(parent_module, layer_name):
            delattr(parent_module, layer_name)
            removed_paths.add(target_path)
            print(f"🗑️ 已移除层: {target_path}")
            
            # 递归检查父模块是否为空
            current_module = parent_module
            for i in reversed(range(len(path_parts)-1)):
                module_part = path_parts[i]
                ancestor = model
                for p in path_parts[:i+1]:
                    ancestor = getattr(ancestor, p)
                
                # 如果当前模块没有子模块，则删除
                if len(list(ancestor.children())) == 0:
                    grandparent = model
                    for p in path_parts[:i]:
                        grandparent = getattr(grandparent, p)
                    delattr(grandparent, module_part)
                    removed_paths.add(".".join(path_parts[:i+1]))
                    print(f"🗑️ 移除空父模块: {'.'.join(path_parts[:i+1])}")

    # 二次验证确保移除成功
    for path in removed_paths:
        clean_path = path.replace("model.diffusion_model.", "")
        try:
            model.get_submodule(clean_path)
            print(f"⚠️ 警告: 层未被正确移除: {path}")
        except AttributeError:
            continue

############################################
# 5. 权重加载：仅加载剪枝 diffusion_model 的权重
############################################
# def load_pruned_diffusion_weights(pruned_diffusion_model, original_state_dict, original_diffusion_model):
#     new_state_dict = {}
#     for name, module in pruned_diffusion_model.named_modules():
#         try:
#             orig_module = original_diffusion_model.get_submodule(name)
#         except Exception:
#             continue
#         for param_name, param in module.named_parameters(recurse=False):
#             orig_key = f"{name}.{param_name}"
#             if orig_key in original_state_dict:
#                 new_state_dict[orig_key] = original_state_dict[orig_key]
#     pruned_diffusion_model.load_state_dict(new_state_dict, strict=False)
#     return pruned_diffusion_model
def load_pruned_diffusion_weights(pruned_model, original_state_dict):
    # 构建路径映射表
    path_mapping = {}
    for new_path, _ in pruned_model.named_modules():
        orig_path = new_path.replace("_", ".")  # 处理数字索引模块
        path_mapping[new_path] = f"model.diffusion_model.{orig_path}"
    
    # 加载权重
    new_state_dict = {}
    for new_path, param in pruned_model.named_parameters():
        orig_path = path_mapping.get(new_path, new_path)
        if orig_path in original_state_dict:
            new_state_dict[new_path] = original_state_dict[orig_path]
    
    pruned_model.load_state_dict(new_state_dict, strict=False)
    return pruned_model


def load_weights_after_pruning(model, original_state_dict):
    """
    剪枝后加载权重
    """
    model.load_state_dict(original_state_dict, strict=False)
    print("权重加载完成（非严格模式）")
    return model
############################################
# # 6. 微调训练（整个模型）
# ############################################

class PrunedFCDiffusionModel(pl.LightningModule):  # 改为继承LightningModule
    def __init__(self, original_model, pruned_diffusion_model, learning_rate=1e-5, sd_locked=True):
        super().__init__()
        # 保留原始模型引用
        self.original_model = original_model
        
        # 替换diffusion_model
        self.diffusion_model = pruned_diffusion_model
        self.control_model = original_model.control_model
        self.cond_stage_model = original_model.cond_stage_model
        self.first_stage_model = original_model.first_stage_model
        
        # 继承必要属性和参数
        self.learning_rate = learning_rate
        self.sd_locked = sd_locked
        self.first_stage_key = original_model.first_stage_key
        self.control_mode = original_model.control_mode
        self.ucg_training = original_model.ucg_training if hasattr(original_model, 'ucg_training') else {}
        self.use_scheduler = original_model.use_scheduler if hasattr(original_model, 'use_scheduler') else False

         # 确保剪枝模型的参数可训练
        for param in self.diffusion_model.parameters():
            param.requires_grad = True  # 强制解冻剪枝部分

    # def forward(self, x, c):
    #     return self.diffusion_model(x, c)
    # def forward(self, x, c, *args, **kwargs):
    #     print(f"self.shorten_cond_schedule:{self.shorten_cond_schedule}")
        
    #     t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
    #     if self.model.conditioning_key is not None:
    #         assert c is not None
    #         if self.cond_stage_trainable:
    #             c = self.get_learned_conditioning(c)
    #         if self.shorten_cond_schedule:  # TODO: drop this option
    #             tc = self.cond_ids[t].to(self.device)
    #             c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
    #     return self.p_losses(x, c, t, *args, **kwargs)


    def forward(self, x, c, *args, **kwargs):
        # 生成随机时间步 (与原始实现一致)
        t = torch.randint(
            0, self.original_model.num_timesteps, 
            (x.shape[0],), 
            device=self.device
        ).long()

        # 处理条件信息
        if self.original_model.cond_stage_trainable:
            c = self.original_model.get_learned_conditioning(c)

        # 调用核心损失计算
        return self.original_model.p_losses(x, c, t, *args, **kwargs)
    @torch.no_grad()
    def get_input(self, batch, k, *args, **kwargs):
        print(f"k:{k}")
        return self.original_model.get_input(batch, k, *args, **kwargs)

    def shared_step(self, batch):
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)
        return loss, {"loss": loss}


    def configure_optimizers(self):
        # 收集所有需要训练的参数
        trainable_params = []
        # 扩散模型参数（剪枝后）
        trainable_params += list(self.diffusion_model.parameters())
        # 其他模块参数（根据 sd_locked 决定）
        if not self.sd_locked:
            trainable_params += list(self.control_model.parameters())
            trainable_params += list(self.cond_stage_model.parameters())
            trainable_params += list(self.first_stage_model.parameters())
        
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        # 前向传播（确保输出保留梯度）
        # print(f'self.first_stage_key:{self.first_stage_key}')
        # x, c = self.get_input(batch, self.first_stage_key)
        # output = self(x, c)
        loss, loss_dict = self.shared_step(batch)
        # 计算损失（假设模型直接返回损失值）
        # loss = output
        self.log("train_loss", loss, prog_bar=True)
        return loss


# 在剪枝前添加格式处理
def clean_layer_names(layers):
    cleaned = []
    for layer in layers:
        # 移除类型后缀和多余前缀
        clean = layer.replace("model.diffusion_model.", "")
        clean = re.sub(r"_\w+$", "", clean)  # 移除_Conv2d等后缀
        clean = 'diffusion_model.' + clean
        cleaned.append(clean)
    return cleaned


############################################
# 7. 主流程
############################################
if __name__ == "__main__":
    # 加载配置与检查点
    yaml_file_path = "configs/model_config.yaml"
    # ckpt_file_path = r'D:\paper\FCDiffusion_code-main\lightning_logs\fcdiffusion_mid_pass_checkpoint\epoch=11-step=241999.ckpt'
    ckpt_file_path = r"D:\paper\FRE_FCD\lightning_logs\low\epoch=5-step=17999_SA.ckpt"
    config = OmegaConf.load(yaml_file_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sd = load_model_from_config(config, ckpt_file_path, device)
    
    # 注册钩子，仅在 diffusion_model 内采集激活数据
    register_hooks(model)
    
    # 遍历测试数据（图像与文本）
    directory_path = r'D:\paper\FCDiffusion_code-main\datasets\test'
    image_files, text_contents = traverse_images_and_texts(directory_path)
    test_res_num = 1
    dataset = TestDataset(image_files[0], text_contents[0], test_res_num)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False)
    
    # 前向传播采集激活值activation_stats
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # 假设 model.get_input 用于数据预处理（例如 "jpg" 图像）
            x, c = model.get_input(batch, 'jpg')
            _ = model(x, c)
    
    # 移除所有钩子，防止后续干扰
    for handle in handles:
        handle.remove()
    
    # 解析激活统计文件，计算每一层的影响力，并筛选出需要保留的层
    text = read_text_from_file("activation_stats_low.txt")
    layers = parse_text(text)
    layers_influences = calculate_layer_influences(layers, alpha=2.0, beta=1.0)
    # 这里设置的阈值可根据实际激活统计信息调整
    influence_threshold = 1.3
    layers_to_keep,prune_layers = filter_layers_by_influence(layers_influences, threshold=influence_threshold)
    # print(f"根据激活值筛选后，在 diffusion_model 中保留的层: {layers_to_keep}")
    
    # # 在调用 create_pruned_diffusion_model 前添加以下代码
    # cleaned_layers = []
    # for layer in prune_layers:
    #     # 移除前缀并替换分隔符
    #     clean_layer = layer.replace("model.diffusion_model.", "").replace(".", "/")
    #     # clean_layer = layer.replace(".", "/")
    #     # 添加类型后缀
    #     module_type = layer.split("_")[-1]
    #     cleaned_layers.append(f"{clean_layer}")
    # layers_to_remove = cleaned_layers
    # 构造剪枝后的 diffusion_model（UNet 模块）
    
    # prune_layers = clean_layer_names(prune_layers)
    # # print(f"layers_to_remove:{prune_layers}")
    # # pruned_diffusion_model = create_pruned_diffusion_model(model.model.diffusion_model, prune_layers)
    # print(f"剪枝前:{model.model.diffusion_model}")
    # remove_layers_from_model(model.model, prune_layers)
    # print(f"剪枝后:{model.model.diffusion_model}")
    # # print(f"pruned_diffusion_model:{pruned_diffusion_model}")
    # # pruned_diffusion_model = load_pruned_diffusion_weights(pruned_diffusion_model, c, model.model.diffusion_model)
    # pruned_diffusion_model = load_weights_after_pruning(model.model.diffusion_model,sd)
    # # 组装包含剪枝 diffusion_model 的完整 FCDiffusion 模型，其它模块保持原样
    # pruned_full_model = PrunedFCDiffusionModel(model, pruned_diffusion_model).to(device)
    

    # pruned_lightning_model = PrunedFCDiffusionModel(
    #     original_model=model,
    #     pruned_diffusion_model=pruned_diffusion_model,
    #     learning_rate=1e-5,
    #     sd_locked=True
    # ).to(device)

    # # 配置训练参数（与原训练脚本一致）
    # batch_size = 4
    # logger_freq = 500
    # val_every_n_train_steps = 1000
    # max_epochs = 100
    #     # 回调配置
    # logger = ImageLogger(root_path='pruned_logs', batch_frequency=logger_freq)
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath='pruned_checkpoints',
    #     every_n_train_steps=val_every_n_train_steps,
    #     save_top_k=-1
    # )
    
    # # 配置Trainer
    # trainer = pl.Trainer(
    #     gpus=1,
    #     precision=32,
    #     max_epochs=max_epochs,
    #     callbacks=[logger, checkpoint_callback],
    #     # enable_progress_bar=True
    # )
    
    # # 开始训练
    # trainer.fit(pruned_lightning_model, dataloader)
    # # 微调训练
    # # trained_pruned_model = fine_tune_pruned_model(pruned_full_model, dataloader, device, epochs=20)
    
    # # 保存最终模型（包括配置信息与剪枝层记录）
    # torch.save({
    #     'state_dict': pruned_lightning_model.state_dict(),
    #     'config': config,
    #     'pruned_layers': layers_to_keep
    # }, 'pruned_final_model.ckpt')
    
    # print("剪枝并微调后的模型已保存到 pruned_final_model.ckpt")
