# # 文件名: my_metrics.py
# # 作用: 封装 CLIP 和 DINO 相似度计算的逻辑

# import torch
# import clip
# from PIL import Image
# from torchvision import transforms

# # ==============================================================================
# # 1. CLIP 相似度计算模块 (基于 calculate_CLIP.py)
# # ==============================================================================

# class ClipSimilarity:
#     """一个封装了CLIP模型以计算图文相似度的类"""
#     def __init__(self, device="cuda"):
#         print("正在加载 CLIP 模型 (ViT-B/32)...")
#         self.device = device
#         # 加载CLIP模型和预处理器
#         self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
#         self.model.eval()
#         # 定义CLIP最大上下文长度
#         self.MAX_CONTEXT_LENGTH = 77
#         # 用于将Tensor转换为PIL Image
#         self.to_pil = transforms.ToPILImage()

#     def _truncate_text(self, text: str) -> str:
#         """确保文本长度在CLIP的token限制内"""
#         tokens = clip.tokenize([text], truncate=True) # 使用内置的truncate更高效
#         # 如果内置截断不够，可以换回您脚本中的循环截断逻辑
#         return clip.decode(tokens[0]).rstrip("endoftext")


#     @torch.no_grad()
#     def calculate(self, image_tensor: torch.Tensor, text: str) -> float:
#         """
#         计算单个图像张量和文本之间的CLIP相似度。
#         :param image_tensor: PyTorch张量, 形状 (C, H, W), 值范围 [0, 1].
#         :param text: 文本字符串.
#         :return: 相似度得分 (float).
#         """
#         # 将PyTorch Tensor转为PIL Image以进行预处理
#         # 确保张量在CPU上并且值范围是 [0, 1]
#         image_pil = self.to_pil(image_tensor.cpu())
        
#         # 预处理图像并增加batch维度
#         image_input = self.preprocess(image_pil).unsqueeze(0).to(self.device)
#         image_features = self.model.encode_image(image_input)
#         image_features /= image_features.norm(dim=-1, keepdim=True) # 归一化

#         # 预处理文本
#         # text = self._truncate_text(text)
#         text_tokens = clip.tokenize([text], truncate=True).to(self.device)
#         text_features = self.model.encode_text(text_tokens)
#         text_features /= text_features.norm(dim=-1, keepdim=True) # 归一化

#         # 计算余弦相似度
#         similarity = (image_features @ text_features.T).item()
#         return similarity

# # ==============================================================================
# # 2. DINO 结构相似度计算模块 (基于 calculate_DINO_VIT.py)
# # ==============================================================================

# class DinoVitSimilarity:
#     """一个封装了DINO-ViT模型以计算图像结构相似度的类"""
#     def __init__(self, model_name="./dino-vits16", device="cuda"):
#         from transformers import ViTFeatureExtractor, AutoModel
#         print(f"正在加载 DINO-ViT 模型 ({model_name})...")
#         # 您提供的脚本中路径是相对的，这里改为Hugging Face模型名，您可改回本地路径
#         # model_path = "../Super_FCD/dino-vits16/" 
#         model_path =  model_name 
#         self.device = device
#         # 加载DINO模型和特征提取器
#         self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)
#         self.model = AutoModel.from_pretrained(model_path).to(self.device)
#         self.model.eval()
#         self.to_pil = transforms.ToPILImage()

#     @torch.no_grad()
#     def _extract_features(self, image_pil: Image.Image) -> torch.Tensor:
#         """提取单个PIL图像的DINO特征"""
#         inputs = self.feature_extractor(images=image_pil, return_tensors="pt").to(self.device)
#         outputs = self.model(**inputs)
#         return outputs.last_hidden_state

#     @torch.no_grad()
#     def calculate(self, image_tensor1: torch.Tensor, image_tensor2: torch.Tensor) -> float:
#         """
#         计算两个图像张量之间的结构相似度
#         :param image_tensor1: PyTorch张量, 形状 (C, H, W), 值范围 [0, 1].
#         :param image_tensor2: PyTorch张量, 形状 (C, H, W), 值范围 [0, 1].
#         :return: 结构相似度得分 (float).
#         """
#         # 转换为PIL Image
#         image_pil1 = self.to_pil(image_tensor1.cpu())
#         image_pil2 = self.to_pil(image_tensor2.cpu())
        
#         # 提取特征
#         features1 = self._extract_features(image_pil1)
#         features2 = self._extract_features(image_pil2)

#         # 对token维度求平均
#         features1 = features1.mean(dim=1)
#         features2 = features2.mean(dim=1)

#         # 计算结构相似度
#         similarity = torch.cosine_similarity(features1, features2).item()
#         return similarity


# 文件名: my_metrics.py (修正版)
# 作用: 封装 CLIP 和 DINO 相似度计算的逻辑，并确保支持梯度反向传播

import torch
import clip
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# ==============================================================================
# 1. CLIP 相似度计算模块 (修正版)
# ==============================================================================

class ClipSimilarity:
    """一个封装了CLIP模型以计算图文相似度的类 (已修正以支持梯度)"""
    def __init__(self, device="cuda"):
        print("正在加载 CLIP 模型 (ViT-B/32)...")
        self.device = device
        self.model, self.pil_preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        # 【修正】创建直接在Tensor上操作的预处理流程，以保持计算图连续
        # CLIP的预处理器包含Resize, CenterCrop, 和 Normalize。我们在这里复现它们。
        # 从加载的预处理器中提取出关键参数 (分辨率, 均值, 标准差)
        clip_resolution = self.model.visual.input_resolution
        clip_mean = (0.48145466, 0.4578275, 0.42962372)
        clip_std = (0.26862954, 0.26130258, 0.27577711)

        self.tensor_preprocess = transforms.Compose([
            transforms.Resize(clip_resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(clip_resolution),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ])
        print("CLIP 模块已准备好进行可微调的相似度计算。")

    # 【修正】移除了 @torch.no_grad() 装饰器，以允许梯度计算
    def calculate(self, image_tensor: torch.Tensor, text: str) -> torch.Tensor:
        """
        计算单个图像张量和文本之间的CLIP相似度。
        此函数现在是可微的。
        :param image_tensor: PyTorch张量, 形状 (C, H, W) 或 (B, C, H, W), 值范围 [0, 1].
        :param text: 文本字符串.
        :return: 相似度得分 (标量 torch.Tensor).
        """
        # 如果输入是单个图片 (C,H,W)，增加一个批次维度
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
            
        # 【修正】直接在Tensor上进行预处理，不转换为PIL Image
        image_input = self.tensor_preprocess(image_tensor).to(self.device)
        image_features = self.model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 文本处理部分保持不变，因为它已经是基于Tensor的
        text_tokens = clip.tokenize([text], truncate=True).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 【修正】计算相似度并返回Tensor，而不是 .item() 返回的float
        # 乘以100是CLIP的惯例，用于匹配其训练方式
        similarity = (image_features @ text_features.T) * 100
        return similarity.squeeze() # 从 (1,1) 或 (1,) 压缩为标量Tensor

# ==============================================================================
# 2. DINO 结构相似度计算模块 (修正版)
# ==============================================================================

# class DinoVitSimilarity:
#     """一个封装了DINO-ViT模型以计算图像结构相似度的类 (已修正以支持梯度)"""
#     def __init__(self, model_name="./dino-vits16", device="cuda"):
#         from transformers import ViTFeatureExtractor, AutoModel
#         print(f"正在加载 DINO-ViT 模型 ({model_name})...")
#         self.device = device
        
#         # 加载模型
#         self.model = AutoModel.from_pretrained(model_name).to(self.device)
#         self.model.eval()

#         # 【修正】创建直接在Tensor上操作的预处理流程
#         # 从HuggingFace加载的特征提取器用于获取参数
#         feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
#         self.tensor_preprocess = transforms.Compose([
#             transforms.Resize(tuple(feature_extractor.size.values()), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
#             transforms.CenterCrop(tuple(feature_extractor.size.values())),
#             transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
#         ])
#         print("DINO-ViT 模块已准备好进行可微调的相似度计算。")

#     # 【修正】移除了 @torch.no_grad()
#     def _extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
#         """提取单个图像张量的DINO特征"""
#         # 【修正】对Tensor进行预处理
#         processed_tensor = self.tensor_preprocess(image_tensor).to(self.device)
#         # 【修正】直接将处理后的Tensor喂给模型
#         outputs = self.model(pixel_values=processed_tensor)
#         return outputs.last_hidden_state

#     # 【修正】移除了 @torch.no_grad()
#     def calculate(self, image_tensor1: torch.Tensor, image_tensor2: torch.Tensor) -> torch.Tensor:
#         """
#         计算两个图像张量之间的结构相似度。此函数现在是可微的。
#         :param image_tensor1: PyTorch张量, 形状 (C, H, W) 或 (B, C, H, W), 值范围 [0, 1].
#         :param image_tensor2: PyTorch张量, 形状 (C, H, W) 或 (B, C, H, W), 值范围 [0, 1].
#         :return: 结构相似度得分 (标量 torch.Tensor).
#         """
#         # 确保输入有批次维度
#         if image_tensor1.dim() == 3:
#             image_tensor1 = image_tensor1.unsqueeze(0)
#         if image_tensor2.dim() == 3:
#             image_tensor2 = image_tensor2.unsqueeze(0)

#         # 【修正】不再转换为PIL Image，直接传递Tensor
#         features1 = self._extract_features(image_tensor1)
#         features2 = self._extract_features(image_tensor2)

#         # 对token维度求平均
#         features1 = features1.mean(dim=1)
#         features2 = features2.mean(dim=1)

#         # 【修正】计算相似度并返回Tensor
#         similarity = F.cosine_similarity(features1, features2)
#         return similarity.squeeze()


# ==============================================================================
# 2. DINO 结构相似度计算模块 (再次修正版)
# ==============================================================================

class DinoVitSimilarity:
    """一个封装了DINO-ViT模型以计算图像结构相似度的类 (已修正以支持梯度)"""
    def __init__(self, model_name="./dino-vits16", device="cuda"):
        from transformers import ViTFeatureExtractor, AutoModel
        print(f"正在加载 DINO-ViT 模型 ({model_name})...")
        self.device = device
        
        # 加载模型
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # 【修正】创建直接在Tensor上操作的预处理流程
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        
        # 【关键修正】更鲁棒地获取图像尺寸
        # 旧代码假定 feature_extractor.size 是一个字典，但它可能是一个整数。
        size_config = feature_extractor.size
        if isinstance(size_config, dict):
            # 如果是字典，取 'shortest_edge' 的值
            target_size = size_config['shortest_edge']
        else:
            # 如果是整数，直接使用
            target_size = size_config
            
        self.tensor_preprocess = transforms.Compose([
            # 使用修正后的 target_size
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(target_size),
            transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
        ])
        print("DINO-ViT 模块已准备好进行可微调的相似度计算。")

    # 【修正】移除了 @torch.no_grad()
    def _extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """提取单个图像张量的DINO特征"""
        # 【修正】对Tensor进行预处理
        processed_tensor = self.tensor_preprocess(image_tensor).to(self.device)
        # 【修正】直接将处理后的Tensor喂给模型
        outputs = self.model(pixel_values=processed_tensor)
        return outputs.last_hidden_state

    # 【修正】移除了 @torch.no_grad()
    def calculate(self, image_tensor1: torch.Tensor, image_tensor2: torch.Tensor) -> torch.Tensor:
        """
        计算两个图像张量之间的结构相似度。此函数现在是可微的。
        :param image_tensor1: PyTorch张量, 形状 (C, H, W) 或 (B, C, H, W), 值范围 [0, 1].
        :param image_tensor2: PyTorch张量, 形状 (C, H, W) 或 (B, C, H, W), 值范围 [0, 1].
        :return: 结构相似度得分 (标量 torch.Tensor).
        """
        # 确保输入有批次维度
        if image_tensor1.dim() == 3:
            image_tensor1 = image_tensor1.unsqueeze(0)
        if image_tensor2.dim() == 3:
            image_tensor2 = image_tensor2.unsqueeze(0)

        # 【修正】不再转换为PIL Image，直接传递Tensor
        features1 = self._extract_features(image_tensor1)
        features2 = self._extract_features(image_tensor2)

        # 对token维度求平均
        features1 = features1.mean(dim=1)
        features2 = features2.mean(dim=1)

        # 【修正】计算相似度并返回Tensor
        similarity = F.cosine_similarity(features1, features2)
        return similarity.squeeze()