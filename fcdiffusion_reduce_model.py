# #fcdiffusion_reduce_model.py
# import re
# from collections import namedtuple

# # 定义一个数据结构来存储每一层的信息
# LayerStats = namedtuple("LayerStats", ["name", "shape", "mean", "variance"])

# # 从文件中读取文本
# def read_text_from_file(file_path):
#     with open(file_path, "r", encoding="utf-8") as file:
#         text = file.read()
#     return text

# # 解析文本
# def parse_text(text):
#     layers = []
#     # 更灵活的正则表达式，允许多余的空格和不同的括号
#     pattern = re.compile(r"激活值统计\s*[\(（](.*?)[\)）]\s*：\s*形状\s*=\s*[\(（](.*?)[\)）]\s*,\s*均值\s*=\s*([-+]?\d*\.\d+|\d+)\s*,\s*方差\s*=\s*([-+]?\d*\.\d+|\d+)\s*")
#     matches = pattern.findall(text)

#     if not matches:
#         print("正则表达式未匹配到任何内容，请检查输入文件格式！")
#         return layers

#     for match in matches:
#         name, shape, mean, variance = match
#         layers.append(LayerStats(name, shape, float(mean), float(variance)))

#     return layers

# # # 计算影响力
# def calculate_layer_influences(layers, alpha=2.0, beta=1.0):
#     influences = []
#     for layer in layers:
#         influence = alpha * abs(layer.mean) + beta * layer.variance
#         influences.append((layer.name, influence))
#     return influences

# # def calculate_layer_influences(model, alpha=1.0, beta=1.0):
# #     layers = model.layers  # 假设模型有一个layers属性，里面包含了所有的层
# #     influences = []
# #     for layer in layers:
# #         mean = layer.mean  # 假设每个层有mean和variance属性
# #         variance = layer.variance
# #         influence = alpha * abs(mean) + beta * variance
# #         influences.append((layer, influence))
# #     return influences


# def filter_layers_by_influence(layers_influences, threshold=0.1):
#     # 筛选影响力大于阈值的层
#     selected_layers = [layer for layer, influence in layers_influences if influence > threshold]
#     return selected_layers




# def update_model_with_filtered_layers(model, selected_layers):
#     """
#     更新模型结构，仅保留筛选后的层。
#     同时打印出更新前后的模型结构，确保更新正确。
#     """
#     print(f"\n更新模型前的结构：{len(model.layers)}")
#     # for layer in model.layers:
#     #     print(f"  - {layer.name} (mean={layer.mean}, variance={layer.variance})")

#     # 更新模型的层
#     model.layers = selected_layers

#     print(f"\n更新模型后的结构：{len(model.layers)}")
#     # for layer in model.layers:
#     #     print(f"  - {layer.name} (mean={layer.mean}, variance={layer.variance})")

#     return model
# # # 计算影响力并筛选层
# # layers_influences = calculate_layer_influences(model, alpha=2.0, beta=1.0)
# # selected_layers = filter_layers_by_influence(layers_influences, threshold=0.1)

# # # 更新模型结构
# # update_model_with_filtered_layers(model, selected_layers)

# # 排序并生成新的文本格式
# def generate_sorted_text(sorted_influences):
#     sorted_text = "按影响力从低到高排序的结果：\n"
#     for name, influence in sorted_influences:
#         sorted_text += f"{name}: 影响力 = {influence:.4f}\n"
#     return sorted_text



# # 主函数
# def main(input_file_path, output_file_path, alpha=2.0, beta=1.0):
#     # 从文件读取文本
#     text = read_text_from_file(input_file_path)

#     # 解析文本
#     layers = parse_text(text)

#     if not layers:
#         print("未解析到任何层信息，请检查输入文件内容！")
#         return

#     # 计算影响力并排序
#     influences = calculate_influence(layers, alpha, beta)
#     sorted_influences = sorted(influences, key=lambda x: x[1])

#     # 生成排序后的文本
#     sorted_text = generate_sorted_text(sorted_influences)

#     # 将结果写入新文件
#     with open(output_file_path, "w", encoding="utf-8") as file:
#         file.write(sorted_text)

#     print(f"排序结果已保存到文件：{output_file_path}")

# # 示例调用
# if __name__ == "__main__":
#     input_file = "activation_stats.txt"  # 输入文件路径
#     output_file = "act_output2_1.txt"  # 输出文件路径
#     main(input_file, output_file, alpha=2.0, beta=1.0)


# fcdiffusion_reduce_model.py
import re
from collections import namedtuple

import numpy as np

# 定义一个数据结构来存储每一层的信息
LayerStats = namedtuple("LayerStats", ["name", "shape", "mean", "variance"])

# 从文件中读取文本
def read_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

# 解析文本
# def parse_text(text):
#     layers = []
#     # 更灵活的正则表达式，允许多余的空格和不同的括号
#     pattern = re.compile(r"激活值统计\s*[\(（](.*?)[\)）]\s*：\s*形状\s*=\s*[\(（](.*?)[\)）]\s*,\s*均值\s*=\s*([-+]?\d*\.\d+|\d+)\s*,\s*方差\s*=\s*([-+]?\d*\.\d+|\d+)\s*")
#     matches = pattern.findall(text)

#     if not matches:
#         print("正则表达式未匹配到任何内容，请检查输入文件格式！")
#         return layers

#     for match in matches:
#         name, shape, mean, variance = match
#         layers.append(LayerStats(name, shape, float(mean), float(variance)))

#     print(f"layers:{layers}")
#     return layers

import re

# 定义一个类来存储层的统计信息
# class LayerStats:
#     def __init__(self, name, shape, mean, variance):
#         self.name = name
#         self.shape = shape
#         self.mean = mean
#         self.variance = variance

# def parse_text(text):
#     layers = []
#     # 调整正则表达式以更准确地匹配层名称和统计信息
#     pattern = re.compile(
#         r"激活值统计\s*[\(（](.*?)[\)）]\s*：\s*形状\s*=\s*[\(（](.*?)[\)）]\s*,\s*均值\s*=\s*([-+]?\d*\.\d+|\d+)\s*,\s*方差\s*=\s*([-+]?\d*\.\d+|\d+)\s*")
#     matches = pattern.findall(text)

#     if not matches:
#         print("正则表达式未匹配到任何内容，请检查输入文件格式！")
#         return layers

#     for match in matches:
#         name, shape, mean, variance = match
#         print(f"name:{name}")
#         # 清理层名称中的多余空格
#         name = name.strip()
#         layers.append(LayerStats(name, shape, float(mean), float(variance)))
#     print(f"layers:{layers}")
#     return layers

import re
from collections import namedtuple

# 定义一个namedtuple来存储层的统计信息
LayerStats = namedtuple("LayerStats", ["name", "shape", "mean", "variance"])

def parse_text(text):
    layers = []
    print("start parse")
    # 使用更简单的正则表达式匹配每一行
    pattern = re.compile(
        r"激活值统计:(.*),形状:(.*),均值:(.*),方差:(.*)"
    )
    matches = pattern.finditer(text)
    print(f"matches:{matches}")
    if not matches:
        print("正则表达式未匹配到任何内容，请检查输入文件格式！")
        return layers
    
    

    for match in matches:
            name = match.group(1)
            shape = match.group(2)
            mean = match.group(3)
            variance = match.group(4)
            # 清理层名称中的多余空格
            # print(f"name:{name}")
            # f.write(name)
            name = name.strip()
            # print(f"name:{name}")
            layers.append(LayerStats(name, shape, float(mean), float(variance)))
    # with open("match_layers.txt",'w') as f:
    #     f.write(layers)
    # print(f"layers:{layers}")
    return layers


# # 计算每一层的影响力
# def calculate_layer_influences(layers, alpha=2.0, beta=1.0):
#     influences = []
#     for layer in layers:
#         influence = alpha * abs(layer.mean) + beta * layer.variance
#         influences.append((layer.name, influence))
#     with open("all_influence_layers.txt", "w",encoding='utf-8') as f:
#         f.write(f"{influences}")
    
#     return influences

def calculate_layer_influences(layers, alpha=1.0, beta=1.0):
    # 计算全局均值和方差
    all_means = [layer.mean for layer in layers]
    all_vars = [layer.variance for layer in layers]
    global_mean = np.mean(all_means)
    global_var = np.mean(all_vars)
    
    # 标准化并加权
    # layer_influences = {}
    layer_influences = []
    for layer in layers:
        norm_mean = layer.mean / global_mean
        norm_var = layer.variance / global_var
        influence = alpha * norm_mean + beta * norm_var
        # layer_influences[layer.name] = influence
        layer_influences.append((layer.name, influence))
    with open("all_influence_layers.txt", "w",encoding='utf-8') as f:
        f.write(f"{layer_influences}")
    return layer_influences



def filter_layers(influences, method='topk', k=0.5, n_std=1.0):
    scores = list(influences.values())
    if method == 'topk':
        threshold = np.percentile(scores, (1 - k) * 100)
    elif method == 'std':
        mean = np.mean(scores)
        std = np.std(scores)
        threshold = mean + n_std * std
    return {name: score for name, score in influences.items() if score >= threshold}
# # 筛选影响力大于阈值的层
# def filter_layers_by_influence(layers_influences, threshold=0.1):
#     selected_layers = [layer_name for layer_name, influence in layers_influences if influence > threshold 
#                        and "diffusion_model" in layer_name and ('Conv2D' or 'Resblock') in layer_name ]
#     with open("selected_layers.txt", "w",encoding='utf-8') as f:
#         f.write(f"{selected_layers}")
#     return selected_layers


# # 计算每一层的影响力 
# def calculate_layer_influences(layers, alpha=2.0, beta=1.0):
#     """
#     layers: 列表，每个元素应包含属性：
#        - name: 层的名称（可以包含前缀）
#        - mean: 激活值均值
#        - variance: 激活值方差
#     """
#     influences = []
#     for layer in layers:
#         if hasattr(layer, "mean") and hasattr(layer, "variance"):
#             influence = alpha * abs(layer.mean) + beta * layer.variance
#             influences.append((layer.name, influence))
#     influences.sort(key=lambda x: x[1], reverse=True)
#     with open("all_influence_layers.txt", "w", encoding="utf-8") as f:
#         f.write(f"{influences}\n")
#     return influences

# 筛选出剪枝时需要保留和剪枝的层
def filter_layers_by_influence(layers_influences, threshold=1):
    # candidate_keywords = ["conv2d", "resblock", "linear"]
    candidate_keywords = ["conv2d", "resblock"]
    # 仅保留属于 diffusion_model 的层（过滤条件：名称中含有 'diffusion_model'）
    diffusion_layers = [
        (name, influence) for name, influence in layers_influences 
        if "diffusion_model" in name.lower()
    ]
    # print(f"diffusion_layers:{diffusion_layers}")
    
    # 在 diffusion_model 内的层中，候选层：名称中包含候选关键字
    candidate_layers = [
        layer_name for layer_name, _ in diffusion_layers 
        if any(keyword in layer_name.lower() for keyword in candidate_keywords)
    ]
    
    # print(f"candidate_layers:{candidate_layers}")
    # 保留层：候选层中影响力大于阈值的层
    keep_layers = [
        layer_name for layer_name, influence in diffusion_layers 
        if abs(influence) > threshold and any(keyword in layer_name.lower() for keyword in candidate_keywords)
    ]
    
    # 剪枝层：候选层中不在保留层内的部分
    prune_layers = [layer for layer in candidate_layers if layer not in keep_layers]
    
    with open("keep_layers.txt", "w", encoding="utf-8") as f:
        f.write(f"{keep_layers}")
    with open("prune_layers.txt", "w", encoding="utf-8") as f:
        f.write(f"{prune_layers}")
    
    return keep_layers, prune_layers


# 更新模型结构，仅保留筛选后的层
def update_model_with_filtered_layers(model, selected_layers):
    """
    更新模型结构，仅保留筛选后的层。
    同时打印出更新前后的模型结构，确保更新正确。
    """
    print(f"\n更新模型前的结构：{len(model.layers)}层")
    original_layers = model.layers
    model.layers = [layer for layer in original_layers if layer.name in selected_layers]

    print(f"\n更新模型后的结构：{len(model.layers)}层")
    return model

# 排序并生成新的文本格式
def generate_sorted_text(sorted_influences):
    sorted_text = "按影响力从低到高排序的结果：\n"
    for name, influence in sorted_influences:
        sorted_text += f"{name}: 影响力 = {influence:.4f}\n"
    return sorted_text

# 主函数
def main(input_file_path, model):
    """
    主函数，从文本文件中读取层次信息一次，计算影响力，筛选层次，并更新模型结构。
    """
    # 从文件读取文本
    text = read_text_from_file(input_file_path)

    # 解析文本
    layers = parse_text(text)

    if not layers:
        print("未解析到任何层信息，请检查输入文件内容！")
        return

    # 计算每一层的影响力
    layers_influences = calculate_layer_influences(layers, alpha=1.0, beta=1.0)
    sorted_influences = sorted(layers_influences, key=lambda x: x[1])

    # 筛选影响力大于阈值的层
    # selected_layers = filter_layers_by_influence(layers_influences, threshold=0.1)

    # # 更新模型结构
    # updated_model = update_model_with_filtered_layers(model, selected_layers)

    # # 生成排序后的文本
    # sorted_text = generate_sorted_text(sorted_influences)

    # # 打印排序结果
    # print(sorted_text)

    # return updated_model


# # 定义学生模型类
# class StudentModel(nn.Module):
#     def __init__(self, original_model, layers_to_keep):
#         super(StudentModel, self).__init__()
#         self.layers = nn.ModuleList()
#         self.layer_names = []  # 用于记录保留的层名称
#         self.original_model = original_model  # 保留原始模型的引用

#         # 遍历原始模型的模块，提取需要保留的层
#         for name, module in original_model.named_modules():
#             if name in layers_to_keep:
#                 self.layers.append(module)
#                 self.layer_names.append(name)
#         print(f"保留的层: {self.layer_names}")

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x

#     def load_weights_from_original_model(self):
#         """从原始模型中加载权重"""
#         for student_layer, original_layer_name in zip(self.layers, self.layer_names):
#             original_layer = self.original_model.get_submodule(original_layer_name)
#             student_layer.load_state_dict(original_layer.state_dict())


def create_pruned_diffusion_model(original_diffusion_model, layers_to_keep):
    class PrunedDiffusionModel(nn.Module):
        def __init__(self, parent_module, parent_path=""):
            super().__init__()
            self.layers = nn.ModuleDict()
            self._build_pruned_structure(parent_module, parent_path)

        def _build_pruned_structure(self, module, current_path):
            """递归构建修剪后的模型结构"""
            for name, child_module in module.named_children():
                # 构建完整路径（兼容两种格式）
                full_path = f"{current_path}.{name}" if current_path else name
                formatted_path_v1 = f"model.diffusion_model.{full_path}_{type(child_module).__name__}"
                # formatted_path_v2 = f"{full_path}_{type(child_module).__name__}"

                # 检查是否在保留列表中
                if any(layer in formatted_path_v1 for layer in layers_to_keep):
                    self.layers[full_path] = child_module
                    print(f"✅ 保留层: {full_path} (匹配到 {formatted_path_v1} )")
                else:
                    # 递归处理子模块
                    sub_module = PrunedDiffusionModel(child_module, full_path)
                    if len(sub_module.layers) > 0:
                        self.layers[full_path] = sub_module
                        print(f"🔍 保留子树: {full_path}")

        def forward(self, x, c=None):
            for name, layer in self.layers.items():
                if isinstance(layer, PrunedDiffusionModel):
                    x = layer(x, c)
                else:
                    x = layer(x)
            return x

    # 初始化并打印最终结构
    pruned_model = PrunedDiffusionModel(original_diffusion_model)
    print("\n最终剪枝结构：")
    for name, module in pruned_model.layers.items():
        print(f"└─ {name} ({type(module).__name__})")
    return pruned_model

# 示例调用
if __name__ == "__main__":
        # 筛选重要层
    influence_threshold = 0.1  # 设置影响力值的阈值
    input_file = "activation_stats.txt"  # 输入文件路径
    
    text = read_text_from_file(input_file)
    
    # 解析文本
    layers = parse_text(text)
    layers_influences = calculate_layer_influences(layers, alpha=1.0, beta=1.0)
    
    layers_to_keep = filter_layers_by_influence(layers_influences, threshold=1.3)
    
    
    # # 构建学生模型
    # student_model = StudentModel(model, layers_to_keep).to(device)
    
    # # 从原始模型中加载权重
    # student_model.load_weights_from_original_model()
    # # 保存学生模型的权重
    # student_ckpt_path = "student_model_checkpoint.ckpt"
    # torch.save(student_model.state_dict(), student_ckpt_path)
    # print(f"学生模型已保存到 {student_ckpt_path}")
    