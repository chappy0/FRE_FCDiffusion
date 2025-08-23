import sys
import os
import torch
from fcdiffusion.model import create_model

assert len(sys.argv) == 3, 'Args are wrong.'

input_path = sys.argv[1]
output_path = sys.argv[2]

assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

# run this script to initialize the entire FCDiffusion model.
# python tool_add_control_sd21.py ./models/v2-1_512-ema-pruned.ckpt ./models/FCDiffusion_ini.ckpt
# The FCNet in FCDiffusion is mainly initialized from the pretrained LDM UNet encoder, except for some newly added parameters.

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


model = create_model(config_path='configs/student_model_config.yaml')

pretrained_weights = torch.load(input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():

    is_control, name = get_node_name(k, 'control_')
    if is_control:
        copy_k = 'model.diffusion_' + name   # model.diffusion_model.input_blocks
    else:
        copy_k = k
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

# for key in list(target_dict.keys()):  # 使用 list() 避免遍历时修改字典
#     if "proj_in.weight" in key or "proj_out.weight" in key:  # 根据实际层名调整匹配规则
#         # 获取原始权重形状
#         original_weight = target_dict[key]
#         out_ch, in_ch = original_weight.shape[:2]
        
#         # 将 2D 权重转换为 4D（适用于 Conv2d 1x1）
#         target_dict[key] = original_weight.view(out_ch, in_ch, 1, 1)
        
model.load_state_dict(target_dict, strict=False)
torch.save(model.state_dict(), output_path)
print('Done.')
