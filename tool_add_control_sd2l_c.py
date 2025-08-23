import sys
import os
import torch
import torch.nn as nn
from fcdiffusion.model import create_model

def get_node_name(name, parent_name):
    """Check if the node name starts with the parent name and return the suffix."""
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

def initialize_fcdiffusion_weights(input_path, output_path):
    """Initialize FCDiffusion model weights from a pretrained SD model."""
    # Validate paths
    assert os.path.exists(input_path), 'Input model does not exist.'
    assert not os.path.exists(output_path), 'Output filename already exists.'
    assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

    # Create FCDiffusion model
    model = create_model(config_path='configs/model_config.yaml')

    # Load pretrained weights
    pretrained_weights = torch.load(input_path)
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']

    # Get scratch weights
    scratch_dict = model.state_dict()

    # Prepare target dictionary to hold migrated weights
    target_dict = {}

    # Mapping and migration logic
    for k_target in scratch_dict.keys():
        # Check if the weight belongs to the control model, diffusion model, VAE, or text encoder
        is_control, name_in_control = get_node_name(k_target, 'control_model.')  
        is_diffusion, name_in_diffusion = get_node_name(k_target, 'model.diffusion_model.')
        is_vae, name_in_vae = get_node_name(k_target, 'first_stage_model.')  
        is_text_encoder, name_in_text_encoder = get_node_name(k_target, 'cond_stage_model.')  

        source_key = None  

        if is_control:
            # Handle control model weights
            if k_target.endswith('.shared_kv.weight'):
                original_key = name_in_control.replace('.shared_kv.weight', '.to_k.weight')
                source_key = f'model.diffusion_model.{original_key}'
                print(f"Initializing control model weight '{k_target}' from SD's '{source_key}' (original to_k)")
            elif k_target.endswith('.to_k.weight') or k_target.endswith('.to_v.weight'):
                print(f"Skipping '{k_target}' as it's part of a shared KV projection in control model.")
                continue
            else:
                source_key = f'model.diffusion_model.{name_in_control}'
        elif is_diffusion:
            # Handle main UNet weights
            if '.shared_kv.weight' in k_target:
                original_key = k_target.replace('.shared_kv.weight', '.to_k.weight')
                source_key = original_key
                print(f"Initializing main UNet weight '{k_target}' from SD's '{source_key}' (original to_k)")
            elif '.to_k.weight' in k_target or '.to_v.weight' in k_target:
                print(f"Skipping '{k_target}' (main UNet) as it's part of a shared KV projection.")
                continue
            else:
                source_key = k_target
        elif is_vae:
            # Handle VAE weights
            if k_target.endswith('.shared_kv.weight'):
                original_key = name_in_vae.replace('.shared_kv.weight', '.to_k.weight')
                source_key = f'first_stage_model.{original_key}'
                print(f"Initializing VAE weight '{k_target}' from SD's '{source_key}' (original to_k)")
            elif k_target.endswith('.to_k.weight') or k_target.endswith('.to_v.weight'):
                print(f"Skipping '{k_target}' (VAE) as it's part of a shared KV projection.")
                continue
            else:
                source_key = k_target
        elif is_text_encoder:
            # Handle text encoder weights
            if k_target.endswith('.shared_kv.weight'):
                original_key = name_in_text_encoder.replace('.shared_kv.weight', '.to_k.weight')
                source_key = f'cond_stage_model.{original_key}'
                print(f"Initializing text encoder weight '{k_target}' from SD's '{source_key}' (original to_k)")
            elif k_target.endswith('.to_k.weight') or k_target.endswith('.to_v.weight'):
                print(f"Skipping '{k_target}' (text encoder) as it's part of a shared KV projection.")
                continue
            else:
                source_key = k_target
        else:
            # Handle other modules
            source_key = k_target

        # Migrate weights if source key exists and shapes match
        if source_key and source_key in pretrained_weights:
            if scratch_dict[k_target].shape == pretrained_weights[source_key].shape:
                target_dict[k_target] = pretrained_weights[source_key].clone()
                print(f"Copied weight from SD's '{source_key}' to FCDiffusion's '{k_target}'.")
            else:
                print(f"Shape mismatch for '{k_target}' ({scratch_dict[k_target].shape} vs {pretrained_weights[source_key].shape}). Using scratch weights.")
                target_dict[k_target] = scratch_dict[k_target].clone()
        else:
            print(f"Key '{source_key}' not found in SD weights or shape mismatch. Using scratch weights for '{k_target}'.")
            target_dict[k_target] = scratch_dict[k_target].clone()

    # Load target dictionary into the model
    model.load_state_dict(target_dict, strict=False)

    # Initialize unmatched weights
    for name, param in model.named_parameters():
        if name not in target_dict or torch.all(target_dict[name] == 0):
            if 'weight' in name:
                if 'norm' in name:
                    nn.init.ones_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    torch.save(model.state_dict(), output_path)
    print(f"Weights successfully saved to {output_path}")

if __name__ == "__main__":
    assert len(sys.argv) == 3, 'Usage: script.py <input_path> <output_path>'
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    initialize_fcdiffusion_weights(input_path, output_path)