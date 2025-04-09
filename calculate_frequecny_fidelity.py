import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tools.dct_util import dct_2d, high_pass, idct_2d, low_pass, low_pass_and_shuffle

def compute_frequency_fidelity(generated_image, target_image, control_mode='low_pass'):
    """
    Compute the frequency fidelity (MSE) for each frequency band: low, mini, mid, and high.
    :param generated_image: The generated image from the student model.
    :param target_image: The target image for comparison.
    :param control_mode: The control mode for frequency band extraction ('low_pass', 'mini_pass', 'mid_pass', 'high_pass').
    :return: Fidelity for each frequency band.
    """
    # 将图像移动到 GPU 上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generated_image = generated_image.to(device)
    target_image = target_image.to(device)
    
    # Apply DCT to both generated and target images
    generated_dct = dct_2d(generated_image.squeeze(0), norm='ortho')
    target_dct = dct_2d(target_image.squeeze(0), norm='ortho')
    
    # Extract frequency bands from both generated and target images
    generated_bands = extract_frequency_bands(generated_dct.unsqueeze(0), control_mode)
    target_bands = extract_frequency_bands(target_dct.unsqueeze(0), control_mode)
    
    # Compute the fidelity (MSE) for each frequency band
    fidelity = {}
    for band in ['low', 'mini', 'mid', 'high']:
        generated_band = generated_bands.get(band, None)
        target_band = target_bands.get(band, None)
        
        if generated_band is not None and target_band is not None:
            fidelity[band] = F.mse_loss(generated_band, target_band)
    
    return fidelity

def extract_frequency_bands(z0_dct, control_mode='low_pass'):
    """
    Extracts frequency bands (low, mini, mid, high) based on the specified control mode.
    :param z0_dct: The DCT of the input image.
    :param control_mode: The control mode to apply ('low_pass', 'mini_pass', 'mid_pass', 'high_pass').
    :return: Extracted frequency bands as a dictionary.
    """
    frequency_bands = {}

    # 确保输入张量在 GPU 上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z0_dct = z0_dct.to(device)
    
    # Low-frequency (LF): Low-pass filter
    if control_mode == 'low_pass':
        z0_dct_filter = low_pass(z0_dct, 30)
        frequency_bands['low'] = idct_2d(z0_dct_filter.squeeze(0), norm='ortho')

    # Mini-frequency (MF): Low-pass and shuffle filter
    elif control_mode == 'mini_pass':
        z0_dct_filter = low_pass_and_shuffle(z0_dct, 10)
        frequency_bands['mini'] = idct_2d(z0_dct_filter.squeeze(0), norm='ortho')

    # Mid-frequency (MF): High-pass after low-pass filter
    elif control_mode == 'mid_pass':
        z0_dct_filter = high_pass(low_pass(z0_dct, 40), 20)
        frequency_bands['mid'] = idct_2d(z0_dct_filter.squeeze(0), norm='ortho')

    # High-frequency (HF): High-pass filter
    elif control_mode == 'high_pass':
        z0_dct_filter = high_pass(z0_dct, 50)
        frequency_bands['high'] = idct_2d(z0_dct_filter.squeeze(0), norm='ortho')

    return frequency_bands

def load_image(image_path, transform=None):
    """
    Load an image from a file path and apply transformations.
    :param image_path: Path to the image file.
    :param transform: Optional transformations to apply.
    :return: Image tensor.
    """
    image = Image.open(image_path).convert('RGB')
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def compute_folder_fidelity(generated_folder, target_folder, control_mode='low_pass'):
    """
    Compute the frequency fidelity for all images in two folders.
    :param generated_folder: Path to the folder containing generated images.
    :param target_folder: Path to the folder containing target images.
    :param control_mode: The control mode for frequency band extraction.
    :return: Average fidelity for each frequency band.
    """
    # Get list of image files in both folders
    generated_files = sorted([os.path.join(generated_folder, f) for f in os.listdir(generated_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    target_files = sorted([os.path.join(target_folder, f) for f in os.listdir(target_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    # Ensure the same number of images in both folders
    assert len(generated_files) == len(target_files), "Folders must contain the same number of images"
    
    # Initialize average fidelity
    avg_fidelity = {'low': 0.0, 'mini': 0.0, 'mid': 0.0, 'high': 0.0}
    count = 0
    
    # Process each pair of images
    for generated_file, target_file in zip(generated_files, target_files):
        # Load images
        generated_image = load_image(generated_file)
        target_image = load_image(target_file)
        
        # Compute fidelity
        fidelity = compute_frequency_fidelity(generated_image, target_image, control_mode)
        
        # Accumulate fidelity
        for band in ['low', 'mini', 'mid', 'high']:
            if band in fidelity:
                avg_fidelity[band] += fidelity[band].item()
        
        count += 1
    
    # Compute average fidelity
    for band in ['low', 'mini', 'mid', 'high']:
        if count > 0:
            avg_fidelity[band] /= count
    
    return avg_fidelity

if __name__ == "__main__":
    generated_folder = r"D:\paper\FCDiffusion_code-main\datasets\test_mini"  # Path to generated images folder
    target_folder = r"D:\paper\FCDiffusion_code-main\datasets\original_images"              # Path to target images folder
    control_mode = 'mini_pass'  # Choose 'low_pass', 'mini_pass', 'mid_pass', or 'high_pass'
    
    avg_fidelity = compute_folder_fidelity(generated_folder, target_folder, control_mode)
    for band, fidelity_value in avg_fidelity.items():
        print(f"Average fidelity for {band} frequency band: {fidelity_value}")