Of course, here is the English version of the README for your project.

-----

# DKD Project Code



## Quick Start Guide

### 1\. Environment Setup

We recommend creating and activating an isolated virtual environment using Conda.

```bash
# 1. Create and activate the Conda environment
conda create -n DKD python=3.8
conda activate DKD

# 2. Install PyTorch and related dependencies (adjust according to your CUDA version)
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# 3. Install all other required libraries
pip install -r requirements.txt
```

### 2\. Data & Pre-trained Model Preparation

#### **Dataset**

Image-text pairs are required for training. You can prepare a dataset like **LAION Aesthetics 6.5+** and organize it using the following structure.

```
DKD/
└── datasets/
    └── training_data/
        ├── image/
        │   ├── 000000000.jpg
        │   └── ...
        └── prompt/
            ├── 000000000.txt
            └── ...
```

#### **Pre-trained Models**

The following two pre-trained models are required for distillation training:

1.  **Stable Diffusion v2-1-base**:

      * **File**: `v2-1_512-ema-pruned.ckpt`
      * **Download**: [huggingface.co/stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main)
      * **Path**: `./models/v2-1_512-ema-pruned.ckpt`

2.  **OpenCLIP**:

      * **File**: `open_clip_pytorch_model.bin`
      * **Download**: [huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main)
      * **Path**: `./models/CLIP-ViT-H-14/open_clip_pytorch_model.bin`

Please ensure your `./models` directory structure is as follows:

```
DKD/
└── models/
    ├── v2-1_512-ema-pruned.ckpt
    └── CLIP-ViT-H-14/
        └── open_clip_pytorch_model.bin
```

### 3\. Model Distillation Training

1.  **Modify Script Paths**: Open the training script `fcdffusion_distill_final.py` and update the model and dataset paths to your local paths.

      * `teacher_model_path`: Path to the teacher model's checkpoint file (.ckpt).
      * `output_path`: Path where the trained student model weights will be saved.
      * `dataset_path`: Path to the root directory of your dataset.

2.  **Start Training**:

    ```bash
    python fcdffusion_distill_final.py
    ```

### 4\. Model Testing & Validation

Use the `fcdiffusion_student_test.py` script to load the trained student model for inference and validation.

1.  **Modify Script Paths**: Open the test script `fcdiffusion_student_test.py` and modify the relevant paths.

      * `student_model_path`: Path to the student model weights trained in the previous step.
      * `input_image_path`: The source image for testing.
      * `output_dir`: The directory where test results will be saved.

2.  **Start Testing**:

    ```bash
    python fcdiffusion_student_test.py
    ```



