# s4s_trainer.py

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import logging
from tqdm import tqdm

# 假设您的教师模型、U-Net等都可通过以下方式导入
from ldm.util import instantiate_from_config
from stages_step_optim import NoiseScheduleVP 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- 配置 ---
CONFIG_PATH = "configs/stable-diffusion/v2-inference.yaml"
CKPT_PATH = r"D:\paper\FCDiffusion_code-main\models\v2-1_512-ema-pruned.ckpt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 输入：阶段一和阶段二的产出 ---
# 由30步最优教师生成的“黄金”潜变量轨迹数据集
TEACHER_TRAJECTORY_DATA_PATH = "teacher_golden_dataset/latents.npy" 
STUDENT_SCHEDULE_PATH = "msos_optimized_schedules/student_nfe4_schedule.txt"

# --- 输出：阶段三的产出 ---
OUTPUT_DIR = "s4s_learned_coeffs"
NFE = 4
ORDER = 4 # S4S求解器的阶数，决定了coeffs的形状

# --- 训练参数 ---
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 256

def get_s4s_update(x_curr, eps_t, history_eps, coeffs_step):
    """ 计算单步S4S更新 """
    correction = torch.zeros_like(eps_t)
    num_history_to_use = min(len(history_eps), len(coeffs_step))
    for i in range(num_history_to_use):
        correction += coeffs_step[i] * (eps_t - history_eps[i])
    d_t = eps_t + correction
    return d_t

def s4s_solver_trajectory(x_T, ts, model, coeffs):
    """ 使用给定的S4S系数，完整地走一条NFE步的轨迹 """
    x_curr = x_T.clone()
    eps_history = []
    
    # 确保模型在正确的设备和评估模式
    model.to(DEVICE).eval()

    for i in range(len(ts) - 1):
        t_curr, t_next = ts[i], ts[i+1]
        
        # 预测eps
        time_cond = torch.full((x_curr.shape[0],), t_curr * 999, device=DEVICE, dtype=torch.long)
        # 注意：这里简化了CFG，实际训练时应传入conditioning
        # 为了专注于系数学习，我们假设输入是无条件的
        with torch.no_grad():
            eps_pred = model.apply_model(x_curr, time_cond, None) # 简化为无条件

        # 计算S4S方向
        d_t = get_s4s_update(eps_pred, eps_history, coeffs[i])
        
        # DDIM更新
        alpha_t = model.alphas_cumprod[int(t_curr*999)].sqrt()
        alpha_next = model.alphas_cumprod[int(t_next*999)].sqrt()
        sigma_t = (1 - alpha_t**2).sqrt()
        sigma_next = (1 - alpha_next**2).sqrt()
        
        x_curr = (alpha_next / alpha_t) * x_curr + (sigma_next - alpha_next * sigma_t / alpha_t) * d_t

        eps_history.insert(0, eps_pred.detach())
        if len(eps_history) > ORDER - 1:
            eps_history.pop()
            
    return x_curr


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载U-Net模型 (只需要它的apply_model)
    logging.info("Loading U-Net model...")
    config = OmegaConf.load(CONFIG_PATH)
    model = instantiate_from_config(config.model)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.to(DEVICE)

    # 2. 加载教师数据集和学生调度
    logging.info(f"Loading teacher dataset from {TEACHER_TRAJECTORY_DATA_PATH}")
    teacher_data = np.load(TEACHER_TRAJECTORY_DATA_PATH)
    # 假设数据格式: [num_samples, latent_channels, H, W]
    # 我们需要初始噪声x_T和教师最终生成的x_0
    x_T_teacher = torch.randn_like(torch.from_numpy(teacher_data[:1])) # 假设所有样本从相同的高斯噪声开始
    x_0_teacher = torch.from_numpy(teacher_data).to(DEVICE).float()
    
    dataset = TensorDataset(x_0_teacher)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    logging.info(f"Loading student schedule from {STUDENT_SCHEDULE_PATH}")
    student_ts = torch.from_numpy(np.loadtxt(STUDENT_SCHEDULE_PATH, dtype=np.float32)).to(DEVICE)

    # 3. 初始化可学习的S4S系数
    # 形状: [NFE, Order-1]
    s4s_coeffs = torch.randn(NFE, ORDER - 1, device=DEVICE, requires_grad=True)

    # 4. 设置优化器
    optimizer = torch.optim.AdamW([s4s_coeffs], lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    logging.info("🚀 Starting S4S coefficient training...")
    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0
        for batch_x0_teacher, in pbar:
            batch_x_T = x_T_teacher.repeat(batch_x0_teacher.shape[0], 1, 1, 1)
            
            optimizer.zero_grad()
            
            # 使用当前系数，计算学生的最终输出
            x_0_student = s4s_solver_trajectory(batch_x_T, student_ts, model, s4s_coeffs)
            
            # 计算与教师输出的MSE损失
            loss = F.mse_loss(x_0_student, batch_x0_teacher)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        scheduler.step()
        logging.info(f"Epoch {epoch+1} average loss: {total_loss / len(dataloader):.6f}")

    # 5. 保存学到的系数
    final_coeffs_np = s4s_coeffs.detach().cpu().numpy()
    output_path = os.path.join(OUTPUT_DIR, f"student_nfe{NFE}_coeffs.npy")
    np.save(output_path, final_coeffs_np)
    
    logging.info(f"\n🎉🎉🎉 S4S training complete! Coefficients saved to: {output_path}")

if __name__ == '__main__':
    # 注意：运行此脚本前，您需要先生成教师的黄金数据集
    # 例如：用一个脚本加载teacher_nfe30_schedule.txt，生成一批图像，并保存它们的潜变量x_0
    main()