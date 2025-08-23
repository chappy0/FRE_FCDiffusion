# # # # # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # # # # import matplotlib.patches as patches

# # # # # # # # # # # # --- 设置环境 ---
# # # # # # # # # # # # 为了正确显示中文和负号
# # # # # # # # # # # plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是常用的支持中文的字体
# # # # # # # # # # # plt.rcParams['axes.unicode_minus'] = False

# # # # # # # # # # # # --- 创建画布 ---
# # # # # # # # # # # fig, ax = plt.subplots(figsize=(16, 9))
# # # # # # # # # # # ax.axis('off') # 隐藏坐标轴

# # # # # # # # # # # # --- 定义样式 ---
# # # # # # # # # # # bbox_style = dict(boxstyle="round,pad=0.8", fc="#eaf4ff", ec="#6a89cc", lw=1.5)
# # # # # # # # # # # arrow_style = dict(arrowstyle="->,head_width=0.4,head_length=0.8", color="black", lw=1.5)
# # # # # # # # # # # section_font = {'fontsize': 18, 'fontweight': 'bold', 'color': '#333333'}
# # # # # # # # # # # text_font = {'fontsize': 14, 'ha': 'center', 'va': 'center'}

# # # # # # # # # # # # --- 绘制各个模块 ---

# # # # # # # # # # # # 1. 左侧：空间域
# # # # # # # # # # # ax.text(0.15, 0.9, "空间域 (Spatial Domain)", ha="center", **section_font)
# # # # # # # # # # # ax.text(0.15, 0.5, "原始图像\n(Original Image)", bbox=bbox_style, **text_font)

# # # # # # # # # # # # 2. 右侧：生成/重构
# # # # # # # # # # # ax.text(0.85, 0.9, "生成/重构 (Generation/Reconstruction)", ha="center", **section_font)
# # # # # # # # # # # ax.text(0.85, 0.5, "生成/编辑后的图像\n(Generated/Edited Image)", bbox=bbox_style, **text_font)

# # # # # # # # # # # # 3. 中间：变换域
# # # # # # # # # # # ax.text(0.5, 0.9, "变换域 (Transformed Domain)", ha="center", **section_font)
# # # # # # # # # # # # 绘制中心的抽象频谱图
# # # # # # # # # # # ax.add_patch(patches.Rectangle((0.35, 0.25), 0.3, 0.5, facecolor='whitesmoke', edgecolor='black', lw=1.5))
# # # # # # # # # # # ax.add_patch(patches.Circle((0.5, 0.5), 0.06, color='#a2d2ff'))
# # # # # # # # # # # ax.text(0.5, 0.5, "低频\n(LF)", ha='center', va='center', fontsize=12, fontweight='bold')
# # # # # # # # # # # ax.text(0.5, 0.65, "结构, 轮廓, 内容\n(Structure, Shape, Content)", ha='center', va='center', fontsize=11, color='darkblue')
# # # # # # # # # # # ax.text(0.5, 0.35, "高频 (HF)\n细节, 纹理, 风格\n(Details, Texture, Style)", ha='center', va='center', fontsize=11, color='darkred')


# # # # # # # # # # # # 4. 下方：操控模块
# # # # # # # # # # # p_diffusion = ax.text(0.5, 0.05, "扩散模型 (Diffusion Model)\n引导 / 过滤 / 修改\n(Guide / Filter / Modify)", bbox=bbox_style, **text_font)

# # # # # # # # # # # # 5. 绘制箭头和连线
# # # # # # # # # # # # 正向变换
# # # # # # # # # # # ax.annotate("正向变换\n(Forward Transform)", xy=(0.35, 0.5), xytext=(0.25, 0.5),
# # # # # # # # # # #             arrowprops=arrow_style, va="center", ha="center", fontsize=12)

# # # # # # # # # # # # 逆向变换
# # # # # # # # # # # ax.annotate("逆向变换\n(Inverse Transform)", xy=(0.75, 0.5), xytext=(0.65, 0.5),
# # # # # # # # # # #             arrowprops=arrow_style, va="center", ha="center", fontsize=12)

# # # # # # # # # # # # 引导箭头
# # # # # # # # # # # ax.annotate("", xy=(p_diffusion.get_position()[0], p_diffusion.get_bbox_patch().get_boxstyle().pad * 2 + p_diffusion.get_window_extent().height / fig.dpi / 2 + 0.05),
# # # # # # # # # # #             xytext=(0.5, 0.25), xycoords='data', textcoords='data',
# # # # # # # # # # #             arrowprops=dict(arrowstyle="fancy", color="#555555",
# # # # # # # # # # #                             connectionstyle="arc3,rad=0", lw=2))
# # # # # # # # # # # ax.text(0.5, 0.18, "操控", ha='center', va='center', fontsize=12, fontweight='bold')


# # # # # # # # # # # # --- 添加总标题并保存 ---
# # # # # # # # # # # fig.suptitle("生成模型中频域分析与操控的通用框架", fontsize=22, fontweight='bold')
# # # # # # # # # # # plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局防止标题重叠

# # # # # # # # # # # # 保存为高清png文件
# # # # # # # # # # # plt.savefig("frequency_domain_framework.png", dpi=300, bbox_inches='tight')

# # # # # # # # # # # print("图片已成功生成并保存为 'frequency_domain_framework.png'")


# # # # # # # # # # # 步骤1: 安装必要的库 (在Google Colab中通常已预装)
# # # # # # # # # # # 如果您在本地运行，请先在终端/命令行中执行: pip install matplotlib numpy scikit-image
# # # # # # # # # # try:
# # # # # # # # # #     import matplotlib.pyplot as plt
# # # # # # # # # #     import numpy as np
# # # # # # # # # #     from skimage.data import camera
# # # # # # # # # #     from skimage.transform import resize
# # # # # # # # # # except ImportError:
# # # # # # # # # #     print("正在安装必要的库，请稍候...")
# # # # # # # # # #     import subprocess
# # # # # # # # # #     import sys
# # # # # # # # # #     subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "numpy", "scikit-image"])
# # # # # # # # # #     import matplotlib.pyplot as plt
# # # # # # # # # #     import numpy as np
# # # # # # # # # #     from skimage.data import camera
# # # # # # # # # #     from skimage.transform import resize

# # # # # # # # # # # 步骤2: 核心代码，生成图像
# # # # # # # # # # # --- 准备图像 ---
# # # # # # # # # # # 使用 scikit-image 中的标准测试图像 "camera"
# # # # # # # # # # image = camera()
# # # # # # # # # # # 调整尺寸以便快速处理
# # # # # # # # # # image = resize(image, (256, 256), anti_aliasing=True)

# # # # # # # # # # # --- 进行傅里叶变换 ---
# # # # # # # # # # fft_image = np.fft.fft2(image)
# # # # # # # # # # shifted_fft = np.fft.fftshift(fft_image)
# # # # # # # # # # magnitude_spectrum = np.log(1 + np.abs(shifted_fft))

# # # # # # # # # # # --- 创建滤波器 (掩模) ---
# # # # # # # # # # rows, cols = image.shape
# # # # # # # # # # crow, ccol = rows // 2 , cols // 2
# # # # # # # # # # radius = 30
# # # # # # # # # # y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
# # # # # # # # # # low_pass_mask = x*x + y*y <= radius*radius
# # # # # # # # # # high_pass_mask = ~low_pass_mask

# # # # # # # # # # # --- 应用滤波器并重构图像 ---
# # # # # # # # # # fft_low = shifted_fft * low_pass_mask
# # # # # # # # # # img_low_freq = np.real(np.fft.ifft2(np.fft.ifftshift(fft_low)))

# # # # # # # # # # fft_high = shifted_fft * high_pass_mask
# # # # # # # # # # img_high_freq = np.real(np.fft.ifft2(np.fft.ifftshift(fft_high)))

# # # # # # # # # # # --- 绘图 ---
# # # # # # # # # # # 设置美观的字体和样式
# # # # # # # # # # plt.style.use('seaborn-v0_8-whitegrid')
# # # # # # # # # # fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor='white')
# # # # # # # # # # ax = axes.ravel()

# # # # # # # # # # cmap_gray = 'gray'
# # # # # # # # # # title_fontsize = 16

# # # # # # # # # # ax[0].imshow(image, cmap=cmap_gray)
# # # # # # # # # # ax[0].set_title('(a) Original Image', fontsize=title_fontsize)

# # # # # # # # # # ax[1].imshow(magnitude_spectrum, cmap=cmap_gray)
# # # # # # # # # # ax[1].set_title('(b) Fourier Spectrum', fontsize=title_fontsize)

# # # # # # # # # # ax[2].imshow(img_low_freq, cmap=cmap_gray)
# # # # # # # # # # ax[2].set_title('(e) Low-Frequency Recon.', fontsize=title_fontsize)

# # # # # # # # # # ax[3].imshow(low_pass_mask, cmap=cmap_gray)
# # # # # # # # # # ax[3].set_title('(c) Low-Pass Filter', fontsize=title_fontsize)

# # # # # # # # # # ax[4].imshow(high_pass_mask, cmap=cmap_gray)
# # # # # # # # # # ax[4].set_title('(d) High-Pass Filter', fontsize=title_fontsize)

# # # # # # # # # # ax[5].imshow(img_high_freq, cmap=cmap_gray)
# # # # # # # # # # ax[5].set_title('(f) High-Frequency Recon.', fontsize=title_fontsize)

# # # # # # # # # # for a in ax:
# # # # # # # # # #     a.axis('off')

# # # # # # # # # # fig.suptitle('Frequency Domain Decomposition of an Image', fontsize=22, fontweight='bold')
# # # # # # # # # # plt.tight_layout(rect=[0, 0, 1, 0.95])

# # # # # # # # # # # 直接显示图像
# # # # # # # # # # plt.show()

# # # # # # # # # # # 自动保存文件到当前目录
# # # # # # # # # # # 在Colab中，文件会保存在左侧的“文件”面板里
# # # # # # # # # # file_name = "professional_frequency_decomposition.png"
# # # # # # # # # # fig.savefig(file_name, dpi=300, bbox_inches='tight', facecolor='white')
# # # # # # # # # # print(f"\n图片已成功生成并保存为 '{file_name}'")


# # # # # # # # import numpy as np
# # # # # # # # import matplotlib.pyplot as plt

# # # # # # # # # 步骤1: 创建一个复合信号
# # # # # # # # # 参数设置
# # # # # # # # sampling_rate = 1000  # 采样率 (Hz)
# # # # # # # # duration = 2          # 信号时长 (s)
# # # # # # # # t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False) # 时间轴

# # # # # # # # # 分量1: 低频信号 (5 Hz)
# # # # # # # # freq1 = 5
# # # # # # # # amp1 = 1.0
# # # # # # # # signal1 = amp1 * np.sin(2 * np.pi * freq1 * t)

# # # # # # # # # 分量2: 高频信号 (50 Hz)，振幅较小
# # # # # # # # freq2 = 50
# # # # # # # # amp2 = 0.5
# # # # # # # # signal2 = amp2 * np.sin(2 * np.pi * freq2 * t)

# # # # # # # # # 噪声
# # # # # # # # noise_amp = 0.15
# # # # # # # # noise = noise_amp * np.random.randn(len(t))

# # # # # # # # # 复合信号 = 低频 + 高频 + 噪声
# # # # # # # # composite_signal = signal1 + signal2 + noise

# # # # # # # # # 步骤2: 进行傅里叶变换 (FFT)
# # # # # # # # n = len(composite_signal)
# # # # # # # # fft_vals = np.fft.fft(composite_signal)
# # # # # # # # fft_freq = np.fft.fftfreq(n, d=1/sampling_rate)

# # # # # # # # # 只取正频率部分进行可视化
# # # # # # # # positive_n = n // 2
# # # # # # # # positive_freqs = fft_freq[:positive_n]
# # # # # # # # positive_fft_vals = np.abs(fft_vals[:positive_n]) * (2/n) # 归一化幅度

# # # # # # # # # 步骤3: 频域滤波与时域重构
# # # # # # # # # 定义滤波器
# # # # # # # # cutoff_freq = 20  # 以20Hz为界

# # # # # # # # # 低通滤波
# # # # # # # # fft_low = fft_vals.copy()
# # # # # # # # fft_low[np.abs(fft_freq) > cutoff_freq] = 0  # 将高于截止频率的频谱分量置零
# # # # # # # # signal_low_recon = np.fft.ifft(fft_low)

# # # # # # # # # 高通滤波
# # # # # # # # fft_high = fft_vals.copy()
# # # # # # # # fft_high[np.abs(fft_freq) <= cutoff_freq] = 0 # 将低于等于截止频率的频谱分量置零
# # # # # # # # signal_high_recon = np.fft.ifft(fft_high)

# # # # # # # # # 步骤4: 绘图
# # # # # # # # plt.style.use('seaborn-v0_8-whitegrid') # 使用专业美观的绘图风格
# # # # # # # # fig, axes = plt.subplots(2, 2, figsize=(14, 9))
# # # # # # # # fig.suptitle('1D Signal Decomposition in the Frequency Domain', fontsize=20, fontweight='bold')

# # # # # # # # # (a) 原始信号
# # # # # # # # axes[0, 0].plot(t, composite_signal, color='k', lw=1)
# # # # # # # # axes[0, 0].set_title('(a) Original Composite Signal', fontsize=14)
# # # # # # # # axes[0, 0].set_xlabel('Time [s]', fontsize=12)
# # # # # # # # axes[0, 0].set_ylabel('Amplitude', fontsize=12)
# # # # # # # # axes[0, 0].set_xlim(0, duration)
# # # # # # # # axes[0, 0].grid(True)

# # # # # # # # # (b) 频谱图
# # # # # # # # axes[0, 1].plot(positive_freqs, positive_fft_vals, color='crimson', lw=1.5)
# # # # # # # # axes[0, 1].set_title('(b) Frequency Spectrum (FFT)', fontsize=14)
# # # # # # # # axes[0, 1].set_xlabel('Frequency [Hz]', fontsize=12)
# # # # # # # # axes[0, 1].set_ylabel('Magnitude', fontsize=12)
# # # # # # # # axes[0, 1].set_xlim(0, 100) # 显示到100Hz
# # # # # # # # axes[0, 1].grid(True)

# # # # # # # # # (c) 低频分量
# # # # # # # # axes[1, 0].plot(t, signal_low_recon.real, color='mediumblue', lw=1)
# # # # # # # # axes[1, 0].set_title('(c) Reconstructed Low-Frequency Component', fontsize=14)
# # # # # # # # axes[1, 0].set_xlabel('Time [s]', fontsize=12)
# # # # # # # # axes[1, 0].set_ylabel('Amplitude', fontsize=12)
# # # # # # # # axes[1, 0].set_xlim(0, duration)
# # # # # # # # axes[1, 0].grid(True)

# # # # # # # # # (d) 高频分量
# # # # # # # # axes[1, 1].plot(t, signal_high_recon.real, color='darkorange', lw=1)
# # # # # # # # axes[1, 1].set_title('(d) Reconstructed High-Frequency Component', fontsize=14)
# # # # # # # # axes[1, 1].set_xlabel('Time [s]', fontsize=12)
# # # # # # # # axes[1, 1].set_ylabel('Amplitude', fontsize=12)
# # # # # # # # axes[1, 1].set_xlim(0, duration)
# # # # # # # # axes[1, 1].grid(True)


# # # # # # # # plt.tight_layout(rect=[0, 0.03, 1, 0.94])
# # # # # # # # plt.show()

# # # # # # # # # 自动保存文件
# # # # # # # # file_name = "professional_signal_decomposition.png"
# # # # # # # # fig.savefig(file_name, dpi=300, facecolor='white', bbox_inches='tight')
# # # # # # # # print(f"\n图片已成功生成并保存为 '{file_name}'")



# # # # # # # # import numpy as np
# # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # import pywt # 需要安装 PyWavelets 库: pip install PyWavelets

# # # # # # # # # 步骤1: 创建一个非平稳信号 (高频信号只在后半段出现)
# # # # # # # # sampling_rate = 1000
# # # # # # # # duration = 2
# # # # # # # # t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# # # # # # # # # 低频分量 (15 Hz)，持续存在
# # # # # # # # signal1 = np.sin(2 * np.pi * 15 * t)
# # # # # # # # # 高频分量 (80 Hz)，只在后半段出现
# # # # # # # # signal2 = np.sin(2 * np.pi * 80 * t)
# # # # # # # # window = np.zeros_like(t)
# # # # # # # # window[t > 1] = 1 # 从1秒后开始
# # # # # # # # composite_signal = signal1 + signal2 * window

# # # # # # # # # 步骤2: 进行连续小波变换 (CWT)
# # # # # # # # wavelet = 'cmor1.5-1.0' # Morlet小波
# # # # # # # # scales = np.arange(1, 128) # 定义尺度范围
# # # # # # # # coefficients, frequencies = pywt.cwt(composite_signal, scales, wavelet, 1/sampling_rate)

# # # # # # # # # 步骤3: 绘图
# # # # # # # # fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2]})
# # # # # # # # fig.suptitle('Wavelet Transform (CWT) Analysis', fontsize=18, fontweight='bold')

# # # # # # # # # 绘制原始信号
# # # # # # # # axs[0].plot(t, composite_signal, 'k', lw=1)
# # # # # # # # axs[0].set_title('Original Non-Stationary Signal', fontsize=14)
# # # # # # # # axs[0].set_xlabel('Time [s]', fontsize=12)
# # # # # # # # axs[0].set_ylabel('Amplitude', fontsize=12)
# # # # # # # # axs[0].grid(True)

# # # # # # # # # 绘制小波时频谱图 (Scalogram)
# # # # # # # # im = axs[1].pcolormesh(t, frequencies, np.abs(coefficients), cmap='viridis', shading='gouraud')
# # # # # # # # axs[1].set_title('Wavelet Scalogram', fontsize=14)
# # # # # # # # axs[1].set_xlabel('Time [s]', fontsize=12)
# # # # # # # # axs[1].set_ylabel('Frequency [Hz]', fontsize=12)
# # # # # # # # axs[1].set_yscale('log') # 使用对数坐标以便观察
# # # # # # # # fig.colorbar(im, ax=axs[1], label='Magnitude')

# # # # # # # # plt.tight_layout(rect=[0, 0, 1, 0.95])
# # # # # # # # plt.show()

# # # # # # # # # 保存文件
# # # # # # # # fig.savefig("professional_wavelet_decomposition.png", dpi=300)


# # # # # # # import numpy as np
# # # # # # # import matplotlib.pyplot as plt
# # # # # # # from scipy import signal as scipysignal # 需要安装 SciPy: pip install scipy

# # # # # # # # 复用之前创建的非平稳信号
# # # # # # # sampling_rate = 1000
# # # # # # # duration = 2
# # # # # # # t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
# # # # # # # signal1 = np.sin(2 * np.pi * 15 * t)
# # # # # # # signal2 = np.sin(2 * np.pi * 80 * t)
# # # # # # # window = np.zeros_like(t)
# # # # # # # window[t > 1] = 1
# # # # # # # composite_signal = signal1 + signal2 * window

# # # # # # # # 步骤2: 进行STFT
# # # # # # # f, t_stft, Zxx = scipysignal.stft(composite_signal, fs=sampling_rate, nperseg=128)

# # # # # # # # 步骤3: 绘图
# # # # # # # fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2]})
# # # # # # # fig.suptitle('Short-Time Fourier Transform (STFT) Analysis', fontsize=18, fontweight='bold')

# # # # # # # # 绘制原始信号
# # # # # # # axs[0].plot(t, composite_signal, 'k', lw=1)
# # # # # # # axs[0].set_title('Original Non-Stationary Signal', fontsize=14)
# # # # # # # axs[0].set_xlabel('Time [s]', fontsize=12)
# # # # # # # axs[0].set_ylabel('Amplitude', fontsize=12)
# # # # # # # axs[0].grid(True)

# # # # # # # # 绘制STFT频谱图
# # # # # # # im = axs[1].pcolormesh(t_stft, f, np.abs(Zxx), vmin=0, shading='gouraud')
# # # # # # # axs[1].set_title('STFT Spectrogram', fontsize=14)
# # # # # # # axs[1].set_xlabel('Time [s]', fontsize=12)
# # # # # # # axs[1].set_ylabel('Frequency [Hz]', fontsize=12)
# # # # # # # fig.colorbar(im, ax=axs[1], label='Magnitude')

# # # # # # # plt.tight_layout(rect=[0, 0, 1, 0.95])
# # # # # # # plt.show()

# # # # # # # # 保存文件
# # # # # # # fig.savefig("professional_stft_decomposition.png", dpi=300)


# # # # # # import numpy as np
# # # # # # import matplotlib.pyplot as plt
# # # # # # from scipy.fft import dct, fft, fftfreq

# # # # # # # 使用之前的信号，但这里用平稳信号（高低频都持续存在）更能体现DCT和FFT的对比
# # # # # # sampling_rate = 1000
# # # # # # duration = 2
# # # # # # t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
# # # # # # signal1 = np.sin(2 * np.pi * 15 * t)
# # # # # # signal2 = np.sin(2 * np.pi * 50 * t)
# # # # # # stationary_signal = signal1 + signal2

# # # # # # # 步骤2: 计算DCT和FFT
# # # # # # # DCT
# # # # # # dct_vals = dct(stationary_signal, type=2, norm='ortho')
# # # # # # # FFT
# # # # # # n = len(stationary_signal)
# # # # # # fft_vals = np.abs(fft(stationary_signal))[:n//2] * (2/n)
# # # # # # fft_freqs = fftfreq(n, 1/sampling_rate)[:n//2]

# # # # # # # 步骤3: 绘图
# # # # # # fig, axs = plt.subplots(3, 1, figsize=(12, 10))
# # # # # # fig.suptitle('DCT vs. FFT Comparison', fontsize=18, fontweight='bold')

# # # # # # # 绘制原始信号
# # # # # # axs[0].plot(t, stationary_signal, 'k', lw=1)
# # # # # # axs[0].set_title('Original Stationary Signal', fontsize=14)
# # # # # # axs[0].set_xlabel('Time [s]')
# # # # # # axs[0].set_ylabel('Amplitude')

# # # # # # # 绘制FFT频谱
# # # # # # axs[1].plot(fft_freqs, fft_vals, 'crimson')
# # # # # # axs[1].set_title('FFT Spectrum', fontsize=14)
# # # # # # axs[1].set_xlabel('Frequency [Hz]')
# # # # # # axs[1].set_ylabel('Magnitude')
# # # # # # axs[1].set_xlim(0, 100)

# # # # # # # 绘制DCT系数
# # # # # # axs[2].plot(dct_vals, 'mediumblue')
# # # # # # axs[2].set_title('DCT Coefficients', fontsize=14)
# # # # # # axs[2].set_xlabel('Coefficient Index (Frequency-like)')
# # # # # # axs[2].set_ylabel('Magnitude')
# # # # # # axs[2].set_xlim(0, 200)

# # # # # # plt.subplots_adjust(hspace=6)
# # # # # # plt.tight_layout(rect=[0, 0, 1, 0.95])
# # # # # # plt.show()

# # # # # # # 保存文件
# # # # # # fig.savefig("professional_dct_vs_fft.png", dpi=300)


# # # # # # 步骤0: 安装必要的库
# # # # # # 在终端/命令行中执行，或在Colab/Jupyter Notebook的单元格中执行
# # # # # # !pip install numpy matplotlib scipy pywavelets pyrtools
# # # # # try:
# # # # #     import numpy as np
# # # # #     import matplotlib.pyplot as plt
# # # # #     from scipy import signal as scipysignal
# # # # #     import pywt
# # # # #     import pyrtools as pt
# # # # # except ImportError:
# # # # #     print("正在安装必要的库 (scipy, pywavelets, pyrtools)...")
# # # # #     import subprocess
# # # # #     import sys
# # # # #     subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "pywavelets", "pyrtools"])
# # # # #     import numpy as np
# # # # #     import matplotlib.pyplot as plt
# # # # #     from scipy import signal as scipysignal
# # # # #     import pywt
# # # # #     import pyrtools as pt


# # # # # # --- Part 1: 一维时变信号分析 (STFT & Wavelet) ---

# # # # # # 1.1 创建信号
# # # # # sampling_rate = 1000
# # # # # duration = 2
# # # # # t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
# # # # # signal1 = np.sin(2 * np.pi * 15 * t)
# # # # # signal2 = np.sin(2 * np.pi * 80 * t)
# # # # # window = np.zeros_like(t)
# # # # # window[t > 1] = 1
# # # # # nonstationary_signal = signal1 + signal2 * window

# # # # # # 1.2 STFT 计算
# # # # # f_stft, t_stft, Zxx = scipysignal.stft(nonstationary_signal, fs=sampling_rate, nperseg=128)

# # # # # # 1.3 Wavelet 计算
# # # # # wavelet = 'cmor1.5-1.0'
# # # # # scales = np.arange(1, 128)
# # # # # coefficients, frequencies = pywt.cwt(nonstationary_signal, scales, wavelet, 1/sampling_rate)


# # # # # # --- Part 2: 二维方向性分析 (Steerable Pyramid) ---

# # # # # # 2.1 创建带方向的图像
# # # # # image_size = 256
# # # # # x = np.linspace(-1, 1, image_size)
# # # # # y = np.linspace(-1, 1, image_size)
# # # # # X, Y = np.meshgrid(x, y)
# # # # # # 创建45度角的正弦波条纹
# # # # # oriented_image = np.sin(2 * np.pi * (X + Y) * 5)

# # # # # # 2.2 Steerable Pyramid 分解
# # # # # # order=1 表示分解为2个方向 (0, 90度)
# # # # # # order=3 表示分解为4个方向 (0, 45, 90, 135度)
# # # # # pyramid = pt.pyramids.SteerablePyramidSpace(oriented_image, height=4, order=3)
# # # # # # 提取第1个尺度下，第2个方向(45度)的子带响应
# # # # # oriented_subband = pyramid.pyr_coeffs[('band', 1)]


# # # # # # --- Part 3: 绘图 ---
# # # # # fig, axes = plt.subplots(2, 2, figsize=(15, 12))
# # # # # fig.suptitle('Comparison of Joint Local-Feature Transforms', fontsize=20, fontweight='bold')

# # # # # # (a) STFT
# # # # # im_a = axes[0, 0].pcolormesh(t_stft, f_stft, np.abs(Zxx), vmin=0, shading='gouraud', cmap='magma')
# # # # # axes[0, 0].set_title('(a) STFT: Fixed Resolution', fontsize=14)
# # # # # axes[0, 0].set_xlabel('Time [s]', fontsize=12)
# # # # # axes[0, 0].set_ylabel('Frequency [Hz]', fontsize=12)
# # # # # fig.colorbar(im_a, ax=axes[0, 0])

# # # # # # (b) Wavelet
# # # # # im_b = axes[0, 1].pcolormesh(t, frequencies, np.abs(coefficients), cmap='magma', shading='gouraud')
# # # # # axes[0, 1].set_title('(b) Wavelet: Multi-Resolution', fontsize=14)
# # # # # axes[0, 1].set_xlabel('Time [s]', fontsize=12)
# # # # # axes[0, 1].set_ylabel('Frequency [Hz]', fontsize=12)
# # # # # axes[0, 1].set_yscale('log')
# # # # # fig.colorbar(im_b, ax=axes[0, 1])

# # # # # # (c) Oriented Image
# # # # # axes[1, 0].imshow(oriented_image, cmap='gray')
# # # # # axes[1, 0].set_title('(c) Original Image with 45° Orientation', fontsize=14)
# # # # # axes[1, 0].axis('off')

# # # # # # (d) Steerable Pyramid Sub-band
# # # # # axes[1, 1].imshow(oriented_subband, cmap='gray')
# # # # # axes[1, 1].set_title("(d) Steerable Pyramid: 45° Sub-band", fontsize=14)
# # # # # axes[1, 1].axis('off')

# # # # # plt.tight_layout(rect=[0, 0, 1, 0.95])
# # # # # plt.show()

# # # # # # 保存文件
# # # # # fig.savefig("professional_local_analysis_comparison.png", dpi=300)


# # # # import numpy as np
# # # # import matplotlib.pyplot as plt
# # # # from scipy.fft import dct, fft, fftfreq

# # # # # --- 信号生成和计算 (这部分不变) ---
# # # # sampling_rate = 1000
# # # # duration = 2
# # # # t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
# # # # signal1 = np.sin(2 * np.pi * 15 * t)
# # # # signal2 = np.sin(2 * np.pi * 50 * t)
# # # # stationary_signal = signal1 + signal2

# # # # # DCT
# # # # dct_vals = dct(stationary_signal, type=2, norm='ortho')
# # # # # FFT
# # # # n = len(stationary_signal)
# # # # fft_vals = np.abs(fft(stationary_signal))[:n//2] * (2/n)
# # # # fft_freqs = fftfreq(n, 1/sampling_rate)[:n//2]


# # # # import numpy as np
# # # # import matplotlib.pyplot as plt
# # # # from scipy.fft import dct, fft, fftfreq
# # # # from matplotlib.gridspec import GridSpec # 导入GridSpec

# # # # # --- 信号生成和计算 (这部分不变) ---
# # # # # ... (代码同上) ...

# # # # # --- 绘图 (使用 GridSpec 的修正版) ---
# # # # fig = plt.figure(figsize=(12, 12))
# # # # fig.suptitle('DCT vs. FFT Comparison', fontsize=18, fontweight='bold')

# # # # # 1. 关键修正：创建一个3行1列的网格，并设置较大的垂直间距 hspace
# # # # gs = GridSpec(3, 1, figure=fig, hspace=0.8)

# # # # # 2. 将每个子图精确地放置在网格的不同行
# # # # ax1 = fig.add_subplot(gs[0, 0])
# # # # ax2 = fig.add_subplot(gs[1, 0])
# # # # ax3 = fig.add_subplot(gs[2, 0])

# # # # # 3. 在每个子图上绘制内容
# # # # # 绘制原始信号
# # # # ax1.plot(t, stationary_signal, 'k', lw=1)
# # # # ax1.set_title('Original Stationary Signal', fontsize=14)
# # # # ax1.set_xlabel('Time [s]')
# # # # ax1.set_ylabel('Amplitude')

# # # # # 绘制FFT频谱
# # # # ax2.plot(fft_freqs, fft_vals, 'crimson')
# # # # ax2.set_title('FFT Spectrum', fontsize=14)
# # # # ax2.set_xlabel('Frequency [Hz]')
# # # # ax2.set_ylabel('Magnitude')
# # # # ax2.set_xlim(0, 100)

# # # # # 绘制DCT系数
# # # # ax3.plot(dct_vals, 'mediumblue')
# # # # ax3.set_title('DCT Coefficients', fontsize=14)
# # # # ax3.set_xlabel('Coefficient Index (Frequency-like)')
# # # # ax3.set_ylabel('Magnitude')
# # # # ax3.set_xlim(0, 200)

# # # # # 调整主标题的位置以避免与子图重叠
# # # # fig.subplots_adjust(top=0.92)

# # # # plt.show()

# # # # # 保存文件
# # # # fig.savefig("professional_dct_vs_fft_v3_gridspec.png", dpi=300)


# # # # 步骤0: 安装必要的库
# # # # 在终端/命令行中执行，或在Colab/Jupyter Notebook的单元格中执行
# # # # !pip install numpy matplotlib scipy pywavelets pyrtools
# # # try:
# # #     import numpy as np
# # #     import matplotlib.pyplot as plt
# # #     from scipy import signal as scipysignal
# # #     import pywt
# # #     import pyrtools as pt
# # # except ImportError:
# # #     print("正在安装必要的库 (scipy, pywavelets, pyrtools)...")
# # #     import subprocess
# # #     import sys
# # #     subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "pywavelets", "pyrtools"])
# # #     import numpy as np
# # #     import matplotlib.pyplot as plt
# # #     from scipy import signal as scipysignal
# # #     import pywt
# # #     import pyrtools as pt


# # # # --- Part 1: 一维时变信号分析 (STFT & Wavelet) ---

# # # # 1.1 创建信号
# # # sampling_rate = 1000
# # # duration = 2
# # # t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
# # # signal1 = np.sin(2 * np.pi * 15 * t)
# # # signal2 = np.sin(2 * np.pi * 80 * t)
# # # window = np.zeros_like(t)
# # # window[t > 1] = 1
# # # nonstationary_signal = signal1 + signal2 * window

# # # # 1.2 STFT 计算
# # # f_stft, t_stft, Zxx = scipysignal.stft(nonstationary_signal, fs=sampling_rate, nperseg=128)

# # # # 1.3 Wavelet 计算
# # # wavelet = 'cmor1.5-1.0'
# # # scales = np.arange(1, 128)
# # # coefficients, frequencies = pywt.cwt(nonstationary_signal, scales, wavelet, 1/sampling_rate)


# # # # --- Part 2: 二维方向性分析 (Steerable Pyramid) ---

# # # # 2.1 创建带方向的图像
# # # image_size = 256
# # # x = np.linspace(-1, 1, image_size)
# # # y = np.linspace(-1, 1, image_size)
# # # X, Y = np.meshgrid(x, y)
# # # oriented_image = np.sin(2 * np.pi * (X + Y) * 5)

# # # # 2.2 Steerable Pyramid 分解
# # # pyramid = pt.pyramids.SteerablePyramidSpace(oriented_image, height=4, order=3)

# # # # 2.3 关键修正：使用正确的key来提取子带
# # # # 我们选择第1层(level=1), 第1个方向(orientation_index=1, 对应45度)
# # # # 您也可以尝试其他key，例如 (0, 0), (2, 3) 等
# # # key_to_display = (1, 1) 
# # # oriented_subband = pyramid.pyr_coeffs[key_to_display]


# # # # --- Part 3: 绘图 ---
# # # fig, axes = plt.subplots(2, 2, figsize=(15, 12))
# # # fig.suptitle('Comparison of Joint Local-Feature Transforms', fontsize=20, fontweight='bold')

# # # # (a) STFT
# # # im_a = axes[0, 0].pcolormesh(t_stft, f_stft, np.abs(Zxx), vmin=0, shading='gouraud', cmap='magma')
# # # axes[0, 0].set_title('(a) STFT: Fixed Resolution', fontsize=14)
# # # axes[0, 0].set_xlabel('Time [s]', fontsize=12)
# # # axes[0, 0].set_ylabel('Frequency [Hz]', fontsize=12)
# # # fig.colorbar(im_a, ax=axes[0, 0])

# # # # (b) Wavelet
# # # im_b = axes[0, 1].pcolormesh(t, frequencies, np.abs(coefficients), cmap='magma', shading='gouraud')
# # # axes[0, 1].set_title('(b) Wavelet: Multi-Resolution', fontsize=14)
# # # axes[0, 1].set_xlabel('Time [s]', fontsize=12)
# # # axes[0, 1].set_ylabel('Frequency [Hz]', fontsize=12)
# # # axes[0, 1].set_yscale('log')
# # # fig.colorbar(im_b, ax=axes[0, 1])

# # # # (c) Oriented Image
# # # axes[1, 0].imshow(oriented_image, cmap='gray')
# # # axes[1, 0].set_title('(c) Original Image with 45° Orientation', fontsize=14)
# # # axes[1, 0].axis('off')

# # # # (d) Steerable Pyramid Sub-band
# # # axes[1, 1].imshow(oriented_subband, cmap='gray')
# # # axes[1, 1].set_title(f"(d) Steerable Pyramid: Sub-band {key_to_display}", fontsize=14)
# # # axes[1, 1].axis('off')

# # # plt.tight_layout(rect=[0, 0, 1, 0.95])
# # # plt.show()

# # # # 保存文件
# # # fig.savefig("professional_local_analysis_comparison_v2.png", dpi=300)

# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy.fft import dct, fft, fftfreq

# # # --- 信号生成和计算 (这部分不变) ---
# # sampling_rate = 1000
# # duration = 2
# # t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
# # signal1 = np.sin(2 * np.pi * 15 * t)
# # signal2 = np.sin(2 * np.pi * 50 * t)
# # stationary_signal = signal1 + signal2

# # # DCT
# # dct_vals = dct(stationary_signal, type=2, norm='ortho')
# # # FFT
# # n = len(stationary_signal)
# # fft_vals = np.abs(fft(stationary_signal))[:n//2] * (2/n)
# # fft_freqs = fftfreq(n, 1/sampling_rate)[:n//2]


# # # --- 绘图 (最终手动定位版) ---
# # # 1. 创建 Figure 和 Axes 对象
# # fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# # # 2. 关键修正：不再使用 fig.suptitle()
# # #    改用 fig.text() 在绝对坐标上放置主标题
# # #    (0.5, 0.96) 表示水平居中，垂直方向在96%的高度位置
# # fig.text(0.5, 0.96, 'DCT vs. FFT Comparison', 
# #          ha='center',            # 水平居中
# #          va='center',            # 垂直居中
# #          fontsize=20,            # 字体大小
# #          fontweight='bold')      # 字体粗细

# # # 3. 绘制各个子图
# # # 绘制原始信号
# # axs[0].plot(t, stationary_signal, 'k', lw=1)
# # axs[0].set_title('Original Stationary Signal', fontsize=14)
# # axs[0].set_xlabel('Time [s]')
# # axs[0].set_ylabel('Amplitude')

# # # 绘制FFT频谱
# # axs[1].plot(fft_freqs, fft_vals, 'crimson')
# # axs[1].set_title('FFT Spectrum', fontsize=14)
# # axs[1].set_xlabel('Frequency [Hz]')
# # axs[1].set_ylabel('Magnitude')
# # axs[1].set_xlim(0, 100)

# # # 绘制DCT系数
# # axs[2].plot(dct_vals, 'mediumblue')
# # axs[2].set_title('DCT Coefficients', fontsize=14)
# # axs[2].set_xlabel('Coefficient Index (Frequency-like)')
# # axs[2].set_ylabel('Magnitude')
# # axs[2].set_xlim(0, 200)

# # # 4. 关键修正：移除所有自动布局，让手动布局生效
# # # plt.tight_layout() # 移除或注释掉
# # # plt.subplots_adjust() # 移除或注释掉

# # # 5. 为了给底部标签和主标题留出空间，可以轻微调整整体布局
# # fig.subplots_adjust(top=0.9, bottom=0.1, hspace=0.7)


# # plt.show()

# # # 保存文件
# # fig.savefig("professional_dct_vs_fft_final.png", dpi=300)

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal as scipysignal
# import pywt
# import pyrtools as pt

# # --- Part 1: 一维时变信号分析 (STFT & Wavelet) ---

# # 1.1 创建信号
# sampling_rate = 1000
# duration = 2
# t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
# signal1 = np.sin(2 * np.pi * 15 * t)
# signal2 = np.sin(2 * np.pi * 80 * t)
# window = np.zeros_like(t)
# window[t > 1] = 1
# nonstationary_signal = signal1 + signal2 * window

# # 1.2 STFT 计算
# f_stft, t_stft, Zxx = scipysignal.stft(nonstationary_signal, fs=sampling_rate, nperseg=128)

# # 1.3 Wavelet 计算
# wavelet = 'cmor1.5-1.0'
# scales = np.arange(1, 128)
# coefficients, frequencies = pywt.cwt(nonstationary_signal, scales, wavelet, 1/sampling_rate)


# # --- Part 2: 二维方向性分析 (Steerable Pyramid) ---

# # 2.1 创建带方向的图像
# image_size = 256
# x = np.linspace(-1, 1, image_size)
# y = np.linspace(-1, 1, image_size)
# X, Y = np.meshgrid(x, y)
# oriented_image = np.sin(2 * np.pi * (X + Y) * 5)

# # 2.2 Steerable Pyramid 分解
# pyramid = pt.pyramids.SteerablePyramidSpace(oriented_image, height=4, order=3)
# key_to_display = (1, 1) 
# oriented_subband = pyramid.pyr_coeffs[key_to_display]


# # --- Part 3: 绘图 (恢复为默认风格) ---

# # 关键修正：不再使用任何 plt.style.use()
# fig, axes = plt.subplots(2, 2, figsize=(15, 12))
# fig.suptitle('Comparison of Joint Local-Feature Transforms', fontsize=20, fontweight='bold')

# # (a) STFT
# # 关键修正：将cmap从'magma'改回默认的'viridis'
# im_a = axes[0, 0].pcolormesh(t_stft, f_stft, np.abs(Zxx), vmin=0, shading='gouraud', cmap='viridis')
# axes[0, 0].set_title('(a) STFT: Fixed Resolution', fontsize=14)
# axes[0, 0].set_xlabel('Time [s]', fontsize=12)
# axes[0, 0].set_ylabel('Frequency [Hz]', fontsize=12)
# fig.colorbar(im_a, ax=axes[0, 0])

# # (b) Wavelet
# # 关键修正：将cmap从'magma'改回默认的'viridis'
# im_b = axes[0, 1].pcolormesh(t, frequencies, np.abs(coefficients), cmap='viridis', shading='gouraud')
# axes[0, 1].set_title('(b) Wavelet: Multi-Resolution', fontsize=14)
# axes[0, 1].set_xlabel('Time [s]', fontsize=12)
# axes[0, 1].set_ylabel('Frequency [Hz]', fontsize=12)
# axes[0, 1].set_yscale('log')
# fig.colorbar(im_b, ax=axes[0, 1])

# # (c) Oriented Image
# axes[1, 0].imshow(oriented_image, cmap='gray')
# axes[1, 0].set_title('(c) Original Image with 45° Orientation', fontsize=14)
# axes[1, 0].axis('off')

# # (d) Steerable Pyramid Sub-band
# axes[1, 1].imshow(oriented_subband, cmap='gray')
# axes[1, 1].set_title(f"(d) Steerable Pyramid: Sub-band {key_to_display}", fontsize=14)
# axes[1, 1].axis('off')

# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()

# # 保存文件 (移除了facecolor='white'参数)
# fig.savefig("professional_local_analysis_comparison_default_style.png", dpi=300)


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipysignal
import pywt

# --- 1. 创建共同的输入信号 ---
sampling_rate = 1000
duration = 2
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# 信号包含一个持续的15Hz低频和一个仅在后半段出现的80Hz高频
original_signal = np.sin(2 * np.pi * 15 * t)
window = np.zeros_like(t)
window[t > 1] = 1
original_signal += np.sin(2 * np.pi * 80 * t) * window


# --- 2. 使用STFT/Gabor进行分解和重构 ---
# STFT正变换
f_stft, t_stft, Zxx = scipysignal.stft(original_signal, fs=sampling_rate, nperseg=256)
# 在频域进行滤波
cutoff_freq = 40  # 以40Hz为界
Zxx_low = Zxx.copy()
Zxx_low[f_stft > cutoff_freq, :] = 0
Zxx_high = Zxx.copy()
Zxx_high[f_stft <= cutoff_freq, :] = 0
# STFT逆变换重构
_, stft_low_recon = scipysignal.istft(Zxx_low, fs=sampling_rate)
_, stft_high_recon = scipysignal.istft(Zxx_high, fs=sampling_rate)


# --- 3. 使用小波变换进行分解和重构 ---
wavelet = 'db8'
level = 5
coeffs = pywt.wavedec(original_signal, wavelet, level=level)
# 重构低频
coeffs_low = [coeffs[0]] + [np.zeros_like(d) for d in coeffs[1:]]
wavelet_low_recon = pywt.waverec(coeffs_low, wavelet)
# 重构高频
coeffs_high = [np.zeros_like(coeffs[0])] + coeffs[1:]
wavelet_high_recon = pywt.waverec(coeffs_high, wavelet)


# --- 4. 绘图 (并排对比) ---
# 校正信号长度差异
min_len = min(len(t), len(stft_low_recon), len(wavelet_low_recon))
t, original_signal = t[:min_len], original_signal[:min_len]
stft_low_recon, stft_high_recon = stft_low_recon[:min_len], stft_high_recon[:min_len]
wavelet_low_recon, wavelet_high_recon = wavelet_low_recon[:min_len], wavelet_high_recon[:min_len]

# 创建图表
fig = plt.figure(figsize=(16, 12))
gs = plt.GridSpec(3, 2, figure=fig, hspace=0.6, wspace=0.2)

# 主标题
fig.text(0.5, 0.96, 'Waveform Decomposition: STFT/Gabor vs. Wavelet', 
         ha='center', va='center', fontsize=20, fontweight='bold')

# 顶行：原始信号 (横跨两列)
ax_orig = fig.add_subplot(gs[0, :])
ax_orig.plot(t, original_signal, 'k', lw=1)
ax_orig.set_title('Original Non-Stationary Signal', fontsize=16)
ax_orig.set_ylabel('Amplitude', fontsize=12)

# 左列：STFT/Gabor分解结果
ax_stft_low = fig.add_subplot(gs[1, 0], sharex=ax_orig)
ax_stft_low.plot(t, stft_low_recon, 'mediumblue')
ax_stft_low.set_title('Low-Freq Component (via STFT)', fontsize=14)
ax_stft_low.set_ylabel('Amplitude', fontsize=12)

ax_stft_high = fig.add_subplot(gs[2, 0], sharex=ax_orig)
ax_stft_high.plot(t, stft_high_recon, 'crimson')
ax_stft_high.set_title('High-Freq Component (via STFT)', fontsize=14)
ax_stft_high.set_xlabel('Time [s]', fontsize=12)
ax_stft_high.set_ylabel('Amplitude', fontsize=12)

# 右列：小波分解结果
ax_wave_low = fig.add_subplot(gs[1, 1], sharex=ax_orig)
ax_wave_low.plot(t, wavelet_low_recon, 'mediumblue')
ax_wave_low.set_title('Low-Freq Component (via Wavelet)', fontsize=14)

ax_wave_high = fig.add_subplot(gs[2, 1], sharex=ax_orig)
ax_wave_high.plot(t, wavelet_high_recon, 'crimson')
ax_wave_high.set_title('High-Freq Component (via Wavelet)', fontsize=14)
ax_wave_high.set_xlabel('Time [s]', fontsize=12)

# 隐藏y轴标签以保持整洁
plt.setp(ax_wave_low.get_yticklabels(), visible=False)
plt.setp(ax_wave_high.get_yticklabels(), visible=False)

fig.subplots_adjust(top=0.88)
plt.show()

# 保存文件
fig.savefig("stft_vs_wavelet_decomposition.png", dpi=300)