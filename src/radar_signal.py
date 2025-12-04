# src/radar_signal.py
import numpy as np
from scipy.fft import fft

def generate_radar_signal(f_c=77e9, B=100e6, T=0.01, fs=1e6, N_chirps=10):
    t = np.arange(0, T, 1 / fs) # 生成等差数列
    chirp_freq = f_c + (B / T) * t # 线性调频信号的频率
    signal = np.sin(2 * np.pi * chirp_freq * t)
    return signal

def simulate_distance_doppler_map(signal, fs, noise_level=0.1):
    from scipy.fft import fft
    n_fft = 64
    hop_length = 32
    window = np.hanning(n_fft)
    noisy_signal = signal + np.random.randn(len(signal)) * noise_level
    stft_data = []
    for i in range(0, len(noisy_signal) - n_fft + 1, hop_length):
        chunk = noisy_signal[i:i+n_fft]
        fft_chunk = fft(chunk * window)
        magnitude = np.abs(fft_chunk[:n_fft//2])  # 对称，只取正半频谱，向下取整
        stft_data.append(magnitude)
    stft_data = np.array(stft_data).T       # 转为二维数组：形状为 (n_fft//2, num_frames)，按列
    return np.log(stft_data + 1e-8)

def create_dataset():
    fs = 1e6  # 采样率 1 MHz
    f_c = 77e9  # 载频 77 GHz
    c = 3e8  # 光速
    B = 100e6  # 带宽
    T = 0.01  # Chirp 周期

    X, y = [], []
    original_shape = None

    for i in range(100):
        signal = generate_radar_signal(f_c=f_c, B=B, T=T, fs=fs)

        if i < 50:  # 无目标
            final_signal = signal
            label = 0
        else:  # 有目标（运动目标）
            R = 15.0  # 距离 15 米
            v = 2.5  # 速度 2.5 m/s（产生多普勒）
            tau = 2 * R / c
            delay_samples = int(tau * fs)

            A = 0.005  # 回波

            t_full = np.arange(len(signal)) / fs
            echo = np.zeros_like(signal)

            if delay_samples < len(signal):
                # 计算有效时间（回波存在的时间段）
                t_echo = t_full[delay_samples:] - tau  # 相对于目标的时间

                # 原始 chirp 相位：phi_tx(t) = 2π [f_c * t + (B/(2T)) * t^2]
                # 回波相位：phi_rx(t) = 2π [f_c * (t - tau) + (B/(2T)) * (t - tau)^2] + 2π * f_d * (t - tau)
                wavelength = c / f_c
                f_d = 2 * v / wavelength

                phase = np.pi * np.random.rand()  # 改为随机相位，破坏相干性
                echo[delay_samples:] = A * np.sin(phase)  # 固定相位 → 随机相位

            final_signal = signal + echo
            label = 1

        # 噪声
        dd_map = simulate_distance_doppler_map(final_signal, fs=fs, noise_level=0.02)
        if original_shape is None:
            original_shape = dd_map.shape

        X.append(dd_map.flatten())
        y.append(label)

    return np.array(X), np.array(y), original_shape