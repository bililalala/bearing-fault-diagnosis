"""
数据预处理函数：
- 滑动窗口重叠采样 (第3章, 式3.7)
- PAA 分段聚合近似 (第4章, 式4.3)
- GAF 格拉姆角场编码 (第4章, 式4.1-4.2)
- FFT 快速傅里叶变换 (第5章, 式5.11)
- VMD 变分模态分解 (第5章, 5.2.6节)
"""
import numpy as np
from scipy.fft import fft
from scipy.signal import hilbert


# ============================================================
# 滑动窗口重叠采样 (论文第3章, p.33, 式3.7)
# ============================================================
def sliding_window(signal: np.ndarray, window_size: int, step: int) -> np.ndarray:
    """
    滑动窗口重叠采样，增加样本数量。
    论文式3.7: n = [(N-1)/s] + 1
    其中 N=102400, s=500, window_size 由 N/s 推导约为1024。

    Args:
        signal:  (total_len,) 一维信号
        window_size: 每个样本的窗口长度
        step: 滑动步长 s

    Returns:
        (num_samples, window_size) 采样后的样本矩阵
    """
    total_len = len(signal)
    n_samples = (total_len - window_size) // step + 1
    samples = np.zeros((n_samples, window_size), dtype=signal.dtype)
    for i in range(n_samples):
        start = i * step
        samples[i] = signal[start:start + window_size]
    return samples


# ============================================================
# PAA 分段聚合近似 (论文第4章, p.40-41, 式4.3)
# ============================================================
def paa_reduce(signal: np.ndarray, num_segments: int) -> np.ndarray:
    """
    分段聚合近似 (Piecewise Aggregate Approximation)。
    将长度为 n 的信号划分为 m 个等长区间，每区间取均值。
    论文式4.3: q_i = (1/k) * Σ s_j

    Args:
        signal:  (n,) 一维信号
        num_segments: 分段数 m

    Returns:
        (m,) 降维后的序列
    """
    n = len(signal)
    segment_len = n // num_segments
    if segment_len == 0:
        raise ValueError(f"num_segments ({num_segments}) must be <= signal length ({n})")
    # 截断到整除长度
    signal = signal[:segment_len * num_segments]
    segments = signal.reshape(num_segments, segment_len)
    return segments.mean(axis=1)


# ============================================================
# GAF 格拉姆角场 (论文第4章, p.40, 式4.1-4.2)
# ============================================================
def gaf_encode(signal: np.ndarray, method: str = "summation") -> np.ndarray:
    """
    格拉姆角场 (Gramian Angular Field) 编码。
    将一维时间序列编码为二维 GAF 矩阵。
    论文式4.1: x̃_t = (x_t - mean(X)) / (max(X) - min(X))  归一化到 [-1,1]
    论文式4.2: φ = arccos(x̃_t), r = t/N                        极坐标映射

    Args:
        signal:  (n,) 一维信号（已可选经PAA降维）
        method: "summation" (GASF) 或 "difference" (GADF)

    Returns:
        (n, n) GAF矩阵
    """
    # 归一化到 [-1, 1] (式4.1)
    signal_min = signal.min()
    signal_max = signal.max()
    if signal_max - signal_min < 1e-10:
        # 常数信号的边界情况
        normalized = np.zeros_like(signal)
    else:
        normalized = (signal - signal.mean()) / (signal_max - signal_min)
        normalized = np.clip(normalized, -1.0, 1.0)

    # 极坐标映射 (式4.2)
    phi = np.arccos(normalized)  # φ ∈ [0, π]

    # GAF矩阵
    if method == "summation":
        # GASF: cos(φ_i + φ_j)
        gaf = np.cos(phi[:, None] + phi[None, :])
    elif method == "difference":
        # GADF: sin(φ_i - φ_j)
        gaf = np.sin(phi[:, None] - phi[None, :])
    else:
        raise ValueError(f"Unknown GAF method: {method}")

    return gaf


# ============================================================
# FFT 快速傅里叶变换 (论文第5章, p.54, 式5.11)
# ============================================================
def fft_transform(signal: np.ndarray, n_fft: int = None) -> np.ndarray:
    """
    快速傅里叶变换，将时域信号转换到频域。
    论文式5.11: X(k) = Σ x(n) * W_N^{kn}

    Args:
        signal:  (n,) 一维时域信号
        n_fft:   FFT点数，默认等于信号长度

    Returns:
        (n_fft//2 + 1,) 单边频谱幅值
    """
    if n_fft is None:
        n_fft = len(signal)
    spectrum = fft(signal, n=n_fft)
    magnitude = np.abs(spectrum[:n_fft // 2 + 1])
    return magnitude


# ============================================================
# VMD 变分模态分解 (论文第5章, p.54, 5.2.6节)
# ============================================================
def vmd_decompose(
    signal: np.ndarray,
    K: int = 3,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7,
    max_iter: int = 500,
    init: int = 1,
    DC: int = 0,
) -> np.ndarray:
    """
    变分模态分解 (Variational Mode Decomposition)。
    论文第5章用VMD提取时域特征，再输入CoAtNet。
    论文参数参考了文献[98]，此处实现标准VMD算法。

    Args:
        signal:  (n,) 一维信号
        K:       模态数（论文未明确，文献[98]参考值）
        alpha:   惩罚因子
        tau:     噪声容忍度
        tol:     收敛容差
        max_iter:最大迭代次数
        init:    初始化方式 (1=all zero, 2=all one, 3=random)
        DC:      是否包含直流分量

    Returns:
        (K, n) 各模态分量矩阵
    """
    n = len(signal)
    # 频域初始化
    f = np.fft.fftfreq(n)[:n // 2 + 1]
    omega_k = np.linspace(0, 0.5, K)  # 初始中心频率均匀分布

    # 信号频谱
    f_hat = np.fft.fft(signal)

    # 初始化模态 (频域)
    if init == 1:
        u_hat = np.zeros((K, n), dtype=complex)
        for k in range(K):
            u_hat[k] = f_hat / K
    else:
        u_hat = np.random.randn(K, n) + 1j * np.random.randn(K, n)
        u_hat *= np.abs(f_hat)[None, :] / np.abs(u_hat).max(axis=1, keepdims=True)

    u_hat_prev = u_hat.copy()
    omega_k_prev = omega_k.copy()

    # 拉格朗日乘子 (频域)
    lambda_hat = np.zeros(n, dtype=complex)

    # 预计算频率平方
    freq_sq = np.arange(n) ** 2

    for it in range(max_iter):
        # 更新每个模态
        for k in range(K):
            # 计算残差
            sum_u = np.sum(u_hat, axis=0) - u_hat[k]
            residual = f_hat - sum_u + lambda_hat / 2

            # Wiener滤波更新模态 (频域)
            u_hat[k] = residual / (1 + alpha * (freq_sq - omega_k[k]) ** 2)

            # 更新中心频率
            power = np.abs(u_hat[k]) ** 2
            omega_k[k] = np.sum(freq_sq[:n // 2 + 1] * power[:n // 2 + 1]) / (np.sum(power[:n // 2 + 1]) + 1e-10)

        # 更新拉格朗日乘子
        if tau > 0:
            sum_u = np.sum(u_hat, axis=0)
            lambda_hat = lambda_hat + tau * (f_hat - sum_u)

        # 收敛检查
        diff = np.sum(np.abs(u_hat - u_hat_prev) ** 2) / (np.sum(np.abs(u_hat_prev) ** 2) + 1e-10)
        if diff < tol:
            break

        u_hat_prev = u_hat.copy()
        omega_k_prev = omega_k.copy()

    # 逆变换到实域
    modes = np.zeros((K, n))
    for k in range(K):
        modes[k] = np.fft.ifft(u_hat[k]).real

    return modes
