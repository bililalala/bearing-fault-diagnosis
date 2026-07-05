"""
共用工具函数：种子设置、设备选择、早停机制、LSR损失函数
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 随机种子 (论文未给定，需自行固定并记录)
# ============================================================
def set_seed(seed: int = 42) -> None:
    """固定所有随机种子，保证可重复性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 设备选择
# ============================================================
def get_device() -> torch.device:
    """自动选择可用设备 (CUDA > MPS > CPU)。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# 早停机制
# ============================================================
class EarlyStopping:
    """
    早停机制：监控验证指标，patience 轮无改善则停止。
    论文第3/4/5章均隐含或显式使用了早停。
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = float("inf") if mode == "min" else float("-inf")

    def __call__(self, score: float) -> bool:
        """返回 True 表示应停止训练。"""
        if self.best_score is None:
            self.best_score = score
            self._reset_counter()
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self._reset_counter()
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

    def _reset_counter(self):
        self.counter = 0


# ============================================================
# 标签平滑正则化损失 (论文第3章, p.31, 式3.6)
# ============================================================
class LabelSmoothingLoss(nn.Module):
    """
    标签平滑正则化 (Label Smoothing Regularization, LSR)
    论文第3章用来缓解小样本过拟合 (式3.6):
        L_LSR = -Σ (y_i*(1-α) + α/N) * log(p_i)

    参数:
        alpha: 平滑因子（论文未明确，建议 0.05~0.1）
        num_classes: 类别数
    """

    def __init__(self, num_classes: int, alpha: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (batch, num_classes) 模型输出的 logits
            target: (batch,) 整数类别标签
        Returns:
            标量损失值
        """
        log_probs = F.log_softmax(pred, dim=-1)
        # 构造平滑标签: y_smooth = y*(1-α) + α/N
        smooth_target = torch.full_like(log_probs, self.alpha / self.num_classes)
        smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.alpha + self.alpha / self.num_classes)
        loss = (-smooth_target * log_probs).sum(dim=-1).mean()
        return loss


# ============================================================
# 高斯白噪声添加 (论文第5章噪声鲁棒性实验, p.61)
# ============================================================
def add_gaussian_noise(signal: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    向信号添加指定信噪比的高斯白噪声。

    Args:
        signal: (..., seq_len) 原始信号
        snr_db: 信噪比 (dB)，如 -6, -4, -2, 0

    Returns:
        加噪后的信号
    """
    signal_power = signal.pow(2).mean(dim=-1, keepdim=True)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(signal) * torch.sqrt(noise_power)
    return signal + noise
