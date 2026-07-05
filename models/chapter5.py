"""
第5章模型：融合通道注意力机制的CoAtNet-BiLSTM滚动轴承故障诊断方法
============================================================
论文第50-63页

双路并行架构 + 交叉注意力融合:

    原始一维振动信号
    ├─ VMD分解 → 时域特征 → CoAtNet (1D MBConv + Relative Attention)
    │                                    ↓
    │                              时域特征向量
    │                                    ↓
    │                          ┌── Cross-Attention ──┐
    │                          │   特征融合           │
    └─ FFT → 频域特征 → ECA-BiLSTM ──────────────────┘
                                    ↓
                              FC + Softmax → 5类

论文图5.4: 完整模型结构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# ============================================================
# ECA-net 高效通道注意力 (论文第5章 图5.1)
# ============================================================
class ECANet(nn.Module):
    """
    高效通道注意力 (Efficient Channel Attention)。
    论文p.51 式5.1: g_c = (1/(H*W)) * Σ X_c(h,w)
    使用自适应1D卷积学习通道间关系，k=3 (论文p.51)。
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len) 或 (batch, channels, features)
        Returns:
            (batch, channels, ...) 加权后的特征
        """
        # 全局平均池化 → 1D卷积 → sigmoid
        y = self.gap(x)  # (batch, channels, 1)
        y = y.transpose(1, 2)  # (batch, 1, channels)
        y = self.conv(y)        # (batch, 1, channels)
        y = y.transpose(1, 2)   # (batch, channels, 1)
        y = self.sigmoid(y)
        return x * y


# ============================================================
# 1D 深度可分离卷积 (MBConv)
# ============================================================
class DepthwiseConv1d(nn.Module):
    """逐通道卷积。"""
    def __init__(self, channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              stride=stride, padding=kernel_size // 2,
                              groups=channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck 卷积块 (1D版本)。
    论文p.52-53: CoAtNet的 S1/S2 Stage。
    结构: Expand → DepthwiseConv → SE(可选) → Project
    """

    def __init__(self, in_channels: int, out_channels: int,
                 expand_ratio: int = 4, kernel_size: int = 3,
                 stride: int = 1, use_se: bool = False):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []
        # Expand
        if expand_ratio != 1:
            layers.append(nn.Conv1d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())

        # Depthwise
        layers.append(DepthwiseConv1d(hidden_dim, kernel_size, stride))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())

        # SE (可选)
        if use_se:
            squeeze_dim = max(1, hidden_dim // 4)
            layers.append(SELayer1D(hidden_dim, squeeze_dim))

        # Project
        layers.append(nn.Conv1d(hidden_dim, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm1d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class SELayer1D(nn.Module):
    """1D Squeeze-and-Excitation"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, reduction, 1),
            nn.GELU(),
            nn.Conv1d(reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


# ============================================================
# 相对多头自注意力 (论文p.53 式5.8-5.10)
# ============================================================
class RelativeAttention(nn.Module):
    """
    相对注意力机制 (Relative Attention)。
    论文式5.10: Rel-Attention(Q,K,V) = softmax((QK^T + w)/√d) * V
    其中 w 为相对位置矩阵。
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0,
                 max_len: int = 1024):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # 相对位置偏置 (可学习)
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * max_len - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            (batch, seq_len, dim)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        # 相对位置偏置
        rel_pos = self._get_rel_pos_bias(N, x.device)
        attn = attn + rel_pos

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out

    def _get_rel_pos_bias(self, seq_len: int, device) -> torch.Tensor:
        """生成相对位置偏置矩阵。"""
        idx = torch.arange(seq_len, device=device)
        rel_idx = idx[None, :] - idx[:, None]  # (N, N)
        rel_idx = rel_idx + (self.rel_pos_bias.shape[0] // 2)
        rel_idx = torch.clamp(rel_idx, 0, self.rel_pos_bias.shape[0] - 1)
        return self.rel_pos_bias[rel_idx]  # (N, N)


class RelativeAttentionBlock(nn.Module):
    """
    相对注意力模块 (论文图5.3 S3/S4)。
    结构: LN → Multi-Head Relative Attention → +残差 → LN → FFN → +残差
    """

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RelativeAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# CoAtNet 网络 (论文p.52-53, 图5.3)
# ============================================================
class CoAtNet(nn.Module):
    """
    CoAtNet: CNN + Transformer 混合架构 (1D版本)。

    Stage: S0(Conv) → S1(MBConv) → S2(MBConv) → S3(Rel-Attn) → S4(Rel-Attn)
    论文图5.3: 针对一维信号做了适应性改进。
    """

    def __init__(
        self,
        in_channels: int,
        stages_config: list,
    ):
        """
        Args:
            in_channels: 输入通道数
            stages_config: Stage配置列表
        """
        super().__init__()
        self.stages = nn.ModuleList()
        curr_channels = in_channels

        for stage_cfg in stages_config:
            stype = stage_cfg["type"]
            out_ch = stage_cfg["out_channels"]

            if stype == "conv1d":
                self.stages.append(nn.Sequential(
                    nn.Conv1d(curr_channels, out_ch,
                              kernel_size=stage_cfg.get("kernel_size", 3),
                              stride=stage_cfg.get("stride", 2),
                              padding=stage_cfg.get("kernel_size", 3) // 2,
                              bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                ))
            elif stype == "mbconv":
                self.stages.append(MBConvBlock(
                    in_channels=curr_channels,
                    out_channels=out_ch,
                    expand_ratio=stage_cfg.get("expand_ratio", 4),
                    kernel_size=stage_cfg.get("kernel_size", 3),
                ))
            elif stype == "rel_attention":
                self.stages.append(RelativeAttentionBlock(
                    dim=curr_channels,
                    num_heads=stage_cfg.get("num_heads", 8),
                ))
                out_ch = curr_channels  # Rel-Attn 不改变维度

            curr_channels = out_ch

        self.out_channels = curr_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, out_channels', seq_len') 特征
        """
        for stage in self.stages:
            if isinstance(stage, RelativeAttentionBlock):
                # Rel-Attn 需要 (batch, seq_len, dim)
                x = x.transpose(1, 2)
                x = stage(x)
                x = x.transpose(1, 2)
            else:
                x = stage(x)
        return x


# ============================================================
# ECA-BiLSTM 分支 (论文图5.1 + 5.2)
# ============================================================
class ECABiLSTM(nn.Module):
    """
    融合ECA通道注意力的BiLSTM网络。
    论文p.55: 1×1×16 Conv → BN → ReLU → ECA-net → BiLSTM
    """

    def __init__(self, conv_channels: int = 16,
                 eca_kernel_size: int = 3, bilstm_hidden: int = 128,
                 bilstm_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        # 1×1 卷积 (论文p.55: 1×1×16卷积层), 输入固定为1通道的频域信号
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )
        # ECA注意力
        self.eca = ECANet(conv_channels, kernel_size=eca_kernel_size)

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if bilstm_layers > 1 else 0.0,
        )
        self.output_dim = bilstm_hidden * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, features, seq_len) 频域特征
        Returns:
            (batch, output_dim) BiLSTM最后时间步的输出
        """
        x = self.conv(x)      # (batch, 16, seq_len)
        x = self.eca(x)       # (batch, 16, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, 16)
        out, (h_n, c_n) = self.bilstm(x)
        # 取最后时间步
        final = out[:, -1, :]  # (batch, hidden*2)
        return final


# ============================================================
# 交叉注意力融合 (论文p.53-54)
# ============================================================
class CrossAttentionFusion(nn.Module):
    """
    交叉注意力机制 (Cross-Attention)。
    论文p.53-54: 用于融合时域和频域特征。
    将一路作为query，另一路作为key/value进行交叉注意力计算。
    """

    def __init__(self, dim1: int, dim2: int, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        # 将两个分支投影到统一的融合维度
        fused_dim = max(dim1, dim2)
        self.fused_dim = fused_dim
        self.align1 = nn.Linear(dim1, fused_dim) if dim1 != fused_dim else nn.Identity()
        self.align2 = nn.Linear(dim2, fused_dim) if dim2 != fused_dim else nn.Identity()

        self.num_heads = num_heads
        self.head_dim = fused_dim // num_heads

        # 交叉注意力: 时域特征(query) attend 频域特征(key/value)
        self.q_proj = nn.Linear(fused_dim, fused_dim)
        self.k_proj = nn.Linear(fused_dim, fused_dim)
        self.v_proj = nn.Linear(fused_dim, fused_dim)
        self.out_proj = nn.Linear(fused_dim, fused_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, time_feat: torch.Tensor, freq_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_feat: (batch, dim1) 时域特征 (CoAtNet输出)
            freq_feat: (batch, dim2) 频域特征 (BiLSTM输出)
        Returns:
            (batch, fused_dim) 融合特征
        """
        # 对齐维度
        t = self.align1(time_feat).unsqueeze(1)  # (batch, 1, fused_dim)
        f = self.align2(freq_feat).unsqueeze(1)  # (batch, 1, fused_dim)

        fused_dim = t.shape[-1]
        B = t.shape[0]

        q = self.q_proj(t).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(f).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(f).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, 1, fused_dim)
        out = self.out_proj(out).squeeze(1)
        return out


# ============================================================
# 完整模型: CoAtNet-AMBiLSTM
# ============================================================
class CoAtNetAMBiLSTM(nn.Module):
    """
    融合通道注意力机制的 CoAtNet-BiLSTM 故障诊断模型 (论文图5.4)。

    双路并行:
        - 时域路径: VMD → CoAtNet
        - 频域路径: FFT → ECA-BiLSTM
        - 融合: Cross-Attention → FC → Softmax
    """

    def __init__(
        self,
        num_classes: int = 5,
        # CoAtNet 配置
        coatnet_in_channels: int = 1,
        coatnet_stages: list = None,
        # ECA-BiLSTM 配置
        conv_channels: int = 16,
        eca_kernel_size: int = 3,
        bilstm_hidden: int = 128,
        bilstm_layers: int = 2,
        # 交叉注意力
        cross_attn_heads: int = 8,
        cross_attn_dropout: float = 0.1,
        # 分类头
        classifier_dropout: float = 0.5,
    ):
        super().__init__()

        # 默认 CoAtNet 配置
        if coatnet_stages is None:
            coatnet_stages = [
                {"type": "conv1d",    "out_channels": 64,  "kernel_size": 3, "stride": 2},
                {"type": "mbconv",   "out_channels": 96,  "expand_ratio": 4, "kernel_size": 3},
                {"type": "mbconv",   "out_channels": 192, "expand_ratio": 4, "kernel_size": 3},
                {"type": "rel_attention", "out_channels": 192, "num_heads": 8},
                {"type": "rel_attention", "out_channels": 192, "num_heads": 12},
            ]

        # 时域分支: CoAtNet
        self.coatnet = CoAtNet(coatnet_in_channels, coatnet_stages)
        self.time_gap = nn.AdaptiveAvgPool1d(1)

        # 频域分支: ECA-BiLSTM (输入固定为1通道频域信号)
        self.eca_bilstm = ECABiLSTM(
            conv_channels=conv_channels,
            eca_kernel_size=eca_kernel_size,
            bilstm_hidden=bilstm_hidden,
            bilstm_layers=bilstm_layers,
        )

        # 交叉注意力融合
        self.cross_attn = CrossAttentionFusion(
            dim1=self.coatnet.out_channels,
            dim2=self.eca_bilstm.output_dim,
            num_heads=cross_attn_heads,
            dropout=cross_attn_dropout,
        )

        fused_dim = max(self.coatnet.out_channels, self.eca_bilstm.output_dim)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(fused_dim, num_classes),
        )

    def forward(self, time_signal: torch.Tensor, freq_signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_signal: (batch, 1, seq_len) VMD分解后的时域信号
            freq_signal: (batch, freq_len, seq_len) FFT频域特征
        Returns:
            (batch, num_classes) logits
        """
        # 时域路径
        time_feat = self.coatnet(time_signal)  # (batch, C, seq_len')
        time_feat = self.time_gap(time_feat).squeeze(-1)  # (batch, C)

        # 频域路径
        freq_feat = self.eca_bilstm(freq_signal)  # (batch, D)

        # 交叉注意力融合
        fused = self.cross_attn(time_feat, freq_feat)  # (batch, fused_dim)

        # 分类
        logits = self.classifier(fused)
        return logits


# ============================================================
# 工厂函数
# ============================================================
def create_coatnet_ambilstm(config: dict) -> CoAtNetAMBiLSTM:
    """从配置字典创建 CoAtNet-AMBiLSTM 模型。"""
    model_cfg = config["model"]
    return CoAtNetAMBiLSTM(
        num_classes=model_cfg["num_classes"],
        coatnet_in_channels=1,
        coatnet_stages=model_cfg["coatnet"]["stages"],
        conv_channels=model_cfg["eca_bilstm"]["conv1d_channels"],
        eca_kernel_size=model_cfg["eca_bilstm"]["eca_kernel_size"],
        bilstm_hidden=model_cfg["eca_bilstm"]["bilstm_hidden_size"],
        bilstm_layers=model_cfg["eca_bilstm"]["bilstm_num_layers"],
        cross_attn_heads=model_cfg["cross_attention"]["num_heads"],
        cross_attn_dropout=model_cfg["cross_attention"]["dropout"],
        classifier_dropout=model_cfg["classifier_dropout"],
    )
