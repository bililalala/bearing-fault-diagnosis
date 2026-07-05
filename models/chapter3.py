"""
第3章模型：基于双通道大核CNN和BIGRU的轴承故障诊断方法
============================================================
论文第27-38页

模型架构 (图3.1):
    输入(1D振动信号)
    ├─ Conv Block 1 (大卷积核1×31, 2层) ──┐
    ├─ Conv Block 2.1→2.2→2.3 (小卷积核1×6, 各2层) ──┤
    │                    ↓ 逐元素相乘融合                │
    │              BiGRU (时序特征提取)                  │
    │              全局平均池化 (GAP)                    │
    │              软注意力机制                           │
    │              FC + Softmax → 10类输出               │
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 基础卷积块 (Conv Block)
# ============================================================
class ConvBlock(nn.Module):
    """
    基础卷积块: BN → Conv1d → ReLU → BN → Conv1d → ReLU
    论文第29页图3.2: 每个模块包含若干归一化层(BN)和非线性激活层(ReLU)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: str = "same"):
        super().__init__()
        # 使用字符串 "same" (PyTorch 1.9+) 自动处理奇数/偶数卷积核的padding
        conv_pad = padding if padding == "same" else 0

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=conv_pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=conv_pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


# ============================================================
# 双通道大核CNN特征提取模块 (论文图3.2)
# ============================================================
class DualChannelCNN(nn.Module):
    """
    双通道大核CNN模块。

    Channel 1 (宏观特征):
        Conv Block 1: 2层Conv1d, kernel_size=31 (大卷积核)
        论文p.29: "使网络提取出信号的宏观特征"

    Channel 2 (细微特征):
        Conv Block 2.1 → Conv Block 2.2 → Conv Block 2.3
        每块2层Conv1d, kernel_size=6 (小卷积核串联增加深度)
        论文p.29: "增加了模型深度，提取出信号的细微特征"

    融合: 逐元素相乘 (element-wise multiplication)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 64,
                 kernel_large: int = 31, kernel_small: int = 6):
        super().__init__()
        # Channel 1: 大卷积核路径
        self.channel1 = ConvBlock(in_channels, out_channels, kernel_large)

        # Channel 2: 小卷积核深路径 (三个模块串联)
        self.channel2_block1 = ConvBlock(in_channels, out_channels, kernel_small)
        self.channel2_block2 = ConvBlock(out_channels, out_channels, kernel_small)
        self.channel2_block3 = ConvBlock(out_channels, out_channels, kernel_small)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        feat1 = self.channel1(x)                                                # 大卷积核路径
        feat2 = self.channel2_block3(self.channel2_block2(self.channel2_block1(x)))  # 深路径
        # 论文p.29: "两个通道得到的输出通过逐元素相乘方法来做到特征融合"
        fused = feat1 * feat2
        return fused


# ============================================================
# BiGRU 时序特征提取模块 (论文图3.3)
# ============================================================
class BiGRUModule(nn.Module):
    """
    双向GRU网络，提取轴承振动信号的时序特征。
    论文p.30 式3.1-3.3:
        h_forward  = f(U_forward * h_{t-1} + W_forward * X_t + b_forward)
        h_backward = f(U_backward * h_{t-1} + W_backward * X_t + b_backward)
        h_t = [h_forward, h_backward]
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, seq_len, hidden_size * 2)
        """
        # GRU 期望 (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)
        out, _ = self.gru(x)
        return out  # (batch, seq_len, hidden_size * 2)


# ============================================================
# 软注意力机制 (论文图3.4)
# ============================================================
class SoftAttention(nn.Module):
    """
    软注意力机制。
    论文p.31 式3.4: y = Σ a_i * h_i
    通过为不同输入通道/时间步分配权重，调整模型对输入信息的关注度。
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, feature_dim)
        Returns:
            (batch, feature_dim) 加权聚合后的特征向量
        """
        # 计算注意力权重
        attn_weights = self.attention(x)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # 加权求和 (式3.4)
        weighted = x * attn_weights  # (batch, seq_len, feature_dim)
        out = weighted.sum(dim=1)    # (batch, feature_dim)
        return out


# ============================================================
# 完整模型: 双通道大核CNN + BiGRU + 注意力
# ============================================================
class DualCNNBiGRU(nn.Module):
    """
    基于双通道大核CNN和BIGRU的故障诊断模型 (论文图3.1)。

    输入:   (batch, 1, seq_len) 一维振动信号
    输出:   (batch, num_classes) 各类别预测概率
    """

    def __init__(
        self,
        seq_len: int = 1024,
        in_channels: int = 1,
        cnn_out_channels: int = 64,
        kernel_large: int = 31,
        kernel_small: int = 6,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 2,
        num_classes: int = 10,
        classifier_dropout: float = 0.5,
    ):
        super().__init__()

        # 双通道大核CNN特征提取
        self.dual_cnn = DualChannelCNN(
            in_channels=in_channels,
            out_channels=cnn_out_channels,
            kernel_large=kernel_large,
            kernel_small=kernel_small,
        )

        # BiGRU时序特征提取
        self.bigru = BiGRUModule(
            input_size=cnn_out_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
        )
        gru_output_dim = gru_hidden_size * 2  # 双向

        # 全局平均池化 (论文p.28)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # 软注意力 (论文p.31)
        self.attention = SoftAttention(feature_dim=gru_output_dim)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(gru_output_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, seq_len) 振动信号
        Returns:
            (batch, num_classes) logits
        """
        # 双通道CNN特征提取
        cnn_feat = self.dual_cnn(x)  # (batch, 64, seq_len)

        # BiGRU时序特征提取
        gru_feat = self.bigru(cnn_feat)  # (batch, seq_len, hidden*2)

        # GAP: 在seq_len维度池化
        # gru_feat: (batch, seq_len, hidden*2) → (batch, hidden*2, seq_len)
        pooled = self.gap(gru_feat.transpose(1, 2)).squeeze(-1)  # (batch, hidden*2)

        # 注意力机制 (论文放在GAP之后, 对各时间步加权)
        attended = self.attention(gru_feat)  # (batch, hidden*2)

        # 融合GAP特征和注意力特征
        combined = pooled + attended  # 残差连接

        # 分类
        logits = self.classifier(combined)
        return logits


# ============================================================
# 工厂函数
# ============================================================
def create_dual_cnn_bigru(config: dict) -> DualCNNBiGRU:
    """从配置字典创建模型。"""
    return DualCNNBiGRU(
        seq_len=config.get("seq_len", 1024),
        in_channels=1,
        cnn_out_channels=config["model"]["conv_block1"]["out_channels"],
        kernel_large=config["model"]["conv_block1"]["kernel_size"],
        kernel_small=config["model"]["conv_block2"]["kernel_size"],
        gru_hidden_size=config["model"]["bigru"]["hidden_size"],
        gru_num_layers=config["model"]["bigru"]["num_layers"],
        num_classes=config["dataset"]["num_classes"],
        classifier_dropout=config["model"]["classifier"]["dropout_rate"],
    )
