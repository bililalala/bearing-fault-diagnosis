"""
第4章模型：基于改进GAF与Swin-Transformer的轴承故障诊断方法
============================================================
论文第39-49页

管线:
    一维振动信号 → PAA降维 → GAF编码 → 二维图像 → Swin-Transformer → 分类

模型架构 (论文图4.2):
    Patch Partition → Linear Embedding
    Stage1: Swin-T Block (W-MSA) + Swin-T Block (SW-MSA)
    Stage2: Patch Merging ↓ + Swin-T Block ×2
    Stage3: Patch Merging ↓ + Swin-T Block ×6
    Stage4: Patch Merging ↓ + Swin-T Block ×2
    LayerNorm → GAP → FC → 10类

论文式4.4-4.7: Swin-Transformer Block
论文式4.8: Attention(Q,K,V) = SoftMax(QK^T/√d + B)V
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# ============================================================
# Patch Embedding (论文图4.2 Patch Partition)
# ============================================================
class PatchEmbed(nn.Module):
    """
    将输入图像分割成不重叠的 patch 并线性投影。
    论文p.42: 每个 patch 被视为一个独立区域并创造掩码作为绝对位置编码。
    """

    def __init__(self, img_size: int = 64, patch_size: int = 4,
                 in_channels: int = 3, embed_dim: int = 96):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, C, H, W)
        Returns:
            (batch, num_patches, embed_dim)
        """
        x = self.proj(x)  # (batch, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)
        return x


# ============================================================
# 窗口多头自注意力 (W-MSA / SW-MSA)
# ============================================================
class WindowAttention(nn.Module):
    """
    窗口多头自注意力。
    论文式4.8: Attention(Q,K,V) = SoftMax(QK^T/√d + B)V
    其中 B 为相对位置偏置。
    """

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置表 (论文p.43, 式4.8中的B)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        # 相对位置索引
        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing="ij"))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    (batch * num_windows, window_size^2, dim)
            mask: 窗口注意力mask
        Returns:
            (batch * num_windows, window_size^2, dim)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B, heads, N, N)

        # 相对位置偏置
        rel_pos_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rel_pos_bias = rel_pos_bias.view(self.window_size ** 2, self.window_size ** 2, -1)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()
        attn = attn + rel_pos_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ============================================================
# Swin-Transformer Block (论文图4.3)
# ============================================================
class SwinTransformerBlock(nn.Module):
    """
    Swin-Transformer 基础块。
    论文式4.4-4.7:
        ẑ_l   = W-MSA(LN(z_{l-1})) + z_{l-1}
        z_l    = MLP(LN(ẑ_l)) + ẑ_l
        ẑ_{l+1} = SW-MSA(LN(z_l)) + z_l
        z_{l+1} = MLP(LN(ẑ_{l+1})) + ẑ_{l+1}
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 7,
                 shift_size: int = 0, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, attn_drop, drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden, drop=drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (batch, H*W, dim)
            H, W: 当前特征图高宽
        Returns:
            (batch, H*W, dim)
        """
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # 循环移位 (SW-MSA)
        if self.shift_size > 0:
            x = x.view(B, H, W, C)
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x = x.view(B, H * W, C)

        # 窗口划分
        x_windows = self._window_partition(x, H, W)  # (B*nW, window_size^2, C)
        # 窗口注意力
        attn_mask = self._get_attn_mask(H, W, x.device) if self.shift_size > 0 else None
        attn_out = self.attn(x_windows, attn_mask)
        # 窗口合并
        x = self._window_reverse(attn_out, H, W)

        # 逆循环移位
        if self.shift_size > 0:
            x = x.view(B, H, W, C)
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def _window_partition(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """将特征图划分为不重叠的窗口。"""
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        ws = self.window_size
        # 填充到窗口大小的整数倍
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = H + pad_b, W + pad_r
        x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, ws * ws, C)
        return windows

    def _window_reverse(self, windows: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """将窗口合并回特征图。"""
        ws = self.window_size
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        Hp, Wp = H + pad_b, W + pad_r
        B = int(windows.shape[0] / (Hp * Wp / ws / ws))
        x = windows.view(B, Hp // ws, Wp // ws, ws, ws, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        return x.view(B, H * W, -1)

    def _get_attn_mask(self, H: int, W: int, device) -> Optional[torch.Tensor]:
        """生成SW-MSA的注意力mask。"""
        ws = self.window_size
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        if pad_r == 0 and pad_b == 0 and self.shift_size == 0:
            return None

        Hp, Wp = H + pad_b, W + pad_r
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (slice(0, -ws), slice(-ws, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -ws), slice(-ws, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self._window_partition(img_mask.view(1, Hp * Wp, -1), Hp, Wp)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


# ============================================================
# MLP (论文图4.4)
# ============================================================
class Mlp(nn.Module):
    """
    多层感知器 (MLP)。
    论文p.42: 使用平铺层(LN)和高斯线性激活函数(GELU)。
    """

    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================================
# DropPath (Stochastic Depth)
# ============================================================
class DropPath(nn.Module):
    """随机深度。"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ============================================================
# Patch Merging (下采样)
# ============================================================
class PatchMerging(nn.Module):
    """
    Patch合并层，将2×2相邻patch合并并降维。
    Swin-T 的层次化下采样模块。
    """

    def __init__(self, dim: int, out_dim: int = None):
        super().__init__()
        out_dim = out_dim or dim * 2
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (batch, H*W, dim)
            H, W: 当前特征图尺寸
        Returns:
            (batch, (H/2)*(W/2), out_dim), H/2, W/2
        """
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # 2×2 合并
        x0 = x[:, 0::2, 0::2, :]  # 左上
        x1 = x[:, 1::2, 0::2, :]  # 右上
        x2 = x[:, 0::2, 1::2, :]  # 左下
        x3 = x[:, 1::2, 1::2, :]  # 右下
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2


# ============================================================
# Swin-Transformer Stage
# ============================================================
class SwinStage(nn.Module):
    """一个Swin-Transformer阶段，包含多个Block和可选的PatchMerging。"""

    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int = 7,
                 mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, downsample: bool = False, out_dim: int = None):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim, num_heads=num_heads, window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
            )

        if downsample:
            self.downsample = PatchMerging(dim, out_dim)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        for blk in self.blocks:
            x = blk(x, H, W)
        if self.downsample:
            x, H, W = self.downsample(x, H, W)
        return x, H, W


# ============================================================
# 完整 PGAF-Swin 模型
# ============================================================
class PGAFSwin(nn.Module):
    """
    基于改进GAF与Swin-Transformer的故障诊断模型 (论文图4.5)。

    输入: (batch, 3, H, W) GAF编码后的二维图像
    输出: (batch, num_classes) logits
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 96,
        depths: list = None,
        num_heads: list = None,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        depths = depths or [2, 2, 6, 2]
        num_heads = num_heads or [3, 6, 12, 24]
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (len(depths) - 1))

        # Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_drop = nn.Dropout(drop_rate)

        # 绝对位置编码 (论文p.42)
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)

        # Stochastic Depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 四个Stage
        H = W = img_size // patch_size
        self.stage1 = SwinStage(
            dim=embed_dim, depth=depths[0], num_heads=num_heads[0],
            window_size=window_size, mlp_ratio=mlp_ratio, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[0:depths[0]],
            downsample=True, out_dim=embed_dim * 2,
        )
        H, W = H // 2, W // 2
        self.stage2 = SwinStage(
            dim=embed_dim * 2, depth=depths[1], num_heads=num_heads[1],
            window_size=window_size, mlp_ratio=mlp_ratio, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[depths[0]:sum(depths[:2])],
            downsample=True, out_dim=embed_dim * 4,
        )
        H, W = H // 2, W // 2
        self.stage3 = SwinStage(
            dim=embed_dim * 4, depth=depths[2], num_heads=num_heads[2],
            window_size=window_size, mlp_ratio=mlp_ratio, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
            downsample=True, out_dim=embed_dim * 8,
        )
        H, W = H // 2, W // 2
        self.stage4 = SwinStage(
            dim=embed_dim * 8, depth=depths[3], num_heads=num_heads[3],
            window_size=window_size, mlp_ratio=mlp_ratio, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
            downsample=False,
        )

        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, H, W) GAF图像
        Returns:
            (batch, num_classes) logits
        """
        # 如果输入是1通道GAF图像，复制到3通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        H = x.shape[2] // self.patch_embed.patch_size[0]
        W = x.shape[3] // self.patch_embed.patch_size[1]

        x = self.patch_embed(x)
        x = self.pos_drop(x + self.absolute_pos_embed)

        x, H, W = self.stage1(x, H, W)
        x, H, W = self.stage2(x, H, W)
        x, H, W = self.stage3(x, H, W)
        x, H, W = self.stage4(x, H, W)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2)).squeeze(-1)
        x = self.head(x)
        return x


# ============================================================
# 工厂函数
# ============================================================
def create_pgaf_swin(config: dict) -> PGAFSwin:
    """从配置字典创建PGAF-Swin模型。"""
    swin_cfg = config["model"]["swin"]
    return PGAFSwin(
        img_size=config["preprocessing"]["image"]["size"],
        patch_size=swin_cfg["patch_size"],
        in_channels=config["preprocessing"]["image"]["channels"],
        num_classes=config["model"]["num_classes"],
        embed_dim=swin_cfg["embed_dim"],
        depths=swin_cfg["depths"],
        num_heads=swin_cfg["num_heads"],
        window_size=swin_cfg["window_size"],
        mlp_ratio=swin_cfg["mlp_ratio"],
        drop_rate=swin_cfg["drop_rate"],
        attn_drop_rate=swin_cfg["attn_drop_rate"],
        drop_path_rate=swin_cfg["drop_path_rate"],
    )
