"""
CWRU 滚动轴承数据集加载类
用于第3章（双通道大核CNN+BiGRU）和第4章（GAF+Swin-Transformer）
============================================================
论文引用：
- 第3章 p.32-33: 数据集介绍
- 第4章 p.44-45: 数据集介绍
CWRU数据格式：每个故障状态对应一个 .mat 文件
下载地址：https://engineering.case.edu/bearingdatacenter/download-data-file
"""
import os
import glob
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset


# CWRU 12kHz 驱动端数据文件命名模式 (需根据实际下载文件调整)
# 典型文件: XXXX_DE_time, XXXX_FE_time, XXXX_BA_time
# 故障类型和损伤直径对应关系
#   内圈故障 (IR): IR007, IR014, IR021
#   外圈故障 (OR): OR007@6, OR014@6, OR021@6
#   滚动体故障 (B): B007, B014, B021
#   正常 (Normal): Normal_X


class CWRUDataset(Dataset):
    """
    CWRU 轴承数据集。

    论文表3.1 数据集介绍:
        - 正常: 标签0
        - 内圈故障: 0.007inch→1, 0.014inch→2, 0.021inch→3
        - 外圈故障: 0.007inch→4, 0.014inch→5, 0.021inch→6
        - 滚动体故障: 0.007inch→7, 0.014inch→8, 0.021inch→9
        共10类
    """

    # 文件标识到标签的映射 (12kHz DE, 工况0/1/2/3)
    FILE_PATTERNS = {
        "normal": 0,
        "IR007": 1, "IR014": 2, "IR021": 3,
        "OR007": 4, "OR014": 5, "OR021": 6,
        "B007":  7, "B014":  8, "B021":  9,
    }

    def __init__(
        self,
        data_dir: str,
        load_condition: int = None,     # None=所有工况, 0/1/2/3=指定工况
        window_size: int = 1024,
        sliding_step: int = 500,
        samples_per_class: int = 200,
        train: bool = True,
        train_ratio: float = 0.7,
        transform=None,
    ):
        """
        Args:
            data_dir:         CWRU数据根目录 (含 .mat 文件)
            load_condition:   工况 0/1/2/3, 或None加载全部
            window_size:      每个样本长度
            sliding_step:     滑动步长 (论文 s=500)
            samples_per_class:每类最大样本数 (论文200)
            train:            True=训练集, False=测试集
            train_ratio:      训练集比例 (论文0.7)
            transform:        可选的变换函数
        """
        self.data_dir = data_dir
        self.load_condition = load_condition
        self.window_size = window_size
        self.sliding_step = sliding_step
        self.samples_per_class = samples_per_class
        self.train = train
        self.train_ratio = train_ratio
        self.transform = transform

        self.samples = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        """加载并预处理CWRU数据。"""
        # 递归搜索子目录 — 兼容嵌套的目录结构
        mat_files = glob.glob(os.path.join(self.data_dir, "**", "*.mat"), recursive=True)
        if not mat_files:
            raise FileNotFoundError(
                f"No .mat files found in {self.data_dir}. "
                "Please download CWRU dataset from "
                "https://engineering.case.edu/bearingdatacenter/download-data-file"
            )

        for mat_path in mat_files:
            fname = os.path.basename(mat_path)

            # 识别故障类型（不区分大小写）
            label = None
            for pattern, lbl in self.FILE_PATTERNS.items():
                if pattern.lower() in fname.lower():
                    label = lbl
                    break
            if label is None:
                continue  # 跳过无法识别的文件

            # 加载数据
            mat_data = loadmat(mat_path)
            # 变量名格式: X{number}_DE_time, 如 X118_DE_time
            signal = None
            for key in mat_data:
                val = mat_data[key]
                if isinstance(val, np.ndarray) and val.size > 1000:
                    if "DE_time" in key:
                        signal = val.flatten()
                        break

            if signal is None:
                # fallback: 取第一个足够长的一维数组
                for key in mat_data:
                    val = mat_data[key]
                    if isinstance(val, np.ndarray) and val.size > 1000 and val.ndim <= 2:
                        signal = val.flatten()
                        break

            if signal is None or len(signal) < self.window_size:
                continue

            # 如果文件含多个工况，根据 load_condition 筛选
            # CWRU通常一个文件对应一个工况，此处做简单处理

            # 滑动窗口采样
            n_windows = min(
                (len(signal) - self.window_size) // self.sliding_step + 1,
                self.samples_per_class,
            )
            for i in range(n_windows):
                start = i * self.sliding_step
                window = signal[start:start + self.window_size]
                self.samples.append(window)
                self.labels.append(label)

        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        # 按训练/测试划分
        indices = np.arange(len(self.samples))
        np.random.seed(42)
        np.random.shuffle(indices)
        split = int(len(indices) * self.train_ratio)
        if self.train:
            take = indices[:split]
        else:
            take = indices[split:]

        self.samples = self.samples[take]
        self.labels = self.labels[take]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.samples[idx]).unsqueeze(0)  # (1, window_size)
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


class CWRUDatasetFewShot(CWRUDataset):
    """
    小样本CWRU数据集 (论文第3章 p.35-37 小样本对比实验)。
    通过 alpha 参数控制训练样本比例。

    论文设置:
        alpha ∈ {0.1, 0.2, 0.4, 0.5, 0.7}
        其中 alpha=0.7 为正常样本量
    """

    def __init__(self, alpha: float = 0.1, **kwargs):
        super().__init__(train=True, **kwargs)
        self._filter_by_alpha(alpha)

    def _filter_by_alpha(self, alpha: float):
        """按比例 alpha 采样每个类别。"""
        all_indices = np.arange(len(self.samples))
        selected = []
        for cls in np.unique(self.labels):
            cls_indices = all_indices[self.labels == cls]
            n_select = max(1, int(len(cls_indices) * alpha))
            selected.append(
                np.random.choice(cls_indices, size=n_select, replace=False)
            )
        selected = np.concatenate(selected)
        self.samples = self.samples[selected]
        self.labels = self.labels[selected]
