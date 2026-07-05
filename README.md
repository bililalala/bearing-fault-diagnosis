# 基于深度学习和时频域特征融合的滚动轴承故障诊断研究

复现框架 —— 刘鹏 硕士论文 (内蒙古科技大学, 2025)

## 三种方法

| 脚本 | 方法 | 数据集 | 核心组件 |
|------|------|--------|----------|
| `train_chapter3.py` | 双通道大核CNN + BiGRU + 注意力 | CWRU | LargeKernelConv, BiGRU, SoftAttention, LabelSmoothing |
| `train_chapter4.py` | 改进GAF + Swin-Transformer | CWRU | PAA降维, GAF编码, Swin-T, W-MSA/SW-MSA |
| `train_chapter5.py` | CoAtNet-BiLSTM + ECA + Cross-Attention | Paderborn / 试验台 | VMD, FFT, CoAtNet, ECA-net, Cross-Attention |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 下载数据集后运行
python train_chapter3.py --data_dir /path/to/CWRU
python train_chapter4.py --data_dir /path/to/CWRU
python train_chapter5.py --data_dir /path/to/Paderborn
```

## 项目结构

```
├── configs/          # 三个YAML配置文件（含论文所有超参数）
├── data/             # 数据集加载 + 预处理函数
├── models/           # 三个方法的模型实现
├── utils/            # 共用工具（指标/可视化/早停/LSR损失）
├── train_chapter3.py # 方法一独立脚本
├── train_chapter4.py # 方法二独立脚本
├── train_chapter5.py # 方法三独立脚本
└── requirements.txt
```

## 数据集下载

- **CWRU**: https://engineering.case.edu/bearingdatacenter/download-data-file
- **Paderborn**: https://mb.uni-paderborn.de/kat/forschung/kat-datencenter/bearing-datacenter

## 注意事项

- 论文中部分超参数（如 VMD 的 K/α、LSR 平滑因子、PAA 分段数）未明确给出，在配置文件中用 `???` 标注，需根据参考文献反查或实验确定。
- 建议先复现第3章方法（最简单），再逐步推进第4章和第5章。
