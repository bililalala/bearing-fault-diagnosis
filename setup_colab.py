#!/usr/bin/env python
"""Colab 一键部署脚本——复制到 Colab 运行即可重建整个项目。"""
import os

PROJECT = '/content/bearing-fault-diagnosis'

# 创建目录
for d in ['', 'configs', 'data', 'models', 'utils']:
    os.makedirs(os.path.join(PROJECT, d), exist_ok=True)

def w(rel, content):
    path = os.path.join(PROJECT, rel)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'  ✓ {rel}')

# =============================================================
w('configs/chapter3_dual_cnn_bigru.yaml', r'''# ============================================================
# 第3章：基于双通道大核CNN和BIGRU的轴承故障诊断方法
# 论文位置：第27-38页
# ============================================================

dataset:
  name: "CWRU"
  data_dir: "/content/CWRU"
  sample_freq: 12000
  signal_length: 102400
  window_size: 1024
  sliding_step: 500
  samples_per_class: 200
  train_ratio: 0.7
  num_classes: 10
  load_conditions: [0, 1, 2, 3]

model:
  name: "DualCNNBiGRU"
  conv_block1:
    kernel_size: 31
    out_channels: 64
    num_layers: 2
  conv_block2:
    kernel_size: 6
    out_channels: 64
    num_blocks: 3
    num_layers_per_block: 2
  bigru:
    hidden_size: 128
    num_layers: 2
    bidirectional: true
    dropout: 0.0
  attention:
    type: "soft"
  classifier:
    dropout_rate: 0.5

training:
  optimizer: "adam"
  learning_rate: 0.001
  batch_size: 64
  epochs: 50
  loss: "label_smoothing"
  label_smoothing_alpha: 0.1
  weight_decay: 0.0
  grad_clip: null
  early_stopping:
    enabled: true
    patience: 10
    monitor: "val_loss"

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1"]
  t_sne:
    enabled: true
    perplexity: 30
  confusion_matrix: true
''')
