#!/usr/bin/env python
"""
第5章训练脚本：融合通道注意力机制的CoAtNet-BiLSTM故障诊断
============================================================
论文第50-63页

双路并行: 信号→VMD(时域)→CoAtNet ─┐
         信号→FFT(频域)→ECA-BiLSTM ─┤ 交叉注意力融合→分类

用法:
    python train_chapter5.py --data_dir /path/to/Paderborn --config configs/chapter5_coatnet_bilstm.yaml
    python train_chapter5.py --data_dir /path/to/Paderborn --eval_only --checkpoint checkpoint_ch5.pt
    python train_chapter5.py --data_dir /path/to/Paderborn --noise_test

依赖: pip install torch numpy scipy pyyaml scikit-learn matplotlib tqdm
"""
import os
import sys
import argparse
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="第5章: CoAtNet-BiLSTM + ECA + Cross-Attention 故障诊断")
    parser.add_argument("--config", type=str, default="configs/chapter5_coatnet_bilstm.yaml", help="配置文件路径")
    parser.add_argument("--data_dir", type=str, required=True, help="帕德博恩/试验台数据集目录")
    parser.add_argument("--output_dir", type=str, default="outputs/chapter5", help="输出目录")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型checkpoint路径")
    parser.add_argument("--eval_only", action="store_true", help="仅评估")
    parser.add_argument("--noise_test", action="store_true", help="噪声鲁棒性实验 (论文p.61)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def preprocess_signal(signal, config):
    """VMD + FFT 双路处理。"""
    from data.preprocessing import vmd_decompose, fft_transform
    vmd_cfg = config["preprocessing"]["vmd"]
    K = vmd_cfg["K"]; alpha = vmd_cfg["alpha"]
    if K == "???" or alpha == "???":
        K = 3; alpha = 2000.0
    modes = vmd_decompose(signal, K=int(K) if isinstance(K, str) else K,
                           alpha=float(alpha) if isinstance(alpha, str) else alpha)
    time_feat = modes[0]
    n_fft_val = config["preprocessing"]["fft"]["n_fft"]
    if n_fft_val == "???":
        n_fft_val = len(signal)
    freq_feat = fft_transform(signal, n_fft=int(n_fft_val) if isinstance(n_fft_val, str) else n_fft_val)
    return time_feat, freq_feat


class DualPathDataset:
    """双路数据集: 原始信号 → (time_feat, freq_feat, label)"""
    def __init__(self, raw_dataset, config):
        import torch
        from tqdm import tqdm
        self.samples = []
        self.segment_len = config["preprocessing"]["signal"]["segment_length"]
        print("  预处理 VMD + FFT ...")
        for signal, label in tqdm(list(raw_dataset)):
            signal_np = signal.squeeze(0).numpy()
            if len(signal_np) > self.segment_len:
                signal_np = signal_np[:self.segment_len]
            else:
                signal_np = np.pad(signal_np, (0, self.segment_len - len(signal_np)))
            try:
                time_feat, freq_feat = preprocess_signal(signal_np, config)
                time_feat = np.pad(time_feat[:self.segment_len], (0, max(0, self.segment_len - len(time_feat))))
                self.samples.append((torch.from_numpy(time_feat).float().unsqueeze(0),
                                     torch.from_numpy(freq_feat).float(), label))
            except Exception:
                continue
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    import torch
    time_signals = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[2] for item in batch])
    freq_signals = [item[1] for item in batch]
    max_len = max(f.shape[0] for f in freq_signals)
    freq_padded = torch.zeros(len(freq_signals), max_len)
    for i, f in enumerate(freq_signals):
        freq_padded[i, :f.shape[0]] = f
    return time_signals, freq_padded.unsqueeze(1), labels


def train_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    from tqdm import tqdm
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
    for time_sig, freq_sig, y in pbar:
        time_sig, freq_sig, y = time_sig.to(device), freq_sig.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(time_sig, freq_sig), y)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
        all_preds.extend(model(time_sig, freq_sig).argmax(dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    from utils.metrics import compute_metrics
    return total_loss / len(loader), compute_metrics(np.array(all_labels), np.array(all_preds))["accuracy"]


def evaluate(model, loader, criterion, device):
    import torch
    from tqdm import tqdm
    model.eval()
    total_loss, all_preds, all_labels, all_features = 0.0, [], [], []
    with torch.no_grad():
        for time_sig, freq_sig, y in tqdm(loader, desc="Evaluating"):
            time_sig, freq_sig, y = time_sig.to(device), freq_sig.to(device), y.to(device)
            logits = model(time_sig, freq_sig)
            loss = criterion(logits, y)
            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_features.append(logits.cpu().numpy())
    from utils.metrics import compute_metrics
    return (total_loss / len(loader), compute_metrics(np.array(all_labels), np.array(all_preds)),
            np.concatenate(all_features, axis=0), np.array(all_labels))


def noise_robustness_test(model, raw_dataset, config, device, output_dir):
    """噪声鲁棒性实验 (论文p.61 图5.12)"""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from utils.common import add_gaussian_noise
    snr_levels = config["noise_experiment"]["snr_levels"]
    criterion = torch.nn.CrossEntropyLoss()
    results = {}
    print("\n--- 噪声鲁棒性实验 ---")
    for snr in snr_levels:
        noisy_data, noisy_labels = [], []
        for signal, label in raw_dataset:
            noisy = add_gaussian_noise(signal, snr).numpy()
            noisy_data.append(noisy)
            noisy_labels.append(label)
        loader = DataLoader(TensorDataset(
            torch.from_numpy(np.array(noisy_data)).float().unsqueeze(1),
            torch.tensor(noisy_labels)),
            batch_size=config["training"]["batch_size"])
        _, metrics, _, _ = evaluate(model, loader, criterion, device)
        results[snr] = metrics["accuracy"]
        print(f"  SNR={snr}dB: Accuracy={metrics['accuracy']:.4f}")
    return results


def main():
    args = parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from data.paderborn import PaderbornDataset
    from models.chapter5 import CoAtNetAMBiLSTM
    from utils.common import set_seed, get_device, EarlyStopping
    from utils.visualize import plot_training_curves, plot_confusion_matrix, plot_tsne
    from utils.metrics import confusion_matrix as cm_fn

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print(f"第5章: CoAtNet-BiLSTM + ECA + Cross-Attention 故障诊断")
    print(f"设备: {device}")
    print("=" * 60)

    # 原始数据集
    ds_kwargs = {"data_dir": args.data_dir, "train_ratio": config["dataset"]["train_ratio"],
                 "val_ratio": config["dataset"]["val_ratio"],
                 "segment_length": config["preprocessing"]["signal"]["segment_length"]}
    raw_train = PaderbornDataset(train=True, val=False, **ds_kwargs)
    raw_test = PaderbornDataset(train=False, val=False, **ds_kwargs)

    # 双路预处理
    train_dataset = DualPathDataset(raw_train, config)
    test_dataset = DualPathDataset(raw_test, config)
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"],
                             shuffle=False, collate_fn=collate_fn)
    print(f"  训练集: {len(train_dataset)} 样本, 测试集: {len(test_dataset)} 样本")

    # 模型
    model = CoAtNetAMBiLSTM(
        num_classes=config["model"]["num_classes"],
        coatnet_in_channels=1,
        coatnet_stages=config["model"]["coatnet"]["stages"],
        conv_channels=config["model"]["eca_bilstm"]["conv1d_channels"],
        eca_kernel_size=config["model"]["eca_bilstm"]["eca_kernel_size"],
        bilstm_hidden=config["model"]["eca_bilstm"]["bilstm_hidden_size"],
        bilstm_layers=config["model"]["eca_bilstm"]["bilstm_num_layers"],
        cross_attn_heads=config["model"]["cross_attention"]["num_heads"],
        cross_attn_dropout=config["model"]["cross_attention"]["dropout"],
        classifier_dropout=config["model"]["classifier_dropout"],
    ).to(device)
    print(f"  参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"],
                                 weight_decay=config["training"].get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=config["training"]["early_stopping"]["patience"])

    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state_dict"])

    if args.noise_test:
        results = noise_robustness_test(model, raw_test, config, device, args.output_dir)
        with open(os.path.join(args.output_dir, "noise_results.txt"), "w") as f:
            for snr, acc in results.items():
                f.write(f"SNR={snr}dB: {acc:.4f}\n")
        return

    if args.eval_only:
        test_loss, metrics, features, labels = evaluate(model, test_loader, criterion, device)
        print(f"\n  测试集: Acc={metrics['accuracy']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        cm = cm_fn(labels, np.argmax(features, axis=1))
        class_names = [config["dataset"]["labels"][i] for i in range(config["model"]["num_classes"])]
        plot_confusion_matrix(cm, class_names, os.path.join(args.output_dir, "confusion_matrix_ch5.png"))
        plot_tsne(features, labels, class_names, os.path.join(args.output_dir, "tsne_ch5.png"), title="Chapter 5: t-SNE")
        return

    # 训练
    best_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, config["training"]["epochs"])
        val_loss, val_metrics, _, _ = evaluate(model, test_loader, criterion, device)
        val_acc = val_metrics["accuracy"]
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)
        scheduler.step(val_loss)
        print(f"  Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "val_metrics": val_metrics, "config": config},
                       os.path.join(args.output_dir, "best_model_ch5.pt"))
            print(f"  ✓ 保存最佳模型 (Acc={best_acc:.4f})")
        if early_stopping(val_loss):
            print(f"  早停触发于 epoch {epoch}"); break

    print(f"\n最佳验证准确率: {best_acc:.4f}")
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model_ch5.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_metrics, features, labels = evaluate(model, test_loader, criterion, device)
    print(f"最佳模型测试集: Acc={test_metrics['accuracy']:.4f}, P={test_metrics['precision']:.4f}, R={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, os.path.join(args.output_dir, "training_curves_ch5.png"))
    preds = features.argmax(axis=1)
    cm = cm_fn(labels, preds)
    class_names = [config["dataset"]["labels"][i] for i in range(config["model"]["num_classes"])]
    plot_confusion_matrix(cm, class_names, os.path.join(args.output_dir, "confusion_matrix_ch5.png"))
    plot_tsne(features, labels, class_names, os.path.join(args.output_dir, "tsne_ch5.png"), title="Chapter 5: t-SNE")
    print(f"\n所有输出已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
