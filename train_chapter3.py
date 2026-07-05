#!/usr/bin/env python
"""
第3章训练脚本：基于双通道大核CNN和BIGRU的轴承故障诊断
============================================================
论文第27-38页

用法:
    python train_chapter3.py --data_dir /path/to/CWRU --config configs/chapter3_dual_cnn_bigru.yaml
    python train_chapter3.py --data_dir /path/to/CWRU --eval_only --checkpoint checkpoint_ch3.pt
    python train_chapter3.py --data_dir /path/to/CWRU --few_shot --alpha 0.1

依赖: pip install torch numpy scipy pyyaml scikit-learn matplotlib tqdm
"""
import os
import sys
import argparse
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="第3章: 双通道大核CNN+BiGRU 滚动轴承故障诊断")
    parser.add_argument("--config", type=str, default="configs/chapter3_dual_cnn_bigru.yaml",
                        help="配置文件路径")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="CWRU数据集目录")
    parser.add_argument("--output_dir", type=str, default="outputs/chapter3",
                        help="输出目录")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型checkpoint路径")
    parser.add_argument("--eval_only", action="store_true",
                        help="仅评估，不训练")
    parser.add_argument("--few_shot", action="store_true",
                        help="启用小样本实验")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="小样本比例 (0.1/0.2/0.4/0.5/0.7)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    return parser.parse_args()


def train_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    from tqdm import tqdm
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    from utils.metrics import compute_metrics
    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    return avg_loss, metrics["accuracy"]


def evaluate(model, loader, criterion, device):
    import torch
    from tqdm import tqdm
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    all_features = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_features.append(logits.cpu().numpy())

    from utils.metrics import compute_metrics
    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    all_features = np.concatenate(all_features, axis=0)
    return avg_loss, metrics, all_features, np.array(all_labels)


def main():
    args = parse_args()

    # Lazy imports — torch is required past this point
    import torch
    from torch.utils.data import DataLoader
    from data.cwru import CWRUDataset, CWRUDatasetFewShot
    from models.chapter3 import DualCNNBiGRU
    from utils.common import set_seed, get_device, EarlyStopping, LabelSmoothingLoss
    from utils.visualize import plot_training_curves, plot_confusion_matrix, plot_tsne
    from utils.metrics import confusion_matrix as cm_fn

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print(f"第3章: 双通道大核CNN+BiGRU 故障诊断")
    print(f"设备: {device}")
    print("=" * 60)

    # 数据集
    dataset_kwargs = {
        "data_dir": args.data_dir,
        "window_size": config["dataset"]["window_size"],
        "sliding_step": config["dataset"]["sliding_step"],
        "samples_per_class": config["dataset"]["samples_per_class"],
        "train_ratio": config["dataset"]["train_ratio"],
    }
    if args.few_shot:
        train_dataset = CWRUDatasetFewShot(alpha=args.alpha, **dataset_kwargs)
        print(f"  小样本模式: alpha={args.alpha}, 训练样本={len(train_dataset)}")
    else:
        train_dataset = CWRUDataset(train=True, **dataset_kwargs)
    test_dataset = CWRUDataset(train=False, **dataset_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    print(f"  训练集: {len(train_dataset)} 样本, 测试集: {len(test_dataset)} 样本")

    # 模型
    model = DualCNNBiGRU(
        seq_len=config["dataset"]["window_size"],
        cnn_out_channels=config["model"]["conv_block1"]["out_channels"],
        kernel_large=config["model"]["conv_block1"]["kernel_size"],
        kernel_small=config["model"]["conv_block2"]["kernel_size"],
        gru_hidden_size=config["model"]["bigru"]["hidden_size"],
        gru_num_layers=config["model"]["bigru"]["num_layers"],
        num_classes=config["dataset"]["num_classes"],
        classifier_dropout=config["model"]["classifier"]["dropout_rate"],
    ).to(device)
    print(f"  参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 损失 & 优化器
    loss_type = config["training"]["loss"]
    if loss_type == "label_smoothing":
        alpha = config["training"]["label_smoothing_alpha"]
        if alpha == "???":
            alpha = 0.1
            print(f"  ⚠️ LSR alpha 未明确，使用默认值 {alpha}")
        criterion = LabelSmoothingLoss(config["dataset"]["num_classes"], alpha=alpha)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=config["training"]["early_stopping"]["patience"])

    # 加载checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  加载checkpoint: {args.checkpoint}")

    # 仅评估
    if args.eval_only:
        test_loss, metrics, features, labels = evaluate(model, test_loader, criterion, device)
        print(f"\n  测试集: Loss={test_loss:.4f}, Acc={metrics['accuracy']:.4f}, "
              f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        cm = cm_fn(labels, np.argmax(features, axis=1))
        class_names = [config["dataset"]["labels"][i] for i in range(config["dataset"]["num_classes"])]
        plot_confusion_matrix(cm, class_names, os.path.join(args.output_dir, "confusion_matrix_ch3.png"))
        plot_tsne(features, labels, class_names, os.path.join(args.output_dir, "tsne_ch3.png"),
                  title="Chapter 3: t-SNE Feature Visualization")
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

        print(f"  Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "val_metrics": val_metrics, "config": config},
                       os.path.join(args.output_dir, "best_model_ch3.pt"))
            print(f"  ✓ 保存最佳模型 (Acc={best_acc:.4f})")

        if early_stopping(val_loss):
            print(f"  早停触发于 epoch {epoch}")
            break

    print(f"\n最佳验证准确率: {best_acc:.4f}")

    # 最终评估与可视化
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model_ch3.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_metrics, features, labels = evaluate(model, test_loader, criterion, device)
    print(f"最佳模型测试集: Acc={test_metrics['accuracy']:.4f}")

    plot_training_curves(train_losses, val_losses, train_accs, val_accs, os.path.join(args.output_dir, "training_curves_ch3.png"))
    preds = features.argmax(axis=1)
    cm = cm_fn(labels, preds)
    class_names = [config["dataset"]["labels"][i] for i in range(config["dataset"]["num_classes"])]
    plot_confusion_matrix(cm, class_names, os.path.join(args.output_dir, "confusion_matrix_ch3.png"))
    plot_tsne(features, labels, class_names, os.path.join(args.output_dir, "tsne_ch3.png"),
              title="Chapter 3: t-SNE Feature Visualization")
    print(f"\n所有输出已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
