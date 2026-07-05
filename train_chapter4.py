#!/usr/bin/env python
"""
第4章训练脚本：基于改进GAF与Swin-Transformer的轴承故障诊断
============================================================
论文第39-49页

管线: 一维信号 → PAA降维 → GAF编码 → 2D图像 → Swin-Transformer → 分类

用法:
    python train_chapter4.py --data_dir /path/to/CWRU --config configs/chapter4_gaf_swin.yaml
    python train_chapter4.py --data_dir /path/to/CWRU --eval_only --checkpoint checkpoint_ch4.pt

依赖: pip install torch numpy scipy pyyaml scikit-learn matplotlib tqdm
"""
import os
import sys
import argparse
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="第4章: 改进GAF + Swin-Transformer 轴承故障诊断")
    parser.add_argument("--config", type=str, default="configs/chapter4_gaf_swin.yaml", help="配置文件路径")
    parser.add_argument("--data_dir", type=str, required=True, help="CWRU数据集目录")
    parser.add_argument("--output_dir", type=str, default="outputs/chapter4", help="输出目录")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型checkpoint路径")
    parser.add_argument("--eval_only", action="store_true", help="仅评估")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


class GAFTransform:
    """GAF编码变换: 1D信号 → PAA → GAF → 2D图像"""
    def __init__(self, paa_segments=64, gaf_method="summation", image_size=64):
        self.paa_segments = paa_segments
        self.gaf_method = gaf_method
        self.image_size = image_size

    def __call__(self, signal):
        import torch
        import torch.nn.functional as F
        from data.preprocessing import paa_reduce, gaf_encode
        sig_np = signal.squeeze(0).numpy()
        if self.paa_segments and self.paa_segments < len(sig_np):
            sig_np = paa_reduce(sig_np, self.paa_segments)
        gaf = gaf_encode(sig_np, method=self.gaf_method)
        gaf_tensor = torch.from_numpy(gaf).float().unsqueeze(0)
        gaf_tensor = F.interpolate(gaf_tensor.unsqueeze(0), size=(self.image_size, self.image_size),
                                    mode="bilinear", align_corners=False).squeeze(0)
        return gaf_tensor


def train_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    from tqdm import tqdm
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
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
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_features.append(logits.cpu().numpy())
    from utils.metrics import compute_metrics
    return (total_loss / len(loader), compute_metrics(np.array(all_labels), np.array(all_preds)),
            np.concatenate(all_features, axis=0), np.array(all_labels))


def main():
    args = parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from data.cwru import CWRUDataset
    from models.chapter4 import PGAFSwin
    from utils.common import set_seed, get_device, EarlyStopping
    from utils.visualize import plot_training_curves, plot_confusion_matrix, plot_tsne
    from utils.metrics import confusion_matrix as cm_fn

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print(f"第4章: 改进GAF + Swin-Transformer 故障诊断")
    print(f"设备: {device}")
    print("=" * 60)

    # GAF变换
    paa_seg = config["preprocessing"]["paa"]["segments"]
    if paa_seg == "???":
        paa_seg = 64
        print(f"  ⚠️ PAA segments 未明确，使用默认值 {paa_seg}")
    img_size = config["preprocessing"]["image"]["size"]
    gaf_transform = GAFTransform(paa_segments=paa_seg, image_size=img_size)

    # 数据集
    dataset_kwargs = {
        "data_dir": args.data_dir,
        "load_condition": config["dataset"]["load_condition"],
        "window_size": 1200,
        "sliding_step": 600,
        "samples_per_class": config["dataset"]["samples_per_file"] // 100,
        "train_ratio": config["dataset"]["train_ratio"],
    }
    train_dataset = CWRUDataset(train=True, transform=gaf_transform, **dataset_kwargs)
    test_dataset = CWRUDataset(train=False, transform=gaf_transform, **dataset_kwargs)
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    print(f"  训练集: {len(train_dataset)} 样本, 测试集: {len(test_dataset)} 样本")

    # 模型
    swin_cfg = config["model"]["swin"]
    model = PGAFSwin(
        img_size=img_size, patch_size=swin_cfg["patch_size"], in_channels=1,
        num_classes=config["model"]["num_classes"], embed_dim=swin_cfg["embed_dim"],
        depths=swin_cfg["depths"], num_heads=swin_cfg["num_heads"],
        window_size=swin_cfg["window_size"], mlp_ratio=swin_cfg["mlp_ratio"],
        drop_rate=swin_cfg["drop_rate"], attn_drop_rate=swin_cfg["attn_drop_rate"],
        drop_path_rate=swin_cfg["drop_path_rate"],
    ).to(device)
    print(f"  参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"],
                                 weight_decay=config["training"].get("weight_decay", 0.05))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    early_stopping = EarlyStopping(patience=config["training"]["early_stopping"]["patience"])

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  加载checkpoint: {args.checkpoint}")

    if args.eval_only:
        test_loss, metrics, features, labels = evaluate(model, test_loader, criterion, device)
        print(f"\n  测试集: Acc={metrics['accuracy']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        cm = cm_fn(labels, np.argmax(features, axis=1))
        class_names = [config["dataset"]["labels"][i] for i in range(config["model"]["num_classes"])]
        plot_confusion_matrix(cm, class_names, os.path.join(args.output_dir, "confusion_matrix_ch4.png"))
        plot_tsne(features, labels, class_names, os.path.join(args.output_dir, "tsne_ch4.png"), title="Chapter 4: t-SNE")
        return

    best_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, config["training"]["epochs"])
        val_loss, val_metrics, _, _ = evaluate(model, test_loader, criterion, device)
        val_acc = val_metrics["accuracy"]
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)
        scheduler.step()
        print(f"  Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "val_metrics": val_metrics, "config": config},
                       os.path.join(args.output_dir, "best_model_ch4.pt"))
            print(f"  ✓ 保存最佳模型 (Acc={best_acc:.4f})")
        if early_stopping(val_loss):
            print(f"  早停触发于 epoch {epoch}"); break

    print(f"\n最佳验证准确率: {best_acc:.4f}")
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_model_ch4.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_metrics, features, labels = evaluate(model, test_loader, criterion, device)
    print(f"最佳模型测试集: Acc={test_metrics['accuracy']:.4f}")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, os.path.join(args.output_dir, "training_curves_ch4.png"))
    preds = features.argmax(axis=1)
    cm = cm_fn(labels, preds)
    class_names = [config["dataset"]["labels"][i] for i in range(config["model"]["num_classes"])]
    plot_confusion_matrix(cm, class_names, os.path.join(args.output_dir, "confusion_matrix_ch4.png"))
    plot_tsne(features, labels, class_names, os.path.join(args.output_dir, "tsne_ch4.png"), title="Chapter 4: t-SNE")
    print(f"\n所有输出已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
