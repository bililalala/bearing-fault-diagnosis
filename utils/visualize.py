"""
可视化工具：t-SNE降维、训练曲线、混淆矩阵热力图
"""
import os
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    save_path: str = "training_curves.png",
):
    """
    绘制训练/验证损失和准确率曲线。
    参照论文图3.6/图5.7/图5.8。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, "b-", label="Train Loss")
    ax1.plot(epochs, val_losses, "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, "b-", label="Train Acc")
    ax2.plot(epochs, val_accs, "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Training curves saved to {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str = "confusion_matrix.png",
    normalize: bool = False,
):
    """
    绘制混淆矩阵热力图。
    参照论文图4.7/图5.9。
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names if len(class_names) <= 10 else [],
        yticklabels=class_names if len(class_names) <= 10 else [],
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在格中显示数值
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {save_path}")


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list = None,
    save_path: str = "tsne.png",
    title: str = "t-SNE Feature Visualization",
    perplexity: float = 30.0,
):
    """
    t-SNE降维可视化。
    参照论文图3.7/图4.8-4.9/图5.10-5.11。
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        embeddings[:, 0], embeddings[:, 1],
        c=labels, cmap="tab10", alpha=0.7, s=10,
    )
    if class_names:
        handles, _ = scatter.legend_elements()
        legend = ax.legend(handles, class_names[:len(handles)], title="Classes",
                          bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.add_artist(legend)

    ax.set_title(title)
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  t-SNE plot saved to {save_path}")
