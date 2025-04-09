import matplotlib.pyplot as plt
import numpy as np
import os


def plot_loss(train_losses, test_losses, save_path="./pic"):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, 'o-', color='#1f77b4', label='Train Loss')

    # If test_losses are available (may not be calculated every epoch)
    if test_losses and len(test_losses) > 0:
        # Create x-axis points for test losses (may be measured less frequently)
        test_epochs = np.linspace(1, len(train_losses), len(test_losses), dtype=int)
        plt.plot(test_epochs, test_losses, 'o-', color='#ff7f0e', label='Test Loss')

    plt.title('Training/Test Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_curves.png'), dpi=300)
    plt.close()


def plot_metrics(metrics_list, plot_frequency, num_epoch, save_path="./pic"):
    # 提取每个指标的数据
    epochs = list(range(plot_frequency, num_epoch + 1, plot_frequency))
    mIoU_values = [metrics.mIoU for metrics in metrics_list]
    Dice_values = [metrics.Dice for metrics in metrics_list]
    HD_values = [metrics.HD for metrics in metrics_list]
    Accuracy_values = [metrics.Accuracy for metrics in metrics_list]
    Recall_values = [metrics.Recall for metrics in metrics_list]
    F1_values = [metrics.F1 for metrics in metrics_list]

    # 绘制第一幅图（mIoU, Dice, Accuracy, Recall, F1）
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, mIoU_values, label='mIoU', marker='o')
    plt.plot(epochs, Dice_values, label='Dice', marker='o')
    plt.plot(epochs, Accuracy_values, label='Accuracy', marker='o')
    plt.plot(epochs, Recall_values, label='Recall', marker='o')
    plt.plot(epochs, F1_values, label='F1', marker='o')

    plt.title('Metrics over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'metrics_plot.png'), dpi=300)
    plt.show()

    # 绘制第二幅图（HD）
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, HD_values, label='HD', marker='o', color='red')

    plt.title('Hausdorff Distance over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Hausdorff Distance')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'HD_plot.png'), dpi=300)
    plt.show()
