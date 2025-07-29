import matplotlib.pyplot as plt
import numpy as np


def plot_loss(train_loss, val_loss, output_dir="outputs"):
    """Plots training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    epochs = np.arange(1, len(train_loss) + 1)
    plt.plot( epochs, train_loss, label="Train Loss")
    plt.plot( epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_curve.png")
    plt.close()


def plot_bleu(bleu_scores, output_dir="outputs"):
    """Plots BLEU score over epochs."""
    plt.figure(figsize=(7, 5))
    epochs = np.arange(1, len(bleu_scores) + 1)
    plt.plot( epochs, bleu_scores, label="BLEU Score")
    plt.xlabel("Epoch")
    plt.ylabel("BLEU Score")
    plt.title("BLEU Score Over Epochs")
    plt.xticks(epochs)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bleu_scores.png")
    plt.close()
