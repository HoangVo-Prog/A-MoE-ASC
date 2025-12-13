import os
from typing import Dict, Optional

import matplotlib.pyplot as plt


def plot_history(history: Dict[str, list], save_dir: Optional[str] = None, prefix: str = "") -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))

    def _name(x: str) -> str:
        return f"{prefix}{x}" if prefix else x

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train")
    if len(history["val_loss"]) > 0:
        plt.plot(epochs[: len(history["val_loss"])], history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per epoch")
    plt.legend()
    plt.grid(True)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, _name("loss_curve.png")), dpi=150, bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.plot(epochs, history["train_f1"], label="train")
    if len(history["val_f1"]) > 0:
        plt.plot(epochs[: len(history["val_f1"])], history["val_f1"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("F1 per epoch")
    plt.legend()
    plt.grid(True)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, _name("f1_curve.png")), dpi=150, bbox_inches="tight")
    plt.show()
