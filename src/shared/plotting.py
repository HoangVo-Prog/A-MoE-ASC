import os
from typing import Dict, Optional
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_history(history: Dict[str, list], save_dir: Optional[str] = None, prefix: str = "") -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))

    def _name(x: str) -> str:
        return f"{prefix}{x}" if prefix else x

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train")
    if len(history["val_loss"]) > 0:
        plt.plot(epochs[: len(history["val_loss"])], history["val_loss"], label="val")

    plt.legend()
    plt.grid(True)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, _name("loss_curve.png")), dpi=150, bbox_inches="tight")
        
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

        
def _print_confusion_matrix(
    y_true,
    y_pred,
    *,
    id2label: Optional[Dict[int, str]] = None,
    normalize: bool = True,
    digits: int = 3,
):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype(np.float32)
        cm = cm / cm.sum(axis=1, keepdims=True)

    if id2label is not None:
        labels = [id2label[i] for i in range(len(id2label))]
    else:
        labels = [str(i) for i in range(cm.shape[0])]

    max_label_len = max(len(l) for l in labels)

    header = " " * (max_label_len + 2)
    for lbl in labels:
        header += f"{lbl:>{max_label_len+2}}"
    print(header)

    for i, row in enumerate(cm):
        row_str = f"{labels[i]:>{max_label_len}} |"
        for val in row:
            if normalize:
                row_str += f"{val:>{max_label_len+2}.{digits}f}"
            else:
                row_str += f"{int(val):>{max_label_len+2}d}"
        print(row_str)

    print()    
