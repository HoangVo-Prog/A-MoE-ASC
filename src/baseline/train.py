import argparse
import json
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
from collections import deque
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AspectSentimentDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        tokenizer,
        max_len_sent: int = 128,
        max_len_term: int = 16,
        label2id: Optional[Dict[str, int]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len_sent = max_len_sent
        self.max_len_term = max_len_term

        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        # sanity check
        for i, s in enumerate(self.samples[:5]):
            if not {"sentence", "aspect", "sentiment"} <= s.keys():
                raise ValueError(f"Invalid sample at index {i}: {s}")

        # build label mapping from TRAIN only
        if label2id is None:
            labels = sorted({s["sentiment"] for s in self.samples})
            self.label2id = {lbl: i for i, lbl in enumerate(labels)}
        else:
            self.label2id = label2id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]

        sentence = item["sentence"]
        term = item["aspect"]
        label = self.label2id[item["sentiment"]]

        sent_enc = self.tokenizer(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=self.max_len_sent,
            return_tensors="pt",
        )

        term_enc = self.tokenizer(
            term,
            truncation=True,
            padding="max_length",
            max_length=self.max_len_term,
            return_tensors="pt",
        )

        return {
            "input_ids_sent": sent_enc["input_ids"].squeeze(0),
            "attention_mask_sent": sent_enc["attention_mask"].squeeze(0),
            "input_ids_term": term_enc["input_ids"].squeeze(0),
            "attention_mask_term": term_enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class BertConcatClassifier(nn.Module):
    """
    BERT based classifier for Aspect Sentiment Classification.

    A single BERT encoder is shared for both the sentence and the aspect term.
    Supported fusion methods for the two CLS embeddings:

        1. "concat": [CLS_sent ; CLS_term]       → dim = 2 * hidden
        2. "add":    CLS_sent + CLS_term         → dim = hidden
        3. "mul":    CLS_sent * CLS_term         → dim = hidden (Hadamard product)

    For "concat" we use a separate classifier head with input size 2 * hidden.
    For "add" and "mul" we use a classifier head with input size hidden.
    """

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1) -> None:
        """
        Args:
            model_name: Name of the pretrained model on Hugging Face hub.
            num_labels: Number of sentiment labels.
            dropout: Dropout probability for the classification head.
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        # Classifier for concat fusion
        self.classifier_concat = nn.Linear(2 * hidden_size, num_labels)

        # Classifier for single hidden dimension fusions (add or mul)
        self.classifier_single = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids_sent: torch.Tensor,
        attention_mask_sent: torch.Tensor,
        input_ids_term: torch.Tensor,
        attention_mask_term: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        fusion_method: str = "concat",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids_sent: Token ids for the sentence.
            attention_mask_sent: Attention mask for the sentence.
            input_ids_term: Token ids for the aspect term.
            attention_mask_term: Attention mask for the aspect term.
            labels: Optional ground truth labels.
            fusion_method: Fusion strategy, one of {"concat", "add", "mul"}.

        Returns:
            A dict with:
                "logits": Tensor of shape [batch_size, num_labels].
                "loss": Cross entropy loss if labels is provided, else None.
        """
        # Encode sentence
        out_sent = self.encoder(
            input_ids=input_ids_sent,
            attention_mask=attention_mask_sent,
        )
        cls_sent = out_sent.last_hidden_state[:, 0, :]  # [batch, hidden]

        # Encode term
        out_term = self.encoder(
            input_ids=input_ids_term,
            attention_mask=attention_mask_term,
        )
        cls_term = out_term.last_hidden_state[:, 0, :]  # [batch, hidden]

        # Fusion
        if fusion_method == "concat":
            fused = torch.cat([cls_sent, cls_term], dim=-1)  # [batch, 2 * hidden]
            logits = self.classifier_concat(self.dropout(fused))

        elif fusion_method == "add":
            fused = cls_sent + cls_term  # [batch, hidden]
            logits = self.classifier_single(self.dropout(fused))

        elif fusion_method == "mul":
            fused = cls_sent * cls_term  # [batch, hidden]
            logits = self.classifier_single(self.dropout(fused))

        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    fusion_method: str = "concat",
    f1_average: str = "macro",
) -> Dict[str, float]:
    """
    Train for one epoch and return metrics: loss, acc, f1.
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(
            input_ids_sent=batch["input_ids_sent"],
            attention_mask_sent=batch["attention_mask_sent"],
            input_ids_term=batch["input_ids_term"],
            attention_mask_term=batch["attention_mask_term"],
            labels=batch["label"],
            fusion_method=fusion_method,
        )
        loss = outputs["loss"]
        logits = outputs["logits"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(batch["label"].detach().cpu().tolist())

    avg_loss = total_loss / max(1, len(dataloader))
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=f1_average)

    return {"loss": avg_loss, "acc": acc, "f1": f1}


def eval_model(
    model: nn.Module,
    dataloader: DataLoader,
    id2label: Optional[Dict[int, str]] = None,
    verbose_report: bool = False,
    fusion_method: str = "concat",
    f1_average: str = "macro",
) -> Dict[str, float]:
    """
    Evaluate and return metrics: loss, acc, f1.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(
                input_ids_sent=batch["input_ids_sent"],
                attention_mask_sent=batch["attention_mask_sent"],
                input_ids_term=batch["input_ids_term"],
                attention_mask_term=batch["attention_mask_term"],
                labels=batch["label"],  # compute loss on eval too
                fusion_method=fusion_method,
            )
            loss = outputs["loss"]
            logits = outputs["logits"]

            total_loss += float(loss.item()) if loss is not None else 0.0

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())

    avg_loss = total_loss / max(1, len(dataloader))
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average=f1_average)

    if verbose_report and id2label is not None:
        target_names = [id2label[i] for i in range(len(id2label))]
        print("Classification report:")
        print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    return {"loss": avg_loss, "acc": acc, "f1": f1}


def plot_history(history: Dict[str, list], save_dir: Optional[str] = None) -> None:
    """
    history keys expected:
      train_loss, val_loss, test_loss,
      train_f1, val_f1, test_f1
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # Loss plot
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.plot(epochs, history["test_loss"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per epoch")
    plt.legend()
    plt.grid(True)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # F1 plot
    plt.figure()
    plt.plot(epochs, history["train_f1"], label="train")
    plt.plot(epochs, history["val_f1"], label="val")
    plt.plot(epochs, history["test_f1"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("F1 per epoch")
    plt.legend()
    plt.grid(True)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "f1_curve.png"), dpi=150, bbox_inches="tight")
    plt.show()


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for ASC training script.
    """
    parser = argparse.ArgumentParser(
        description="BERT classifier for Aspect Sentiment Classification with sentence term fusion"
    )

    # Model and tokenizer
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Name of the pretrained model on Hugging Face",
    )

    # Data paths
    parser.add_argument(
        "--train_path",
        type=str,
        default="dataset/atsa/laptop14/train.json",
        help="Path to training JSON file",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="dataset/atsa/laptop14/val.json",
        help="Path to validation JSON file",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="dataset/atsa/laptop14/test.json",
        help="Path to test JSON file",
    )

    # Sequence lengths
    parser.add_argument(
        "--max_len_sent",
        type=int,
        default=128,
        help="Maximum length for the full sentence",
    )
    parser.add_argument(
        "--max_len_term",
        type=int,
        default=16,
        help="Maximum length for the aspect term",
    )

    # Training hyperparameters
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for validation and test",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability for the classifier",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio over total training steps",
    )

    # Fusion method
    parser.add_argument(
        "--fusion_method",
        type=str,
        default="concat",
        choices=["concat", "add", "mul"],
        help="Fusion method between sentence and term representations",
    )

    # Saving
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_model",
        help="Directory to save the model checkpoint",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="bert_concat_asc.pt",
        help="Filename for the saved model state_dict",
    )

    # Misc
    parser.add_argument(
        "--verbose_report",
        action="store_true",
        help="Print detailed classification report on the test set",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Main training and evaluation pipeline.

    Steps:
        1. Load tokenizer and build datasets.
        2. Initialize model and optimizer.
        3. Train for several epochs and track best validation accuracy.
        4. Evaluate the best model on the test set.
        5. Save the best model state_dict to disk.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Build training dataset first to obtain label2id
    train_dataset = AspectSentimentDataset(
        json_path=args.train_path,
        tokenizer=tokenizer,
        max_len_sent=args.max_len_sent,
        max_len_term=args.max_len_term,
        label2id=None,
    )
    label2id = train_dataset.label2id
    id2label = {v: k for k, v in label2id.items()}
    print("Label mapping:", label2id)

    val_dataset = AspectSentimentDataset(
        json_path=args.val_path,
        tokenizer=tokenizer,
        max_len_sent=args.max_len_sent,
        max_len_term=args.max_len_term,
        label2id=label2id,
    )
    test_dataset = AspectSentimentDataset(
        json_path=args.test_path,
        tokenizer=tokenizer,
        max_len_sent=args.max_len_sent,
        max_len_term=args.max_len_term,
        label2id=label2id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )

    model = BertConcatClassifier(
        model_name=args.model_name,
        num_labels=len(label2id),
        dropout=args.dropout,
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_f1": [],
        "val_f1": [],
        "test_f1": [],
    }

    rolling_k = 3
    val_f1_window = deque(maxlen=rolling_k)
    best_val_f1_rolling = -1.0
    best_state_dict = None
    best_epoch = -1

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            fusion_method=args.fusion_method,
            f1_average="macro",
        )
        print(
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Train macro-F1: {train_metrics['f1']:.4f} | "
            f"Train acc: {train_metrics['acc']:.4f}"
        )

        val_metrics = eval_model(
            model,
            val_loader,
            id2label,
            verbose_report=False,
            fusion_method=args.fusion_method,
            f1_average="macro",
        )
        print(
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val macro-F1: {val_metrics['f1']:.4f} | "
            f"Val acc: {val_metrics['acc']:.4f}"
        )

        test_metrics = eval_model(
            model,
            test_loader,
            id2label,
            verbose_report=args.verbose_report,
            fusion_method=args.fusion_method,
            f1_average="macro",
        )
        print(
            f"Test loss: {test_metrics['loss']:.4f} | "
            f"Test macro-F1: {test_metrics['f1']:.4f} | "
            f"Test acc: {test_metrics['acc']:.4f}"
        )

        # log history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["test_loss"].append(test_metrics["loss"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        history["test_f1"].append(test_metrics["f1"])

        # rolling avg val f1 on last 3 epochs
        val_f1_window.append(val_metrics["f1"])
        val_f1_rolling = sum(val_f1_window) / len(val_f1_window)

        print(f"Val macro-F1 rolling({rolling_k}) = {val_f1_rolling:.4f}")

        # save best by rolling average (only starts being stable after 3 epochs, but still works with len<3)
        if val_f1_rolling > best_val_f1_rolling:
            best_val_f1_rolling = val_f1_rolling
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            print("New best model on rolling val macro-F1, saving in memory")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(DEVICE)
        print(f"Loaded best model at epoch {best_epoch} with rolling val macro-F1 = {best_val_f1_rolling:.4f}")
    else:
        print("Warning: no best_state_dict saved, using last epoch model")

    print("Evaluation best model on test set:")
    test_metrics = eval_model(
        model,
        test_loader,
        id2label,
        verbose_report=args.verbose_report,
        fusion_method=args.fusion_method,
        f1_average="macro",
    )
    print(
        f"Test loss: {test_metrics['loss']:.4f} | "
        f"Test macro-F1: {test_metrics['f1']:.4f} | "
        f"Test acc: {test_metrics['acc']:.4f}"
    )

    # Save model to file
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.output_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Plot curves
    plot_history(history, save_dir=args.output_dir)

if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
