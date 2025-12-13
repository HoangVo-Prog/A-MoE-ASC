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
from sklearn.model_selection import StratifiedKFold
from config import TrainConfig



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


class AspectSentimentDatasetFromSamples(Dataset):
    def __init__(
        self,
        samples: list,
        tokenizer,
        max_len_sent: int,
        max_len_term: int,
        label2id: Dict[str, int],
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len_sent = max_len_sent
        self.max_len_term = max_len_term
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
        self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
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


def plot_history(history: Dict[str, list], save_dir: Optional[str] = None, prefix: str = "") -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))

    def _name(x: str) -> str:
        return f"{prefix}{x}" if prefix else x

    # Loss plot
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per epoch")
    plt.legend()
    plt.grid(True)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, _name("loss_curve.png")), dpi=150, bbox_inches="tight")
    plt.show()

    # F1 plot
    plt.figure()
    plt.plot(epochs, history["train_f1"], label="train")
    plt.plot(epochs, history["val_f1"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("F1 per epoch")
    plt.legend()
    plt.grid(True)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, _name("f1_curve.png")), dpi=150, bbox_inches="tight")
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
    
    parser.add_argument(
        "--k_folds",
        type=int,
        default=0,
        help="If > 1, run Stratified K-fold CV on train_path and ignore val_path",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for CV split and training reproducibility",
    )
    
    parser.add_argument(
        "--rolling_k",
        type=int,
        default=3,
        help="Window size for rolling average val macro-F1",
    )

    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=3,
        help="Early stopping patience on rolling val macro-F1 (2 or 3 recommended)",
    )

    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=0,
        help="Freeze encoder for first N epochs (0 disables)",
    )



    return parser.parse_args()


def build_model(args: argparse.Namespace, num_labels: int) -> nn.Module:
    model = BertConcatClassifier(
        model_name=args.model_name,
        num_labels=num_labels,
        dropout=args.dropout,
    ).to(DEVICE)
    return model


def build_optimizer_and_scheduler(
    args: argparse.Namespace,
    model: nn.Module,
    train_loader: DataLoader,
):
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    # model is BertConcatClassifier, has model.encoder
    for p in model.encoder.parameters():
        p.requires_grad = trainable


def maybe_freeze_encoder(model: nn.Module, epoch_idx_0based: int, freeze_epochs: int) -> None:
    # freeze for epochs: 0..freeze_epochs-1
    if freeze_epochs > 0 and epoch_idx_0based < freeze_epochs:
        set_encoder_trainable(model, False)
    else:
        set_encoder_trainable(model, True)


def train_full_then_test(
    *,
    args: argparse.Namespace,
    tokenizer,
    train_dataset_full: Dataset,
    test_loader: DataLoader,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
):
    print("\n===== Train FULL then Test =====")

    train_loader = DataLoader(
        train_dataset_full,
        batch_size=args.train_batch_size,
        shuffle=True,
    )

    model = build_model(args, num_labels=len(label2id))
    optimizer, scheduler = build_optimizer_and_scheduler(args, model, train_loader)

    out = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        id2label=id2label,
        tag="[FULL] ",
    )

    if out["best_state_dict"] is not None:
        model.load_state_dict(out["best_state_dict"])
        model.to(DEVICE)
        print(f"Loaded best FULL model at epoch {out['best_epoch']}")

    print("\n===== Final TEST evaluation =====")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(
                input_ids_sent=batch["input_ids_sent"],
                attention_mask_sent=batch["attention_mask_sent"],
                input_ids_term=batch["input_ids_term"],
                attention_mask_term=batch["attention_mask_term"],
                labels=None,
                fusion_method=args.fusion_method,
            )
            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())

    print("\nClassification report (TEST):")
    target_names = [id2label[i] for i in range(len(id2label))]
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    save_path = os.path.join(args.output_dir, f"final_{args.output_name}")
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")


def run_training_loop(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer,
    scheduler,
    args: argparse.Namespace,
    id2label: Dict[int, str],
    tag: str = "",
):
    history = {
        "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [],
    }

    rolling_k = args.rolling_k
    val_f1_window = deque(maxlen=rolling_k)

    best_val_f1_rolling = -1.0
    best_state_dict = None
    best_epoch = -1

    patience = args.early_stop_patience
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        print(f"{tag}Epoch {epoch + 1}/{args.epochs}")

        maybe_freeze_encoder(
            model,
            epoch_idx_0based=epoch,
            freeze_epochs=args.freeze_epochs,
        )

        if args.freeze_epochs > 0 and epoch < args.freeze_epochs:
            print(f"Encoder frozen (epoch {epoch + 1}/{args.freeze_epochs})")
        elif args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            print("Encoder unfrozen")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            fusion_method=args.fusion_method,
            f1_average="macro",
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_f1"].append(train_metrics["f1"])

        log = (
            f"Train loss {train_metrics['loss']:.4f} "
            f"F1 {train_metrics['f1']:.4f} "
            f"acc {train_metrics['acc']:.4f}"
        )

        if val_loader is not None:
            val_metrics = eval_model(
                model,
                val_loader,
                id2label,
                verbose_report=False,
                fusion_method=args.fusion_method,
                f1_average="macro",
            )

            history["val_loss"].append(val_metrics["loss"])
            history["val_f1"].append(val_metrics["f1"])

            val_f1_window.append(val_metrics["f1"])
            val_f1_rolling = sum(val_f1_window) / len(val_f1_window)

            log += (
                f" | Val loss {val_metrics['loss']:.4f} "
                f"F1 {val_metrics['f1']:.4f} "
                f"acc {val_metrics['acc']:.4f} "
                f"| Val F1 rolling({rolling_k}) {val_f1_rolling:.4f}"
            )

            if val_f1_rolling > best_val_f1_rolling:
                best_val_f1_rolling = val_f1_rolling
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                epochs_no_improve = 0
                print("New best model on rolling val F1")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement")
                    break
        else:
            val_f1_rolling = None

        print(log)

    return {
        "best_state_dict": best_state_dict,
        "best_epoch": best_epoch,
        "best_val_f1_rolling": best_val_f1_rolling,
        "history": history,
    }


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        model_name=args.model_name,
        fusion_method=args.fusion_method,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        dropout=args.dropout,
        freeze_epochs=args.freeze_epochs,
        rolling_k=args.rolling_k,
        early_stop_patience=args.early_stop_patience,
        k_folds=args.k_folds,
        seed=args.seed,
        max_len_sent=args.max_len_sent,
        max_len_term=args.max_len_term,
        output_dir=args.output_dir,
        output_name=args.output_name,
        verbose_report=args.verbose_report,
    )


def main(args: argparse.Namespace) -> None:
    cfg = build_train_config(args)

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ===== Load full train set =====
    train_dataset_full = AspectSentimentDataset(
        json_path=args.train_path,
        tokenizer=tokenizer,
        max_len_sent=cfg.max_len_sent,
        max_len_term=cfg.max_len_term,
        label2id=None,
    )
    label2id = train_dataset_full.label2id
    id2label = {v: k for k, v in label2id.items()}
    print("Label mapping:", label2id)

    # ===== Fixed test set =====
    test_dataset = AspectSentimentDataset(
        json_path=args.test_path,
        tokenizer=tokenizer,
        max_len_sent=cfg.max_len_sent,
        max_len_term=cfg.max_len_term,
        label2id=label2id,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
    )

    # =====================================================
    # Case 1: No CV, use explicit val_path
    # =====================================================
    if cfg.k_folds <= 1:
        print("Running single split training")

        val_dataset = AspectSentimentDataset(
            json_path=args.val_path,
            tokenizer=tokenizer,
            max_len_sent=cfg.max_len_sent,
            max_len_term=cfg.max_len_term,
            label2id=label2id,
        )

        train_loader = DataLoader(
            train_dataset_full,
            batch_size=cfg.train_batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
        )

        model = build_model(args, num_labels=len(label2id))
        optimizer, scheduler = build_optimizer_and_scheduler(args, model, train_loader)

        out = run_training_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            id2label=id2label,
        )

        model.load_state_dict(out["best_state_dict"])
        model.to(DEVICE)

        final_test = eval_model(
            model,
            test_loader,
            id2label,
            verbose_report=cfg.verbose_report,
            fusion_method=cfg.fusion_method,
            f1_average="macro",
        )

        print(
            f"Final Test loss {final_test['loss']:.4f} "
            f"F1 {final_test['f1']:.4f} "
            f"acc {final_test['acc']:.4f}"
        )

        os.makedirs(cfg.output_dir, exist_ok=True)
        save_path = os.path.join(cfg.output_dir, cfg.output_name)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        return

    # =====================================================
    # Case 2: K-fold Cross Validation
    # =====================================================
    print(f"Running StratifiedKFold with k={cfg.k_folds}")

    samples = train_dataset_full.samples
    y = [label2id[s["sentiment"]] for s in samples]

    skf = StratifiedKFold(
        n_splits=cfg.k_folds,
        shuffle=True,
        random_state=cfg.seed,
    )

    fold_val_f1 = []
    fold_test_f1 = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(samples, y), start=1):
        print(f"\n===== Fold {fold_idx}/{cfg.k_folds} =====")

        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        train_ds = AspectSentimentDatasetFromSamples(
            train_samples,
            tokenizer,
            cfg.max_len_sent,
            cfg.max_len_term,
            label2id,
        )
        val_ds = AspectSentimentDatasetFromSamples(
            val_samples,
            tokenizer,
            cfg.max_len_sent,
            cfg.max_len_term,
            label2id,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train_batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
        )

        model = build_model(args, num_labels=len(label2id))
        optimizer, scheduler = build_optimizer_and_scheduler(args, model, train_loader)

        out = run_training_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            id2label=id2label,
            tag=f"[Fold {fold_idx}] ",
        )

        model.load_state_dict(out["best_state_dict"])
        model.to(DEVICE)

        best_val = eval_model(
            model,
            val_loader,
            id2label,
            verbose_report=False,
            fusion_method=cfg.fusion_method,
            f1_average="macro",
        )
        best_test = eval_model(
            model,
            test_loader,
            id2label,
            verbose_report=False,
            fusion_method=cfg.fusion_method,
            f1_average="macro",
        )

        fold_val_f1.append(best_val["f1"])
        fold_test_f1.append(best_test["f1"])

        print(
            f"Fold {fold_idx} | "
            f"Best rolling Val F1 {out['best_val_f1_rolling']:.4f} | "
            f"Val F1 {best_val['f1']:.4f} | "
            f"Test F1 {best_test['f1']:.4f}"
        )

        save_path = os.path.join(cfg.output_dir, f"fold{fold_idx}_{cfg.output_name}")
        torch.save(model.state_dict(), save_path)
        print(f"Saved fold model to {save_path}")

    import numpy as np
    print("\n===== CV Summary =====")
    print(f"Val macro-F1 mean {np.mean(fold_val_f1):.4f} std {np.std(fold_val_f1):.4f}")
    print(f"Test macro-F1 mean {np.mean(fold_test_f1):.4f} std {np.std(fold_test_f1):.4f}")

    # =====================================================
    # Final train on FULL train then test
    # =====================================================
    train_full_then_test(
        args=args,
        tokenizer=tokenizer,
        train_dataset_full=train_dataset_full,
        test_loader=test_loader,
        label2id=label2id,
        id2label=id2label,
    )


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
