import json
import os
import argparse
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score
from tqdm.auto import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AspectSentimentDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        tokenizer,
        max_len_sent: int = 128,
        max_len_term: int = 16,
        label2id: Dict[str, int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_len_sent = max_len_sent
        self.max_len_term = max_len_term

        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.samples = list(raw.values())

        # Nếu không truyền label2id thì tự build từ dữ liệu
        if label2id is None:
            labels = sorted(list({s["polarity"] for s in self.samples}))
            self.label2id = {lbl: i for i, lbl in enumerate(labels)}
        else:
            self.label2id = label2id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        sentence = item["sentence"]
        term = item["term"]
        label_str = item["polarity"]

        label = self.label2id[label_str]

        # Encode sentence
        sent_enc = self.tokenizer(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=self.max_len_sent,
            return_tensors="pt",
        )
        # Encode aspect term
        term_enc = self.tokenizer(
            term,
            truncation=True,
            padding="max_length",
            max_length=self.max_len_term,
            return_tensors="pt",
        )

        out = {
            "input_ids_sent": sent_enc["input_ids"].squeeze(0),
            "attention_mask_sent": sent_enc["attention_mask"].squeeze(0),
            "input_ids_term": term_enc["input_ids"].squeeze(0),
            "attention_mask_term": term_enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
        return out


class BertConcatClassifier(nn.Module):
    """
    BERT encoder dùng chung cho sentence và term.
    Fusion: concat [CLS]_sent và [CLS]_term.
    """

    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_size, num_labels)

    def forward(
        self,
        input_ids_sent,
        attention_mask_sent,
        input_ids_term,
        attention_mask_term,
        labels=None,
        fusion_method: str = "concat",
    ):
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
        cls_term = out_term.last_hidden_state[:, 0, :]

        # Fusion concat
        if fusion_method == "concat":
            fused = torch.cat([cls_sent, cls_term], dim=-1)  # [batch, 2*hidden]
        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")

        fused = self.dropout(fused)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}


def train_one_epoch(model, dataloader, optimizer, scheduler=None, fusion_method: str = "concat"):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        for k in batch:
            batch[k] = batch[k].to(DEVICE)

        outputs = model(
            input_ids_sent=batch["input_ids_sent"],
            attention_mask_sent=batch["attention_mask_sent"],
            input_ids_term=batch["input_ids_term"],
            attention_mask_term=batch["attention_mask_term"],
            labels=batch["label"],
            fusion_method=fusion_method,
        )
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def eval_model(model, dataloader, id2label=None, verbose_report=False, fusion_method: str = "concat"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            for k in batch:
                batch[k] = batch[k].to(DEVICE)

            outputs = model(
                input_ids_sent=batch["input_ids_sent"],
                attention_mask_sent=batch["attention_mask_sent"],
                input_ids_term=batch["input_ids_term"],
                attention_mask_term=batch["attention_mask_term"],
                labels=None,
                fusion_method=fusion_method,
            )
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)

    if verbose_report and id2label is not None:
        target_names = [id2label[i] for i in range(len(id2label))]
        print("Classification report:")
        print(classification_report(all_labels, all_preds, target_names=target_names))

    return acc


def parse_args():
    parser = argparse.ArgumentParser(
        description="BERT classifier for Aspect Sentiment Classification with sentence term concat fusion"
    )

    # Model and tokenizer
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Tên pretrained model trong HuggingFace",
    )

    # Data paths
    parser.add_argument(
        "--train_path",
        type=str,
        default="/kaggle/working/A-MoE-ASC/dataset/asc/laptop/train.json",
        help="Đường dẫn file train json",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="/kaggle/working/A-MoE-ASC/dataset/asc/laptop/dev.json",
        help="Đường dẫn file validation json",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="/kaggle/working/A-MoE-ASC/dataset/asc/laptop/test.json",
        help="Đường dẫn file test json",
    )

    # Sequence length
    parser.add_argument(
        "--max_len_sent",
        type=int,
        default=128,
        help="Max length cho câu",
    )
    parser.add_argument(
        "--max_len_term",
        type=int,
        default=16,
        help="Max length cho aspect term",
    )

    # Training hyperparameters
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size cho train",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size cho val và test",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Số epoch train",
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
        help="Dropout cho classifier",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Tỉ lệ warmup trên tổng số step",
    )

    # Fusion method (hiện tại mới hỗ trợ concat)
    parser.add_argument(
        "--fusion_method",
        type=str,
        default="concat",
        choices=["concat"],
        help="Phương pháp fusion giữa sentence và term representation",
    )

    # Saving
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saved_model",
        help="Thư mục lưu checkpoint",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="bert_concat_asc.pt",
        help="Tên file model state_dict",
    )

    # Misc
    parser.add_argument(
        "--verbose_report",
        action="store_true",
        help="In classification report trên test set",
    )

    args = parser.parse_args()
    return args


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Tạo dataset train trước để lấy label2id
    temp_train_dataset = AspectSentimentDataset(
        json_path=args.train_path,
        tokenizer=tokenizer,
        max_len_sent=args.max_len_sent,
        max_len_term=args.max_len_term,
        label2id=None,
    )
    label2id = temp_train_dataset.label2id
    id2label = {v: k for k, v in label2id.items()}
    print("Label mapping:", label2id)

    train_dataset = temp_train_dataset
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

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    model = BertConcatClassifier(
        model_name=args.model_name,
        num_labels=len(label2id),
        dropout=args.dropout,
    ).to(DEVICE)

    epochs = args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        avg_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            fusion_method=args.fusion_method,
        )
        print(f"Average train loss: {avg_loss:.4f}")

        val_acc = eval_model(
            model,
            val_loader,
            id2label,
            verbose_report=False,
            fusion_method=args.fusion_method,
        )
        print(f"Validation accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print("New best model on validation, saving in memory")

    # Load best model theo val
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(DEVICE)
        print(f"Loaded best model with val acc = {best_val_acc:.4f}")
    else:
        print("Warning: no best_state_dict saved, using last epoch model")

    print("Evaluation on test set:")
    test_acc = eval_model(
        model,
        test_loader,
        id2label,
        verbose_report=args.verbose_report,
        fusion_method=args.fusion_method,
    )
    print(f"Test accuracy: {test_acc:.4f}")

    # Lưu model ra file
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.output_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
