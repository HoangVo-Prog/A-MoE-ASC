import argparse
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data import AspectSentimentDataset
from model import BertConcatClassifier
from moe import MoEConfig
from utils import get_device


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer, scheduler=None, fusion_method: str = "concat", device: str = "cpu") -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}

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

        total_loss += float(loss.item())

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def eval_model(model: nn.Module, dataloader: DataLoader, id2label: Optional[Dict[int, str]] = None, verbose_report: bool = False, fusion_method: str = "concat", device: str = "cpu") -> float:
    model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASC training with optional MoE FFN replacement")
    p.add_argument("--model_name", type=str, default="bert-base-uncased")

    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--val_path", type=str, required=True)
    p.add_argument("--test_path", type=str, required=True)

    p.add_argument("--max_len_sent", type=int, default=128)
    p.add_argument("--max_len_term", type=int, default=16)

    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.1)

    p.add_argument("--fusion_method", type=str, default="concat", choices=["concat", "add", "mul"])

    p.add_argument("--output_dir", type=str, default="saved_model")
    p.add_argument("--output_name", type=str, default="bert_concat_asc.pt")
    p.add_argument("--verbose_report", action="store_true")

    # MoE options
    p.add_argument("--use_moe", action="store_true")
    p.add_argument("--moe_num_experts", type=int, default=8)
    p.add_argument("--moe_top_k", type=int, default=1)
    p.add_argument("--aux_loss_weight", type=float, default=0.01)

    # flags bạn yêu cầu
    p.add_argument("--freeze_base", action="store_true")
    p.add_argument("--route_mask_pad_tokens", action="store_true")

    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

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

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    moe_cfg = None
    if args.use_moe:
        moe_cfg = MoEConfig(
            num_experts=args.moe_num_experts,
            top_k=args.moe_top_k,
            route_mask_pad_tokens=bool(args.route_mask_pad_tokens),
        )

    model = BertConcatClassifier(
        model_name=args.model_name,
        num_labels=len(label2id),
        dropout=args.dropout,
        use_moe=bool(args.use_moe),
        moe_cfg=moe_cfg,
        freeze_base=bool(args.freeze_base),
        aux_loss_weight=float(args.aux_loss_weight),
    ).to(device)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, fusion_method=args.fusion_method, device=device)
        print(f"Average train loss: {avg_loss:.4f}")

        val_acc = eval_model(model, val_loader, id2label, verbose_report=False, fusion_method=args.fusion_method, device=device)
        print(f"Validation accuracy: {val_acc:.4f}")
        
        test_acc = eval_model(model, test_loader, id2label, verbose_report=args.verbose_report, fusion_method=args.fusion_method, device=device)
        print(f"Test accuracy: {test_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print("New best model on validation, saving in memory")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)
        print(f"Loaded best model with val acc = {best_val_acc:.4f}")
    else:
        print("Warning: no best_state_dict saved, using last epoch model")

    print("Evaluation on test set:")
    test_acc = eval_model(model, test_loader, id2label, verbose_report=args.verbose_report, fusion_method=args.fusion_method, device=device)
    print(f"Test accuracy: {test_acc:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, args.output_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main(parse_args())
