#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split


def read_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        x = json.load(f)
    if not isinstance(x, list):
        raise ValueError(f"{p} must be a JSON list")
    return x


def write_json(p: Path, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def norm(x) -> str:
    return str(x).strip().lower() if x is not None else ""


def conflict_sentence_set(rows):
    s = set()
    for r in rows:
        sent = r.get("sentence")
        if isinstance(sent, str) and sent.strip() and norm(r.get("polarity")) == "conflict":
            s.add(sent)
    return s


def drop_sentences(rows, drop_set):
    return [r for r in rows if r.get("sentence") not in drop_set]


def sentence_majority_label(rows):
    """
    Trả về dict: sentence -> label (majority polarity)
    Nếu hòa: label = "tie"
    """
    sent2pols = defaultdict(list)
    for r in rows:
        sent = r.get("sentence")
        if not isinstance(sent, str) or not sent.strip():
            continue
        pol = norm(r.get("polarity"))
        if pol:
            sent2pols[sent].append(pol)

    sent2label = {}
    for sent, pols in sent2pols.items():
        c = Counter(pols)
        best_cnt = c.most_common(1)[0][1]
        best = [k for k, v in c.items() if v == best_cnt]
        sent2label[sent] = best[0] if len(best) == 1 else "tie"
    return sent2label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir

    train_rows = read_json(data_dir / "train.json")
    test_rows = read_json(data_dir / "test.json")

    # 1) Drop sentence nếu có conflict ở train hoặc test
    drop_set = conflict_sentence_set(train_rows) | conflict_sentence_set(test_rows)
    train_clean = drop_sentences(train_rows, drop_set)
    test_clean = drop_sentences(test_rows, drop_set)

    # 2) Sentence labels để stratify (trên train_clean)
    sent2label = sentence_majority_label(train_clean)
    sentences = list(sent2label.keys())
    labels = [sent2label[s] for s in sentences]

    # 3) Stratified split theo sentence
    train_sents, val_sents = train_test_split(
        sentences,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=labels,
    )
    train_sents = set(train_sents)
    val_sents = set(val_sents)

    train_out = [r for r in train_clean if r.get("sentence") in train_sents]
    val_out = [r for r in train_clean if r.get("sentence") in val_sents]

    # 4) Write outputs
    write_json(out_dir / "train.json", train_out)
    write_json(out_dir / "val.json", val_out)
    write_json(out_dir / "test.json", test_clean)

    print("Done")
    print("Dropped conflict sentences:", len(drop_set))
    print("Train sentences:", len(train_sents), "Val sentences:", len(val_sents))


if __name__ == "__main__":
    main()
