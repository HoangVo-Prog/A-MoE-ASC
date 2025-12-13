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


def get_polarity(row: dict) -> str:
    # hỗ trợ cả schema mới và cũ
    if "polarity" in row:
        return norm(row.get("polarity"))
    return norm(row.get("sentiment"))


def conflict_sentence_set(rows):
    s = set()
    for r in rows:
        sent = r.get("sentence")
        if isinstance(sent, str) and sent.strip():
            if get_polarity(r) == "conflict":
                s.add(sent)
    return s


def drop_sentences(rows, drop_set):
    return [r for r in rows if r.get("sentence") not in drop_set]


def sentence_majority_label(rows):
    sent2pols = defaultdict(list)
    for r in rows:
        sent = r.get("sentence")
        if not isinstance(sent, str) or not sent.strip():
            continue
        pol = get_polarity(r)
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

    # Debug quick counts
    def key_stats(rows, name):
        pols = Counter(get_polarity(r) for r in rows)
        sent_cnt = len({r.get("sentence") for r in rows if isinstance(r.get("sentence"), str)})
        print(f"[{name}] rows={len(rows)} uniq_sentences={sent_cnt} polarity_counts(top)={pols.most_common(6)}")

    key_stats(train_rows, "train_raw")
    key_stats(test_rows, "test_raw")

    # 1) Drop conflict sentence across train + test
    drop_set = conflict_sentence_set(train_rows) | conflict_sentence_set(test_rows)
    train_clean = drop_sentences(train_rows, drop_set)
    test_clean = drop_sentences(test_rows, drop_set)

    key_stats(train_clean, "train_clean")
    key_stats(test_clean, "test_clean")
    print("Dropped conflict sentences:", len(drop_set))

    # 2) Build sentence labels (for stratify)
    sent2label = sentence_majority_label(train_clean)
    sentences = list(sent2label.keys())
    labels = [sent2label[s] for s in sentences]

    print("Sentences usable for split:", len(sentences))
    if len(sentences) == 0:
        # Không có sentence để split
        write_json(out_dir / "train.json", [])
        write_json(out_dir / "val.json", [])
        write_json(out_dir / "test.json", test_clean)
        print("No sentences found after cleaning. Wrote empty train/val.")
        return

    # 3) Stratify nếu 가능: mỗi lớp phải có >=2 samples khi split
    label_counts = Counter(labels)
    min_class = min(label_counts.values())
    can_stratify = (min_class >= 2) and (0.0 < args.val_ratio < 1.0) and (len(sentences) >= 2)

    if can_stratify:
        try:
            train_sents, val_sents = train_test_split(
                sentences,
                test_size=args.val_ratio,
                random_state=args.seed,
                stratify=labels,
            )
        except ValueError as e:
            # fallback nếu sklearn vẫn không chia được
            print("Stratified split failed, fallback to random split. Reason:", str(e))
            train_sents, val_sents = train_test_split(
                sentences,
                test_size=args.val_ratio,
                random_state=args.seed,
                shuffle=True,
                stratify=None,
            )
    else:
        print("Not enough samples per class for stratify, fallback to random split.")
        if len(sentences) == 1:
            train_sents, val_sents = sentences, []
        else:
            train_sents, val_sents = train_test_split(
                sentences,
                test_size=args.val_ratio,
                random_state=args.seed,
                shuffle=True,
                stratify=None,
            )

    train_sents = set(train_sents)
    val_sents = set(val_sents)

    train_out = [r for r in train_clean if r.get("sentence") in train_sents]
    val_out = [r for r in train_clean if r.get("sentence") in val_sents]

    write_json(out_dir / "train.json", train_out)
    write_json(out_dir / "val.json", val_out)
    write_json(out_dir / "test.json", test_clean)

    print("Done")
    print("Train sentences:", len(train_sents), "Val sentences:", len(val_sents))


if __name__ == "__main__":
    main()
