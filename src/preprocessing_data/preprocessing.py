#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def read_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data)}")
    return data


def write_json(path: Path, data: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def norm(x) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def sentences_to_drop_due_to_conflict(rows: List[Dict]) -> set:
    drop = set()
    for r in rows:
        sent = r.get("sentence")
        if not isinstance(sent, str) or not sent.strip():
            continue
        if norm(r.get("polarity")) == "conflict":
            drop.add(sent)
    return drop


def drop_sentences(rows: List[Dict], drop_set: set) -> List[Dict]:
    return [r for r in rows if r.get("sentence") not in drop_set]


def split_sentences(
    sentences: List[str],
    val_ratio: float,
    seed: int,
) -> Tuple[set, set]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    rng = random.Random(seed)
    sentences = list(sentences)
    rng.shuffle(sentences)

    if len(sentences) == 0:
        return set(), set()

    n_val = max(1, int(round(len(sentences) * val_ratio)))
    val_sents = set(sentences[:n_val])
    train_sents = set(sentences[n_val:])
    return train_sents, val_sents


def uniq_sentence_count(rows: List[Dict]) -> int:
    return len(
        {r.get("sentence") for r in rows if isinstance(r.get("sentence"), str)}
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.json"
    test_path = data_dir / "test.json"

    out_dir = Path(args.out_dir) if args.out_dir else data_dir

    train_rows = read_json(train_path)
    test_rows = read_json(test_path)

    # 1) collect sentences to drop (polarity = conflict)
    drop_set = (
        sentences_to_drop_due_to_conflict(train_rows)
        | sentences_to_drop_due_to_conflict(test_rows)
    )

    # 2) drop sentences
    train_clean = drop_sentences(train_rows, drop_set)
    test_clean = drop_sentences(test_rows, drop_set)

    # 3) split train -> train / val by sentence
    train_sentences = sorted(
        {r.get("sentence") for r in train_clean if isinstance(r.get("sentence"), str)}
    )
    train_sents, val_sents = split_sentences(
        train_sentences,
        args.val_ratio,
        args.seed,
    )

    train_out = [r for r in train_clean if r.get("sentence") in train_sents]
    val_out = [r for r in train_clean if r.get("sentence") in val_sents]

    # 4) write output
    write_json(out_dir / "train.json", train_out)
    write_json(out_dir / "val.json", val_out)
    write_json(out_dir / "test.json", test_clean)

    # 5) stats
    print("==== Done ====")
    print(f"Drop sentences (conflict): {len(drop_set)}")
    print(f"Train rows: {len(train_out)} | uniq sentences: {uniq_sentence_count(train_out)}")
    print(f"Val rows: {len(val_out)} | uniq sentences: {uniq_sentence_count(val_out)}")
    print(f"Test rows: {len(test_clean)} | uniq sentences: {uniq_sentence_count(test_clean)}")


if __name__ == "__main__":
    main()
