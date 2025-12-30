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


def get_sentence(row: dict) -> str:
    s = row.get("sentence")
    return s.strip() if isinstance(s, str) else ""


def key_stats(rows, name):
    pols = Counter(get_polarity(r) for r in rows)
    sent_cnt = len({get_sentence(r) for r in rows if get_sentence(r)})
    print(f"[{name}] rows={len(rows)} uniq_sentences={sent_cnt} polarity_counts(top)={pols.most_common(6)}")


def sentence_strata_key(pols_for_sentence):
    """
    pols_for_sentence: list[str] polarity của các aspect trong cùng một sentence.
    strata = "<major>|neu0/neu1"
    """
    c = Counter(pols_for_sentence)
    has_neu = 1 if c.get("neutral", 0) > 0 else 0

    if not c:
        return f"unknown|neu{has_neu}"

    most = c.most_common()
    top_cnt = most[0][1]
    tops = [k for k, v in c.items() if v == top_cnt]

    if len(tops) == 1:
        major = tops[0]
    else:
        # tie rule ổn định, tránh tạo class giả
        if "neutral" in tops:
            major = "neutral"
        else:
            order = ["positive", "negative", "neutral"]
            tops_sorted = sorted(tops, key=lambda x: order.index(x) if x in order else 999)
            major = tops_sorted[0]

    return f"{major}|neu{has_neu}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default=None)

    # tuỳ chọn thêm: nếu bạn muốn chặn leakage do trùng sentence giữa train và test
    ap.add_argument("--dedup_train_vs_test", action="store_true")
    ap.add_argument("--no_split_train_val", action="store_true",
                help="If set, do not split train into train/val. Write val.json as empty list.")


    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir

    train_rows = read_json(data_dir / "train.json")
    test_rows = read_json(data_dir / "test.json")

    key_stats(train_rows, "train_raw")
    key_stats(test_rows, "test_raw")

    # 0) Optional: loại train sentences trùng hệt test sentences để tránh leakage (nếu dataset có duplicate)
    if args.dedup_train_vs_test:
        test_sents = {get_sentence(r) for r in test_rows if get_sentence(r)}
        before = len(train_rows)
        train_rows = [r for r in train_rows if get_sentence(r) not in test_sents]
        print("Dedup train vs test by sentence:", before, "->", len(train_rows))

    # 1) Drop conflict instances (benchmark SemEval14 không dùng conflict)
    train_clean = [r for r in train_rows if get_polarity(r) != "conflict"]
    test_clean = [r for r in test_rows if get_polarity(r) != "conflict"]

    key_stats(train_clean, "train_clean")
    key_stats(test_clean, "test_clean")
    
    # 2.x) Optional: do not split train/val
    if args.no_split_train_val:
        write_json(out_dir / "train.json", train_clean)
        write_json(out_dir / "val.json", [])
        write_json(out_dir / "test.json", test_clean)

        print("Done (no train/val split)")
        key_stats(train_clean, "train_out")
        key_stats([], "val_out")
        key_stats(test_clean, "test_out")
        return

    # 3) Build sentence -> polarity list để group split theo sentence
    sent2pols = defaultdict(list)
    for r in train_clean:
        sent = get_sentence(r)
        if not sent:
            continue
        pol = get_polarity(r)
        if pol:
            sent2pols[sent].append(pol)

    sentences = list(sent2pols.keys())
    strata = [sentence_strata_key(sent2pols[s]) for s in sentences]

    print("Sentences usable for split:", len(sentences))
    if len(sentences) == 0:
        write_json(out_dir / "train.json", [])
        write_json(out_dir / "val.json", [])
        write_json(out_dir / "test.json", test_clean)
        print("No sentences found after cleaning. Wrote empty train/val. Test kept cleaned.")
        return

    strata_counts = Counter(strata)
    print("Strata counts (top):", strata_counts.most_common(12))

    # 4) Stratify nếu đủ điều kiện
    can_stratify = (0.0 < args.val_ratio < 1.0) and (len(sentences) >= 2) and (min(strata_counts.values()) >= 2)

    if len(sentences) == 1:
        train_sents, val_sents = sentences, []
    else:
        if can_stratify:
            try:
                train_sents, val_sents = train_test_split(
                    sentences,
                    test_size=args.val_ratio,
                    random_state=args.seed,
                    stratify=strata,
                )
            except ValueError as e:
                print("Stratified split failed, fallback to random split. Reason:", str(e))
                train_sents, val_sents = train_test_split(
                    sentences,
                    test_size=args.val_ratio,
                    random_state=args.seed,
                    shuffle=True,
                    stratify=None,
                )
        else:
            print("Not enough samples per stratum for stratify, fallback to random split.")
            train_sents, val_sents = train_test_split(
                sentences,
                test_size=args.val_ratio,
                random_state=args.seed,
                shuffle=True,
                stratify=None,
            )

    train_sents = set(train_sents)
    val_sents = set(val_sents)

    train_out = [r for r in train_clean if get_sentence(r) in train_sents]
    val_out = [r for r in train_clean if get_sentence(r) in val_sents]

    write_json(out_dir / "train.json", train_out)
    write_json(out_dir / "val.json", val_out)
    write_json(out_dir / "test.json", test_clean)

    print("Done")
    print("Train sentences:", len(train_sents), "Val sentences:", len(val_sents))
    key_stats(train_out, "train_out")
    key_stats(val_out, "val_out")
    key_stats(test_clean, "test_out")


if __name__ == "__main__":
    main()
