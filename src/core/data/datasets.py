from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class AspectSentimentDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        tokenizer,
        max_len_sent: int,
        max_len_term: int,
        label2id: Optional[Dict[str, int]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len_sent = max_len_sent
        self.max_len_term = max_len_term

        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        for i, s in enumerate(self.samples[:5]):
            if not {"sentence", "aspect", "sentiment"} <= s.keys():
                raise ValueError(f"Invalid sample at index {i}: {s}")

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


def _majority_with_tiebreak(
    pols: Sequence[str],
    prefer_order_no_neu: Tuple[str, ...] = ("positive", "negative", "neutral"),
) -> str:
    c = Counter(pols)
    if not c:
        return "neutral"

    max_cnt = max(c.values())
    tied = [p for p, cnt in c.items() if cnt == max_cnt]

    if len(tied) == 1:
        return tied[0]

    if "neutral" in tied:
        return "neutral"

    for p in prefer_order_no_neu:
        if p in tied:
            return p

    return sorted(tied)[0]


class _SubsetAspectSentimentDataset(Dataset):
    def __init__(
        self,
        base_samples: List[dict],
        indices: List[int],
        tokenizer,
        max_len_sent: int,
        max_len_term: int,
        label2id: Dict[str, int],
    ) -> None:
        self._base_samples = base_samples
        self._indices = indices
        self.tokenizer = tokenizer
        self.max_len_sent = max_len_sent
        self.max_len_term = max_len_term
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self._indices[idx]
        item = self._base_samples[real_idx]

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

    @property
    def base_indices(self) -> List[int]:
        return self._indices


class AspectSentimentDatasetKFold(Dataset):
    """
    Full-train dataset with sentence-level K-fold split (phân rã KFoldConfig).

    Parameters:
        k_folds: số fold, ví dụ 5
        seed: seed dùng cho shuffle sentence
        shuffle: có shuffle sentence indices trước khi chia fold hay không

    Usage:
        base = AspectSentimentDatasetKFold(
            json_path="train.json",
            tokenizer=tok,
            max_len_sent=128,
            max_len_term=16,
            k_folds=config.k_folds,
            seed=config.seed,
            shuffle=True,
        )
        train_ds, val_ds = base.get_fold(0)
    """

    def __init__(
        self,
        json_path: str,
        tokenizer,
        max_len_sent: int,
        max_len_term: int,
        k_folds: int,
        seed: int,
        shuffle: bool = True,
        label2id: Optional[Dict[str, int]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len_sent = max_len_sent
        self.max_len_term = max_len_term

        self.k_folds = int(k_folds)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)

        with open(json_path, "r", encoding="utf-8") as f:
            self.samples: List[dict] = json.load(f)

        if label2id is None:
            labels = sorted({s["sentiment"] for s in self.samples})
            self.label2id = {lbl: i for i, lbl in enumerate(labels)}
        else:
            self.label2id = label2id

        (
            self._sent_list,
            self._sent_to_row_indices,
            self._sent_strata,
        ) = self._build_sentence_groups_and_strata(self.samples)

        self._folds = self._build_folds(
            n_sents=len(self._sent_list),
            strata=self._sent_strata,
            k_folds=self.k_folds,
            seed=self.seed,
            shuffle=self.shuffle,
        )

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

    def num_folds(self) -> int:
        return len(self._folds)

    def get_fold(self, fold_idx: int) -> Tuple[Dataset, Dataset]:
        if fold_idx < 0 or fold_idx >= len(self._folds):
            raise IndexError(f"fold_idx out of range: {fold_idx}")

        val_sent_indices = set(self._folds[fold_idx])
        train_sent_indices = [i for i in range(len(self._sent_list)) if i not in val_sent_indices]

        train_row_indices = self._sent_indices_to_row_indices(train_sent_indices)
        val_row_indices = self._sent_indices_to_row_indices(sorted(val_sent_indices))

        train_ds = _SubsetAspectSentimentDataset(
            base_samples=self.samples,
            indices=train_row_indices,
            tokenizer=self.tokenizer,
            max_len_sent=self.max_len_sent,
            max_len_term=self.max_len_term,
            label2id=self.label2id,
        )
        val_ds = _SubsetAspectSentimentDataset(
            base_samples=self.samples,
            indices=val_row_indices,
            tokenizer=self.tokenizer,
            max_len_sent=self.max_len_sent,
            max_len_term=self.max_len_term,
            label2id=self.label2id,
        )
        return train_ds, val_ds

    def _sent_indices_to_row_indices(self, sent_indices: List[int]) -> List[int]:
        out: List[int] = []
        for si in sent_indices:
            sent = self._sent_list[si]
            out.extend(self._sent_to_row_indices[sent])
        return out

    @staticmethod
    def _build_sentence_groups_and_strata(
        samples: List[dict],
    ) -> Tuple[List[str], Dict[str, List[int]], List[str]]:
        sent_to_rows: Dict[str, List[int]] = defaultdict(list)
        sent_to_pols: Dict[str, List[str]] = defaultdict(list)

        for i, s in enumerate(samples):
            sent = s["sentence"]
            pol = s["sentiment"]
            sent_to_rows[sent].append(i)
            sent_to_pols[sent].append(pol)

        sent_list = list(sent_to_rows.keys())

        sent_strata: List[str] = []
        for sent in sent_list:
            pols = sent_to_pols[sent]
            major = _majority_with_tiebreak(pols)
            has_neu = any(p == "neutral" for p in pols)
            strata_key = f"{major}|neu{1 if has_neu else 0}"
            sent_strata.append(strata_key)

        return sent_list, dict(sent_to_rows), sent_strata

    @staticmethod
    def _build_folds(
        n_sents: int,
        strata: List[str],
        k_folds: int,
        seed: int,
        shuffle: bool,
    ) -> List[List[int]]:
        if k_folds <= 1:
            raise ValueError("k_folds must be >= 2 for K-fold")

        if n_sents < k_folds:
            raise ValueError(f"Not enough sentences for k-fold: n_sents={n_sents} < k_folds={k_folds}")

        counts = Counter(strata)
        can_stratify = all(v >= k_folds for v in counts.values())

        rng = random.Random(seed)
        indices = list(range(n_sents))
        if shuffle:
            rng.shuffle(indices)

        if not can_stratify:
            folds: List[List[int]] = [[] for _ in range(k_folds)]
            for j, si in enumerate(indices):
                folds[j % k_folds].append(si)
            return folds

        strata_to_indices: Dict[str, List[int]] = defaultdict(list)
        for si in indices:
            strata_to_indices[strata[si]].append(si)

        folds = [[] for _ in range(k_folds)]
        for _, group in strata_to_indices.items():
            for j, si in enumerate(group):
                folds[j % k_folds].append(si)

        if shuffle:
            for f in folds:
                rng.shuffle(f)

        return folds
