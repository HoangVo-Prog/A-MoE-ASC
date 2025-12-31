from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
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


def _majority_with_tiebreak(
    pols: Sequence[str],
    prefer_order_no_neu: Tuple[str, ...] = ("positive", "negative", "neutral"),
) -> str:
    """
    Replicate the tie break logic:
    - majority vote on polarity list
    - if tie and there is 'neutral' among tied, choose 'neutral'
    - else choose by fixed priority: positive > negative > neutral
    """
    c = Counter(pols)
    if not c:
        return "neutral"

    max_cnt = max(c.values())
    tied = [p for p, cnt in c.items() if cnt == max_cnt]

    if len(tied) == 1:
        return tied[0]

    # tie case
    if "neutral" in tied:
        return "neutral"

    for p in prefer_order_no_neu:
        if p in tied:
            return p

    # fallback
    return sorted(tied)[0]


class _SubsetAspectSentimentDataset(Dataset):
    """
    A lightweight subset view that preserves the same __getitem__ output format.
    """
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
    def base_indices(self):
        return self._indices

class AspectSentimentDatasetKFold(Dataset):
    """
    Full-train dataset with sentence-level K-fold split.

    Usage:
        base = AspectSentimentDatasetKFold(
            json_path="train.json",
            tokenizer=tok,
            max_len_sent=128,
            max_len_term=16,
            kfold=KFoldConfig(k=5, seed=42, shuffle=True),
        )

        train_ds, val_ds = base.get_fold(0)  # both are Dataset with same __getitem__ format
    """

    def __init__(
        self,
        config,
        json_path,
        tokenizer,
        max_len_sent: int,
        max_len_term: int,
        label2id: Optional[Dict[str, int]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len_sent = max_len_sent
        self.max_len_term = max_len_term
        self.kfold = config.kfold

        with open(json_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        self.samples: List[dict] = samples

        # Build label mapping from TRAIN only (full train list)
        if label2id is None:
            labels = sorted({s["sentiment"] for s in self.samples})
            self.label2id = {lbl: i for i, lbl in enumerate(labels)}
        else:
            self.label2id = label2id

        # Precompute sentence grouping, strata labels, and folds
        (
            self._sent_list,
            self._sent_to_row_indices,
            self._sent_strata,
        ) = self._build_sentence_groups_and_strata(self.samples)

        self._folds = self._build_folds(
            n_sents=len(self._sent_list),
            strata=self._sent_strata,
            kfold=self.kfold,
        )

    def __len__(self) -> int:
        # Base dataset length is full train rows
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Allows using the base dataset directly if needed
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
        """
        Return (train_dataset, val_dataset) for a specific fold.
        Both datasets return the same dict format as __getitem__.
        Split is sentence-level, so no sentence appears in both.
        """
        if fold_idx < 0 or fold_idx >= len(self._folds):
            raise IndexError(f"fold_idx out of range: {fold_idx} not in [0, {len(self._folds) - 1}]")

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
        """
        Replicate preprocessing sentence-level split logic:
        - group rows by sentence
        - for each sentence, compute major polarity and whether neutral exists
        - strata_key = f"{major}|neu0" or f"{major}|neu1"
        """
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
    def _build_folds(n_sents: int, strata: List[str], kfold) -> List[List[int]]:
        """
        Build K folds on sentence indices.
        Prefer stratified folds if each stratum has at least K samples.
        Else fallback to plain KFold-like splitting.
        """
        if kfold.k <= 1:
            raise ValueError("k must be >= 2 for K-fold")

        if n_sents < kfold.k:
            raise ValueError(f"Not enough sentences for k-fold: n_sents={n_sents} < k={kfold.k}")

        # Check stratify feasibility: each stratum count >= K
        counts = Counter(strata)
        can_stratify = all(v >= kfold.k for v in counts.values())

        rng = random.Random(kfold.seed)
        indices = list(range(n_sents))

        if kfold.shuffle:
            rng.shuffle(indices)

        if not can_stratify:
            # Simple KFold split on shuffled sentence indices
            folds: List[List[int]] = [[] for _ in range(kfold.k)]
            for j, si in enumerate(indices):
                folds[j % kfold.k].append(si)
            return folds

        # Stratified KFold on sentence level:
        # We implement manually to avoid hard dependency on sklearn.
        # Approach:
        # - group sentence indices by strata
        # - distribute each group's indices round-robin into K folds
        strata_to_indices: Dict[str, List[int]] = defaultdict(list)
        for si in indices:
            strata_to_indices[strata[si]].append(si)

        folds = [[] for _ in range(kfold.k)]
        for key, group in strata_to_indices.items():
            # group is already shuffled if cfg.shuffle
            for j, si in enumerate(group):
                folds[j % kfold.k].append(si)

        # Optional: shuffle within each fold for nicer randomness
        if kfold.shuffle:
            for f in folds:
                rng.shuffle(f)

        return folds
