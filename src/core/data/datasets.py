from __future__ import annotations

import json
import random
import re
import string
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


def _find_subsequence(haystack: List[int], needle: List[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("-", " ")
    s = s.strip(string.punctuation)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _compute_aspect_span(
    *,
    tokenizer,
    sent_enc,
    term: str,
    sentence: str,
    max_len_sent: int,
    max_len_term: int,
):
    sentence_norm = _normalize_text(sentence)
    term_norm = _normalize_text(term)
    term_raw = (term or "").lower()

    term_enc = tokenizer(
        term_norm,
        truncation=True,
        padding="max_length",
        max_length=max_len_term,
        return_tensors="pt",
        add_special_tokens=False,
    )
    sent_ids = sent_enc["input_ids"].squeeze(0).tolist()
    sent_mask = sent_enc["attention_mask"].squeeze(0).tolist()

    valid_len = int(sum(sent_mask))
    if valid_len <= 0:
        aspect_start = 0
        aspect_end = 0
        aspect_mask = torch.zeros(max_len_sent, dtype=torch.long)
        return (
            aspect_start,
            aspect_end,
            aspect_mask,
            [],
            "NOT_FOUND_RAW",
            False,
            {
                "sentence_norm": sentence_norm,
                "term_norm": term_norm,
                "content_ids": [],
                "term_ids": [],
                "sent_tokens": [],
                "term_tokens": [],
                "token_match_idx": -1,
                "valid_len": valid_len,
                "sep_idx": -1,
            },
        )

    # Match aspect token IDs inside the sentence content region [CLS] ... [SEP].
    sep_id = getattr(tokenizer, "sep_token_id", None)
    sep_idx = None
    if sep_id is not None:
        try:
            sep_idx = sent_ids.index(sep_id)
        except ValueError:
            sep_idx = None
    if sep_idx is None:
        sep_idx = min(valid_len - 1, max_len_sent - 1)

    content_start = 1
    content_end = max(content_start, sep_idx)

    content_ids = sent_ids[content_start:content_end]
    sent_tokens = tokenizer.convert_ids_to_tokens(content_ids)
    term_tokens = tokenizer.tokenize(term_raw)

    token_match_idx = _find_subsequence(sent_tokens, term_tokens)
    if token_match_idx >= 0 and len(term_tokens) > 0:
        aspect_start = content_start + token_match_idx
        aspect_end = aspect_start + len(term_tokens)
        if aspect_start >= max_len_sent:
            aspect_start = 0
            aspect_end = 0
            aspect_mask = torch.zeros(max_len_sent, dtype=torch.long)
            return (
                aspect_start,
                aspect_end,
                aspect_mask,
                [],
                "TRUNCATED",
                False,
                {
                    "sentence_norm": sentence_norm,
                    "term_norm": term_norm,
                    "content_ids": content_ids,
                    "term_ids": [],
                    "sent_tokens": sent_tokens,
                    "term_tokens": term_tokens,
                    "token_match_idx": token_match_idx,
                    "valid_len": valid_len,
                    "sep_idx": sep_idx,
                },
            )

        aspect_end = min(aspect_end, max_len_sent)
        aspect_mask = torch.zeros(max_len_sent, dtype=torch.long)
        if aspect_end > aspect_start:
            aspect_mask[aspect_start:aspect_end] = 1

        matched_tokens = sent_tokens[token_match_idx : token_match_idx + len(term_tokens)]
        return (
            aspect_start,
            aspect_end,
            aspect_mask,
            matched_tokens,
            "OK",
            True,
            {
                "sentence_norm": sentence_norm,
                "term_norm": term_norm,
                "content_ids": content_ids,
                "term_ids": [],
                "sent_tokens": sent_tokens,
                "term_tokens": term_tokens,
                "token_match_idx": token_match_idx,
                "valid_len": valid_len,
                "sep_idx": sep_idx,
            },
        )

    term_ids = term_enc["input_ids"].squeeze(0).tolist()
    term_mask = term_enc["attention_mask"].squeeze(0).tolist()
    term_len = int(sum(term_mask))
    term_ids = term_ids[:term_len]

    match_idx = _find_subsequence(content_ids, term_ids)
    if match_idx < 0 or term_len <= 0:
        raw_found = term_norm != "" and term_norm in sentence_norm
        truncated = (valid_len >= max_len_sent) or (sep_idx >= max_len_sent - 1)
        if not raw_found:
            fail_reason = "NOT_FOUND_RAW"
        elif truncated:
            fail_reason = "TRUNCATED"
        else:
            fail_reason = "TOKEN_MISMATCH"
        aspect_start = 0
        aspect_end = 0
        aspect_mask = torch.zeros(max_len_sent, dtype=torch.long)
        return (
            aspect_start,
            aspect_end,
            aspect_mask,
            [],
            fail_reason,
            False,
            {
                "sentence_norm": sentence_norm,
                "term_norm": term_norm,
                "content_ids": content_ids,
                "term_ids": term_ids,
                "sent_tokens": sent_tokens,
                "term_tokens": term_tokens,
                "token_match_idx": token_match_idx,
                "valid_len": valid_len,
                "sep_idx": sep_idx,
            },
        )

    aspect_start = content_start + match_idx
    aspect_end = aspect_start + term_len
    if aspect_start >= max_len_sent:
        aspect_start = 0
        aspect_end = 0
        aspect_mask = torch.zeros(max_len_sent, dtype=torch.long)
        return (
            aspect_start,
            aspect_end,
            aspect_mask,
            [],
            "TRUNCATED",
            False,
            {
                "sentence_norm": sentence_norm,
                "term_norm": term_norm,
                "content_ids": content_ids,
                "term_ids": term_ids,
                "sent_tokens": sent_tokens,
                "term_tokens": term_tokens,
                "token_match_idx": token_match_idx,
                "valid_len": valid_len,
                "sep_idx": sep_idx,
            },
        )

    aspect_end = min(aspect_end, max_len_sent)
    aspect_mask = torch.zeros(max_len_sent, dtype=torch.long)
    if aspect_end > aspect_start:
        aspect_mask[aspect_start:aspect_end] = 1

    matched_tokens = tokenizer.convert_ids_to_tokens(sent_ids[aspect_start:aspect_end])
    return (
        aspect_start,
        aspect_end,
        aspect_mask,
        matched_tokens,
        "OK",
        True,
        {
            "sentence_norm": sentence_norm,
            "term_norm": term_norm,
            "content_ids": content_ids,
            "term_ids": term_ids,
            "sent_tokens": sent_tokens,
            "term_tokens": term_tokens,
            "token_match_idx": token_match_idx,
            "valid_len": valid_len,
            "sep_idx": sep_idx,
        },
    )


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

        self._debug_span_prints = 0
        self._debug_span_limit = 0
        self._debug_epoch = None
        self._debug_split = ""
        self._debug_batch_idx = 0
        self._debug_seen = 0

        self.total_samples = 0
        self.matched_samples = 0
        self.matched_mask_sum = 0.0
        self.token_mismatch_count = 0
        self.truncated_count = 0
        self.not_found_raw_count = 0

    def __len__(self) -> int:
        return len(self.samples)

    def begin_debug(self, *, epoch: int, split: str, batch_idx: int = 0, max_samples: int = 5) -> None:
        self._debug_span_prints = 0
        self._debug_span_limit = int(max_samples)
        self._debug_epoch = int(epoch)
        self._debug_split = str(split)
        self._debug_batch_idx = int(batch_idx)
        self._debug_seen = 0

    def reset_match_stats(self) -> None:
        self.total_samples = 0
        self.matched_samples = 0
        self.matched_mask_sum = 0.0
        self.token_mismatch_count = 0
        self.truncated_count = 0
        self.not_found_raw_count = 0

    def update_match_stats(self, *, total: int, matched: int, matched_mask_sum: float) -> None:
        self.total_samples += int(total)
        self.matched_samples += int(matched)
        self.matched_mask_sum += float(matched_mask_sum)

    @property
    def match_rate(self) -> float:
        if self.total_samples <= 0:
            return 0.0
        return float(self.matched_samples) / float(self.total_samples)

    def get_match_stats(self) -> Dict[str, float]:
        avg_mask = (
            float(self.matched_mask_sum) / float(self.matched_samples)
            if self.matched_samples > 0
            else 0.0
        )
        return {"match_rate": self.match_rate, "avg_mask_sum": avg_mask}

    def get_diag_stats(self) -> Dict[str, float]:
        return {
            "total": float(self.total_samples),
            "matched": float(self.matched_samples),
            "match_rate": self.match_rate,
            "token_mismatch": float(self.token_mismatch_count),
            "truncated": float(self.truncated_count),
            "not_found_raw": float(self.not_found_raw_count),
        }

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

        (
            aspect_start,
            aspect_end,
            aspect_mask_sent,
            matched_tokens,
            fail_reason,
            matched,
            diag,
        ) = _compute_aspect_span(
            tokenizer=self.tokenizer,
            sent_enc=sent_enc,
            term=term,
            sentence=sentence,
            max_len_sent=self.max_len_sent,
            max_len_term=self.max_len_term,
        )

        self.total_samples += 1
        if matched:
            self.matched_samples += 1
            self.matched_mask_sum += float(aspect_mask_sent.sum().item())
        else:
            if fail_reason == "TOKEN_MISMATCH":
                self.token_mismatch_count += 1
            elif fail_reason == "TRUNCATED":
                self.truncated_count += 1
            elif fail_reason == "NOT_FOUND_RAW":
                self.not_found_raw_count += 1

        if self._debug_epoch is not None:
            sample_idx = self._debug_seen
            self._debug_seen += 1

            if (
                self._debug_epoch == 0
                and self._debug_split in {"val", "test"}
                and self._debug_batch_idx == 0
                and sample_idx < 10
                and fail_reason != "OK"
            ):
                sentence_norm = diag.get("sentence_norm", "")
                term_norm = diag.get("term_norm", "")
                content_ids = diag.get("content_ids", [])
                term_ids = diag.get("term_ids", [])
                valid_len = diag.get("valid_len", 0)
                sep_idx = diag.get("sep_idx", -1)

                sent_piece_tokens = diag.get("sent_tokens") or self.tokenizer.tokenize(sentence_norm or sentence)
                term_piece_tokens = diag.get("term_tokens") or self.tokenizer.tokenize(term_norm or term)

                token_match_idx = int(diag.get("token_match_idx", -1))
                if token_match_idx < 0:
                    token_match_idx = _find_subsequence(sent_piece_tokens, term_piece_tokens)
                id_match_idx = _find_subsequence(content_ids, term_ids)

                sent_piece_tokens_view = (
                    sent_piece_tokens
                    if len(sent_piece_tokens) <= 40
                    else (sent_piece_tokens[:40] + ["..."] + sent_piece_tokens[-10:])
                )
                content_ids_view = (
                    content_ids
                    if len(content_ids) <= 80
                    else (content_ids[:60] + ["..."] + content_ids[-20:])
                )
                decoded_content = self.tokenizer.decode(content_ids)[:200]
                decoded_term = self.tokenizer.decode(term_ids)

                raw_found_idx = -1
                raw_found = False
                if term_norm:
                    raw_found_idx = sentence_norm.find(term_norm)
                    raw_found = raw_found_idx >= 0

                token_found = token_match_idx >= 0
                reason = ""
                if raw_found and token_found and id_match_idx < 0:
                    reason = "ID_MAPPING_OR_SPECIAL_TOKENS"
                elif raw_found and not token_found:
                    reason = "TOKENIZATION_SPLIT_DIFF"
                elif not raw_found:
                    reason = "NORM_MISMATCH_OR_TEXT_DIFF"

                if valid_len >= self.max_len_sent and raw_found:
                    reason = reason + "+POSSIBLE_TRUNCATION" if reason else "POSSIBLE_TRUNCATION"

                def _special_chars_view(s: str) -> str:
                    out = []
                    for ch in s:
                        if not (ch.isalnum() or ch.isspace()):
                            out.append(f"{ch}->U+{ord(ch):04X}")
                    return " ".join(out)

                sentence_raw_lower = sentence.lower()
                aspect_raw_lower = term.lower()
                raw_idx_in_sentence = sentence_raw_lower.find(aspect_raw_lower)
                raw_substring = (
                    sentence[raw_idx_in_sentence : raw_idx_in_sentence + len(term)]
                    if raw_idx_in_sentence >= 0
                    else ""
                )

                token_span = ""
                if token_found:
                    token_span = sent_piece_tokens[token_match_idx : token_match_idx + len(term_piece_tokens)]

                block = [
                    f"[HAGMoE span diag] epoch={self._debug_epoch} split={self._debug_split} "
                    f"batch={self._debug_batch_idx} sample={sample_idx}",
                    f"  max_len_sent={self.max_len_sent} valid_len={valid_len} sep_idx={sep_idx}",
                    f"  sentence_raw: {sentence}",
                    f"  aspect_raw: {term}",
                    f"  sentence_norm: {sentence_norm}",
                    f"  aspect_norm: {term_norm}",
                    f"  fail_reason: {fail_reason}",
                    f"  sent_piece_tokens: {sent_piece_tokens_view}",
                    f"  term_piece_tokens: {term_piece_tokens}",
                    f"  sent_piece_ids: {content_ids_view}",
                    f"  term_piece_ids: {term_ids}",
                    f"  decoded_content_snippet: {decoded_content}",
                    f"  decoded_term: {decoded_term}",
                    f"  raw_found_substring: {raw_found} idx={raw_found_idx}",
                    f"  token_found_subsequence: {token_found} idx={token_match_idx} span={token_span}",
                    f"  id_found_subsequence: {id_match_idx}",
                    f"  aspect_raw_specials: {_special_chars_view(term)}",
                    f"  sentence_raw_specials: {_special_chars_view(raw_substring)}",
                    f"  suggested_reason: {reason}",
                ]
                print("\n".join(block))

        return {
            "input_ids_sent": sent_enc["input_ids"].squeeze(0),
            "attention_mask_sent": sent_enc["attention_mask"].squeeze(0),
            "input_ids_term": term_enc["input_ids"].squeeze(0),
            "attention_mask_term": term_enc["attention_mask"].squeeze(0),
            "aspect_start": torch.tensor(aspect_start, dtype=torch.long),
            "aspect_end": torch.tensor(aspect_end, dtype=torch.long),
            "aspect_mask_sent": aspect_mask_sent,
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

        self._debug_span_prints = 0
        self._debug_span_limit = 0
        self._debug_epoch = None
        self._debug_split = ""
        self._debug_batch_idx = 0
        self._debug_seen = 0

        self.total_samples = 0
        self.matched_samples = 0
        self.matched_mask_sum = 0.0
        self.token_mismatch_count = 0
        self.truncated_count = 0
        self.not_found_raw_count = 0

    def __len__(self) -> int:
        return len(self._indices)

    def begin_debug(self, *, epoch: int, split: str, batch_idx: int = 0, max_samples: int = 5) -> None:
        self._debug_span_prints = 0
        self._debug_span_limit = int(max_samples)
        self._debug_epoch = int(epoch)
        self._debug_split = str(split)
        self._debug_batch_idx = int(batch_idx)
        self._debug_seen = 0

    def reset_match_stats(self) -> None:
        self.total_samples = 0
        self.matched_samples = 0
        self.matched_mask_sum = 0.0
        self.token_mismatch_count = 0
        self.truncated_count = 0
        self.not_found_raw_count = 0

    def update_match_stats(self, *, total: int, matched: int, matched_mask_sum: float) -> None:
        self.total_samples += int(total)
        self.matched_samples += int(matched)
        self.matched_mask_sum += float(matched_mask_sum)

    @property
    def match_rate(self) -> float:
        if self.total_samples <= 0:
            return 0.0
        return float(self.matched_samples) / float(self.total_samples)

    def get_match_stats(self) -> Dict[str, float]:
        avg_mask = (
            float(self.matched_mask_sum) / float(self.matched_samples)
            if self.matched_samples > 0
            else 0.0
        )
        return {"match_rate": self.match_rate, "avg_mask_sum": avg_mask}

    def get_diag_stats(self) -> Dict[str, float]:
        return {
            "total": float(self.total_samples),
            "matched": float(self.matched_samples),
            "match_rate": self.match_rate,
            "token_mismatch": float(self.token_mismatch_count),
            "truncated": float(self.truncated_count),
            "not_found_raw": float(self.not_found_raw_count),
        }

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

        (
            aspect_start,
            aspect_end,
            aspect_mask_sent,
            matched_tokens,
            fail_reason,
            matched,
            diag,
        ) = _compute_aspect_span(
            tokenizer=self.tokenizer,
            sent_enc=sent_enc,
            term=term,
            sentence=sentence,
            max_len_sent=self.max_len_sent,
            max_len_term=self.max_len_term,
        )

        self.total_samples += 1
        if matched:
            self.matched_samples += 1
            self.matched_mask_sum += float(aspect_mask_sent.sum().item())
        else:
            if fail_reason == "TOKEN_MISMATCH":
                self.token_mismatch_count += 1
            elif fail_reason == "TRUNCATED":
                self.truncated_count += 1
            elif fail_reason == "NOT_FOUND_RAW":
                self.not_found_raw_count += 1

        if self._debug_epoch is not None:
            sample_idx = self._debug_seen
            self._debug_seen += 1

            if (
                self._debug_epoch == 0
                and self._debug_split in {"val", "test"}
                and self._debug_batch_idx == 0
                and sample_idx < 10
                and fail_reason != "OK"
            ):
                sentence_norm = diag.get("sentence_norm", "")
                term_norm = diag.get("term_norm", "")
                content_ids = diag.get("content_ids", [])
                term_ids = diag.get("term_ids", [])
                valid_len = diag.get("valid_len", 0)
                sep_idx = diag.get("sep_idx", -1)

                sent_piece_tokens = diag.get("sent_tokens") or self.tokenizer.tokenize(sentence_norm or sentence)
                term_piece_tokens = diag.get("term_tokens") or self.tokenizer.tokenize(term_norm or term)

                token_match_idx = int(diag.get("token_match_idx", -1))
                if token_match_idx < 0:
                    token_match_idx = _find_subsequence(sent_piece_tokens, term_piece_tokens)
                id_match_idx = _find_subsequence(content_ids, term_ids)

                sent_piece_tokens_view = (
                    sent_piece_tokens
                    if len(sent_piece_tokens) <= 40
                    else (sent_piece_tokens[:40] + ["..."] + sent_piece_tokens[-10:])
                )
                content_ids_view = (
                    content_ids
                    if len(content_ids) <= 80
                    else (content_ids[:60] + ["..."] + content_ids[-20:])
                )
                decoded_content = self.tokenizer.decode(content_ids)[:200]
                decoded_term = self.tokenizer.decode(term_ids)

                raw_found_idx = -1
                raw_found = False
                if term_norm:
                    raw_found_idx = sentence_norm.find(term_norm)
                    raw_found = raw_found_idx >= 0

                token_found = token_match_idx >= 0
                reason = ""
                if raw_found and token_found and id_match_idx < 0:
                    reason = "ID_MAPPING_OR_SPECIAL_TOKENS"
                elif raw_found and not token_found:
                    reason = "TOKENIZATION_SPLIT_DIFF"
                elif not raw_found:
                    reason = "NORM_MISMATCH_OR_TEXT_DIFF"

                if valid_len >= self.max_len_sent and raw_found:
                    reason = reason + "+POSSIBLE_TRUNCATION" if reason else "POSSIBLE_TRUNCATION"

                def _special_chars_view(s: str) -> str:
                    out = []
                    for ch in s:
                        if not (ch.isalnum() or ch.isspace()):
                            out.append(f"{ch}->U+{ord(ch):04X}")
                    return " ".join(out)

                sentence_raw_lower = sentence.lower()
                aspect_raw_lower = term.lower()
                raw_idx_in_sentence = sentence_raw_lower.find(aspect_raw_lower)
                raw_substring = (
                    sentence[raw_idx_in_sentence : raw_idx_in_sentence + len(term)]
                    if raw_idx_in_sentence >= 0
                    else ""
                )

                token_span = ""
                if token_found:
                    token_span = sent_piece_tokens[token_match_idx : token_match_idx + len(term_piece_tokens)]

                block = [
                    f"[HAGMoE span diag] epoch={self._debug_epoch} split={self._debug_split} "
                    f"batch={self._debug_batch_idx} sample={sample_idx}",
                    f"  max_len_sent={self.max_len_sent} valid_len={valid_len} sep_idx={sep_idx}",
                    f"  sentence_raw: {sentence}",
                    f"  aspect_raw: {term}",
                    f"  sentence_norm: {sentence_norm}",
                    f"  aspect_norm: {term_norm}",
                    f"  fail_reason: {fail_reason}",
                    f"  sent_piece_tokens: {sent_piece_tokens_view}",
                    f"  term_piece_tokens: {term_piece_tokens}",
                    f"  sent_piece_ids: {content_ids_view}",
                    f"  term_piece_ids: {term_ids}",
                    f"  decoded_content_snippet: {decoded_content}",
                    f"  decoded_term: {decoded_term}",
                    f"  raw_found_substring: {raw_found} idx={raw_found_idx}",
                    f"  token_found_subsequence: {token_found} idx={token_match_idx} span={token_span}",
                    f"  id_found_subsequence: {id_match_idx}",
                    f"  aspect_raw_specials: {_special_chars_view(term)}",
                    f"  sentence_raw_specials: {_special_chars_view(raw_substring)}",
                    f"  suggested_reason: {reason}",
                ]
                print("\n".join(block))

        return {
            "input_ids_sent": sent_enc["input_ids"].squeeze(0),
            "attention_mask_sent": sent_enc["attention_mask"].squeeze(0),
            "input_ids_term": term_enc["input_ids"].squeeze(0),
            "attention_mask_term": term_enc["attention_mask"].squeeze(0),
            "aspect_start": torch.tensor(aspect_start, dtype=torch.long),
            "aspect_end": torch.tensor(aspect_end, dtype=torch.long),
            "aspect_mask_sent": aspect_mask_sent,
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

        self._debug_span_prints = 0
        self._debug_span_limit = 0
        self._debug_epoch = None
        self._debug_split = ""
        self._debug_batch_idx = 0
        self._debug_seen = 0

        self.total_samples = 0
        self.matched_samples = 0
        self.matched_mask_sum = 0.0
        self.token_mismatch_count = 0
        self.truncated_count = 0
        self.not_found_raw_count = 0

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

    def begin_debug(self, *, epoch: int, split: str, batch_idx: int = 0, max_samples: int = 5) -> None:
        self._debug_span_prints = 0
        self._debug_span_limit = int(max_samples)
        self._debug_epoch = int(epoch)
        self._debug_split = str(split)
        self._debug_batch_idx = int(batch_idx)
        self._debug_seen = 0

    def reset_match_stats(self) -> None:
        self.total_samples = 0
        self.matched_samples = 0
        self.matched_mask_sum = 0.0
        self.token_mismatch_count = 0
        self.truncated_count = 0
        self.not_found_raw_count = 0

    def update_match_stats(self, *, total: int, matched: int, matched_mask_sum: float) -> None:
        self.total_samples += int(total)
        self.matched_samples += int(matched)
        self.matched_mask_sum += float(matched_mask_sum)

    @property
    def match_rate(self) -> float:
        if self.total_samples <= 0:
            return 0.0
        return float(self.matched_samples) / float(self.total_samples)

    def get_match_stats(self) -> Dict[str, float]:
        avg_mask = (
            float(self.matched_mask_sum) / float(self.matched_samples)
            if self.matched_samples > 0
            else 0.0
        )
        return {"match_rate": self.match_rate, "avg_mask_sum": avg_mask}

    def get_diag_stats(self) -> Dict[str, float]:
        return {
            "total": float(self.total_samples),
            "matched": float(self.matched_samples),
            "match_rate": self.match_rate,
            "token_mismatch": float(self.token_mismatch_count),
            "truncated": float(self.truncated_count),
            "not_found_raw": float(self.not_found_raw_count),
        }

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

        (
            aspect_start,
            aspect_end,
            aspect_mask_sent,
            matched_tokens,
            fail_reason,
            matched,
            diag,
        ) = _compute_aspect_span(
            tokenizer=self.tokenizer,
            sent_enc=sent_enc,
            term=term,
            sentence=sentence,
            max_len_sent=self.max_len_sent,
            max_len_term=self.max_len_term,
        )

        self.total_samples += 1
        if matched:
            self.matched_samples += 1
            self.matched_mask_sum += float(aspect_mask_sent.sum().item())
        else:
            if fail_reason == "TOKEN_MISMATCH":
                self.token_mismatch_count += 1
            elif fail_reason == "TRUNCATED":
                self.truncated_count += 1
            elif fail_reason == "NOT_FOUND_RAW":
                self.not_found_raw_count += 1

        if self._debug_epoch is not None:
            sample_idx = self._debug_seen
            self._debug_seen += 1

            if (
                self._debug_epoch == 0
                and self._debug_split in {"val", "test"}
                and self._debug_batch_idx == 0
                and sample_idx < 10
                and fail_reason != "OK"
            ):
                sentence_norm = diag.get("sentence_norm", "")
                term_norm = diag.get("term_norm", "")
                content_ids = diag.get("content_ids", [])
                term_ids = diag.get("term_ids", [])
                valid_len = diag.get("valid_len", 0)
                sep_idx = diag.get("sep_idx", -1)

                sent_piece_tokens = diag.get("sent_tokens") or self.tokenizer.tokenize(sentence_norm or sentence)
                term_piece_tokens = diag.get("term_tokens") or self.tokenizer.tokenize(term_norm or term)

                token_match_idx = int(diag.get("token_match_idx", -1))
                if token_match_idx < 0:
                    token_match_idx = _find_subsequence(sent_piece_tokens, term_piece_tokens)
                id_match_idx = _find_subsequence(content_ids, term_ids)

                sent_piece_tokens_view = (
                    sent_piece_tokens
                    if len(sent_piece_tokens) <= 40
                    else (sent_piece_tokens[:40] + ["..."] + sent_piece_tokens[-10:])
                )
                content_ids_view = (
                    content_ids
                    if len(content_ids) <= 80
                    else (content_ids[:60] + ["..."] + content_ids[-20:])
                )
                decoded_content = self.tokenizer.decode(content_ids)[:200]
                decoded_term = self.tokenizer.decode(term_ids)

                raw_found_idx = -1
                raw_found = False
                if term_norm:
                    raw_found_idx = sentence_norm.find(term_norm)
                    raw_found = raw_found_idx >= 0

                token_found = token_match_idx >= 0
                reason = ""
                if raw_found and token_found and id_match_idx < 0:
                    reason = "ID_MAPPING_OR_SPECIAL_TOKENS"
                elif raw_found and not token_found:
                    reason = "TOKENIZATION_SPLIT_DIFF"
                elif not raw_found:
                    reason = "NORM_MISMATCH_OR_TEXT_DIFF"

                if valid_len >= self.max_len_sent and raw_found:
                    reason = reason + "+POSSIBLE_TRUNCATION" if reason else "POSSIBLE_TRUNCATION"

                def _special_chars_view(s: str) -> str:
                    out = []
                    for ch in s:
                        if not (ch.isalnum() or ch.isspace()):
                            out.append(f"{ch}->U+{ord(ch):04X}")
                    return " ".join(out)

                sentence_raw_lower = sentence.lower()
                aspect_raw_lower = term.lower()
                raw_idx_in_sentence = sentence_raw_lower.find(aspect_raw_lower)
                raw_substring = (
                    sentence[raw_idx_in_sentence : raw_idx_in_sentence + len(term)]
                    if raw_idx_in_sentence >= 0
                    else ""
                )

                token_span = ""
                if token_found:
                    token_span = sent_piece_tokens[token_match_idx : token_match_idx + len(term_piece_tokens)]

                block = [
                    f"[HAGMoE span diag] epoch={self._debug_epoch} split={self._debug_split} "
                    f"batch={self._debug_batch_idx} sample={sample_idx}",
                    f"  max_len_sent={self.max_len_sent} valid_len={valid_len} sep_idx={sep_idx}",
                    f"  sentence_raw: {sentence}",
                    f"  aspect_raw: {term}",
                    f"  sentence_norm: {sentence_norm}",
                    f"  aspect_norm: {term_norm}",
                    f"  fail_reason: {fail_reason}",
                    f"  sent_piece_tokens: {sent_piece_tokens_view}",
                    f"  term_piece_tokens: {term_piece_tokens}",
                    f"  sent_piece_ids: {content_ids_view}",
                    f"  term_piece_ids: {term_ids}",
                    f"  decoded_content_snippet: {decoded_content}",
                    f"  decoded_term: {decoded_term}",
                    f"  raw_found_substring: {raw_found} idx={raw_found_idx}",
                    f"  token_found_subsequence: {token_found} idx={token_match_idx} span={token_span}",
                    f"  id_found_subsequence: {id_match_idx}",
                    f"  aspect_raw_specials: {_special_chars_view(term)}",
                    f"  sentence_raw_specials: {_special_chars_view(raw_substring)}",
                    f"  suggested_reason: {reason}",
                ]
                print("\n".join(block))

        return {
            "input_ids_sent": sent_enc["input_ids"].squeeze(0),
            "attention_mask_sent": sent_enc["attention_mask"].squeeze(0),
            "input_ids_term": term_enc["input_ids"].squeeze(0),
            "attention_mask_term": term_enc["attention_mask"].squeeze(0),
            "aspect_start": torch.tensor(aspect_start, dtype=torch.long),
            "aspect_end": torch.tensor(aspect_end, dtype=torch.long),
            "aspect_mask_sent": aspect_mask_sent,
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
