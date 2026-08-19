"""Reward dataset and collator for weighted DPO alignment.

Reads a reward table (one row per generated sample) and renders each row into the
framework-first layout GenCDR was trained on, exposing the CDR completion span so
the trainer can compute the completion-only log-ratio.

Two modes:

- ``"single"`` — one chain (IgGenCDR / NanoGenCDR). Required columns:
  ``meta_framework_source``, ``meta_fr1``..``meta_fr4``, ``meta_cdr1``..``meta_cdr3``,
  ``reward``. Optional ``meta_chain_type`` ("H" default / "L") and ``meta_scheme``.

- ``"paired"`` — cognate heavy+light (p-IgGenCDR). Required columns:
  ``meta_framework_source``, ``meta_h_fr1``..``meta_h_fr4``, ``meta_l_fr1``..``meta_l_fr4``,
  ``meta_h_cdr1``..``meta_h_cdr3``, ``meta_l_cdr1``..``meta_l_cdr3``, ``reward``.
  Optional ``meta_scheme``.

The completion span is ``(first <HPRED>|<LPRED>) + 1`` through the last ``<EOS>``
(inclusive), which is a single contiguous block for both single and paired layouts
(for paired it spans both chains' CDR blocks, including the second pred tag). Paired
alignment generalises the manuscript's single-chain method and is not itself part of
the published experiments.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from gencdr.rendering import render_paired, render_single
from gencdr.tokenizer import TOK_EOS, TOK_HPRED, TOK_LPRED

FR_KEYS = ("fr1", "fr2", "fr3", "fr4")
CDR_KEYS = ("cdr1", "cdr2", "cdr3")


class FrameworkCDRRewardDataset(Dataset):
    """Reward-labelled framework/CDR samples for weighted DPO alignment."""

    def __init__(self, df: pd.DataFrame, mode: str = "single", group_by: str = "source") -> None:
        """Validate columns and normalise dtypes for the requested mode.

        Parameters
        ----------
        df : pandas.DataFrame
            Reward table (see module docstring for required columns).
        mode : str
            ``"single"`` or ``"paired"``.
        group_by : str
            Scaffold grouping key: ``"source"`` (``meta_framework_source``) or
            ``"prompt"`` (source plus the framework sequences).
        """
        super().__init__()
        mode = str(mode).lower()
        if mode not in {"single", "paired"}:
            raise ValueError("mode must be one of {'single', 'paired'}")
        group_by = str(group_by).lower()
        if group_by not in {"source", "prompt"}:
            raise ValueError("group_by must be one of {'source', 'prompt'}")

        if mode == "single":
            req = (
                ["meta_framework_source"]
                + [f"meta_{k}" for k in FR_KEYS]
                + [f"meta_{k}" for k in CDR_KEYS]
                + ["reward"]
            )
        else:
            req = (
                ["meta_framework_source"]
                + [f"meta_h_{k}" for k in FR_KEYS]
                + [f"meta_l_{k}" for k in FR_KEYS]
                + [f"meta_h_{k}" for k in CDR_KEYS]
                + [f"meta_l_{k}" for k in CDR_KEYS]
                + ["reward"]
            )
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns for mode='{mode}': {missing}")

        self.mode = mode
        self.group_by = group_by
        self.df = df.fillna("").reset_index(drop=True)
        str_cols = [c for c in req if c != "reward"]
        self.df[str_cols] = self.df[str_cols].astype(str)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.df)

    def _segments(self, row: pd.Series, prefix: str) -> Dict[str, str]:
        """Collect the seven FR/CDR segment strings for one chain from a row."""
        return {
            **{k: row[f"meta_{prefix}{k}"] for k in FR_KEYS},
            **{k: row[f"meta_{prefix}{k}"] for k in CDR_KEYS},
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one rendered-ready record (group label, segments, reward)."""
        row = self.df.iloc[idx]
        source = row["meta_framework_source"]
        scheme = row["meta_scheme"] if "meta_scheme" in self.df.columns and row["meta_scheme"] else None

        if self.mode == "single":
            segs = self._segments(row, prefix="")
            chain_type = (
                row["meta_chain_type"] if "meta_chain_type" in self.df.columns and row["meta_chain_type"] else "H"
            )
            if self.group_by == "prompt":
                group = "::".join([source, *(segs[k] for k in FR_KEYS)])
            else:
                group = source
            return {
                "group": group,
                "chain_type": chain_type,
                "segments": segs,
                "scheme": scheme,
                "reward": float(row["reward"]),
            }

        h_segs = self._segments(row, prefix="h_")
        l_segs = self._segments(row, prefix="l_")
        if self.group_by == "prompt":
            fr_key = "::".join([*(h_segs[k] for k in FR_KEYS), *(l_segs[k] for k in FR_KEYS)])
            group = f"{source}::{fr_key}"
        else:
            group = source
        return {
            "group": group,
            "h_segments": h_segs,
            "l_segments": l_segs,
            "scheme": scheme,
            "reward": float(row["reward"]),
        }


@dataclass
class FrameworkCDRRewardCollator:
    """Render + tokenise a batch and extract per-sequence CDR completion spans."""

    tokenizer: PreTrainedTokenizerFast
    mode: str = "single"
    order: str = "L-first"
    include_scheme_token: bool = False
    max_length: Optional[int] = 256
    pad_to_multiple_of: int = 8
    eos_id: int = field(init=False)
    hpred_id: int = field(init=False)
    lpred_id: int = field(init=False)

    def __post_init__(self) -> None:
        """Resolve special-token ids and validate the tokenizer/config."""
        self.mode = str(self.mode).lower()
        if self.mode not in {"single", "paired"}:
            raise ValueError("mode must be one of {'single', 'paired'}")
        if self.order not in {"L-first", "H-first"}:
            raise ValueError("order must be 'L-first' or 'H-first'")
        self.max_length = (
            int(self.max_length)
            if self.max_length is not None
            else int(getattr(self.tokenizer, "model_max_length", 1024))
        )

        self.eos_id = int(self.tokenizer.convert_tokens_to_ids(TOK_EOS))
        self.hpred_id = int(self.tokenizer.convert_tokens_to_ids(TOK_HPRED))
        self.lpred_id = int(self.tokenizer.convert_tokens_to_ids(TOK_LPRED))
        if self.tokenizer.pad_token_id is None:
            raise RuntimeError("Tokenizer must define pad_token_id for batching/padding.")
        if min(self.eos_id, self.hpred_id, self.lpred_id) < 0:
            raise RuntimeError("Special token ids not found in tokenizer vocab (EOS/HPRED/LPRED).")

    def _render(self, x: Dict[str, Any]) -> str:
        """Render one record into the framework-first string for tokenisation."""
        if self.mode == "single":
            sample = {"chain_type": x["chain_type"], "segments": x["segments"], "scheme": x.get("scheme")}
            return render_single(sample, include_scheme=self.include_scheme_token)

        h_sample = {"chain_type": "H", "segments": x["h_segments"], "scheme": x.get("scheme")}
        l_sample = {"chain_type": "L", "segments": x["l_segments"], "scheme": x.get("scheme")}
        return render_paired(h_sample, l_sample, order=self.order, include_scheme=self.include_scheme_token)

    def _completion_span_from_ids(self, ids: List[int], valid_len: int) -> tuple:
        """Return the unshifted (start, end-exclusive) completion span for one sequence.

        Spans from the first ``<HPRED>``/``<LPRED>`` (excluded) through the last
        ``<EOS>`` (included), i.e. the contiguous CDR completion block.
        """
        ids = ids[:valid_len]

        pred_pos = None
        for i, t in enumerate(ids):
            if t == self.hpred_id or t == self.lpred_id:
                pred_pos = i
                break
        if pred_pos is None:
            raise ValueError("No <HPRED> or <LPRED> found (cannot build completion span).")

        eos_pos = None
        for i in range(len(ids) - 1, -1, -1):
            if ids[i] == self.eos_id:
                eos_pos = i
                break
        if eos_pos is None:
            eos_pos = len(ids) - 1  # truncation removed EOS; use last available token

        start = pred_pos + 1
        end = eos_pos + 1  # include EOS
        if end <= start:
            raise ValueError(f"Invalid completion span: pred_pos={pred_pos}, eos_pos={eos_pos}, len={len(ids)}")
        return start, end

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a batch into tensors for the DPO trainer.

        Returns a dict with ``input_ids``, ``attention_mask``, ``completion_spans``
        (``[B, 2]`` unshifted), ``rewards`` and ``group_labels``.
        """
        texts = [self._render(x) for x in batch]
        rewards = [x["reward"] for x in batch]
        groups = [x["group"] for x in batch]

        enc = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_token_type_ids=False,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        num_seqs = int(input_ids.size(0))
        completion_spans = torch.full((num_seqs, 2), -1, dtype=torch.long)
        for i in range(num_seqs):
            valid_len = int(attention_mask[i].sum().item())
            start, end = self._completion_span_from_ids(input_ids[i].tolist(), valid_len=valid_len)
            completion_spans[i, 0] = start
            completion_spans[i, 1] = end

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_spans": completion_spans,
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "group_labels": groups,
        }
