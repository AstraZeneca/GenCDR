"""Shared helpers for region-wise log-likelihood scoring of antibody sequences.

These functions cover:

- abnumber-based FR/CDR segmentation with sequence reconciliation,
- input normalization,
- Spearman/Pearson correlation over paired pandas Series,
- region (full/FR/CDR) log-likelihood aggregation from per-position loss.

``segment_sequence_by_scheme`` requires the optional ``abnumber`` dependency
(install with the ``scoring`` extra: ``pip install gencdr[scoring]``); it is
imported lazily so the rest of the package works without it.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch


def segment_sequence_by_scheme(sequence: str, scheme: str) -> Optional[Dict[str, str]]:
    """Split a sequence into FR/CDR segments via abnumber, reconciled to the input.

    Returns a dict with keys fr1, cdr1, fr2, cdr2, fr3, cdr3, fr4 whose concatenation
    equals ``sequence``, or None if the sequence cannot be annotated/reconciled.

    Requires the optional ``abnumber`` dependency (``pip install gencdr[scoring]``).
    """
    try:
        from abnumber import Chain
    except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
        raise ImportError(
            "segment_sequence_by_scheme requires the optional 'abnumber' dependency. "
            "Install it with: pip install gencdr[scoring]"
        ) from exc

    try:
        ch = Chain(sequence, scheme=scheme)
    except Exception:
        return None

    seg = {
        "fr1": ch.fr1_seq or "",
        "cdr1": ch.cdr1_seq or "",
        "fr2": ch.fr2_seq or "",
        "cdr2": ch.cdr2_seq or "",
        "fr3": ch.fr3_seq or "",
        "cdr3": ch.cdr3_seq or "",
        "fr4": ch.fr4_seq or "",
    }

    recon = seg["fr1"] + seg["cdr1"] + seg["fr2"] + seg["cdr2"] + seg["fr3"] + seg["cdr3"] + seg["fr4"]
    if recon == sequence:
        return seg

    # Recover extra prefix/suffix (folded into fr1/fr4) when abnumber trims residues.
    pos = sequence.find(recon)
    if pos == -1:
        return None
    seg["fr1"] = sequence[:pos] + seg["fr1"]
    seg["fr4"] = seg["fr4"] + sequence[pos + len(recon) :]
    recon2 = seg["fr1"] + seg["cdr1"] + seg["fr2"] + seg["cdr2"] + seg["fr3"] + seg["cdr3"] + seg["fr4"]
    return seg if recon2 == sequence else None


def normalize_sequence(value: object) -> str:
    """Coerce a value to an uppercased, stripped sequence string."""
    return str(value or "").strip().upper()


def spearman_corr(x: pd.Series, y: pd.Series) -> Optional[float]:
    """Spearman correlation via rank-transform; None if fewer than 2 paired points."""
    paired = pd.concat([x, y], axis=1).dropna()
    if len(paired) < 2:
        return None
    corr = paired.iloc[:, 0].rank().corr(paired.iloc[:, 1].rank())
    return None if pd.isna(corr) else float(corr)


def pearson_corr(x: pd.Series, y: pd.Series) -> Optional[float]:
    """Pearson correlation; None if fewer than 2 paired points."""
    paired = pd.concat([x, y], axis=1).dropna()
    if len(paired) < 2:
        return None
    corr = paired.iloc[:, 0].corr(paired.iloc[:, 1])
    return None if pd.isna(corr) else float(corr)


def compute_region_ll_from_loss(
    per_pos_loss: torch.Tensor,
    aa_mask: torch.Tensor,
    segment_lengths_list: List[List[int]],
    offset: int = 1,
) -> Tuple[List[float], List[float], List[float]]:
    """Compute (ll_full, ll_fr, ll_cdr) from per-position loss ``[B, L-1]`` and an aa mask.

    ``offset`` is the shift index of the first amino-acid target:
    1 for models whose shifted labels begin with a control token,
    0 for models whose shifted labels begin directly with aa0.
    """
    B = per_pos_loss.size(0)
    l_minus1 = per_pos_loss.size(1)
    device = per_pos_loss.device

    fr_mask = torch.zeros(B, l_minus1, dtype=torch.bool, device=device)
    cdr_mask = torch.zeros(B, l_minus1, dtype=torch.bool, device=device)

    for b in range(B):
        fr1_len, cdr1_len, fr2_len, cdr2_len, fr3_len, cdr3_len, fr4_len = segment_lengths_list[b]
        cursor = offset

        fr_mask[b, cursor : cursor + fr1_len] = True
        cursor += fr1_len
        cdr_mask[b, cursor : cursor + cdr1_len] = True
        cursor += cdr1_len
        fr_mask[b, cursor : cursor + fr2_len] = True
        cursor += fr2_len
        cdr_mask[b, cursor : cursor + cdr2_len] = True
        cursor += cdr2_len
        fr_mask[b, cursor : cursor + fr3_len] = True
        cursor += fr3_len
        cdr_mask[b, cursor : cursor + cdr3_len] = True
        cursor += cdr3_len
        fr_mask[b, cursor : cursor + fr4_len] = True

    fr_mask = fr_mask & aa_mask
    cdr_mask = cdr_mask & aa_mask

    out_full: List[float] = []
    out_fr: List[float] = []
    out_cdr: List[float] = []

    for b in range(B):
        full_count = aa_mask[b].sum().item()
        fr_count = fr_mask[b].sum().item()
        cdr_count = cdr_mask[b].sum().item()

        ll_full = -(per_pos_loss[b] * aa_mask[b].float()).sum().item() / max(full_count, 1)
        ll_fr = -(per_pos_loss[b] * fr_mask[b].float()).sum().item() / max(fr_count, 1)
        ll_cdr = -(per_pos_loss[b] * cdr_mask[b].float()).sum().item() / max(cdr_count, 1)

        out_full.append(ll_full)
        out_fr.append(ll_fr)
        out_cdr.append(ll_cdr)

    return out_full, out_fr, out_cdr
