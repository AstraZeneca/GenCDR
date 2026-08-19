"""Generate CDRs from framework regions with a GenCDR model and write results to CSV.

Two entry points, used by the ``gencdr`` CLI:
- ``generate_single``: single-chain (IgGenCDR / NanoGenCDR) CDR generation
- ``generate_paired``: joint heavy+light (p-IgGenCDR) CDR generation

Framework inputs may come from a JSON file (one framework set) or a CSV file (one
framework set per row). Outputs are written as CSV — no parquet dependency.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from gencdr.generator import GenCDRGenerator

FR_KEYS = ("fr1", "fr2", "fr3", "fr4")


def _full_sequence(fr: Dict[str, str], cdr1: Optional[str], cdr2: Optional[str], cdr3: Optional[str]) -> Optional[str]:
    """Assemble the full chain sequence from frameworks + CDRs, or None if any CDR is missing."""
    if cdr1 is None or cdr2 is None or cdr3 is None:
        return None
    return fr["fr1"] + cdr1 + fr["fr2"] + cdr2 + fr["fr3"] + cdr3 + fr["fr4"]


def _load_single_inputs(
    frameworks_json: Optional[Path],
    in_csv: Optional[Path],
    chain_type: str,
) -> List[Dict[str, Any]]:
    """Return a list of {id, chain_type, frameworks, cdr1, cdr2} input records for single-chain generation."""
    if (frameworks_json is None) == (in_csv is None):
        raise ValueError("Provide exactly one of frameworks_json or in_csv.")

    records: List[Dict[str, Any]] = []
    if frameworks_json is not None:
        data = json.loads(Path(frameworks_json).read_text())
        missing = [k for k in FR_KEYS if k not in data]
        if missing:
            raise ValueError(f"Frameworks JSON missing keys: {missing}")
        records.append(
            {
                "id": str(data.get("id", "seq0")),
                "chain_type": str(data.get("chain_type", chain_type)),
                "frameworks": {k: str(data[k]) for k in FR_KEYS},
                "cdr1": (str(data["cdr1"]) if data.get("cdr1") else None),
                "cdr2": (str(data["cdr2"]) if data.get("cdr2") else None),
            }
        )
        return records

    df = pd.read_csv(in_csv)
    missing = [k for k in FR_KEYS if k not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing framework columns: {missing}")
    for i, row in df.iterrows():
        records.append(
            {
                "id": str(row["id"]) if "id" in df.columns and pd.notna(row.get("id")) else f"seq{i}",
                "chain_type": str(row["chain_type"]) if "chain_type" in df.columns else chain_type,
                "frameworks": {k: str(row[k]) for k in FR_KEYS},
                "cdr1": str(row["cdr1"]) if "cdr1" in df.columns and pd.notna(row.get("cdr1")) else None,
                "cdr2": str(row["cdr2"]) if "cdr2" in df.columns and pd.notna(row.get("cdr2")) else None,
            }
        )
    return records


def generate_single(
    model_dir: str,
    out_csv: Path,
    chain_type: str,
    n_samples: int,
    frameworks_json: Optional[Path] = None,
    in_csv: Optional[Path] = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 256,
    scheme: Optional[str] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    fp: int = 16,
    include_scheme_token: bool = False,
    cdr1_temperature: Optional[float] = None,
    cdr2_temperature: Optional[float] = None,
    cdr3_temperature: Optional[float] = None,
) -> Path:
    """Generate single-chain CDRs for each framework set and write a CSV of results."""
    inputs = _load_single_inputs(frameworks_json, in_csv, chain_type)
    gen = GenCDRGenerator(model_dir=model_dir, device=device, fp=fp, include_scheme_token=include_scheme_token)

    rows: List[Dict[str, Any]] = []
    for rec in inputs:
        results = gen.generate_cdrs_from_frameworks(
            chain_type=rec["chain_type"],
            frameworks=rec["frameworks"],
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            known_cdr1=rec["cdr1"],
            known_cdr2=rec["cdr2"],
            scheme=scheme,
            seed=seed,
            cdr1_temperature=cdr1_temperature,
            cdr2_temperature=cdr2_temperature,
            cdr3_temperature=cdr3_temperature,
        )
        fr = rec["frameworks"]
        for sample_idx, res in enumerate(results):
            rows.append(
                {
                    "id": rec["id"],
                    "chain_type": rec["chain_type"],
                    "sample_idx": sample_idx,
                    "fr1": fr["fr1"],
                    "fr2": fr["fr2"],
                    "fr3": fr["fr3"],
                    "fr4": fr["fr4"],
                    "cdr1": res.get("cdr1"),
                    "cdr2": res.get("cdr2"),
                    "cdr3": res.get("cdr3"),
                    "full_sequence": _full_sequence(fr, res.get("cdr1"), res.get("cdr2"), res.get("cdr3")),
                    "parsed_ok": bool(res.get("parsed_ok", False)),
                    "sep_count": int(res.get("sep_count", 0)),
                    "issues": ",".join(res.get("issues", [])),
                }
            )

    out_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    n_ok = int(out_df["parsed_ok"].sum()) if len(out_df) else 0
    print(f"Generated {len(out_df)} sequences ({n_ok} parsed_ok) from {len(inputs)} framework set(s) -> {out_csv}")
    return out_csv


def _load_paired_inputs(
    frameworks_json: Optional[Path],
    in_csv: Optional[Path],
) -> List[Dict[str, Any]]:
    """Return a list of {id, h_frameworks, l_frameworks} records for paired generation."""
    if (frameworks_json is None) == (in_csv is None):
        raise ValueError("Provide exactly one of frameworks_json or in_csv.")

    records: List[Dict[str, Any]] = []
    if frameworks_json is not None:
        data = json.loads(Path(frameworks_json).read_text())
        if "H" not in data or "L" not in data:
            raise ValueError("Paired frameworks JSON must contain 'H' and 'L' objects.")
        for chain in ("H", "L"):
            miss = [k for k in FR_KEYS if k not in data[chain]]
            if miss:
                raise ValueError(f"Paired frameworks JSON chain {chain} missing keys: {miss}")
        records.append(
            {
                "id": str(data.get("id", "pair0")),
                "h_frameworks": {k: str(data["H"][k]) for k in FR_KEYS},
                "l_frameworks": {k: str(data["L"][k]) for k in FR_KEYS},
            }
        )
        return records

    df = pd.read_csv(in_csv)
    h_cols = [f"h_{k}" for k in FR_KEYS]
    l_cols = [f"l_{k}" for k in FR_KEYS]
    missing = [c for c in h_cols + l_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing paired framework columns: {missing}")
    for i, row in df.iterrows():
        records.append(
            {
                "id": str(row["id"]) if "id" in df.columns and pd.notna(row.get("id")) else f"pair{i}",
                "h_frameworks": {k: str(row[f"h_{k}"]) for k in FR_KEYS},
                "l_frameworks": {k: str(row[f"l_{k}"]) for k in FR_KEYS},
            }
        )
    return records


def generate_paired(
    model_dir: str,
    out_csv: Path,
    n_samples: int,
    frameworks_json: Optional[Path] = None,
    in_csv: Optional[Path] = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 256,
    order: str = "L-first",
    scheme: Optional[str] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    fp: int = 16,
    include_scheme_token: bool = False,
) -> Path:
    """Jointly generate paired H+L CDRs for each framework set and write a CSV of results."""
    inputs = _load_paired_inputs(frameworks_json, in_csv)
    gen = GenCDRGenerator(model_dir=model_dir, device=device, fp=fp, include_scheme_token=include_scheme_token)

    rows: List[Dict[str, Any]] = []
    for rec in inputs:
        results = gen.generate_paired_cdrs_from_frameworks(
            h_frameworks=rec["h_frameworks"],
            l_frameworks=rec["l_frameworks"],
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            order=order,
            scheme=scheme,
            seed=seed,
        )
        hf = rec["h_frameworks"]
        lf = rec["l_frameworks"]
        for sample_idx, res in enumerate(results):
            h = res["H"]
            light = res["L"]
            rows.append(
                {
                    "id": rec["id"],
                    "sample_idx": sample_idx,
                    "order": res.get("order", order),
                    "h_cdr1": h.get("cdr1"),
                    "h_cdr2": h.get("cdr2"),
                    "h_cdr3": h.get("cdr3"),
                    "l_cdr1": light.get("cdr1"),
                    "l_cdr2": light.get("cdr2"),
                    "l_cdr3": light.get("cdr3"),
                    "h_full_sequence": _full_sequence(hf, h.get("cdr1"), h.get("cdr2"), h.get("cdr3")),
                    "l_full_sequence": _full_sequence(lf, light.get("cdr1"), light.get("cdr2"), light.get("cdr3")),
                    "parsed_ok": bool(res.get("parsed_ok", False)),
                    "issues": ",".join(res.get("issues", [])),
                }
            )

    out_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    n_ok = int(out_df["parsed_ok"].sum()) if len(out_df) else 0
    print(f"Generated {len(out_df)} paired samples ({n_ok} parsed_ok) from {len(inputs)} framework set(s) -> {out_csv}")
    return out_csv
