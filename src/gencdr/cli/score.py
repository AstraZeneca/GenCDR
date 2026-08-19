"""Score antibody sequences from a CSV with GenCDR region log-likelihoods.

Computes, per sequence:
- full sequence AA log-likelihood (FR + CDR)
- framework-only AA log-likelihood (tokens before the prediction tag)
- CDR-only AA log-likelihood (tokens after the prediction tag)
- CDR3 / no-CDR3 split log-likelihoods

Region splitting uses abnumber with the selected numbering scheme (requires the
optional 'scoring' extra). matplotlib is imported lazily and only when plotting.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from gencdr.generator import GenCDRGenerator
from gencdr.scoring import normalize_sequence, pearson_corr, segment_sequence_by_scheme, spearman_corr


def _collect_items_for_scoring(
    records: List[Dict[str, Any]],
    sequence_column: str,
    scheme: str,
    allow_empty_cdr: bool,
    chain_type: str,
) -> Tuple[List[Dict[str, str]], List[int], Dict[int, str]]:
    """Build valid items for scoring and notes for skipped rows."""
    items: List[Dict[str, str]] = []
    valid_row_indices: List[int] = []
    notes: Dict[int, str] = {}

    for row_idx, row in enumerate(records):
        seq = normalize_sequence(row.get(sequence_column, ""))
        if not seq:
            notes[row_idx] = "empty_sequence"
            continue

        segs = segment_sequence_by_scheme(seq, scheme=scheme)
        if segs is None:
            notes[row_idx] = f"abnumber_failed_{scheme}"
            continue

        if not allow_empty_cdr and (len(segs["cdr1"]) == 0 or len(segs["cdr2"]) == 0 or len(segs["cdr3"]) == 0):
            notes[row_idx] = "empty_cdr"
            continue

        items.append({"chain_type": chain_type, **segs})
        valid_row_indices.append(row_idx)

    return items, valid_row_indices, notes


def _corr_pair(df_num: pd.DataFrame, value_col: str, target_col: str) -> Dict[str, Optional[float]]:
    """Compute Spearman/Pearson for one value-vs-target pair."""
    return {
        "spearman": spearman_corr(df_num[value_col], df_num[target_col]),
        "pearson": pearson_corr(df_num[value_col], df_num[target_col]),
    }


def _populate_score_columns(
    df: pd.DataFrame,
    valid_row_indices: List[int],
    score_by_region: Dict[str, List[float]],
    score_cdr3_split: Dict[str, List[float]],
    reduction: str,
    col_full: str,
    col_fr: str,
    col_cdr: str,
    col_no_cdr3: str,
    col_cdr3: str,
    ppl_full: str,
    ppl_fr: str,
    ppl_cdr: str,
    ppl_no_cdr3: str,
    ppl_cdr3: str,
) -> None:
    """Write region scores (and optionally perplexities) back to the dataframe."""
    for row_idx, ll_full, ll_fr, ll_cdr, ll_no_cdr3, ll_cdr3 in zip(
        valid_row_indices,
        score_by_region["full"],
        score_by_region["framework"],
        score_by_region["cdr"],
        score_cdr3_split["no_cdr3"],
        score_cdr3_split["cdr3"],
    ):
        df.at[row_idx, col_full] = float(ll_full)
        df.at[row_idx, col_fr] = float(ll_fr)
        df.at[row_idx, col_cdr] = float(ll_cdr)
        df.at[row_idx, col_no_cdr3] = float(ll_no_cdr3)
        df.at[row_idx, col_cdr3] = float(ll_cdr3)
        if reduction == "mean":
            # With ll = -mean_nll, perplexity is exp(mean_nll) = exp(-ll).
            df.at[row_idx, ppl_full] = float(math.exp(-float(ll_full)))
            df.at[row_idx, ppl_fr] = float(math.exp(-float(ll_fr)))
            df.at[row_idx, ppl_cdr] = float(math.exp(-float(ll_cdr)))
            df.at[row_idx, ppl_no_cdr3] = float(math.exp(-float(ll_no_cdr3)))
            df.at[row_idx, ppl_cdr3] = float(math.exp(-float(ll_cdr3)))


def _make_scatter_and_spearman(
    df: pd.DataFrame,
    value_col: str,
    ll_cols: Dict[str, str],
    plot_path: Path,
    spearman_json: Optional[Path],
    ppl_cols: Optional[Dict[str, str]] = None,
) -> Dict[str, Optional[float]]:
    """Generate value-vs-perplexity scatter plots and return Spearman/Pearson correlations by region."""
    # matplotlib is only needed for plotting; import lazily with a headless backend.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if value_col not in df.columns:
        raise KeyError(f"CSV does not contain value column '{value_col}'")

    df_num = df.copy()
    df_num[value_col] = pd.to_numeric(df_num[value_col], errors="coerce")
    for col in ll_cols.values():
        df_num[col] = pd.to_numeric(df_num[col], errors="coerce")
    if ppl_cols:
        for col in ppl_cols.values():
            df_num[col] = pd.to_numeric(df_num[col], errors="coerce")

    regions = ("full", "framework", "cdr")
    ll_corr = {k: _corr_pair(df_num, value_col, ll_cols[k]) for k in regions}
    spearman = {k: ll_corr[k]["spearman"] for k in regions}
    pearson = {k: ll_corr[k]["pearson"] for k in regions}

    # Plot perplexity if available, otherwise fall back to log-likelihood
    plot_cols = ppl_cols if ppl_cols else ll_cols
    y_label_prefix = "Perplexity" if ppl_cols else "Log-Likelihood"

    # Correlations on perplexity (sign-flipped relative to LL)
    if ppl_cols:
        ppl_corr = {k: _corr_pair(df_num, value_col, ppl_cols[k]) for k in regions}
        spearman_ppl = {k: ppl_corr[k]["spearman"] for k in regions}
        pearson_ppl = {k: ppl_corr[k]["pearson"] for k in regions}
    else:
        spearman_ppl = spearman
        pearson_ppl = pearson

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    region_order = [
        ("full", "Full Sequence"),
        ("framework", "Framework"),
        ("cdr", "CDR"),
    ]

    for ax, (key, title) in zip(axes, region_order):
        col = plot_cols[key]
        points = df_num[[value_col, col]].dropna()
        ax.scatter(points[value_col], points[col], alpha=0.7, s=18)
        rho = spearman_ppl[key]
        r = pearson_ppl[key]
        rho_txt = "NA" if rho is None else f"{rho:.4f}"
        r_txt = "NA" if r is None else f"{r:.4f}"
        ax.set_title(f"{title}\nSpearman={rho_txt}  Pearson={r_txt}")
        ax.set_xlabel(value_col)
        ax.set_ylabel(f"{y_label_prefix} ({key})")
        ax.grid(alpha=0.25)

    fig.suptitle(f"{value_col} vs GenCDR {y_label_prefix}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    if spearman_json is not None:
        payload = {
            "value_column": value_col,
            "ll_columns": ll_cols,
            "spearman_ll": spearman,
            "pearson_ll": pearson,
        }
        if ppl_cols:
            payload["ppl_columns"] = ppl_cols
            payload["spearman_ppl"] = spearman_ppl
            payload["pearson_ppl"] = pearson_ppl
        spearman_json.parent.mkdir(parents=True, exist_ok=True)
        with spearman_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    return spearman


def score_csv(
    csv_path: Path,
    out_csv: Path,
    model_dir: Path,
    sequence_column: str,
    scheme: str,
    chain_type: str,
    device: Optional[str],
    fp: int,
    batch_size: int,
    reduction: str,
    include_scheme_token: bool,
    allow_empty_cdr: bool,
    plot_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Score antibody sequences in a CSV with GenCDR region likelihoods."""
    if chain_type not in {"H", "L"}:
        raise ValueError("chain_type must be 'H' or 'L'")

    df = pd.read_csv(csv_path)
    if sequence_column not in df.columns:
        raise KeyError(f"CSV missing required column: {sequence_column}")

    records = df.to_dict(orient="records")
    items, valid_row_indices, notes = _collect_items_for_scoring(
        records=records,
        sequence_column=sequence_column,
        scheme=scheme,
        allow_empty_cdr=allow_empty_cdr,
        chain_type=chain_type,
    )

    generator = GenCDRGenerator(
        model_dir=str(model_dir),
        device=device,
        fp=fp,
        include_scheme_token=include_scheme_token,
    )

    score_by_region = generator.batch_log_likelihood_regions_from_segments(
        items=items,
        scheme=scheme,
        reduction=reduction,
        batch_size=batch_size,
    )

    score_cdr3_split = generator.batch_log_likelihood_cdr3_split_from_segments(
        items=items,
        scheme=scheme,
        reduction=reduction,
        batch_size=batch_size,
    )

    col_full = f"gencdr_log_likelihood_full_{reduction}"
    col_fr = f"gencdr_log_likelihood_framework_{reduction}"
    col_cdr = f"gencdr_log_likelihood_cdr_{reduction}"
    col_no_cdr3 = f"gencdr_log_likelihood_no_cdr3_{reduction}"
    col_cdr3 = f"gencdr_log_likelihood_cdr3_{reduction}"
    ppl_full = "gencdr_perplexity_full"
    ppl_fr = "gencdr_perplexity_framework"
    ppl_cdr = "gencdr_perplexity_cdr"
    ppl_no_cdr3 = "gencdr_perplexity_no_cdr3"
    ppl_cdr3 = "gencdr_perplexity_cdr3"

    df[col_full] = pd.NA
    df[col_fr] = pd.NA
    df[col_cdr] = pd.NA
    df[col_no_cdr3] = pd.NA
    df[col_cdr3] = pd.NA
    df[ppl_full] = pd.NA
    df[ppl_fr] = pd.NA
    df[ppl_cdr] = pd.NA
    df[ppl_no_cdr3] = pd.NA
    df[ppl_cdr3] = pd.NA

    _populate_score_columns(
        df=df,
        valid_row_indices=valid_row_indices,
        score_by_region=score_by_region,
        score_cdr3_split=score_cdr3_split,
        reduction=reduction,
        col_full=col_full,
        col_fr=col_fr,
        col_cdr=col_cdr,
        col_no_cdr3=col_no_cdr3,
        col_cdr3=col_cdr3,
        ppl_full=ppl_full,
        ppl_fr=ppl_fr,
        ppl_cdr=ppl_cdr,
        ppl_no_cdr3=ppl_no_cdr3,
        ppl_cdr3=ppl_cdr3,
    )

    note_col = "gencdr_scoring_note"
    if notes:
        df[note_col] = ""
        for row_idx, msg in notes.items():
            df.at[row_idx, note_col] = msg

    df["gencdr_model_dir"] = str(model_dir)
    df["gencdr_scheme"] = scheme
    df["gencdr_chain_type"] = chain_type
    df["gencdr_include_scheme_token"] = bool(include_scheme_token)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if reduction != "mean":
        print("Perplexity columns left empty because reduction!='mean' (perplexity needs mean token NLL).")

    if plot_config is not None:
        value_column = str(plot_config["value_column"])
        plot_path = plot_config.get("plot_path")
        spearman_json = plot_config.get("spearman_json")
        resolved_plot_path = Path(plot_path) if plot_path else out_csv.with_suffix(".value_vs_ll.png")
        ll_cols = {
            "full": col_full,
            "framework": col_fr,
            "cdr": col_cdr,
        }
        ppl_cols_dict = (
            {
                "full": ppl_full,
                "framework": ppl_fr,
                "cdr": ppl_cdr,
            }
            if reduction == "mean"
            else None
        )
        spearman = _make_scatter_and_spearman(
            df=df,
            value_col=value_column,
            ll_cols=ll_cols,
            plot_path=resolved_plot_path,
            spearman_json=spearman_json,
            ppl_cols=ppl_cols_dict,
        )
        print(
            "Spearman correlations "
            f"(value={value_column}): full={spearman['full']} framework={spearman['framework']} cdr={spearman['cdr']}"
        )
        print(f"Wrote plot: {resolved_plot_path}")
        if spearman_json:
            print(f"Wrote Spearman JSON: {spearman_json}")

    print(f"Scored {csv_path} | rows={len(df)} valid={len(valid_row_indices)} skipped={len(notes)} -> {out_csv}")
    return out_csv
