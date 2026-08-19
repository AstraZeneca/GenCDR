#!/usr/bin/env python3
"""Align a GenCDR model with weighted DPO from a reward CSV.

This runs the scaffold-grouped weighted DPO objective from the GenCDR manuscript.
It needs only a reward CSV — no structure prediction or oracle code. Each row is one
generated sample described by its framework/CDR segments plus a scalar ``reward``;
rows sharing ``meta_framework_source`` form one scaffold group.

Requires the 'align' extra:

    pip install "gencdr[align]"          # or: poetry install --extras align

Usage (single-chain):

    python examples/align_cdrs.py --model iggencdr \
        --reward-csv examples/rewards/single_rewards.csv \
        --output-dir /tmp/gencdr_aligned

Equivalent packaged CLI:

    gencdr align --model iggencdr \
        --reward-csv examples/rewards/single_rewards.csv \
        --output-dir /tmp/gencdr_aligned

`--model` may be a local model directory or a short name resolved against the local
checkpoint cache ($GENCDR_HOME, default ~/.cache/gencdr). The reference policy defaults
to the model being aligned. Model weights are not shipped with this package.

Required CSV columns (single-chain):
    meta_framework_source, meta_fr1..meta_fr4, meta_cdr1..meta_cdr3, reward
    (optional: meta_chain_type [H/L], meta_scheme)

For paired alignment (p-IgGenCDR), pass --paired and provide meta_h_* / meta_l_*
columns instead. Paired alignment extends the single-chain method described in the
manuscript and is not itself part of the published experiments.
"""

from pathlib import Path
from typing import List, Optional

import typer


def main(
    model: str = typer.Option(..., "--model", "-m", help="Policy model directory or short name."),
    reward_csv: List[Path] = typer.Option(
        ..., "--reward-csv", help="Reward CSV path (repeat --reward-csv to combine several)."
    ),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Directory for the aligned model + logs."),
    ref_model: Optional[str] = typer.Option(None, "--ref-model", help="Frozen reference model (default: the policy)."),
    paired: bool = typer.Option(False, "--paired", help="Paired heavy+light alignment (p-IgGenCDR)."),
    beta: float = typer.Option(0.15, "--beta", help="DPO KL strength (manuscript default 0.15)."),
    max_epochs: int = typer.Option(5, "--max-epochs", "-e", help="Maximum training epochs."),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="Batch size."),
) -> None:
    """Run weighted DPO alignment from reward CSV(s) and report the best checkpoint."""
    # Imported here so the heavy training dependencies load only when actually aligning.
    from gencdr.alignment.aligner import WeightedDPOAligner

    aligner = WeightedDPOAligner(
        model=model,
        output_dir=str(output_dir),
        ref_model=ref_model,
        mode="paired" if paired else "single",
        beta=beta,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    best = aligner.run([str(p) for p in reward_csv])
    print(f"Aligned model saved to {output_dir}")
    print(f"Best checkpoint (by val Spearman): {best or '(none — check logs)'}")


if __name__ == "__main__":
    typer.run(main)
