#!/usr/bin/env python3
"""Score antibody sequences with region log-likelihoods.

This example scores a fully segmented sample directly (no abnumber needed). To score
raw sequences from a CSV, use the CLI instead, which segments with abnumber:

    gencdr score --model iggencdr --csv-path sequences.csv --sequence-column heavy_chain

Usage:
    python examples/score_sequences.py --model iggencdr

`--model` may be a local model directory or a short name resolved against the local
checkpoint cache ($GENCDR_HOME, default ~/.cache/gencdr).
"""

from typing import Optional

import typer
from gencdr import GenCDRGenerator

# A fully segmented single-chain sample (heavy). Segments concatenate to the full chain.
EXAMPLE_ITEM = {
    "chain_type": "H",
    "fr1": "QVQLVESGGGLVQPGGSLRLSCAAS",
    "cdr1": "GFTFSSYA",
    "fr2": "WVRQAPGKGLEWVS",
    "cdr2": "ISGSGGST",
    "fr3": "RFTISRDNSKNTLYLQMNSLRAEDTAVYYCAK",
    "cdr3": "DRGLYYFDY",
    "fr4": "WGQGTLVTVSS",
}


def main(
    model: str = typer.Option(..., "--model", "-m", help="Model directory or short name."),
    device: Optional[str] = typer.Option(None, "--device", help="cuda|cpu|mps (auto if unset)."),
) -> None:
    """Score the example segmented sample and print full-sequence and per-region log-likelihoods."""
    gen = GenCDRGenerator.from_pretrained(model, device=device)

    ll = gen.log_likelihood_from_segments(EXAMPLE_ITEM, reduction="mean")
    print(f"Full-sequence mean log-likelihood: {ll:.4f}")

    regions = gen.batch_log_likelihood_regions_from_segments([EXAMPLE_ITEM], reduction="mean")
    print(f"Framework LL: {regions['framework'][0]:.4f}")
    print(f"CDR LL:       {regions['cdr'][0]:.4f}")


if __name__ == "__main__":
    typer.run(main)
