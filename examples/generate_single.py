#!/usr/bin/env python3
"""Generate single-chain CDRs from framework regions (IgGenCDR / NanoGenCDR).

Usage:
    python examples/generate_single.py --model /path/to/iggencdr
    python examples/generate_single.py --model nanogencdr --frameworks examples/frameworks/nanobody_vhh.json

`--model` may be a local model directory or a short name resolved against the local
checkpoint cache ($GENCDR_HOME, default ~/.cache/gencdr). Weights are not shipped with
this package.
"""

import json
from pathlib import Path
from typing import Optional

import typer
from gencdr import GenCDRGenerator

DEFAULT_FRAMEWORKS = Path(__file__).parent / "frameworks" / "heavy_vh.json"
SCHEMES = ("imgt", "kabat", "chothia")


def main(
    model: str = typer.Option(..., "--model", "-m", help="Model directory or short name (iggencdr / nanogencdr)."),
    frameworks: Path = typer.Option(DEFAULT_FRAMEWORKS, "--frameworks", help="Framework JSON file."),
    n_samples: int = typer.Option(5, "--n-samples", "-n", help="Samples to generate."),
    temperature: float = typer.Option(1.0, "--temperature", "-t", help="Sampling temperature."),
    top_p: float = typer.Option(0.95, "--top-p", help="Nucleus sampling top-p."),
    scheme: str = typer.Option("imgt", "--scheme", help="Numbering scheme: imgt|kabat|chothia."),
    seed: int = typer.Option(0, "--seed", help="RNG seed for reproducibility."),
    device: Optional[str] = typer.Option(None, "--device", help="cuda|cpu|mps (auto if unset)."),
) -> None:
    """Generate single-chain CDRs from a frameworks JSON and print the parsed samples."""
    if scheme.lower() not in SCHEMES:
        raise typer.BadParameter(f"scheme must be one of {SCHEMES}.")

    data = json.loads(frameworks.read_text())
    fw = {k: data[k] for k in ("fr1", "fr2", "fr3", "fr4")}
    chain_type = data.get("chain_type", "H")

    gen = GenCDRGenerator.from_pretrained(model, device=device)
    results = gen.generate_cdrs_from_frameworks(
        chain_type=chain_type,
        frameworks=fw,
        n_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
        scheme=scheme.lower(),
        seed=seed,
    )

    for i, res in enumerate(results):
        status = "ok" if res["parsed_ok"] else f"issues={res['issues']}"
        print(f"[{i}] CDR1={res['cdr1']}  CDR2={res['cdr2']}  CDR3={res['cdr3']}  ({status})")


if __name__ == "__main__":
    typer.run(main)
