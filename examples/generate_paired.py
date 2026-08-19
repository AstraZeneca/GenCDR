#!/usr/bin/env python3
"""Jointly generate paired heavy+light CDRs from both frameworks (p-IgGenCDR).

Usage:
    python examples/generate_paired.py --model p-iggencdr

`--model` may be a local model directory or a short name resolved against the local
checkpoint cache ($GENCDR_HOME, default ~/.cache/gencdr). Weights are not shipped with
this package.
"""

import json
from pathlib import Path
from typing import Optional

import typer
from gencdr import GenCDRGenerator

DEFAULT_FRAMEWORKS = Path(__file__).parent / "frameworks" / "paired_hl.json"


def main(
    model: str = typer.Option(..., "--model", "-m", help="Model directory or short name (p-iggencdr)."),
    frameworks: Path = typer.Option(DEFAULT_FRAMEWORKS, "--frameworks", help="Paired framework JSON file."),
    n_samples: int = typer.Option(5, "--n-samples", "-n", help="Samples to generate."),
    temperature: float = typer.Option(1.0, "--temperature", "-t", help="Sampling temperature."),
    top_p: float = typer.Option(0.95, "--top-p", help="Nucleus sampling top-p."),
    order: str = typer.Option("L-first", "--order", help="Chain order: L-first|H-first."),
    seed: int = typer.Option(0, "--seed", help="RNG seed for reproducibility."),
    device: Optional[str] = typer.Option(None, "--device", help="cuda|cpu|mps (auto if unset)."),
) -> None:
    """Jointly generate paired heavy+light CDRs from a paired frameworks JSON and print them."""
    if order not in ("L-first", "H-first"):
        raise typer.BadParameter("order must be 'L-first' or 'H-first'.")

    data = json.loads(frameworks.read_text())
    h_frameworks = {k: data["H"][k] for k in ("fr1", "fr2", "fr3", "fr4")}
    l_frameworks = {k: data["L"][k] for k in ("fr1", "fr2", "fr3", "fr4")}

    gen = GenCDRGenerator.from_pretrained(model, device=device)
    results = gen.generate_paired_cdrs_from_frameworks(
        h_frameworks=h_frameworks,
        l_frameworks=l_frameworks,
        n_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
        order=order,
        seed=seed,
    )

    for i, res in enumerate(results):
        h, light = res["H"], res["L"]
        status = "ok" if res["parsed_ok"] else f"issues={res['issues']}"
        print(f"[{i}] H: {h['cdr1']} | {h['cdr2']} | {h['cdr3']}")
        print(f"    L: {light['cdr1']} | {light['cdr2']} | {light['cdr3']}  ({status})")


if __name__ == "__main__":
    typer.run(main)
