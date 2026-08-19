"""``gencdr`` command-line entry point.

Subcommands:
  - generate         single-chain (IgGenCDR / NanoGenCDR) CDR generation from frameworks
  - generate-paired  joint heavy+light (p-IgGenCDR) CDR generation
  - score            region log-likelihood scoring of sequences in a CSV
  - align            weighted DPO alignment from reward CSV(s) (requires the 'align' extra)

`--model` accepts either a local model directory or a short name resolved against the
local checkpoint cache (see gencdr.checkpoints).
"""

from pathlib import Path
from typing import List, Optional

import typer

from gencdr.checkpoints import resolve_checkpoint

app = typer.Typer(add_completion=False, help="GenCDR: antibody CDR generation and log-likelihood scoring.")

SCHEMES = ("imgt", "kabat", "chothia")


def _resolve_model(model: str) -> str:
    """Resolve a model name/path to a local directory, raising a clean CLI error on failure."""
    try:
        return str(resolve_checkpoint(model))
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def generate(
    model: str = typer.Option(..., "--model", "-m", help="Model directory or name (iggencdr/nanogencdr)."),
    out_csv: Path = typer.Option(..., "--out-csv", "-o", help="Output CSV path."),
    chain_type: str = typer.Option("H", "--chain-type", help="Chain type: 'H' or 'L'."),
    n_samples: int = typer.Option(10, "--n-samples", "-n", help="Samples per framework set."),
    frameworks_json: Optional[Path] = typer.Option(
        None, "--frameworks", help="JSON file with one framework set {fr1,fr2,fr3,fr4}."
    ),
    in_csv: Optional[Path] = typer.Option(None, "--in-csv", help="CSV with fr1..fr4 columns (one set per row)."),
    temperature: float = typer.Option(1.0, "--temperature", "-t", help="Sampling temperature."),
    top_p: float = typer.Option(0.95, "--top-p", help="Nucleus sampling top-p."),
    max_new_tokens: int = typer.Option(256, "--max-new-tokens", help="Max new tokens to generate."),
    scheme: Optional[str] = typer.Option(None, "--scheme", help="Numbering scheme: imgt|kabat|chothia."),
    seed: Optional[int] = typer.Option(None, "--seed", help="RNG seed for reproducibility."),
    device: Optional[str] = typer.Option(None, "--device", help="Torch device (cuda|cpu|mps); auto if unset."),
    fp: int = typer.Option(16, "--fp", help="Precision: 16 enables autocast on CUDA."),
    include_scheme_token: bool = typer.Option(False, "--include-scheme-token", help="Emit scheme token in prompt."),
    cdr1_temperature: Optional[float] = typer.Option(None, "--cdr1-temperature", help="Per-region temperature (CDR1)."),
    cdr2_temperature: Optional[float] = typer.Option(None, "--cdr2-temperature", help="Per-region temperature (CDR2)."),
    cdr3_temperature: Optional[float] = typer.Option(None, "--cdr3-temperature", help="Per-region temperature (CDR3)."),
) -> None:
    """Generate single-chain CDRs from framework regions."""
    from gencdr.cli.generate import generate_single

    if chain_type not in ("H", "L"):
        raise typer.BadParameter("chain_type must be 'H' or 'L'.")
    if scheme is not None and scheme.lower() not in SCHEMES:
        raise typer.BadParameter(f"scheme must be one of {SCHEMES}.")

    generate_single(
        model_dir=_resolve_model(model),
        out_csv=out_csv,
        chain_type=chain_type,
        n_samples=n_samples,
        frameworks_json=frameworks_json,
        in_csv=in_csv,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        scheme=(scheme.lower() if scheme else None),
        seed=seed,
        device=device,
        fp=fp,
        include_scheme_token=include_scheme_token,
        cdr1_temperature=cdr1_temperature,
        cdr2_temperature=cdr2_temperature,
        cdr3_temperature=cdr3_temperature,
    )


@app.command("generate-paired")
def generate_paired_cmd(
    model: str = typer.Option(..., "--model", "-m", help="Model directory or name (p-iggencdr)."),
    out_csv: Path = typer.Option(..., "--out-csv", "-o", help="Output CSV path."),
    n_samples: int = typer.Option(10, "--n-samples", "-n", help="Samples per framework set."),
    frameworks_json: Optional[Path] = typer.Option(
        None, "--frameworks", help="JSON with {'H': {fr1..fr4}, 'L': {fr1..fr4}}."
    ),
    in_csv: Optional[Path] = typer.Option(None, "--in-csv", help="CSV with h_fr1..h_fr4, l_fr1..l_fr4 columns."),
    temperature: float = typer.Option(1.0, "--temperature", "-t", help="Sampling temperature."),
    top_p: float = typer.Option(0.95, "--top-p", help="Nucleus sampling top-p."),
    max_new_tokens: int = typer.Option(256, "--max-new-tokens", help="Max new tokens (covers both chains)."),
    order: str = typer.Option("L-first", "--order", help="Chain order: L-first|H-first."),
    scheme: Optional[str] = typer.Option(None, "--scheme", help="Numbering scheme: imgt|kabat|chothia."),
    seed: Optional[int] = typer.Option(None, "--seed", help="RNG seed for reproducibility."),
    device: Optional[str] = typer.Option(None, "--device", help="Torch device (cuda|cpu|mps); auto if unset."),
    fp: int = typer.Option(16, "--fp", help="Precision: 16 enables autocast on CUDA."),
    include_scheme_token: bool = typer.Option(False, "--include-scheme-token", help="Emit scheme token in prompt."),
) -> None:
    """Jointly generate paired heavy+light CDRs from both frameworks (p-IgGenCDR)."""
    from gencdr.cli.generate import generate_paired

    if order not in ("L-first", "H-first"):
        raise typer.BadParameter("order must be 'L-first' or 'H-first'.")
    if scheme is not None and scheme.lower() not in SCHEMES:
        raise typer.BadParameter(f"scheme must be one of {SCHEMES}.")

    generate_paired(
        model_dir=_resolve_model(model),
        out_csv=out_csv,
        n_samples=n_samples,
        frameworks_json=frameworks_json,
        in_csv=in_csv,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        order=order,
        scheme=(scheme.lower() if scheme else None),
        seed=seed,
        device=device,
        fp=fp,
        include_scheme_token=include_scheme_token,
    )


@app.command()
def score(
    model: str = typer.Option(..., "--model", "-m", help="Model directory or name."),
    csv_path: Path = typer.Option(..., "--csv-path", help="Input CSV with a sequence column."),
    out_csv: Optional[Path] = typer.Option(
        None, "--out-csv", "-o", help="Output CSV (default: <input>_gencdr_scored)."
    ),
    sequence_column: str = typer.Option("sequence", "--sequence-column", help="Sequence column name."),
    scheme: str = typer.Option("imgt", "--scheme", help="Numbering scheme: imgt|kabat|chothia."),
    chain_type: str = typer.Option("H", "--chain-type", help="Chain token for scoring: 'H' or 'L'."),
    device: Optional[str] = typer.Option(None, "--device", help="Torch device (cuda|cpu|mps); auto if unset."),
    fp: int = typer.Option(16, "--fp", help="Precision: 16 enables autocast on CUDA."),
    batch_size: int = typer.Option(256, "--batch-size", help="Batch size for scoring."),
    reduction: str = typer.Option("mean", "--reduction", help="Reduction: mean|sum."),
    include_scheme_token: bool = typer.Option(False, "--include-scheme-token", help="Emit scheme token in render."),
    allow_empty_cdr: bool = typer.Option(False, "--allow-empty-cdr", help="Keep rows where a CDR is empty."),
    value_column: str = typer.Option("label", "--value-column", help="Numeric column for correlation plots."),
    no_plot: bool = typer.Option(False, "--no-plot", help="Disable scatter plot / Spearman computation."),
    plot_path: Optional[Path] = typer.Option(None, "--plot-path", help="Output PNG path for the scatter plot."),
    spearman_json: Optional[Path] = typer.Option(None, "--spearman-json", help="Output JSON path for correlations."),
) -> None:
    """Score sequences in a CSV with region log-likelihoods (requires the 'scoring' extra)."""
    from gencdr.cli.score import score_csv

    if chain_type not in ("H", "L"):
        raise typer.BadParameter("chain_type must be 'H' or 'L'.")
    if scheme.lower() not in SCHEMES:
        raise typer.BadParameter(f"scheme must be one of {SCHEMES}.")
    if reduction not in ("mean", "sum"):
        raise typer.BadParameter("reduction must be 'mean' or 'sum'.")

    resolved_out = out_csv or csv_path.with_name(f"{csv_path.stem}_gencdr_scored{csv_path.suffix}")
    plot_config = None
    if not no_plot:
        plot_config = {"value_column": value_column, "plot_path": plot_path, "spearman_json": spearman_json}

    score_csv(
        csv_path=csv_path,
        out_csv=resolved_out,
        model_dir=Path(_resolve_model(model)),
        sequence_column=sequence_column,
        scheme=scheme.lower(),
        chain_type=chain_type,
        device=device,
        fp=fp,
        batch_size=batch_size,
        reduction=reduction,
        include_scheme_token=include_scheme_token,
        allow_empty_cdr=allow_empty_cdr,
        plot_config=plot_config,
    )


@app.command()
def align(
    model: str = typer.Option(..., "--model", "-m", help="Policy model directory or name to align."),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Directory for aligned model + logs."),
    reward_csv: List[Path] = typer.Option(
        ..., "--reward-csv", help="Reward CSV path (repeat --reward-csv to combine several)."
    ),
    ref_model: Optional[str] = typer.Option(
        None, "--ref-model", help="Frozen reference model (default: the policy model)."
    ),
    paired: bool = typer.Option(False, "--paired", help="Paired heavy+light alignment (p-IgGenCDR)."),
    order: str = typer.Option("L-first", "--order", help="Paired chain order: L-first|H-first."),
    group_by: str = typer.Option("source", "--group-by", help="Scaffold grouping key: source|prompt."),
    include_scheme_token: bool = typer.Option(False, "--include-scheme-token", help="Emit scheme token in render."),
    beta: float = typer.Option(0.15, "--beta", help="DPO KL strength (manuscript default 0.15)."),
    learning_rate: float = typer.Option(2e-6, "--learning-rate", "--lr", help="AdamW learning rate."),
    weight_decay: float = typer.Option(0.01, "--weight-decay", help="AdamW weight decay."),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="Batch size."),
    max_epochs: int = typer.Option(5, "--max-epochs", "-e", help="Maximum training epochs."),
    val_size: float = typer.Option(0.2, "--val-size", help="Validation fraction."),
    no_stratify: bool = typer.Option(False, "--no-stratify", help="Disable reward-stratified val split."),
    max_length: int = typer.Option(256, "--max-length", help="Max tokenised sequence length."),
    precision: str = typer.Option("32-true", "--precision", help="Lightning precision (e.g. 32-true, bf16-mixed)."),
    num_workers: int = typer.Option(4, "--num-workers", help="DataLoader workers."),
    seed: int = typer.Option(42, "--seed", help="RNG seed."),
    exp_name: str = typer.Option("gencdr-wdpo", "--exp-name", help="Run/checkpoint name prefix."),
    num_devices: Optional[int] = typer.Option(None, "--num-devices", help="Number of devices (default: auto)."),
) -> None:
    """Align a GenCDR model with weighted DPO from reward CSV(s) (requires the 'align' extra)."""
    from gencdr.cli.align import align_cdrs

    if order not in ("L-first", "H-first"):
        raise typer.BadParameter("order must be 'L-first' or 'H-first'.")
    if group_by not in ("source", "prompt"):
        raise typer.BadParameter("group_by must be 'source' or 'prompt'.")

    align_cdrs(
        model=_resolve_model(model),
        reward_csvs=reward_csv,
        output_dir=output_dir,
        ref_model=(_resolve_model(ref_model) if ref_model else None),
        mode=("paired" if paired else "single"),
        order=order,
        group_by=group_by,
        include_scheme_token=include_scheme_token,
        beta=beta,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        val_size=val_size,
        stratified_val=not no_stratify,
        max_length=max_length,
        precision=precision,
        num_workers=num_workers,
        seed=seed,
        exp_name=exp_name,
        num_devices=num_devices,
    )


def run() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    run()
