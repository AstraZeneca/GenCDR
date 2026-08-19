"""Run weighted DPO alignment from reward CSV(s).

Thin CLI wrapper around ``gencdr.alignment.aligner.WeightedDPOAligner``. Imported
lazily from the CLI so the ``align`` extra (pytorch-lightning, torchmetrics) is only
required when this subcommand is used.
"""

from pathlib import Path
from typing import List, Optional


def align_cdrs(
    model: str,
    reward_csvs: List[Path],
    output_dir: Path,
    ref_model: Optional[str],
    mode: str,
    order: str,
    group_by: str,
    include_scheme_token: bool,
    beta: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    val_size: float,
    stratified_val: bool,
    max_length: int,
    precision: str,
    num_workers: int,
    seed: int,
    exp_name: str,
    num_devices: Optional[int],
) -> str:
    """Build and run the weighted DPO aligner; return the best checkpoint path."""
    from gencdr.alignment.aligner import WeightedDPOAligner

    aligner = WeightedDPOAligner(
        model=model,
        output_dir=str(output_dir),
        ref_model=ref_model,
        mode=mode,
        order=order,
        group_by=group_by,
        include_scheme_token=include_scheme_token,
        exp_name=exp_name,
        beta=beta,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        val_size=val_size,
        stratified_val=stratified_val,
        max_length=max_length,
        precision=precision,
        num_workers=num_workers,
        seed=seed,
        num_devices=num_devices,
    )
    best = aligner.run([str(p) for p in reward_csvs])
    print(f"Alignment complete. Aligned model saved to {output_dir}")
    print(f"Best checkpoint (by val Spearman): {best or '(none — check logs)'}")
    return best
