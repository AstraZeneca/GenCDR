"""Weighted DPO alignment for GenCDR models.

Scaffold-grouped weighted DPO (the GenCDR manuscript objective) driven entirely by a
reward CSV: no structure prediction, oracle, or experiment-tracking code. Works for
single-chain models (IgGenCDR / NanoGenCDR) and, as an extension beyond the published
experiments, for the paired model (p-IgGenCDR).

Layout:

- ``loss`` — pure loss math (``grouped_weighted_dpo_loss``, ``completion_mask_unshifted``);
  torch-only, always importable.
- ``dataset`` — ``FrameworkCDRRewardDataset`` / ``FrameworkCDRRewardCollator``; torch +
  transformers, always importable.
- ``trainer`` / ``aligner`` — the Lightning training loop and the CSV-driven runner.
  These require the ``align`` extra (``pip install "gencdr[align]"``) and are imported
  lazily so this subpackage can be imported for the loss/dataset pieces without
  ``pytorch-lightning`` installed.
"""

import importlib
from typing import TYPE_CHECKING

from gencdr.alignment.dataset import FrameworkCDRRewardCollator, FrameworkCDRRewardDataset
from gencdr.alignment.loss import completion_mask_unshifted, grouped_weighted_dpo_loss

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gencdr.alignment.aligner import WeightedDPOAligner
    from gencdr.alignment.trainer import WeightedDPOModule

__all__ = [
    "FrameworkCDRRewardCollator",
    "FrameworkCDRRewardDataset",
    "completion_mask_unshifted",
    "grouped_weighted_dpo_loss",
    "WeightedDPOModule",
    "WeightedDPOAligner",
]

_LAZY = {
    "WeightedDPOModule": ("gencdr.alignment.trainer", "WeightedDPOModule"),
    "WeightedDPOAligner": ("gencdr.alignment.aligner", "WeightedDPOAligner"),
}


def __getattr__(name: str):
    """Lazily import the Lightning-backed classes (require the ``align`` extra)."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            f"'{name}' requires the training dependencies. Install them with "
            'pip install "gencdr[align]" (adds pytorch-lightning and torchmetrics).'
        ) from exc
    return getattr(module, attr)
