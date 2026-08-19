"""GenCDR: framework-conditioned antibody CDR generation and log-likelihood scoring.

Public API:
    - GenCDRGenerator: load a checkpoint and generate/score CDRs
    - decode_cdrs_with_flags_single_chain, decode_paired_cdrs_with_flags: parse outputs
    - resolve_checkpoint: map a name/path to a local model directory
    - segment_sequence_by_scheme: FR/CDR segmentation (requires the 'scoring' extra)

Model families (single tokenizer / rendering scheme):
    - IgGenCDR    single-chain heavy or light (H/L)
    - NanoGenCDR  nanobody / VHH (heavy-only)
    - p-IgGenCDR  cognate paired heavy+light, jointly generated
"""

from gencdr.checkpoints import cache_root, resolve_checkpoint
from gencdr.generator import (
    GenCDRGenerator,
    IgGenCDRGenerator,
    PerRegionTemperatureProcessor,
    decode_cdrs_with_flags_single_chain,
    decode_paired_cdrs_with_flags,
)

__version__ = "0.2.0"

__all__ = [
    "GenCDRGenerator",
    "IgGenCDRGenerator",
    "PerRegionTemperatureProcessor",
    "decode_cdrs_with_flags_single_chain",
    "decode_paired_cdrs_with_flags",
    "resolve_checkpoint",
    "cache_root",
    "__version__",
]
