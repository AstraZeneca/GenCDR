"""Local checkpoint resolution for GenCDR models.

Weights are NOT shipped with this package. Resolution rules for
``resolve_checkpoint(name_or_path)``:

1. If ``name_or_path`` is an existing directory containing a model, use it directly.
2. Otherwise treat it as a short model name and look it up under the local cache root
   (``$GENCDR_HOME``, default ``~/.cache/gencdr``), i.e. ``$GENCDR_HOME/<name>``.

A model directory is expected to hold the HuggingFace files produced at export time:
``config.json``, the weights (``model.safetensors`` or ``pytorch_model.bin``),
``tokenizer.json`` and ``tokenizer_config.json`` / ``special_tokens_map.json``.

HuggingFace Hub download is not yet wired up (see ``download_from_hub``); once model
weights are published, resolution can fall back to a Hub snapshot without any change to
callers, since loading already goes through ``from_pretrained``.
"""

import os
from pathlib import Path
from typing import Dict, Iterable

# Canonical short names -> subdirectory under the cache root. Aliases map to the same dir.
MODEL_ALIASES: Dict[str, str] = {
    "iggencdr": "iggencdr",
    "nanogencdr": "nanogencdr",
    "nano": "nanogencdr",
    "p-iggencdr": "p-iggencdr",
    "piggencdr": "p-iggencdr",
    "paired": "p-iggencdr",
}

# Files that indicate a directory holds model weights (any one is sufficient).
_WEIGHT_FILES: Iterable[str] = ("model.safetensors", "pytorch_model.bin")
_CONFIG_FILE = "config.json"


def cache_root() -> Path:
    """Return the local checkpoint cache root ($GENCDR_HOME, default ~/.cache/gencdr)."""
    env = os.environ.get("GENCDR_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "gencdr"


def _looks_like_model_dir(path: Path) -> bool:
    """Return True if `path` is a directory containing a config and at least one weight file."""
    if not path.is_dir():
        return False
    if not (path / _CONFIG_FILE).is_file():
        return False
    return any((path / w).is_file() for w in _WEIGHT_FILES)


def resolve_checkpoint(name_or_path: str) -> Path:
    """Resolve a directory path or short model name to a local model directory.

    Parameters
    ----------
    name_or_path : str
        Either a filesystem path to a model directory, or a short model name
        (see MODEL_ALIASES) resolved under the cache root.

    Returns
    -------
    Path
        Path to the resolved local model directory.

    Raises
    ------
    FileNotFoundError
        If no matching local model directory can be found.
    """
    candidate = Path(name_or_path).expanduser()
    if candidate.is_dir():
        if not _looks_like_model_dir(candidate):
            raise FileNotFoundError(
                f"Directory '{candidate}' does not look like a model directory "
                f"(expected {_CONFIG_FILE} and one of {list(_WEIGHT_FILES)})."
            )
        return candidate

    key = name_or_path.strip().lower()
    subdir = MODEL_ALIASES.get(key, key)
    resolved = cache_root() / subdir
    if _looks_like_model_dir(resolved):
        return resolved

    known = ", ".join(sorted(set(MODEL_ALIASES.values())))
    raise FileNotFoundError(
        f"Could not resolve checkpoint '{name_or_path}'. Looked for a model directory at "
        f"'{resolved}'. Place the model files there, or pass an explicit directory path. "
        f"Set $GENCDR_HOME to change the cache root (currently '{cache_root()}'). "
        f"Known model names: {known}."
    )


def download_from_hub(name_or_path: str, revision: str = "main") -> Path:
    """Download a checkpoint from the HuggingFace Hub (not yet implemented).

    When GenCDR weights are published to the Hub, this will call
    ``huggingface_hub.snapshot_download`` and return the local snapshot path, so that
    ``resolve_checkpoint`` can transparently fall back to it. Loading itself already goes
    through ``from_pretrained``, so no changes are needed at the call sites.
    """
    raise NotImplementedError(
        "HuggingFace Hub download is not yet available for GenCDR models. "
        "Provide a local model directory or set $GENCDR_HOME. See gencdr.checkpoints."
    )
