"""Resolve local GenCDR checkpoints and download published models on demand."""

import hashlib
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable
from urllib.request import urlopen

RELEASE_TAG = "v1.0.0"
RELEASE_URL = f"https://github.com/AstraZeneca/GenCDR/releases/download/{RELEASE_TAG}"

# Canonical short names -> subdirectory under the cache root. Aliases map to the same dir.
MODEL_ALIASES: Dict[str, str] = {
    "iggencdr": "iggencdr",
    "nanogencdr": "nanogencdr",
    "nano": "nanogencdr",
    "p-iggencdr": "p-iggencdr",
    "piggencdr": "p-iggencdr",
    "paired": "p-iggencdr",
}

_MODEL_SHA256: Dict[str, str] = {
    "iggencdr": "9f7fd9bb2e056881a61f978adf815a4028ab99f9c8e746db736e6b178a9a1f29",
    "p-iggencdr": "4b21fb04dc610990f384267b21342df4740dea5afc89180097168890d4feb944",
    "nanogencdr": "3c4c2f9640841edc639449c5cbc5fad78e6d1315dd4af50099f58a41704c2cdc",
}

_ARCHIVE_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
}
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


def _download_model(model_name: str, destination: Path) -> Path:
    """Download and verify one published model archive, then atomically populate its cache."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"Cannot download model into '{destination}': that cache path already exists but is not a valid model. "
            "Move or remove the incomplete directory, then retry."
        )

    url = f"{RELEASE_URL}/{model_name}.zip"
    archive_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{model_name}-", suffix=".zip", delete=False
        ) as f:
            archive_path = Path(f.name)
            digest = hashlib.sha256()
            with urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    f.write(chunk)
                    digest.update(chunk)

        if digest.hexdigest() != _MODEL_SHA256[model_name]:
            raise ValueError(f"SHA-256 verification failed for downloaded model '{model_name}'.")

        with tempfile.TemporaryDirectory(dir=destination.parent, prefix=f".{model_name}-") as temp_dir:
            staging_dir = Path(temp_dir) / "model"
            staging_dir.mkdir()
            with zipfile.ZipFile(archive_path) as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if len(names) != len(set(names)) or set(names) != _ARCHIVE_FILES:
                    raise ValueError(f"Downloaded archive for '{model_name}' has an unexpected file layout.")
                for info in infos:
                    if info.is_dir() or info.flag_bits & 0x1:
                        raise ValueError(f"Downloaded archive for '{model_name}' contains an unsupported entry.")
                archive.extractall(staging_dir)
            if not _looks_like_model_dir(staging_dir):
                raise ValueError(f"Downloaded archive for '{model_name}' is missing required model files.")
            os.replace(staging_dir, destination)
    finally:
        if archive_path is not None:
            archive_path.unlink(missing_ok=True)

    return destination


def resolve_checkpoint(name_or_path: str) -> Path:
    """Resolve a local model directory or short model name, downloading known models if needed.

    Parameters
    ----------
    name_or_path : str
        Either a filesystem path to a model directory, or a short model name (see
        ``MODEL_ALIASES``) resolved under the cache root.

    Returns
    -------
    Path
        Path to the resolved local model directory.

    Raises
    ------
    FileNotFoundError
        If a local model directory or published model cannot be resolved.
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
    if subdir in _MODEL_SHA256:
        return _download_model(subdir, resolved)

    known = ", ".join(sorted(set(MODEL_ALIASES.values())))
    raise FileNotFoundError(
        f"Could not resolve checkpoint '{name_or_path}'. Looked for a model directory at "
        f"'{resolved}'. Place the model files there, or pass an explicit directory path. "
        f"Set $GENCDR_HOME to change the cache root (currently '{cache_root()}'). "
        f"Known model names: {known}."
    )
