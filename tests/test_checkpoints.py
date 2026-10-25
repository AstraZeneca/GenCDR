"""Tests for local checkpoint resolution and release downloads."""

import hashlib
import io
import zipfile
from unittest.mock import patch

import pytest

import gencdr.checkpoints as checkpoints


def _model_archive() -> bytes:
    """Build a small valid checkpoint archive for download tests."""
    data = io.BytesIO()
    files = {
        "config.json": b"{}",
        "generation_config.json": b"{}",
        "model.safetensors": b"weights",
        "special_tokens_map.json": b"{}",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
    }
    with zipfile.ZipFile(data, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return data.getvalue()


def test_resolve_checkpoint_downloads_and_caches_alias(tmp_path, monkeypatch):
    """A model name downloads once and an alias resolves to the canonical cache."""
    content = _model_archive()
    monkeypatch.setenv("GENCDR_HOME", str(tmp_path))
    monkeypatch.setitem(checkpoints._MODEL_SHA256, "nanogencdr", hashlib.sha256(content).hexdigest())
    with patch("gencdr.checkpoints.urlopen", return_value=io.BytesIO(content)) as mock_urlopen:
        resolved = checkpoints.resolve_checkpoint("nano")
        cached = checkpoints.resolve_checkpoint("nanogencdr")

    assert resolved == tmp_path / "nanogencdr"
    assert cached == resolved
    assert (resolved / "model.safetensors").read_bytes() == b"weights"
    mock_urlopen.assert_called_once_with(
        f"{checkpoints.RELEASE_URL}/nanogencdr.zip",
        timeout=60,
    )


def test_resolve_checkpoint_verifies_checksum_before_extracting(tmp_path, monkeypatch):
    """A failed checksum does not leave an extracted model or partial archive."""
    monkeypatch.setenv("GENCDR_HOME", str(tmp_path))
    monkeypatch.setitem(checkpoints._MODEL_SHA256, "iggencdr", "0" * 64)
    with patch("gencdr.checkpoints.urlopen", return_value=io.BytesIO(_model_archive())):
        with pytest.raises(ValueError, match="SHA-256 verification failed"):
            checkpoints.resolve_checkpoint("iggencdr")

    assert list(tmp_path.iterdir()) == []


def test_resolve_checkpoint_rejects_unexpected_archive_layout(tmp_path, monkeypatch):
    """Only the expected root-level files are extracted from a release archive."""
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("../outside.txt", b"bad")
    content = data.getvalue()
    monkeypatch.setenv("GENCDR_HOME", str(tmp_path))
    monkeypatch.setitem(checkpoints._MODEL_SHA256, "p-iggencdr", hashlib.sha256(content).hexdigest())
    with patch("gencdr.checkpoints.urlopen", return_value=io.BytesIO(content)):
        with pytest.raises(ValueError, match="unexpected file layout"):
            checkpoints.resolve_checkpoint("p-iggencdr")

    assert list(tmp_path.iterdir()) == []
    assert not (tmp_path.parent / "outside.txt").exists()


def test_resolve_checkpoint_preserves_local_directory_behavior(tmp_path, monkeypatch):
    """An explicit local checkpoint path is returned without network access."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setenv("GENCDR_HOME", str(tmp_path / "cache"))
    with patch("gencdr.checkpoints.urlopen") as mock_urlopen:
        assert checkpoints.resolve_checkpoint(str(model_dir)) == model_dir
    mock_urlopen.assert_not_called()


def test_resolve_checkpoint_unknown_name_does_not_download(tmp_path, monkeypatch):
    """Unknown model names keep the local-resolution error instead of making a request."""
    monkeypatch.setenv("GENCDR_HOME", str(tmp_path))
    with patch("gencdr.checkpoints.urlopen") as mock_urlopen:
        with pytest.raises(FileNotFoundError, match="Known model names"):
            checkpoints.resolve_checkpoint("unknown")
    mock_urlopen.assert_not_called()
