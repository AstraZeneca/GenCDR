"""Tests for the GenCDR tokenizer module."""

import tempfile
from pathlib import Path

import pytest
from gencdr.tokenizer import (
    AA_ALPHABET,
    DEFAULT_RESERVED_N,
    SCHEME_TOK_BY_NAME,
    TOK_BOS,
    TOK_EOS,
    TOK_H,
    TOK_HPRED,
    TOK_L,
    TOK_LPRED,
    TOK_PAD,
    TOK_SCHEME_CHOTHIA,
    TOK_SCHEME_IMGT,
    TOK_SCHEME_KABAT,
    TOK_SEP,
    TOK_UNK,
    build_tokenizer_json,
    save_tokenizer_dir,
)
from transformers import PreTrainedTokenizerFast


@pytest.fixture
def temp_tokenizer_dir():
    """Fixture to create a temporary directory for tokenizer files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_build_tokenizer_json_structure():
    """Test that build_tokenizer_json returns valid structure."""
    tok_json = build_tokenizer_json()

    assert "version" in tok_json
    assert "model" in tok_json
    assert "added_tokens" in tok_json
    assert "pre_tokenizer" in tok_json
    assert "post_processor" in tok_json
    assert "decoder" in tok_json

    assert tok_json["model"]["type"] == "BPE"
    assert tok_json["model"]["unk_token"] == TOK_UNK
    assert "vocab" in tok_json["model"]
    assert "merges" in tok_json["model"]
    assert len(tok_json["model"]["merges"]) == 0


def test_build_tokenizer_json_vocab_contains_specials():
    """Test that vocab includes all special tokens."""
    tok_json = build_tokenizer_json()
    vocab = tok_json["model"]["vocab"]

    assert TOK_PAD in vocab
    assert TOK_BOS in vocab
    assert TOK_EOS in vocab
    assert TOK_SEP in vocab
    assert TOK_UNK in vocab

    assert TOK_H in vocab
    assert TOK_L in vocab
    assert TOK_HPRED in vocab
    assert TOK_LPRED in vocab

    assert TOK_SCHEME_IMGT in vocab
    assert TOK_SCHEME_KABAT in vocab
    assert TOK_SCHEME_CHOTHIA in vocab


def test_build_tokenizer_json_vocab_contains_aa():
    """Test that vocab includes all amino acids."""
    tok_json = build_tokenizer_json()
    vocab = tok_json["model"]["vocab"]

    for aa in AA_ALPHABET:
        assert aa in vocab, f"Amino acid {aa} not in vocab"


@pytest.mark.parametrize("reserved_n", [0, 5, 10, 20])
def test_build_tokenizer_json_reserved_tokens(reserved_n):
    """Test that reserved tokens are included based on reserved_n parameter."""
    tok_json = build_tokenizer_json(reserved_n=reserved_n)
    vocab = tok_json["model"]["vocab"]

    for i in range(reserved_n):
        reserved_tok = f"<RESERVED_{i}>"
        assert reserved_tok in vocab, f"Reserved token {reserved_tok} not in vocab"

    if reserved_n < 20:
        extra_reserved = f"<RESERVED_{reserved_n}>"
        assert extra_reserved not in vocab


def test_build_tokenizer_json_added_tokens():
    """Test that added_tokens list is properly formatted."""
    tok_json = build_tokenizer_json()
    added = tok_json["added_tokens"]

    assert isinstance(added, list)
    assert len(added) > 0

    for token_info in added:
        assert "id" in token_info
        assert "special" in token_info
        assert "content" in token_info
        assert token_info["special"] is True


def test_build_tokenizer_json_vocab_size():
    """Test that vocab size matches expected count."""
    reserved_n = 10
    tok_json = build_tokenizer_json(reserved_n=reserved_n)
    vocab = tok_json["model"]["vocab"]

    # Core specials: 5 (PAD, BOS, EOS, SEP, UNK)
    # Chain tokens: 4 (H, L, HPRED, LPRED)
    # Scheme tokens: 3 (IMGT, KABAT, CHOTHIA)
    # Reserved: reserved_n
    # AA alphabet: 26
    expected_size = 5 + 4 + 3 + reserved_n + len(AA_ALPHABET)

    assert len(vocab) == expected_size


def test_save_tokenizer_dir_creates_directory(temp_tokenizer_dir):
    """Test that save_tokenizer_dir creates the output directory."""
    out_dir = temp_tokenizer_dir / "tokenizer"
    assert not out_dir.exists()

    save_tokenizer_dir(out_dir)

    assert out_dir.exists()
    assert out_dir.is_dir()


def test_save_tokenizer_dir_creates_required_files(temp_tokenizer_dir):
    """Test that save_tokenizer_dir creates all required files."""
    out_dir = temp_tokenizer_dir / "tokenizer"
    save_tokenizer_dir(out_dir)

    # Check for HuggingFace tokenizer files
    assert (out_dir / "tokenizer_config.json").exists()
    assert (out_dir / "special_tokens_map.json").exists()
    assert (out_dir / "tokenizer.json").exists()


def test_save_tokenizer_dir_returns_path(temp_tokenizer_dir):
    """Test that save_tokenizer_dir returns the output path."""
    out_dir = temp_tokenizer_dir / "tokenizer"
    result = save_tokenizer_dir(out_dir)

    assert result == out_dir
    assert isinstance(result, Path)


@pytest.mark.parametrize("model_max_length", [512, 1024, 2048])
def test_save_tokenizer_dir_model_max_length(temp_tokenizer_dir, model_max_length):
    """Test that model_max_length parameter is respected."""
    out_dir = temp_tokenizer_dir / "tokenizer"
    save_tokenizer_dir(out_dir, model_max_length=model_max_length)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(out_dir), local_files_only=True)
    assert tokenizer.model_max_length == model_max_length


def test_save_tokenizer_dir_with_string_path(temp_tokenizer_dir):
    """Test that save_tokenizer_dir accepts string paths."""
    out_dir = str(temp_tokenizer_dir / "tokenizer")
    result = save_tokenizer_dir(out_dir)

    assert isinstance(result, Path)
    assert result.exists()


def test_tokenizer_special_tokens(temp_tokenizer_dir):
    """Test that all special tokens are correctly configured."""
    out_dir = temp_tokenizer_dir / "tokenizer"
    save_tokenizer_dir(out_dir)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(out_dir), local_files_only=True)

    # Core specials
    assert tokenizer.pad_token == TOK_PAD
    assert tokenizer.bos_token == TOK_BOS
    assert tokenizer.eos_token == TOK_EOS
    assert tokenizer.sep_token == TOK_SEP
    assert tokenizer.unk_token == TOK_UNK

    # IDs should be valid
    assert tokenizer.pad_token_id is not None
    assert tokenizer.bos_token_id is not None
    assert tokenizer.eos_token_id is not None
    assert tokenizer.sep_token_id is not None
    assert tokenizer.unk_token_id is not None


def test_tokenizer_additional_special_tokens(temp_tokenizer_dir):
    """Test that additional special tokens are present."""
    out_dir = temp_tokenizer_dir / "tokenizer"
    save_tokenizer_dir(out_dir)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(out_dir), local_files_only=True)

    expected_additional = {
        TOK_H,
        TOK_L,
        TOK_HPRED,
        TOK_LPRED,
        TOK_SCHEME_IMGT,
        TOK_SCHEME_KABAT,
        TOK_SCHEME_CHOTHIA,
        *{f"<RESERVED_{i}>" for i in range(DEFAULT_RESERVED_N)},
    }

    add_specials = set(tokenizer.additional_special_tokens)
    missing = expected_additional - add_specials

    assert not missing, f"Missing additional specials: {missing}"


def test_tokenizer_special_token_ids_valid(temp_tokenizer_dir):
    """Test that special tokens have valid (non-UNK) IDs."""
    out_dir = temp_tokenizer_dir / "tokenizer"
    save_tokenizer_dir(out_dir)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(out_dir), local_files_only=True)

    base_specials = [TOK_PAD, TOK_BOS, TOK_EOS, TOK_SEP]
    for t in base_specials:
        tid = tokenizer.convert_tokens_to_ids(t)
        assert tid is not None and tid != tokenizer.unk_token_id, f"Token {t} has invalid/UNK id"

    unk_id = tokenizer.convert_tokens_to_ids(TOK_UNK)
    assert unk_id == tokenizer.unk_token_id, "UNK token id does not equal tokenizer.unk_token_id"

    additional = [TOK_H, TOK_L, TOK_HPRED, TOK_LPRED, TOK_SCHEME_IMGT, TOK_SCHEME_KABAT, TOK_SCHEME_CHOTHIA]
    for t in additional:
        tid = tokenizer.convert_tokens_to_ids(t)
        assert tid is not None and tid != tokenizer.unk_token_id, f"Token {t} has invalid/UNK id"


def test_tokenizer_encode_decode_roundtrip(temp_tokenizer_dir):
    """Test that encoding and decoding round-trip works correctly."""
    out_dir = temp_tokenizer_dir / "tokenizer"
    save_tokenizer_dir(out_dir)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(out_dir), local_files_only=True)

    test_str = f"{TOK_BOS}{TOK_H}ACDEFGHIKLMNPQRSTVWY{TOK_SEP}{TOK_HPRED}XYZ{TOK_EOS}"
    ids = tokenizer.encode(test_str, add_special_tokens=False)
    dec = tokenizer.decode(ids, skip_special_tokens=False)

    assert dec.startswith(f"{TOK_BOS}{TOK_H}"), "Decoded string doesn't start correctly"
    assert dec.endswith(TOK_EOS), "Decoded string doesn't end correctly"
    assert TOK_SEP in dec, "Separator token missing in decoded string"
    assert TOK_HPRED in dec, "HPRED token missing in decoded string"


def test_tokenizer_unknown_character(temp_tokenizer_dir):
    """Test that unknown characters map to UNK token."""
    out_dir = temp_tokenizer_dir / "tokenizer"
    save_tokenizer_dir(out_dir)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(out_dir), local_files_only=True)

    unk_ids = tokenizer.encode(f"{TOK_BOS}!@#{TOK_EOS}", add_special_tokens=False)
    unk_dec = tokenizer.decode(unk_ids, skip_special_tokens=False)

    assert TOK_UNK in unk_dec, "Unknown characters did not map to <UNK>"
    assert "!" not in unk_dec, "Unknown character '!' should not appear in decoded string"
    assert "@" not in unk_dec, "Unknown character '@' should not appear in decoded string"


def test_tokenizer_amino_acid_encoding(temp_tokenizer_dir):
    """Test that all amino acids can be encoded."""
    out_dir = temp_tokenizer_dir / "tokenizer"
    save_tokenizer_dir(out_dir)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(out_dir), local_files_only=True)

    # Encode all amino acids
    aa_string = "".join(AA_ALPHABET)
    ids = tokenizer.encode(aa_string, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)

    for aa in AA_ALPHABET:
        assert aa in decoded, f"Amino acid {aa} not in decoded string"


def test_tokenizer_chain_tokens(temp_tokenizer_dir):
    """Test that chain tokens work correctly."""
    out_dir = temp_tokenizer_dir / "tokenizer"
    save_tokenizer_dir(out_dir)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(out_dir), local_files_only=True)

    # Test heavy chain
    h_str = f"{TOK_H}ACDE"
    h_ids = tokenizer.encode(h_str, add_special_tokens=False)
    h_dec = tokenizer.decode(h_ids, skip_special_tokens=False)
    assert TOK_H in h_dec, "Heavy chain token not in decoded string"

    # Test light chain
    l_str = f"{TOK_L}FGHI"
    l_ids = tokenizer.encode(l_str, add_special_tokens=False)
    l_dec = tokenizer.decode(l_ids, skip_special_tokens=False)
    assert TOK_L in l_dec, "Light chain token not in decoded string"

    # Test prediction tokens
    pred_str = f"{TOK_HPRED}ABC{TOK_LPRED}DEF"
    pred_ids = tokenizer.encode(pred_str, add_special_tokens=False)
    pred_dec = tokenizer.decode(pred_ids, skip_special_tokens=False)
    assert TOK_HPRED in pred_dec, "HPRED token not in decoded string"
    assert TOK_LPRED in pred_dec, "LPRED token not in decoded string"


def test_tokenizer_scheme_tokens(temp_tokenizer_dir):
    """Test that numbering scheme tokens work correctly."""
    out_dir = temp_tokenizer_dir / "tokenizer"
    save_tokenizer_dir(out_dir)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(out_dir), local_files_only=True)

    for _, scheme_tok in SCHEME_TOK_BY_NAME.items():
        test_str = f"{scheme_tok}ACDE"
        ids = tokenizer.encode(test_str, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        assert scheme_tok in decoded, f"Scheme token {scheme_tok} not in decoded string"


def test_tokenizer_padding(temp_tokenizer_dir):
    """Test tokenizer padding functionality."""
    out_dir = temp_tokenizer_dir / "tokenizer"
    save_tokenizer_dir(out_dir, model_max_length=128)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(out_dir), local_files_only=True)

    # Encode two sequences of different lengths with padding
    seqs = ["ACDE", "FGHIKLMN"]
    encoded = tokenizer(seqs, padding="longest", add_special_tokens=False, return_tensors="pt")

    assert "input_ids" in encoded
    assert "attention_mask" in encoded

    assert encoded["input_ids"].shape[0] == 2
    assert encoded["input_ids"].shape[1] == encoded["attention_mask"].shape[1]

    assert (encoded["input_ids"][0] == tokenizer.pad_token_id).any()


def test_tokenizer_batch_encode(temp_tokenizer_dir):
    """Test batch encoding with tokenizer."""
    out_dir = temp_tokenizer_dir / "tokenizer"
    save_tokenizer_dir(out_dir)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(out_dir), local_files_only=True)

    seqs = [
        f"{TOK_H}ACDE",
        f"{TOK_L}FGHI",
        f"{TOK_HPRED}KLMN",
    ]

    encoded = tokenizer(seqs, padding=True, add_special_tokens=False, return_tensors="pt")

    assert encoded["input_ids"].shape[0] == 3
    assert encoded["attention_mask"].shape[0] == 3


def test_tokenizer_vocab_size(temp_tokenizer_dir):
    """Test that tokenizer vocab size is correct."""
    out_dir = temp_tokenizer_dir / "tokenizer"
    reserved_n = 10
    save_tokenizer_dir(out_dir, reserved_n=reserved_n)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(out_dir), local_files_only=True)

    # Expected: 5 core + 4 chain + 3 scheme + reserved_n + 26 AA
    expected_size = 5 + 4 + 3 + reserved_n + len(AA_ALPHABET)
    assert tokenizer.vocab_size == expected_size


def test_tokenizer_skip_special_tokens(temp_tokenizer_dir):
    """Test decoding with skip_special_tokens option."""
    out_dir = temp_tokenizer_dir / "tokenizer"
    save_tokenizer_dir(out_dir)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(out_dir), local_files_only=True)

    test_str = f"{TOK_BOS}{TOK_H}ACDE{TOK_SEP}{TOK_HPRED}FGHI{TOK_EOS}"
    ids = tokenizer.encode(test_str, add_special_tokens=False)

    dec_with = tokenizer.decode(ids, skip_special_tokens=False)
    assert TOK_BOS in dec_with
    assert TOK_H in dec_with
    assert TOK_SEP in dec_with

    dec_skip = tokenizer.decode(ids, skip_special_tokens=True)

    assert len(dec_skip) <= len(dec_with)


def test_tokenizer_reserved_tokens_accessible(temp_tokenizer_dir):
    """Test that reserved tokens are accessible and have valid IDs."""
    out_dir = temp_tokenizer_dir / "tokenizer"
    reserved_n = 5
    save_tokenizer_dir(out_dir, reserved_n=reserved_n)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(out_dir), local_files_only=True)

    for i in range(reserved_n):
        reserved_tok = f"<RESERVED_{i}>"
        tok_id = tokenizer.convert_tokens_to_ids(reserved_tok)
        assert tok_id is not None
        assert tok_id != tokenizer.unk_token_id

        ids = tokenizer.encode(reserved_tok, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        assert reserved_tok in decoded


def test_tokenizer_consistent_across_saves(temp_tokenizer_dir):
    """Test that saving tokenizer twice produces consistent results."""
    out_dir1 = temp_tokenizer_dir / "tokenizer1"
    out_dir2 = temp_tokenizer_dir / "tokenizer2"

    save_tokenizer_dir(out_dir1)
    save_tokenizer_dir(out_dir2)

    tok1 = PreTrainedTokenizerFast.from_pretrained(str(out_dir1), local_files_only=True)
    tok2 = PreTrainedTokenizerFast.from_pretrained(str(out_dir2), local_files_only=True)

    assert tok1.vocab_size == tok2.vocab_size
    assert tok1.model_max_length == tok2.model_max_length

    test_str = f"{TOK_H}ACDEFGHIKLMNPQRSTVWY"
    ids1 = tok1.encode(test_str, add_special_tokens=False)
    ids2 = tok2.encode(test_str, add_special_tokens=False)
    assert ids1 == ids2
