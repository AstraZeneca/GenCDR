"""Tokenization and vocabulary module for GenCDR.

- ByteLevel BPE with empty merges (char-level).
- Special/control tokens: PAD/BOS/EOS/SEP/UNK, chain tags (H/L), prediction tags (HPRED/LPRED),
  scheme tags (IMGT/KABAT/CHOTHIA), and 10 reserved tokens.
- Exposes global token constants and SCHEME_TOK_BY_NAME for rendering.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from transformers import PreTrainedTokenizerFast

TOK_PAD = "<PAD>"
TOK_BOS = "<BOS>"
TOK_EOS = "<EOS>"
TOK_SEP = "<SEP>"
TOK_UNK = "<UNK>"

TOK_H = "<H>"
TOK_L = "<L>"
TOK_HPRED = "<HPRED>"
TOK_LPRED = "<LPRED>"

TOK_SCHEME_IMGT = "<SCHEME_IMGT>"
TOK_SCHEME_KABAT = "<SCHEME_KABAT>"
TOK_SCHEME_CHOTHIA = "<SCHEME_CHOTHIA>"


SCHEME_TOK_BY_NAME: Dict[str, str] = {
    "imgt": TOK_SCHEME_IMGT,
    "kabat": TOK_SCHEME_KABAT,
    "chothia": TOK_SCHEME_CHOTHIA,
}

DEFAULT_RESERVED_N = 10
DEFAULT_MODEL_MAX_LEN = 1024

AA_ALPHABET: List[str] = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "X",
    "B",
    "Z",
    "J",
    "U",
    "O",
]

CANONICAL_AA_ALPHABET: List[str] = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]


def build_tokenizer_json(reserved_n: int = DEFAULT_RESERVED_N) -> Dict[str, Any]:
    """Return Tokenizers JSON for ByteLevel+BPE (char-level) with specials and reserved tokens."""
    specials = [
        TOK_PAD,
        TOK_BOS,
        TOK_EOS,
        TOK_SEP,
        TOK_H,
        TOK_L,
        TOK_HPRED,
        TOK_LPRED,
        TOK_SCHEME_IMGT,
        TOK_SCHEME_KABAT,
        TOK_SCHEME_CHOTHIA,
        TOK_UNK,
    ] + [f"<RESERVED_{i}>" for i in range(reserved_n)]

    vocab: Dict[str, int] = {}
    for i, tok in enumerate(specials + AA_ALPHABET):
        vocab[tok] = i

    added = [
        {
            "id": vocab[t],
            "special": True,
            "content": t,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
        }
        for t in specials
    ]

    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": TOK_UNK,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "vocab": vocab,
            "merges": [],
        },
    }


def save_tokenizer_dir(
    out_dir: "str | Path", reserved_n: int = DEFAULT_RESERVED_N, model_max_length: int = DEFAULT_MODEL_MAX_LEN
) -> Path:
    """Create a HF-compatible tokenizer directory (tokenizer.json + special_tokens_map + config)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    vocab_json_path = out / "tokenizer_vocab.json"
    vocab_json_path.write_text(json.dumps(build_tokenizer_json(reserved_n=reserved_n), indent=2))

    tok = PreTrainedTokenizerFast(tokenizer_file=str(vocab_json_path))
    tok.pad_token = TOK_PAD
    tok.bos_token = TOK_BOS
    tok.eos_token = TOK_EOS
    tok.sep_token = TOK_SEP
    tok.unk_token = TOK_UNK

    tok.add_special_tokens(
        {
            "additional_special_tokens": [
                TOK_H,
                TOK_L,
                TOK_HPRED,
                TOK_LPRED,
                TOK_SCHEME_IMGT,
                TOK_SCHEME_KABAT,
                TOK_SCHEME_CHOTHIA,
                *[f"<RESERVED_{i}>" for i in range(reserved_n)],
            ]
        }
    )
    tok.model_max_length = int(model_max_length)

    tok.save_pretrained(out)
    return out
