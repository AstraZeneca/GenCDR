"""Tests for GenCDR rendering utilities."""

import random

import pytest
from gencdr.rendering import (
    clamp_span,
    compute_cdr_spans_from_token_ids,
    count_and_first_pos,
    find_block_end,
    find_first_n_positions,
    jitter_boundary_lens,
    parse_pred_block_spans,
    render_paired,
    render_single,
    slice_segments,
    strip_bos_eos,
    validate_pred_block_has_two_seps,
    validate_pred_tag_counts,
)
from gencdr.tokenizer import (
    SCHEME_TOK_BY_NAME,
    TOK_BOS,
    TOK_EOS,
    TOK_H,
    TOK_HPRED,
    TOK_L,
    TOK_LPRED,
    TOK_SEP,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast


@pytest.fixture
def segments():
    """Fixture to provide example antibody chain segments."""
    return {
        "fr1": "HFR1SEQ",
        "fr2": "HFR2SEQ",
        "fr3": "HFR3SEQ",
        "fr4": "HFR4SEQ",
        "cdr1": "HCDR1SEQ",
        "cdr2": "HCDR2SEQ",
        "cdr3": "HCDR3SEQ",
    }


@pytest.mark.parametrize(
    "include_scheme, scheme_type, chain_type",
    [(True, "imgt", "H"), (False, None, "L"), (True, "incorrect", "H"), (True, "kabat", "H"), (True, "chothia", "L")],
)
def test_render_single(segments, include_scheme, scheme_type, chain_type):
    """Test rendering of single antibody chains with various options."""
    if (include_scheme and scheme_type not in SCHEME_TOK_BY_NAME) or chain_type not in ("H", "L"):
        with pytest.raises(KeyError):
            render_single(
                {
                    "segments": segments,
                    "chain_type": chain_type,
                    "scheme": scheme_type,
                },
                include_scheme=include_scheme,
            )
    else:
        rendered = render_single(
            {
                "segments": segments,
                "chain_type": chain_type,
                "scheme": scheme_type,
            },
            include_scheme=include_scheme,
        )

        assert isinstance(rendered, str)
        assert rendered.startswith(TOK_BOS)
        assert rendered.endswith(TOK_EOS)
        rendered = rendered[len(TOK_BOS) : -len(TOK_EOS)]
        if include_scheme:
            scheme_tok = SCHEME_TOK_BY_NAME[scheme_type]
            assert rendered.startswith(scheme_tok)
            rendered = rendered[len(scheme_tok) :]
        chain_tok = TOK_H if chain_type == "H" else TOK_L
        assert rendered.startswith(chain_tok)
        rendered = rendered[len(chain_tok) :]

        if chain_type == "H":
            assert TOK_HPRED in rendered
        else:
            assert TOK_LPRED in rendered

        rendered_frameworks, rendered_cdrs = rendered.split(TOK_HPRED if chain_type == "H" else TOK_LPRED)

        parsed_frameworks = rendered_frameworks.split(TOK_SEP)
        assert len(parsed_frameworks) == 4
        assert parsed_frameworks[0] == segments["fr1"]
        assert parsed_frameworks[1] == segments["fr2"]
        assert parsed_frameworks[2] == segments["fr3"]
        assert parsed_frameworks[3] == segments["fr4"]
        parsed_cdrs = rendered_cdrs.split(TOK_SEP)
        assert len(parsed_cdrs) == 3
        assert parsed_cdrs[0] == segments["cdr1"]
        assert parsed_cdrs[1] == segments["cdr2"]
        assert parsed_cdrs[2] == segments["cdr3"]


@pytest.fixture
def segments_h():
    """Fixture to provide example heavy chain segments."""
    return {
        "fr1": "HFR1SEQ",
        "fr2": "HFR2SEQ",
        "fr3": "HFR3SEQ",
        "fr4": "HFR4SEQ",
        "cdr1": "HCDR1SEQ",
        "cdr2": "HCDR2SEQ",
        "cdr3": "HCDR3SEQ",
    }


@pytest.fixture
def segments_l():
    """Fixture to provide example light chain segments."""
    return {
        "fr1": "LFR1SEQ",
        "fr2": "LFR2SEQ",
        "fr3": "LFR3SEQ",
        "fr4": "LFR4SEQ",
        "cdr1": "LCDR1SEQ",
        "cdr2": "LCDR2SEQ",
        "cdr3": "LCDR3SEQ",
    }


@pytest.mark.parametrize(
    "order, include_scheme, h_scheme, l_scheme, expect_error",
    [
        ("L-first", False, "imgt", "kabat", None),
        ("H-first", False, "imgt", "kabat", None),
        ("L-first", False, "incorrect", "incorrect", None),
        ("L-first", True, "imgt", "imgt", None),
        ("H-first", True, "kabat", "kabat", None),
        ("L-first", True, "imgt", "kabat", ValueError),
        ("H-first", True, "incorrect", "incorrect", KeyError),
    ],
)
def test_render_paired_cases(segments_h, segments_l, order, include_scheme, h_scheme, l_scheme, expect_error):
    """Test rendering of paired chains with single shared scheme semantics."""
    h_sample = {"segments": segments_h, "chain_type": "H", "scheme": h_scheme}
    l_sample = {"segments": segments_l, "chain_type": "L", "scheme": l_scheme}

    if expect_error is not None:
        with pytest.raises(expect_error):
            render_paired(h_sample, l_sample, order=order, include_scheme=include_scheme)
        return

    rendered = render_paired(h_sample, l_sample, order=order, include_scheme=include_scheme)

    assert isinstance(rendered, str)
    assert rendered.startswith(TOK_BOS)
    assert rendered.endswith(TOK_EOS)

    if include_scheme:
        scheme_tok = SCHEME_TOK_BY_NAME[h_scheme]
        assert rendered.startswith(TOK_BOS + scheme_tok)
        core = rendered[len(TOK_BOS + scheme_tok) : -len(TOK_EOS)]
    else:
        core = rendered[len(TOK_BOS) : -len(TOK_EOS)]

    assert core.count(TOK_LPRED) == 1
    assert core.count(TOK_HPRED) == 1
    assert core.count(TOK_SEP) == 10

    if order == "L-first":
        expected = (
            TOK_L
            + segments_l["fr1"]
            + TOK_SEP
            + segments_l["fr2"]
            + TOK_SEP
            + segments_l["fr3"]
            + TOK_SEP
            + segments_l["fr4"]
            + TOK_H
            + segments_h["fr1"]
            + TOK_SEP
            + segments_h["fr2"]
            + TOK_SEP
            + segments_h["fr3"]
            + TOK_SEP
            + segments_h["fr4"]
            + TOK_LPRED
            + segments_l["cdr1"]
            + TOK_SEP
            + segments_l["cdr2"]
            + TOK_SEP
            + segments_l["cdr3"]
            + TOK_HPRED
            + segments_h["cdr1"]
            + TOK_SEP
            + segments_h["cdr2"]
            + TOK_SEP
            + segments_h["cdr3"]
        )
    else:
        expected = (
            TOK_H
            + segments_h["fr1"]
            + TOK_SEP
            + segments_h["fr2"]
            + TOK_SEP
            + segments_h["fr3"]
            + TOK_SEP
            + segments_h["fr4"]
            + TOK_L
            + segments_l["fr1"]
            + TOK_SEP
            + segments_l["fr2"]
            + TOK_SEP
            + segments_l["fr3"]
            + TOK_SEP
            + segments_l["fr4"]
            + TOK_HPRED
            + segments_h["cdr1"]
            + TOK_SEP
            + segments_h["cdr2"]
            + TOK_SEP
            + segments_h["cdr3"]
            + TOK_LPRED
            + segments_l["cdr1"]
            + TOK_SEP
            + segments_l["cdr2"]
            + TOK_SEP
            + segments_l["cdr3"]
        )
    assert core == expected


def test_slice_segments():
    """Test slice_segments slicing and mismatch error."""
    fr1, cdr1, fr2, cdr2, fr3, cdr3, fr4 = "AB", "C", "DE", "F", "GHI", "JK", "LMN"
    seq = fr1 + cdr1 + fr2 + cdr2 + fr3 + cdr3 + fr4
    lens = [len(fr1), len(cdr1), len(fr2), len(cdr2), len(fr3), len(cdr3), len(fr4)]

    out = slice_segments(seq, lens)
    assert out["fr1"] == fr1
    assert out["cdr1"] == cdr1
    assert out["fr2"] == fr2
    assert out["cdr2"] == cdr2
    assert out["fr3"] == fr3
    assert out["cdr3"] == cdr3
    assert out["fr4"] == fr4

    bad_lens = lens.copy()
    bad_lens[-1] += 1
    with pytest.raises(RuntimeError, match="Lengths do not sum to sequence length"):
        _ = slice_segments(seq, bad_lens)


def test_jitter_boundary_lens_noop_when_disabled():
    """Zero or negative jitter returns an identical copy."""
    lens = [5, 3, 8, 4, 12, 7, 6]
    out = jitter_boundary_lens(lens, 0)
    assert out == lens
    assert out is not lens  # returns a new list
    assert jitter_boundary_lens(lens, -2) == lens


def test_jitter_boundary_lens_preserves_total_and_nonnegative():
    """Jitter keeps the total length fixed and never produces negative regions."""
    random.seed(0)
    lens = [5, 3, 8, 4, 12, 7, 6]
    total = sum(lens)
    for _ in range(50):
        out = jitter_boundary_lens(lens, 3)
        assert len(out) == len(lens)
        assert sum(out) == total
        assert all(x >= 0 for x in out)


def test_strip_bos_eos():
    """Test strip_bos_eos removes BOS and EOS tokens correctly."""
    text = f"{TOK_BOS}some content{TOK_EOS}"
    assert strip_bos_eos(text) == "some content"

    text = f"{TOK_BOS}some content"
    assert strip_bos_eos(text) == "some content"

    text = f"some content{TOK_EOS}"
    assert strip_bos_eos(text) == "some content"

    text = "some content"
    assert strip_bos_eos(text) == "some content"

    assert strip_bos_eos("") == ""

    text = f"{TOK_BOS}content{TOK_EOS}extra{TOK_EOS}"
    assert strip_bos_eos(text) == "content"

    text = f"{TOK_BOS}{TOK_H}{TOK_SEP}{TOK_HPRED}{TOK_EOS}"
    assert strip_bos_eos(text) == f"{TOK_H}{TOK_SEP}{TOK_HPRED}"


def test_clamp_span():
    """Test clamp_span clamps and validates spans correctly."""
    assert clamp_span(5, 10, 20) == (5, 10)

    assert clamp_span(5, 25, 20) == (5, 20)

    assert clamp_span(25, 30, 20) is None

    assert clamp_span(10, 10, 20) is None

    assert clamp_span(15, 10, 20) is None

    assert clamp_span(5, 10, 0) is None

    assert clamp_span(0, 20, 20) == (0, 20)


def test_count_and_first_pos():
    """Test count_and_first_pos counts tokens and finds first position."""
    ids = [1, 2, 3, 4, 5]
    count, first = count_and_first_pos(ids, 3)
    assert count == 1
    assert first == 2

    ids = [1, 3, 2, 3, 5, 3]
    count, first = count_and_first_pos(ids, 3)
    assert count == 3
    assert first == 1

    ids = [1, 2, 4, 5]
    count, first = count_and_first_pos(ids, 3)
    assert count == 0
    assert first is None

    ids = []
    count, first = count_and_first_pos(ids, 3)
    assert count == 0
    assert first is None

    ids = [3, 1, 2]
    count, first = count_and_first_pos(ids, 3)
    assert count == 1
    assert first == 0


def test_find_block_end():
    """Test find_block_end locates the first stop token."""
    stop_ids = {10, 20, 30}

    ids = [1, 2, 3, 10, 5, 6]
    assert find_block_end(ids, 0, stop_ids) == 3

    ids = [1, 2, 10, 20, 5]
    assert find_block_end(ids, 0, stop_ids) == 2

    ids = [1, 2, 3, 4, 5]
    assert find_block_end(ids, 0, stop_ids) == 5

    ids = [1, 10, 3, 20, 5]
    assert find_block_end(ids, 2, stop_ids) == 3

    ids = [1, 2, 3]
    assert find_block_end(ids, 5, stop_ids) == 3


def test_find_first_n_positions():
    """Test find_first_n_positions finds up to n occurrences."""
    ids = [1, 3, 2, 3, 5, 3, 7]
    positions = find_first_n_positions(ids, 3, 0, 7, 2)
    assert positions == [1, 3]

    positions = find_first_n_positions(ids, 3, 0, 7, 10)
    assert positions == [1, 3, 5]

    positions = find_first_n_positions(ids, 99, 0, 7, 2)
    assert positions == []

    positions = find_first_n_positions(ids, 3, 2, 5, 2)
    assert positions == [3]

    positions = find_first_n_positions(ids, 3, 0, 7, 0)
    assert positions == [1, 3, 5]


def test_validate_pred_tag_counts():
    """Test validate_pred_tag_counts validates HPRED/LPRED counts."""
    validate_pred_tag_counts(1, 1, paired=True)  # Should not raise

    with pytest.raises(ValueError, match="exactly 1 <HPRED> and 1 <LPRED>"):
        validate_pred_tag_counts(0, 1, paired=True)

    with pytest.raises(ValueError, match="exactly 1 <HPRED> and 1 <LPRED>"):
        validate_pred_tag_counts(2, 1, paired=True)

    validate_pred_tag_counts(1, 0, paired=False)

    validate_pred_tag_counts(0, 1, paired=False)

    with pytest.raises(ValueError, match="exactly one of <HPRED> or <LPRED>"):
        validate_pred_tag_counts(1, 1, paired=False)

    with pytest.raises(ValueError, match="exactly one of <HPRED> or <LPRED>"):
        validate_pred_tag_counts(0, 0, paired=False)

    with pytest.raises(ValueError, match="exactly 1 pred tag"):
        validate_pred_tag_counts(2, 0, paired=False)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing span computation."""

    class MockTokenizer:
        def __init__(self):
            self.vocab = {
                TOK_BOS: 1,
                TOK_EOS: 2,
                TOK_H: 3,
                TOK_L: 4,
                TOK_HPRED: 5,
                TOK_LPRED: 6,
                TOK_SEP: 7,
                **{f"<SCHEME_{i}>": 100 + i for i in range(3)},
            }
            # Add alphabet
            for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
                self.vocab[c] = 20 + i

        def get_vocab(self):
            return self.vocab

        def decode(self, ids):
            inv_vocab = {v: k for k, v in self.vocab.items()}
            return "".join(inv_vocab.get(i, f"[{i}]") for i in ids)

    return MockTokenizer()


def test_parse_pred_block_spans(mock_tokenizer):
    """Test parse_pred_block_spans extracts CDR spans correctly."""
    sep_id = mock_tokenizer.vocab[TOK_SEP]
    h_id = mock_tokenizer.vocab[TOK_H]
    stop_ids = {h_id, mock_tokenizer.vocab[TOK_L], mock_tokenizer.vocab[TOK_HPRED]}

    ids = [1, 2, 3, 4, 5, *range(50, 54), sep_id, *range(60, 64), sep_id, *range(70, 74), h_id]
    spans = parse_pred_block_spans(ids, pred_pos=5, sep_id=sep_id, stop_ids=stop_ids, max_len_tokens=100)

    assert spans[0] == (6, 9)  # CDR1: indices 6,7,8 (values 51,52,53)
    assert spans[1] == (10, 14)  # CDR2: indices 10,11,12,13 (values 60,61,62,63)
    assert spans[2] == (15, 19)  # CDR3: indices 15,16,17,18 (values 70,71,72,73)

    spans = parse_pred_block_spans(ids, pred_pos=None, sep_id=sep_id, stop_ids=stop_ids, max_len_tokens=100)
    assert spans == [None, None, None]


def test_validate_pred_block_has_two_seps(mock_tokenizer):
    """Test validate_pred_block_has_two_seps enforces CDR block structure."""
    sep_id = mock_tokenizer.vocab[TOK_SEP]
    hp_id = mock_tokenizer.vocab[TOK_HPRED]
    eos_id = mock_tokenizer.vocab[TOK_EOS]
    stop_ids = {eos_id, hp_id}

    ids = [1, 2, hp_id, 10, sep_id, 20, sep_id, 30, eos_id]
    validate_pred_block_has_two_seps(ids, pred_pos=2, sep_id=sep_id, stop_ids=stop_ids, tokenizer=mock_tokenizer)

    ids = [1, 2, hp_id, 10, sep_id, 20, eos_id]
    with pytest.raises(ValueError, match="Malformed CDR block: expected 2 <SEP>"):
        validate_pred_block_has_two_seps(ids, pred_pos=2, sep_id=sep_id, stop_ids=stop_ids, tokenizer=mock_tokenizer)

    ids = [1, 2, hp_id, 10, 20, eos_id]
    with pytest.raises(ValueError, match="Malformed CDR block: expected 2 <SEP>"):
        validate_pred_block_has_two_seps(ids, pred_pos=2, sep_id=sep_id, stop_ids=stop_ids, tokenizer=mock_tokenizer)

    validate_pred_block_has_two_seps(ids, pred_pos=None, sep_id=sep_id, stop_ids=stop_ids, tokenizer=mock_tokenizer)


@pytest.fixture
def real_tokenizer_fixture(tmp_path):
    """Create a real HuggingFace tokenizer with all special tokens."""
    vocab = {
        TOK_BOS: 0,
        TOK_EOS: 1,
        TOK_H: 2,
        TOK_L: 3,
        TOK_HPRED: 4,
        TOK_LPRED: 5,
        TOK_SEP: 6,
        "<PAD>": 7,
    }

    for _, tok in SCHEME_TOK_BY_NAME.items():
        vocab[tok] = 100 + len(vocab)

    for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY"):
        vocab[aa] = 200 + i

    tokenizer_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="<UNK>"))
    tokenizer_obj.pre_tokenizer = WhitespaceSplit()

    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_obj.save(str(tokenizer_file))

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
    tokenizer.add_special_tokens(
        {
            "bos_token": TOK_BOS,
            "eos_token": TOK_EOS,
            "pad_token": "<PAD>",
            "additional_special_tokens": [TOK_H, TOK_L, TOK_HPRED, TOK_LPRED, TOK_SEP]
            + list(SCHEME_TOK_BY_NAME.values()),
        }
    )

    return tokenizer


def test_compute_cdr_spans_single_heavy_chain(real_tokenizer_fixture):
    """Test compute_cdr_spans_from_token_ids with a single heavy chain rendered text."""
    tokenizer = real_tokenizer_fixture

    segments_h = {
        "fr1": "EVQLVESG",
        "fr2": "MTWVRQAP",
        "fr3": "YYADSVKG",
        "fr4": "WAQGTLVT",
        "cdr1": "GFTFS",
        "cdr2": "INSRG",
        "cdr3": "ARDAYG",
    }

    sample_h = {"chain_type": "H", "segments": segments_h, "scheme": "imgt"}
    rendered = render_single(sample_h, include_scheme=False)

    tokens = []
    i = 0
    while i < len(rendered):
        matched = False
        for special in [TOK_BOS, TOK_EOS, TOK_HPRED, TOK_LPRED, TOK_H, TOK_L, TOK_SEP]:
            if rendered[i:].startswith(special):
                tokens.append(special)
                i += len(special)
                matched = True
                break
        if not matched:
            tokens.append(rendered[i])
            i += 1

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    max_len = len(input_ids)

    spans = compute_cdr_spans_from_token_ids(input_ids, tokenizer, max_len, strict=True)

    assert "H" in spans
    assert "L" in spans
    assert len(spans["H"]) == 3
    assert len(spans["L"]) == 3

    for cdr_span in spans["H"]:
        assert cdr_span is not None
        assert isinstance(cdr_span, tuple)
        assert len(cdr_span) == 2
        start, end = cdr_span
        assert 0 <= start < end <= max_len

    for cdr_span in spans["L"]:
        assert cdr_span is None

    for i, expected in enumerate([segments_h["cdr1"], segments_h["cdr2"], segments_h["cdr3"]]):
        start, end = spans["H"][i]
        cdr_ids = input_ids[start:end]

        cdr_text = "".join(
            [
                tokenizer.convert_ids_to_tokens(id)
                for id in cdr_ids
                if id
                not in [
                    tokenizer.convert_tokens_to_ids(TOK_SEP),
                    tokenizer.convert_tokens_to_ids(TOK_HPRED),
                ]
            ]
        )

        assert expected in cdr_text or cdr_text in expected


def test_compute_cdr_spans_single_light_chain(real_tokenizer_fixture):
    """Test compute_cdr_spans_from_token_ids with a single light chain rendered text."""
    tokenizer = real_tokenizer_fixture

    segments_l = {
        "fr1": "DIQMTQSP",
        "fr2": "WYQQKPGK",
        "fr3": "GVPSRFSG",
        "fr4": "FGQGTKVE",
        "cdr1": "RSSQS",
        "cdr2": "GASSRA",
        "cdr3": "QQYGSS",
    }

    sample_l = {"chain_type": "L", "segments": segments_l, "scheme": "kabat"}
    rendered = render_single(sample_l, include_scheme=True)

    tokens = []
    i = 0
    while i < len(rendered):
        matched = False
        for special in [TOK_BOS, TOK_EOS, TOK_LPRED, TOK_HPRED, TOK_H, TOK_L, TOK_SEP] + list(
            SCHEME_TOK_BY_NAME.values()
        ):
            if rendered[i:].startswith(special):
                tokens.append(special)
                i += len(special)
                matched = True
                break
        if not matched:
            tokens.append(rendered[i])
            i += 1

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    max_len = len(input_ids)

    spans = compute_cdr_spans_from_token_ids(input_ids, tokenizer, max_len, strict=True)

    assert spans["L"][0] is not None
    assert spans["L"][1] is not None
    assert spans["L"][2] is not None

    assert spans["H"][0] is None
    assert spans["H"][1] is None
    assert spans["H"][2] is None


def test_compute_cdr_spans_paired_chains(real_tokenizer_fixture):
    """Test compute_cdr_spans_from_token_ids with paired heavy+light chains."""
    tokenizer = real_tokenizer_fixture

    segments_h = {
        "fr1": "EVQLV",
        "fr2": "MTWVR",
        "fr3": "YYADK",
        "fr4": "WAQGT",
        "cdr1": "GFTF",
        "cdr2": "INSR",
        "cdr3": "ARDY",
    }

    segments_l = {
        "fr1": "DIQMT",
        "fr2": "WYQQK",
        "fr3": "GVPSR",
        "fr4": "FGQGT",
        "cdr1": "RSSQ",
        "cdr2": "GASS",
        "cdr3": "QQYG",
    }

    h_sample = {"chain_type": "H", "segments": segments_h, "scheme": "imgt"}
    l_sample = {"chain_type": "L", "segments": segments_l, "scheme": "imgt"}

    rendered = render_paired(h_sample, l_sample, order="L-first", include_scheme=False)

    tokens = []
    i = 0
    while i < len(rendered):
        matched = False
        for special in [TOK_BOS, TOK_EOS, TOK_HPRED, TOK_LPRED, TOK_H, TOK_L, TOK_SEP]:
            if rendered[i:].startswith(special):
                tokens.append(special)
                i += len(special)
                matched = True
                break
        if not matched:
            tokens.append(rendered[i])
            i += 1

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    max_len = len(input_ids)

    spans = compute_cdr_spans_from_token_ids(input_ids, tokenizer, max_len, strict=True)

    for chain in ["H", "L"]:
        for i in range(3):
            assert spans[chain][i] is not None
            start, end = spans[chain][i]
            assert 0 <= start < end <= max_len


def test_compute_cdr_spans_malformed_input(real_tokenizer_fixture):
    """Test compute_cdr_spans_from_token_ids with malformed inputs in strict mode."""
    tokenizer = real_tokenizer_fixture

    tokens = [TOK_BOS, TOK_H, "A", "B", "C", TOK_SEP, TOK_EOS]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    with pytest.raises(ValueError, match="Inconsistent tags"):
        compute_cdr_spans_from_token_ids(input_ids, tokenizer, len(input_ids), strict=True)

    tokens = [
        TOK_BOS,
        TOK_H,
        TOK_HPRED,
        "A",
        TOK_SEP,
        "B",
        TOK_SEP,
        "C",
        TOK_LPRED,
        "D",
        TOK_SEP,
        "E",
        TOK_SEP,
        "F",
        TOK_EOS,
    ]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    with pytest.raises(ValueError, match="Inconsistent tags"):
        compute_cdr_spans_from_token_ids(input_ids, tokenizer, len(input_ids), strict=True)

    tokens = [TOK_BOS, TOK_H, "A", TOK_HPRED, "C", "D", "E", TOK_EOS]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    with pytest.raises(ValueError, match="Malformed CDR block"):
        compute_cdr_spans_from_token_ids(input_ids, tokenizer, len(input_ids), strict=True)


def test_compute_cdr_spans_non_strict_mode(real_tokenizer_fixture):
    """Test compute_cdr_spans_from_token_ids with non-strict mode for partial sequences."""
    tokenizer = real_tokenizer_fixture

    tokens = [TOK_BOS, TOK_H, "A", TOK_HPRED, "B", TOK_EOS]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    spans = compute_cdr_spans_from_token_ids(input_ids, tokenizer, len(input_ids), strict=False)

    assert "H" in spans
    assert "L" in spans
