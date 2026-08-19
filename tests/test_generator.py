"""Tests for GenCDR generator inference utilities (single-chain, paired, likelihood)."""

import logging
import math
from unittest.mock import MagicMock, patch

import pytest
import torch
from gencdr.generator import (
    GenCDRGenerator,
    IgGenCDRGenerator,
    PerRegionTemperatureProcessor,
    _build_prompt_from_rendered,
    _convert_token_to_id,
    decode_cdrs_with_flags_single_chain,
    decode_paired_cdrs_with_flags,
)
from gencdr.tokenizer import (
    TOK_BOS,
    TOK_EOS,
    TOK_H,
    TOK_HPRED,
    TOK_L,
    TOK_LPRED,
    TOK_PAD,
    TOK_SEP,
)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer with special token mappings."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    def mock_convert(x):
        mapping = {
            TOK_BOS: 1,
            TOK_EOS: 2,
            TOK_H: 3,
            TOK_L: 4,
            TOK_HPRED: 5,
            TOK_LPRED: 6,
            TOK_SEP: 7,
            "<SCHEME_IMGT>": 100,
            "<SCHEME_KABAT>": 101,
            "<SCHEME_CHOTHIA>": 102,
        }
        return mapping.get(x, 99)

    tokenizer.convert_tokens_to_ids = MagicMock(side_effect=mock_convert)

    def mock_encode(text, add_special_tokens=False, return_tensors=None):
        ids = [1, 3, 10, 11, 12, 5, 13, 14, 7, 15, 16, 7, 17, 18, 2]
        if return_tensors == "pt":
            return torch.tensor([ids])
        return ids

    tokenizer.encode = MagicMock(side_effect=mock_encode)

    def mock_batch_encode(texts, add_special_tokens=False, padding=True, truncation=False, return_tensors=None):
        batch_size = len(texts)
        seq_len = 20
        input_ids = torch.randint(1, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        result = {"input_ids": input_ids, "attention_mask": attention_mask}
        return result

    tokenizer.side_effect = mock_batch_encode

    def mock_batch_decode(token_lists, skip_special_tokens=False):
        return [f"{TOK_BOS}{TOK_H}FRAMEWORKS{TOK_HPRED}CDR1{TOK_SEP}CDR2{TOK_SEP}CDR3{TOK_EOS}" for _ in token_lists]

    tokenizer.batch_decode = MagicMock(side_effect=mock_batch_decode)

    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock causal language model."""
    model = MagicMock()
    model.eval = MagicMock(return_value=None)
    model.to = MagicMock(return_value=model)
    model.config = MagicMock()
    model.config.use_cache = True

    def mock_generate(
        input_ids,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        max_new_tokens=256,
        num_return_sequences=1,
        pad_token_id=0,
        eos_token_id=2,
        generator=None,
        logits_processor=None,
    ):
        batch_size = num_return_sequences
        seq_len = input_ids.size(1) + 15
        return torch.randint(1, 100, (batch_size, seq_len))

    model.generate = MagicMock(side_effect=mock_generate)

    def mock_forward(input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        vocab_size = 100
        output = MagicMock()
        output.logits = torch.randn(batch_size, seq_len, vocab_size)
        return output

    model.side_effect = mock_forward

    return model


@pytest.fixture
def generator(tmp_path, mock_tokenizer, mock_model):
    """Create GenCDRGenerator with mocked components."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    with (
        patch("gencdr.generator.PreTrainedTokenizerFast") as mock_tok_class,
        patch("gencdr.generator.AutoModelForCausalLM") as mock_model_class,
    ):
        mock_tok_class.from_pretrained = MagicMock(return_value=mock_tokenizer)
        mock_model_class.from_pretrained = MagicMock(return_value=mock_model)

        gen = GenCDRGenerator(
            model_dir=str(model_dir),
            device="cpu",
            fp=32,
            include_scheme_token=False,
        )

    return gen


@pytest.fixture
def frameworks():
    """Create sample framework regions."""
    return {
        "fr1": "EVQLVESGGGLVKAGGSLRLSCAAS",
        "fr2": "MTWVRQAPGKGLEWVSS",
        "fr3": "YYADSVKGQFTISRDKAKKSLDRQMNSLRGEGTAGYYC",
        "fr4": "WAQGTLVTVSS",
    }


@pytest.fixture
def paired_frameworks():
    """Create sample heavy + light framework regions for paired generation."""
    h_frameworks = {"fr1": "EVQLV", "fr2": "MTWVR", "fr3": "YYADK", "fr4": "WGQGT"}
    l_frameworks = {"fr1": "DIQMT", "fr2": "WYQQK", "fr3": "GVPSR", "fr4": "FGQGT"}
    return h_frameworks, l_frameworks


def test_class_alias_matches():
    """The legacy IgGenCDRGenerator name aliases the unified class."""
    assert IgGenCDRGenerator is GenCDRGenerator


def test_convert_token_to_id_returns_int():
    """Test _convert_token_to_id returns integer for valid token."""
    tokenizer = MagicMock()
    tokenizer.convert_tokens_to_ids = MagicMock(return_value=5)

    result = _convert_token_to_id(tokenizer, "<BOS>")

    assert result == 5
    assert isinstance(result, int)


def test_convert_token_to_id_raises_on_non_int():
    """Test _convert_token_to_id raises RuntimeError for non-integer ID."""
    tokenizer = MagicMock()
    tokenizer.convert_tokens_to_ids = MagicMock(return_value=[5, 6])

    with pytest.raises(RuntimeError, match="not an integer"):
        _convert_token_to_id(tokenizer, "<UNKNOWN>")


def test_build_prompt_from_rendered_no_known_cdrs():
    """Test _build_prompt_from_rendered builds prompt without known CDRs."""
    rendered = f"{TOK_BOS}{TOK_H}FRAMEWORKS{TOK_HPRED}{TOK_SEP}{TOK_SEP}{TOK_SEP}{TOK_EOS}"

    prompt = _build_prompt_from_rendered(rendered, None, None)

    assert prompt.startswith(TOK_BOS + TOK_H)
    assert TOK_HPRED in prompt
    assert prompt.endswith(TOK_HPRED)


def test_build_prompt_from_rendered_with_cdr1():
    """Test _build_prompt_from_rendered includes known CDR1."""
    rendered = f"{TOK_BOS}{TOK_H}FRAMEWORKS{TOK_HPRED}{TOK_SEP}{TOK_SEP}{TOK_SEP}{TOK_EOS}"

    prompt = _build_prompt_from_rendered(rendered, "GFTFS", None)

    assert "GFTFS" in prompt
    assert prompt.endswith(f"GFTFS{TOK_SEP}")


def test_build_prompt_from_rendered_with_cdr1_and_cdr2():
    """Test _build_prompt_from_rendered includes known CDR1 and CDR2."""
    rendered = f"{TOK_BOS}{TOK_H}FRAMEWORKS{TOK_HPRED}{TOK_SEP}{TOK_SEP}{TOK_SEP}{TOK_EOS}"

    prompt = _build_prompt_from_rendered(rendered, "GFTFS", "INSRG")

    assert "GFTFS" in prompt
    assert "INSRG" in prompt
    assert prompt.endswith(f"INSRG{TOK_SEP}")


def test_build_prompt_from_rendered_raises_on_no_pred_tag():
    """Test _build_prompt_from_rendered raises when prediction tag not found."""
    rendered = f"{TOK_BOS}{TOK_H}FRAMEWORKS{TOK_EOS}"

    with pytest.raises(RuntimeError, match="Prediction tag not found"):
        _build_prompt_from_rendered(rendered, None, None)


def test_build_prompt_from_rendered_warns_cdr2_without_cdr1(caplog):
    """Test _build_prompt_from_rendered warns when CDR2 provided without CDR1."""
    caplog.set_level(logging.WARNING)

    rendered = f"{TOK_BOS}{TOK_L}FRAMEWORKS{TOK_LPRED}{TOK_SEP}{TOK_SEP}{TOK_SEP}{TOK_EOS}"

    prompt = _build_prompt_from_rendered(rendered, None, "GASSRA")

    assert any("ignoring" in record.message.lower() and "cdr2" in record.message.lower() for record in caplog.records)
    assert "GASSRA" not in prompt


def test_decode_cdrs_with_flags_valid_structure():
    """Test decode_cdrs_with_flags_single_chain parses valid CDR structure."""
    text = f"{TOK_BOS}{TOK_H}FWS{TOK_HPRED}GFTFS{TOK_SEP}INSRG{TOK_SEP}ARDAY{TOK_EOS}"

    result = decode_cdrs_with_flags_single_chain(text)

    assert result["parsed_ok"] is True
    assert result["cdr1"] == "GFTFS"
    assert result["cdr2"] == "INSRG"
    assert result["cdr3"] == "ARDAY"
    assert result["sep_count"] == 3
    assert result["issues"] == []


def test_decode_cdrs_with_flags_no_pred_tag():
    """Test decode_cdrs_with_flags_single_chain handles missing prediction tag."""
    text = f"{TOK_BOS}{TOK_H}FRAMEWORKS{TOK_EOS}"

    result = decode_cdrs_with_flags_single_chain(text)

    assert result["parsed_ok"] is False
    assert "no_pred" in result["issues"]
    assert result["cdr1"] is None
    assert result["cdr2"] is None
    assert result["cdr3"] is None


def test_decode_cdrs_with_flags_under_sep():
    """Test decode_cdrs_with_flags_single_chain handles too few separators."""
    text = f"{TOK_BOS}{TOK_H}FWS{TOK_HPRED}GFTFS{TOK_SEP}INSRG{TOK_EOS}"

    result = decode_cdrs_with_flags_single_chain(text)

    assert result["parsed_ok"] is False
    assert "under_sep" in result["issues"]
    assert result["sep_count"] == 2


def test_decode_cdrs_with_flags_over_sep():
    """Test decode_cdrs_with_flags_single_chain handles too many separators."""
    text = f"{TOK_BOS}{TOK_H}FWS{TOK_HPRED}A{TOK_SEP}B{TOK_SEP}C{TOK_SEP}D{TOK_EOS}"

    result = decode_cdrs_with_flags_single_chain(text)

    assert result["parsed_ok"] is False
    assert "over_sep" in result["issues"]
    assert result["sep_count"] == 4


def test_decode_cdrs_with_flags_special_in_cdr():
    """Test decode_cdrs_with_flags_single_chain detects special tokens in CDRs."""
    text = f"{TOK_BOS}{TOK_H}FWS{TOK_HPRED}GF<H>FS{TOK_SEP}INSRG{TOK_SEP}ARDAY<L>{TOK_EOS}"

    result = decode_cdrs_with_flags_single_chain(text)

    assert result["parsed_ok"] is False
    assert "special_in_cdr1" in result["issues"]
    assert "special_in_cdr3" in result["issues"]


def test_decode_cdrs_with_flags_known_cdr1():
    """Test decode_cdrs_with_flags_single_chain respects known CDR1."""
    text = f"{TOK_BOS}{TOK_H}FWS{TOK_HPRED}GFTFS{TOK_SEP}INSRG{TOK_SEP}ARDAY{TOK_EOS}"

    result = decode_cdrs_with_flags_single_chain(text, known_cdr1="KNOWN1")

    assert result["cdr1"] == "KNOWN1"
    assert result["cdr2"] == "INSRG"


def test_decode_cdrs_with_flags_known_cdr2():
    """Test decode_cdrs_with_flags_single_chain respects known CDR2."""
    text = f"{TOK_BOS}{TOK_H}FWS{TOK_HPRED}GFTFS{TOK_SEP}INSRG{TOK_SEP}ARDAY{TOK_EOS}"

    result = decode_cdrs_with_flags_single_chain(text, known_cdr1="GFTFS", known_cdr2="KNOWN2")

    assert result["cdr1"] == "GFTFS"
    assert result["cdr2"] == "KNOWN2"


def test_generator_init_raises_on_missing_dir():
    """Test GenCDRGenerator raises FileNotFoundError for missing model directory."""
    with pytest.raises(FileNotFoundError, match="Model directory not found"):
        GenCDRGenerator(model_dir="/nonexistent/path")


def _mock_tok_with_specials():
    """Build a bare mock tokenizer that maps the special tokens to fixed ids."""
    mock_tok = MagicMock()
    mock_tok.pad_token_id = 0
    mock_tok.convert_tokens_to_ids = MagicMock(
        side_effect=lambda x: {
            TOK_BOS: 1,
            TOK_EOS: 2,
            TOK_H: 3,
            TOK_L: 4,
            TOK_HPRED: 5,
            TOK_LPRED: 6,
            TOK_SEP: 7,
            "<SCHEME_IMGT>": 100,
            "<SCHEME_KABAT>": 101,
            "<SCHEME_CHOTHIA>": 102,
        }.get(x, 99)
    )
    return mock_tok


def test_generator_init_sets_device_cuda():
    """Test GenCDRGenerator sets device to cuda when available."""
    mock_tok = _mock_tok_with_specials()

    mock_mdl = MagicMock()
    mock_mdl.eval = MagicMock(return_value=None)
    mock_mdl.to = MagicMock(return_value=mock_mdl)
    mock_mdl.config = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("gencdr.generator.PreTrainedTokenizerFast") as mock_tok_class,
        patch("gencdr.generator.AutoModelForCausalLM") as mock_model_class,
        patch("os.path.isdir", return_value=True),
    ):
        mock_tok_class.from_pretrained = MagicMock(return_value=mock_tok)
        mock_model_class.from_pretrained = MagicMock(return_value=mock_mdl)

        gen = GenCDRGenerator(model_dir="/fake/path", device=None)

        assert gen.device == "cuda"


def test_generator_init_sets_device_mps():
    """Test GenCDRGenerator sets device to mps when cuda unavailable but mps available."""
    mock_tok = _mock_tok_with_specials()

    mock_mdl = MagicMock()
    mock_mdl.eval = MagicMock(return_value=None)
    mock_mdl.to = MagicMock(return_value=mock_mdl)
    mock_mdl.config = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.backends.mps.is_built", return_value=True),
        patch("gencdr.generator.PreTrainedTokenizerFast") as mock_tok_class,
        patch("gencdr.generator.AutoModelForCausalLM") as mock_model_class,
        patch("os.path.isdir", return_value=True),
    ):
        mock_tok_class.from_pretrained = MagicMock(return_value=mock_tok)
        mock_model_class.from_pretrained = MagicMock(return_value=mock_mdl)

        gen = GenCDRGenerator(model_dir="/fake/path", device=None)

        assert gen.device == "mps"


def test_generator_init_sets_fp16_flag():
    """Test GenCDRGenerator sets fp16 flag correctly."""
    mock_tok = _mock_tok_with_specials()

    mock_mdl = MagicMock()
    mock_mdl.eval = MagicMock(return_value=None)
    mock_mdl.to = MagicMock(return_value=mock_mdl)
    mock_mdl.config = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("gencdr.generator.PreTrainedTokenizerFast") as mock_tok_class,
        patch("gencdr.generator.AutoModelForCausalLM") as mock_model_class,
        patch("os.path.isdir", return_value=True),
    ):
        mock_tok_class.from_pretrained = MagicMock(return_value=mock_tok)
        mock_model_class.from_pretrained = MagicMock(return_value=mock_mdl)

        gen = GenCDRGenerator(model_dir="/fake/path", device="cuda", fp=16)

        assert gen.use_fp16 is True


def test_generator_init_raises_on_missing_pad_token(tmp_path, mock_model):
    """Test GenCDRGenerator raises when tokenizer has no pad_token_id."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    tokenizer = MagicMock()
    tokenizer.pad_token_id = None
    tokenizer.convert_tokens_to_ids = MagicMock(return_value=1)

    with (
        patch("gencdr.generator.PreTrainedTokenizerFast") as mock_tok_class,
        patch("gencdr.generator.AutoModelForCausalLM") as mock_model_class,
    ):
        mock_tok_class.from_pretrained = MagicMock(return_value=tokenizer)
        mock_model_class.from_pretrained = MagicMock(return_value=mock_model)

        with pytest.raises(RuntimeError, match="pad token"):
            GenCDRGenerator(model_dir=str(model_dir))


def test_from_pretrained_resolves_and_constructs(tmp_path, mock_tokenizer, mock_model):
    """from_pretrained resolves the name via resolve_checkpoint then builds the generator."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    with (
        patch("gencdr.generator.resolve_checkpoint", return_value=model_dir) as mock_resolve,
        patch("gencdr.generator.PreTrainedTokenizerFast") as mock_tok_class,
        patch("gencdr.generator.AutoModelForCausalLM") as mock_model_class,
    ):
        mock_tok_class.from_pretrained = MagicMock(return_value=mock_tokenizer)
        mock_model_class.from_pretrained = MagicMock(return_value=mock_model)

        gen = GenCDRGenerator.from_pretrained("iggencdr", device="cpu", fp=32)

    mock_resolve.assert_called_once_with("iggencdr")
    assert isinstance(gen, GenCDRGenerator)


def test_generator_render_single_text(generator):
    """Test _render_single_text creates rendered text."""
    with patch("gencdr.generator.render_single") as mock_render:
        mock_render.return_value = "rendered_text"

        result = generator._render_single_text(
            chain_type="H", fr1="FR1", fr2="FR2", fr3="FR3", fr4="FR4", cdr1="C1", cdr2="C2", cdr3="C3", scheme="imgt"
        )

        assert result == "rendered_text"
        mock_render.assert_called_once()


def test_generator_prompt_for_cdr_generation(generator, frameworks):
    """Test _prompt_for_cdr_generation builds correct prompt."""
    with patch.object(generator, "_render_single_text") as mock_render:
        mock_render.return_value = f"{TOK_BOS}{TOK_H}FWS{TOK_HPRED}{TOK_EOS}"

        prompt = generator._prompt_for_cdr_generation(
            chain_type="H", frameworks=frameworks, known_cdr1="GFTFS", known_cdr2=None, scheme="imgt"
        )

        assert TOK_HPRED in prompt
        assert "GFTFS" in prompt


def test_generator_prompt_for_whole_chain(generator):
    """Test _prompt_for_whole_chain creates minimal prompt."""
    prompt = generator._prompt_for_whole_chain("H", scheme=None)

    assert prompt.startswith(TOK_BOS)
    assert TOK_H in prompt
    assert len(prompt) < 20


def test_generator_prompt_for_whole_chain_with_scheme(generator):
    """Test _prompt_for_whole_chain includes scheme token when configured."""
    gen_with_scheme = generator
    gen_with_scheme.include_scheme_token = True

    prompt = gen_with_scheme._prompt_for_whole_chain("L", scheme="imgt")

    assert TOK_BOS in prompt
    assert TOK_L in prompt


def test_generate_cdrs_from_frameworks(generator, frameworks):
    """Test generate_cdrs_from_frameworks returns CDR results."""
    with patch.object(generator, "_prompt_for_cdr_generation") as mock_prompt:
        mock_prompt.return_value = f"{TOK_BOS}{TOK_H}prompt{TOK_HPRED}"

        results = generator.generate_cdrs_from_frameworks(
            chain_type="H", frameworks=frameworks, n_samples=2, temperature=1.0, scheme="imgt"
        )

        assert len(results) == 2
        assert all("cdr1" in r for r in results)
        assert all("cdr2" in r for r in results)
        assert all("cdr3" in r for r in results)
        assert all("parsed_ok" in r for r in results)


def test_generate_cdrs_from_frameworks_with_known_cdrs(generator, frameworks):
    """Test generate_cdrs_from_frameworks handles known CDR1 and CDR2."""
    with patch.object(generator, "_prompt_for_cdr_generation") as mock_prompt:
        mock_prompt.return_value = f"{TOK_BOS}{TOK_H}prompt{TOK_HPRED}"

        results = generator.generate_cdrs_from_frameworks(
            chain_type="H", frameworks=frameworks, n_samples=1, known_cdr1="GFTFS", known_cdr2="INSRG", scheme="imgt"
        )

        assert len(results) == 1


@pytest.mark.parametrize("temperature,top_p", [(0.8, 0.9), (1.0, 0.95), (1.5, 1.0)])
def test_generate_cdrs_respects_sampling_params(generator, frameworks, temperature, top_p):
    """Test generate_cdrs_from_frameworks uses specified sampling parameters."""
    with patch.object(generator, "_prompt_for_cdr_generation") as mock_prompt:
        mock_prompt.return_value = f"{TOK_BOS}{TOK_H}prompt{TOK_HPRED}"

        generator.generate_cdrs_from_frameworks(
            chain_type="H", frameworks=frameworks, n_samples=1, temperature=temperature, top_p=top_p, scheme="imgt"
        )

        generator.model.generate.assert_called_once()
        call_kwargs = generator.model.generate.call_args[1]
        assert call_kwargs["temperature"] == temperature
        assert call_kwargs["top_p"] == top_p


def test_generate_cdrs_batch(generator, frameworks):
    """Test generate_cdrs_batch processes multiple inputs."""
    inputs = [
        {"chain_type": "H", "frameworks": frameworks, "cdr1": None, "cdr2": None, "scheme": "imgt"},
        {"chain_type": "L", "frameworks": frameworks, "cdr1": "RSSQS", "cdr2": None, "scheme": "kabat"},
    ]

    with patch.object(generator, "generate_cdrs_from_frameworks") as mock_gen:
        mock_gen.return_value = [{"cdr1": "A", "cdr2": "B", "cdr3": "C", "parsed_ok": True}]

        results = generator.generate_cdrs_batch(inputs, n_samples_per_input=1)

        assert len(results) == 2
        assert mock_gen.call_count == 2


def test_generate_whole_chains(generator):
    """Test generate_whole_chains returns decoded sequences."""
    with patch.object(generator, "_prompt_for_whole_chain") as mock_prompt:
        mock_prompt.return_value = f"{TOK_BOS}{TOK_H}"

        chains = generator.generate_whole_chains(chain_type="H", n_samples=3, scheme="imgt", temperature=1.2)

        assert len(chains) == 3
        assert all(isinstance(c, str) for c in chains)


def test_validate_segment_item_valid(generator):
    """Test _validate_segment_item accepts valid item."""
    item = {"chain_type": "H", "fr1": "A", "cdr1": "B", "fr2": "C", "cdr2": "D", "fr3": "E", "cdr3": "F", "fr4": "G"}

    chain_type, segments = generator._validate_segment_item(item)

    assert chain_type == "H"
    assert segments["fr1"] == "A"
    assert segments["cdr1"] == "B"


def test_validate_segment_item_raises_on_missing_key(generator):
    """Test _validate_segment_item raises KeyError for missing keys."""
    item = {"chain_type": "H", "fr1": "A"}

    with pytest.raises(KeyError, match="Missing required keys"):
        generator._validate_segment_item(item)


def test_validate_segment_item_raises_on_invalid_chain_type(generator):
    """Test _validate_segment_item raises ValueError for invalid chain_type."""
    item = {"chain_type": "X", "fr1": "A", "cdr1": "B", "fr2": "C", "cdr2": "D", "fr3": "E", "cdr3": "F", "fr4": "G"}

    with pytest.raises(ValueError, match="chain_type must be"):
        generator._validate_segment_item(item)


def test_encode_texts(generator):
    """Test _encode_texts encodes text batch with padding."""
    texts = ["text1", "text2", "text3"]

    input_ids, attention_mask = generator._encode_texts(texts)

    assert input_ids.shape[0] == 3
    assert attention_mask.shape[0] == 3
    assert input_ids.shape == attention_mask.shape


def test_masked_ll_for_batch(generator):
    """Test _masked_ll_for_batch computes log-likelihoods."""
    input_ids = torch.randint(1, 100, (2, 20))
    attention_mask = torch.ones(2, 20)

    lls = generator._masked_ll_for_batch(input_ids, attention_mask, reduction="mean")

    assert len(lls) == 2
    assert all(isinstance(ll, float) for ll in lls)


def test_masked_ll_for_batch_sum_reduction(generator):
    """Test _masked_ll_for_batch with sum reduction."""
    input_ids = torch.randint(1, 100, (2, 20))
    attention_mask = torch.ones(2, 20)

    lls = generator._masked_ll_for_batch(input_ids, attention_mask, reduction="sum")

    assert len(lls) == 2


def test_masked_ll_for_batch_raises_on_zero_valid_tokens(generator):
    """Test _masked_ll_for_batch raises when no valid tokens after masking."""
    input_ids = torch.full((1, 10), generator.pad_id)
    attention_mask = torch.zeros(1, 10)

    with pytest.raises(RuntimeError, match="zero valid tokens"):
        generator._masked_ll_for_batch(input_ids, attention_mask)


def test_log_likelihood_from_segments(generator):
    """Test log_likelihood_from_segments computes likelihood for single sample."""
    item = {
        "chain_type": "H",
        "fr1": "EVQLVES",
        "cdr1": "GFTFS",
        "fr2": "MTWVRQ",
        "cdr2": "INSRG",
        "fr3": "YYADSVK",
        "cdr3": "ARDAY",
        "fr4": "WAQGTL",
    }

    with (
        patch.object(generator, "_render_single_text") as mock_render,
        patch.object(generator, "_encode_texts") as mock_encode,
        patch.object(generator, "_masked_ll_for_batch") as mock_ll,
    ):
        mock_render.return_value = "rendered"
        mock_encode.return_value = (torch.randint(1, 100, (1, 20)), torch.ones(1, 20))
        mock_ll.return_value = [-2.5]

        ll = generator.log_likelihood_from_segments(item, scheme="imgt", reduction="mean")

        assert isinstance(ll, float)
        assert math.isclose(ll, -2.5)


def test_batch_log_likelihood_from_segments(generator):
    """Test batch_log_likelihood_from_segments processes multiple items."""
    items = [
        {"chain_type": "H", "fr1": "A", "cdr1": "B", "fr2": "C", "cdr2": "D", "fr3": "E", "cdr3": "F", "fr4": "G"},
        {"chain_type": "L", "fr1": "X", "cdr1": "Y", "fr2": "Z", "cdr2": "W", "fr3": "V", "cdr3": "U", "fr4": "T"},
    ]

    with (
        patch.object(generator, "_render_single_text") as mock_render,
        patch.object(generator, "_encode_texts") as mock_encode,
        patch.object(generator, "_masked_ll_for_batch") as mock_ll,
    ):
        mock_render.return_value = "rendered"
        mock_encode.return_value = (torch.randint(1, 100, (2, 20)), torch.ones(2, 20))
        mock_ll.return_value = [-2.1, -2.3]

        lls = generator.batch_log_likelihood_from_segments(items, scheme="imgt")

        assert len(lls) == 2
        assert all(isinstance(ll, float) for ll in lls)


def test_batch_log_likelihood_empty_input(generator):
    """Test batch_log_likelihood_from_segments handles empty input."""
    lls = generator.batch_log_likelihood_from_segments([], scheme="imgt")

    assert lls == []


def test_log_likelihood_from_lengths(generator):
    """Test log_likelihood_from_lengths computes likelihood from raw sequence."""
    sequence = "EVQLVESGFTFSMTWVRQINSRGYYADSVKARDAYWAQGTL"
    lens_by_scheme = {"imgt": [7, 5, 6, 5, 7, 5, 7]}

    with (
        patch("gencdr.generator.slice_segments") as mock_slice,
        patch.object(generator, "log_likelihood_from_segments") as mock_ll,
    ):
        mock_slice.return_value = {
            "fr1": "EVQLVES",
            "cdr1": "GFTFS",
            "fr2": "MTWVRQ",
            "cdr2": "INSRG",
            "fr3": "YYADSVK",
            "cdr3": "ARDAY",
            "fr4": "WAQGTL",
        }
        mock_ll.return_value = -2.8

        ll = generator.log_likelihood_from_lengths(
            chain_type="H", sequence=sequence, lens_by_scheme=lens_by_scheme, scheme="imgt"
        )

        assert isinstance(ll, float)
        mock_slice.assert_called_once()


@pytest.mark.parametrize("chain_type", ["H", "L"])
def test_generate_cdrs_supports_both_chains(generator, frameworks, chain_type):
    """Test generate_cdrs_from_frameworks works for both heavy and light chains."""
    with patch.object(generator, "_prompt_for_cdr_generation") as mock_prompt:
        mock_prompt.return_value = f"{TOK_BOS}{TOK_H if chain_type == 'H' else TOK_L}prompt"

        results = generator.generate_cdrs_from_frameworks(
            chain_type=chain_type, frameworks=frameworks, n_samples=1, scheme="imgt"
        )

        assert len(results) == 1


@pytest.mark.parametrize("scheme", ["imgt", "kabat", "chothia"])
def test_generate_cdrs_supports_all_schemes(generator, frameworks, scheme):
    """Test generate_cdrs_from_frameworks works with all numbering schemes."""
    with patch.object(generator, "_prompt_for_cdr_generation") as mock_prompt:
        mock_prompt.return_value = f"{TOK_BOS}{TOK_H}prompt{TOK_HPRED}"

        results = generator.generate_cdrs_from_frameworks(
            chain_type="H", frameworks=frameworks, n_samples=1, scheme=scheme
        )

        assert len(results) == 1


def test_masked_ll_from_include_mask_mean_and_sum(generator):
    """_masked_ll_from_include_mask returns negative mean/sum of masked per-position loss."""
    per_pos_loss = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    include_mask = torch.tensor([[True, False, True, False]])

    ll_sum = generator._masked_ll_from_include_mask(
        per_pos_loss=per_pos_loss, include_mask=include_mask, reduction="sum", mask_name="m"
    )
    ll_mean = generator._masked_ll_from_include_mask(
        per_pos_loss=per_pos_loss, include_mask=include_mask, reduction="mean", mask_name="m"
    )

    assert ll_sum.item() == pytest.approx(-(1.0 + 3.0))
    assert ll_mean.item() == pytest.approx(-(1.0 + 3.0) / 2)


def test_masked_ll_from_include_mask_raises_on_zero_count(generator):
    """A mask selecting no tokens raises a descriptive RuntimeError."""
    per_pos_loss = torch.tensor([[1.0, 2.0, 3.0]])
    include_mask = torch.zeros_like(per_pos_loss, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="zero valid tokens"):
        generator._masked_ll_from_include_mask(
            per_pos_loss=per_pos_loss, include_mask=include_mask, reduction="mean", mask_name="cdr"
        )


def test_batch_log_likelihood_regions_empty(generator):
    """Region split returns empty lists for empty input."""
    assert generator.batch_log_likelihood_regions_from_segments([]) == {
        "full": [],
        "framework": [],
        "cdr": [],
    }


def test_batch_log_likelihood_cdr3_split_empty(generator):
    """CDR3 split returns empty lists for empty input."""
    assert generator.batch_log_likelihood_cdr3_split_from_segments([]) == {
        "no_cdr3": [],
        "cdr3": [],
    }


# Crafted single-sample batch used to exercise the shared forward-pass helper.
# input_ids layout: BOS H aa aa HPRED aa SEP aa SEP aa EOS
_CRAFTED_IDS = torch.tensor([[1, 3, 10, 10, 5, 10, 7, 10, 7, 10, 2]])


def _configure_crafted_tokens(generator):
    """Point the generator's token ids at the crafted layout above."""
    generator.aa_token_ids = [10]
    generator.hpred_id = 5
    generator.lpred_id = 6
    generator.sep_id = 7


def test_batch_forward_region_tensors_identifies_pred_and_aa_mask(generator):
    """The shared helper locates the single prediction tag and amino-acid tokens."""
    _configure_crafted_tokens(generator)
    attn = torch.ones_like(_CRAFTED_IDS)

    with patch.object(generator, "_encode_texts", return_value=(_CRAFTED_IDS, attn)):
        per_pos_loss, aa_mask, pred_pos, _, tgt_pos = generator._batch_forward_region_tensors(["x"])

    assert pred_pos.tolist() == [4]
    # Shifted labels: [3,10,10,5,10,7,10,7,10,2] -> aa where label == 10.
    assert aa_mask[0].tolist() == [False, True, True, False, True, False, True, False, True, False]
    assert per_pos_loss.shape == (1, 10)
    assert tgt_pos[0].tolist() == list(range(1, 11))


def test_batch_log_likelihood_regions_mask_split(generator):
    """Framework tokens precede the prediction tag; CDR tokens follow it."""
    _configure_crafted_tokens(generator)
    attn = torch.ones_like(_CRAFTED_IDS)
    captured = {}

    def fake_mask_ll(per_pos_loss, include_mask, reduction, mask_name):
        captured[mask_name] = include_mask[0].tolist()
        return torch.zeros(include_mask.size(0))

    with (
        patch.object(generator, "_encode_texts", return_value=(_CRAFTED_IDS, attn)),
        patch.object(generator, "_render_segment_texts", return_value=["x"]),
        patch.object(generator, "_masked_ll_from_include_mask", side_effect=fake_mask_ll),
    ):
        generator.batch_log_likelihood_regions_from_segments([{"dummy": True}])

    assert captured["full"] == [False, True, True, False, True, False, True, False, True, False]
    assert captured["framework"] == [False, True, True, False, False, False, False, False, False, False]
    assert captured["cdr"] == [False, False, False, False, True, False, True, False, True, False]


def test_batch_log_likelihood_cdr3_split_mask(generator):
    """CDR3 tokens are those amino acids following two <SEP> tokens after the tag."""
    _configure_crafted_tokens(generator)
    attn = torch.ones_like(_CRAFTED_IDS)
    captured = {}

    def fake_mask_ll(per_pos_loss, include_mask, reduction, mask_name):
        captured[mask_name] = include_mask[0].tolist()
        return torch.zeros(include_mask.size(0))

    with (
        patch.object(generator, "_encode_texts", return_value=(_CRAFTED_IDS, attn)),
        patch.object(generator, "_render_segment_texts", return_value=["x"]),
        patch.object(generator, "_masked_ll_from_include_mask", side_effect=fake_mask_ll),
    ):
        generator.batch_log_likelihood_cdr3_split_from_segments([{"dummy": True}])

    # Only the final amino acid (after two <SEP> tokens) is CDR3.
    assert captured["cdr3"] == [False, False, False, False, False, False, False, False, True, False]
    assert captured["no_cdr3"] == [False, True, True, False, True, False, True, False, False, False]


# ---------------------------------------------------------------------------
# Paired generation (p-IgGenCDR)
# ---------------------------------------------------------------------------


def _valid_paired_text(order: str) -> str:
    """Build a well-formed decoded paired output for the given chain order."""
    l_frs = f"{TOK_L}DIQ{TOK_SEP}WYQ{TOK_SEP}GVP{TOK_SEP}FGQ"
    h_frs = f"{TOK_H}EVQ{TOK_SEP}MTW{TOK_SEP}YYA{TOK_SEP}WGQ"
    l_cdrs = f"{TOK_LPRED}RSSQS{TOK_SEP}GASSRA{TOK_SEP}QQYGSS"
    h_cdrs = f"{TOK_HPRED}GFTFS{TOK_SEP}INSRG{TOK_SEP}ARDAY"
    if order == "L-first":
        core = l_frs + h_frs + l_cdrs + h_cdrs
    else:
        core = h_frs + l_frs + h_cdrs + l_cdrs
    return f"{TOK_BOS}{core}{TOK_EOS}"


@pytest.mark.parametrize("order", ["L-first", "H-first"])
def test_decode_paired_cdrs_valid(order):
    """Both chains parse cleanly from a well-formed paired output, order-agnostically."""
    result = decode_paired_cdrs_with_flags(_valid_paired_text(order), order=order)

    assert result["order"] == order
    assert result["parsed_ok"] is True
    assert result["issues"] == []
    assert result["H"]["cdr1"] == "GFTFS"
    assert result["H"]["cdr2"] == "INSRG"
    assert result["H"]["cdr3"] == "ARDAY"
    assert result["L"]["cdr1"] == "RSSQS"
    assert result["L"]["cdr2"] == "GASSRA"
    assert result["L"]["cdr3"] == "QQYGSS"


def test_decode_paired_cdrs_missing_hpred():
    """A missing <HPRED> block flags the heavy chain and fails parsing."""
    text = f"{TOK_BOS}{TOK_L}DIQ{TOK_H}EVQ{TOK_LPRED}RSSQS{TOK_SEP}GASSRA{TOK_SEP}QQYGSS{TOK_EOS}"
    result = decode_paired_cdrs_with_flags(text, order="L-first")

    assert result["parsed_ok"] is False
    assert "H_no_pred" in result["issues"]
    assert result["H"]["cdr1"] is None
    # Light chain still parses.
    assert result["L"]["cdr1"] == "RSSQS"


def test_decode_paired_cdrs_under_sep_light():
    """Too few separators in the light block flags L_under_sep."""
    text = (
        f"{TOK_BOS}{TOK_L}DIQ{TOK_H}EVQ"
        f"{TOK_LPRED}RSSQS{TOK_SEP}GASSRA"  # only 1 SEP -> 2 parts
        f"{TOK_HPRED}GFTFS{TOK_SEP}INSRG{TOK_SEP}ARDAY{TOK_EOS}"
    )
    result = decode_paired_cdrs_with_flags(text, order="L-first")

    assert result["parsed_ok"] is False
    assert "L_under_sep" in result["issues"]
    assert result["L"]["sep_count"] == 2
    # Heavy chain still parses.
    assert result["H"]["cdr3"] == "ARDAY"


def test_decode_paired_cdrs_special_in_heavy_cdr():
    """A stray non-boundary special token inside a heavy CDR is flagged per-chain.

    A ``<PAD>`` is used rather than a chain tag: chain tags (``<H>``/``<L>``) are block
    boundaries for the paired decoder, so they truncate the block instead of surviving inside
    a CDR. A ``<PAD>`` is neither a boundary nor the ``<SEP>`` delimiter, so it stays in the
    CDR1 part and triggers the special-token flag.
    """
    text = (
        f"{TOK_BOS}{TOK_L}DIQ{TOK_H}EVQ"
        f"{TOK_LPRED}RSSQS{TOK_SEP}GASSRA{TOK_SEP}QQYGSS"
        f"{TOK_HPRED}GF{TOK_PAD}FS{TOK_SEP}INSRG{TOK_SEP}ARDAY{TOK_EOS}"
    )
    result = decode_paired_cdrs_with_flags(text, order="L-first")

    assert result["parsed_ok"] is False
    assert "H_special_in_cdr1" in result["issues"]
    assert result["H"]["cdr1"] is None
    # The remaining heavy CDRs still parse cleanly.
    assert result["H"]["cdr2"] == "INSRG"
    assert result["H"]["cdr3"] == "ARDAY"


def test_prompt_for_paired_ends_at_lpred(generator, paired_frameworks):
    """L-first paired prompt ends at <LPRED> and excludes the heavy pred block."""
    h_fr, l_fr = paired_frameworks
    prompt = generator._prompt_for_paired_cdr_generation(h_fr, l_fr, order="L-first")

    assert prompt.endswith(TOK_LPRED)
    assert TOK_HPRED not in prompt
    assert prompt.count(TOK_LPRED) == 1


def test_prompt_for_paired_ends_at_hpred(generator, paired_frameworks):
    """H-first paired prompt ends at <HPRED> and excludes the light pred block."""
    h_fr, l_fr = paired_frameworks
    prompt = generator._prompt_for_paired_cdr_generation(h_fr, l_fr, order="H-first")

    assert prompt.endswith(TOK_HPRED)
    assert TOK_LPRED not in prompt
    assert prompt.count(TOK_HPRED) == 1


def test_generate_paired_cdrs_from_frameworks(generator, paired_frameworks):
    """generate_paired_cdrs_from_frameworks decodes both chains from a paired output."""
    h_fr, l_fr = paired_frameworks
    valid = _valid_paired_text("L-first")

    with patch.object(generator.tokenizer, "batch_decode", return_value=[valid, valid]):
        results = generator.generate_paired_cdrs_from_frameworks(
            h_frameworks=h_fr, l_frameworks=l_fr, n_samples=2, order="L-first"
        )

    assert len(results) == 2
    for r in results:
        assert r["parsed_ok"] is True
        assert r["order"] == "L-first"
        assert r["H"]["cdr3"] == "ARDAY"
        assert r["L"]["cdr1"] == "RSSQS"


def test_generate_paired_cdrs_respects_sampling_params(generator, paired_frameworks):
    """generate_paired_cdrs_from_frameworks forwards temperature/top_p to model.generate."""
    h_fr, l_fr = paired_frameworks
    valid = _valid_paired_text("L-first")

    with patch.object(generator.tokenizer, "batch_decode", return_value=[valid]):
        generator.generate_paired_cdrs_from_frameworks(
            h_frameworks=h_fr, l_frameworks=l_fr, n_samples=1, temperature=0.7, top_p=0.9
        )

    generator.model.generate.assert_called_once()
    call_kwargs = generator.model.generate.call_args[1]
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["top_p"] == 0.9
    assert call_kwargs["num_return_sequences"] == 1


class TestPerRegionTemperatureProcessor:
    """Test per-region (CDR1/2/3) temperature scaling by SEP count since the prompt."""

    def _apply(self, proc, n_seps):
        # Build a completion of n_seps separator tokens followed by one residue token
        # being emitted, appended after a three-token prompt (prompt length three).
        sep = 7
        gen = [sep] * n_seps + [10]
        ids = torch.tensor([[1, 2, 3] + gen])
        scores = torch.ones((1, 5))
        return proc(ids, scores.clone())

    def test_region_selected_by_sep_count(self):
        """0 SEPs -> CDR1 temp, 1 -> CDR2 temp, >=2 -> CDR3 temp (logits /= temp)."""
        proc = PerRegionTemperatureProcessor(
            sep_id=7,
            prompt_len=3,
            base_temperature=1.0,
            cdr1_temperature=2.0,
            cdr2_temperature=4.0,
            cdr3_temperature=5.0,
        )
        assert torch.allclose(self._apply(proc, 0), torch.full((1, 5), 1 / 2.0))
        assert torch.allclose(self._apply(proc, 1), torch.full((1, 5), 1 / 4.0))
        assert torch.allclose(self._apply(proc, 2), torch.full((1, 5), 1 / 5.0))
        assert torch.allclose(self._apply(proc, 3), torch.full((1, 5), 1 / 5.0))

    def test_unset_regions_fall_back_to_base(self):
        """Regions with no explicit temp use base_temperature."""
        proc = PerRegionTemperatureProcessor(sep_id=7, prompt_len=3, base_temperature=1.0, cdr3_temperature=2.0)
        assert torch.allclose(self._apply(proc, 0), torch.full((1, 5), 1.0))  # CDR1 uses base temp
        assert torch.allclose(self._apply(proc, 1), torch.full((1, 5), 1.0))  # CDR2 uses base temp
        assert torch.allclose(self._apply(proc, 2), torch.full((1, 5), 0.5))  # CDR3 uses temp 2.0
