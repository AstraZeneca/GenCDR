"""Tests for the weighted DPO alignment code (loss, dataset/collator, trainer, aligner).

The loss and dataset tests only need torch + transformers. The trainer/aligner tests
require the ``align`` extra (pytorch-lightning + torchmetrics) and are skipped if it is
not installed.
"""

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from gencdr.alignment.dataset import FrameworkCDRRewardCollator, FrameworkCDRRewardDataset
from gencdr.alignment.loss import completion_mask_unshifted, grouped_weighted_dpo_loss
from gencdr.tokenizer import TOK_EOS, TOK_H, TOK_HPRED, TOK_LPRED, save_tokenizer_dir
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    """Build and load the real GenCDR tokenizer once for the module."""
    tok_dir = tmp_path_factory.mktemp("tok")
    save_tokenizer_dir(tok_dir)
    return PreTrainedTokenizerFast.from_pretrained(str(tok_dir), local_files_only=True)


def _single_df(n: int = 6) -> pd.DataFrame:
    """Build a small single-chain reward table with two scaffold sources."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"s{i}",
                "meta_framework_source": "S1" if i % 2 == 0 else "S2",
                "meta_chain_type": "H",
                "meta_scheme": "imgt",
                "meta_fr1": "EVQLVESGGG",
                "meta_fr2": "WVRQAPGK",
                "meta_fr3": "RFTISRDNS",
                "meta_fr4": "WGQGTLVTVSS",
                "meta_cdr1": "GFTFSSYA",
                "meta_cdr2": "ISGSGGST",
                "meta_cdr3": f"ARDL{'G' * (i + 1)}",
                "reward": float(i) / n,
            }
        )
    return pd.DataFrame(rows)


def _paired_df(n: int = 6) -> pd.DataFrame:
    """Build a small paired reward table with two scaffold sources."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"p{i}",
                "meta_framework_source": "P1" if i % 2 == 0 else "P2",
                "meta_scheme": "imgt",
                "meta_h_fr1": "EVQLVESGGG",
                "meta_h_fr2": "WVRQAPGK",
                "meta_h_fr3": "RFTISRDNS",
                "meta_h_fr4": "WGQGTLVTVSS",
                "meta_h_cdr1": "GFTFSSYA",
                "meta_h_cdr2": "ISGSGGST",
                "meta_h_cdr3": f"ARDL{'G' * (i + 1)}",
                "meta_l_fr1": "DIQMTQSPSS",
                "meta_l_fr2": "WYQQKP",
                "meta_l_fr3": "GVPSRFSGS",
                "meta_l_fr4": "FGQGTKVEIK",
                "meta_l_cdr1": "QSISSY",
                "meta_l_cdr2": "AAS",
                "meta_l_cdr3": f"QQSY{'S' * (i + 1)}",
                "reward": float(i) / n,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# completion_mask_unshifted
# ---------------------------------------------------------------------------
def test_completion_mask_basic():
    """A well-formed span selects exactly [start, end)."""
    spans = torch.tensor([[1, 4]], dtype=torch.long)
    mask = completion_mask_unshifted(spans, seq_len=5)
    assert mask.tolist() == [[False, True, True, True, False]]


def test_completion_mask_invalid_range_all_false():
    """An empty/reversed range (end <= start) yields an all-False row."""
    spans = torch.tensor([[3, 3], [4, 2]], dtype=torch.long)
    mask = completion_mask_unshifted(spans, seq_len=5)
    assert mask.sum().item() == 0


def test_completion_mask_clamps_out_of_bounds():
    """Out-of-range start/end are clamped to [0, seq_len]."""
    spans = torch.tensor([[-3, 2], [2, 100]], dtype=torch.long)
    mask = completion_mask_unshifted(spans, seq_len=5)
    assert mask[0].tolist() == [True, True, False, False, False]
    assert mask[1].tolist() == [False, False, True, True, True]


# ---------------------------------------------------------------------------
# grouped_weighted_dpo_loss
# ---------------------------------------------------------------------------
def test_grouped_loss_matches_manual_two_groups():
    """The grouped loss equals the size-weighted sum of per-group soft cross-entropy."""
    seq_logits = torch.tensor([2.0, 0.5, -1.0, 3.0])
    rewards = torch.tensor([1.0, 0.0, 2.0, -1.0])
    groups = ["a", "a", "b", "b"]

    ce_a = F.cross_entropy(seq_logits[:2].unsqueeze(0), torch.softmax(rewards[:2], 0).unsqueeze(0))
    ce_b = F.cross_entropy(seq_logits[2:].unsqueeze(0), torch.softmax(rewards[2:], 0).unsqueeze(0))
    expected = 0.5 * ce_a + 0.5 * ce_b

    got = grouped_weighted_dpo_loss(seq_logits, rewards, groups)
    assert torch.allclose(got, expected, atol=1e-6)


def test_grouped_loss_single_group_is_soft_ce():
    """With one group the loss is a plain soft-target cross-entropy over the batch."""
    seq_logits = torch.tensor([0.3, -0.7, 1.1])
    rewards = torch.tensor([0.5, 0.2, 0.9])
    groups = ["g", "g", "g"]

    expected = F.cross_entropy(seq_logits.unsqueeze(0), torch.softmax(rewards, 0).unsqueeze(0))
    got = grouped_weighted_dpo_loss(seq_logits, rewards, groups)
    assert torch.allclose(got, expected, atol=1e-6)


def test_grouped_loss_minimised_when_logits_track_rewards():
    """Loss is lower when the logits' softmax matches the reward softmax targets."""
    rewards = torch.tensor([2.0, 0.0, -1.0])
    groups = ["g", "g", "g"]
    aligned = grouped_weighted_dpo_loss(rewards.clone(), rewards, groups)
    misaligned = grouped_weighted_dpo_loss(-rewards, rewards, groups)
    assert aligned < misaligned


def test_grouped_loss_empty_batch_is_zero():
    """An empty batch returns a scalar zero (no NaNs)."""
    loss = grouped_weighted_dpo_loss(torch.empty(0), torch.empty(0), [])
    assert loss.item() == 0.0


def test_grouped_loss_is_differentiable():
    """The loss produces gradients on the sequence logits."""
    seq_logits = torch.tensor([0.2, -0.1, 0.4, 0.9], requires_grad=True)
    rewards = torch.tensor([1.0, 0.0, 0.5, -0.5])
    loss = grouped_weighted_dpo_loss(seq_logits, rewards, ["a", "b", "a", "b"])
    loss.backward()
    assert seq_logits.grad is not None
    assert torch.isfinite(seq_logits.grad).all()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def test_dataset_single_getitem():
    """Single-mode records expose group, segments, chain type, scheme and reward."""
    ds = FrameworkCDRRewardDataset(_single_df(4), mode="single", group_by="source")
    assert len(ds) == 4
    rec = ds[0]
    assert rec["group"] == "S1"
    assert rec["chain_type"] == "H"
    assert rec["scheme"] == "imgt"
    assert set(rec["segments"]) == {"fr1", "fr2", "fr3", "fr4", "cdr1", "cdr2", "cdr3"}
    assert rec["reward"] == pytest.approx(0.0)


def test_dataset_single_group_by_prompt():
    """group_by='prompt' folds the framework sequences into the group label."""
    ds = FrameworkCDRRewardDataset(_single_df(2), mode="single", group_by="prompt")
    rec = ds[0]
    assert rec["group"].startswith("S1::")
    assert "EVQLVESGGG" in rec["group"]


def test_dataset_paired_getitem():
    """Paired-mode records expose separate heavy/light segments."""
    ds = FrameworkCDRRewardDataset(_paired_df(2), mode="paired", group_by="source")
    rec = ds[0]
    assert rec["group"] == "P1"
    assert set(rec["h_segments"]) == {"fr1", "fr2", "fr3", "fr4", "cdr1", "cdr2", "cdr3"}
    assert set(rec["l_segments"]) == {"fr1", "fr2", "fr3", "fr4", "cdr1", "cdr2", "cdr3"}


def test_dataset_missing_columns_raise():
    """Missing required columns raise a clear KeyError."""
    df = _single_df(2).drop(columns=["meta_cdr3"])
    with pytest.raises(KeyError):
        FrameworkCDRRewardDataset(df, mode="single")


def test_dataset_rejects_bad_mode():
    """An unknown mode is rejected."""
    df = _single_df(2)
    with pytest.raises(ValueError):
        FrameworkCDRRewardDataset(df, mode="triple")


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------
def _ids(tokenizer, tok):
    return int(tokenizer.convert_tokens_to_ids(tok))


def test_collator_single_completion_span(tokenizer):
    """Single-chain completion span runs from just after <HPRED> through <EOS>."""
    ds = FrameworkCDRRewardDataset(_single_df(4), mode="single", group_by="source")
    collator = FrameworkCDRRewardCollator(tokenizer=tokenizer, mode="single")
    batch = collator([ds[i] for i in range(4)])

    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["completion_spans"].shape == (4, 2)
    assert batch["rewards"].shape == (4,)
    assert batch["group_labels"] == ["S1", "S2", "S1", "S2"]

    hpred_id, eos_id, h_id = _ids(tokenizer, TOK_HPRED), _ids(tokenizer, TOK_EOS), _ids(tokenizer, TOK_H)
    for i in range(4):
        start, end = batch["completion_spans"][i].tolist()
        ids = batch["input_ids"][i].tolist()
        # token immediately before the completion is the prediction tag
        assert ids[start - 1] == hpred_id
        # completion ends on EOS and contains no chain tag or further pred tag
        assert ids[end - 1] == eos_id
        assert hpred_id not in ids[start:end]
        assert h_id not in ids[start:end]
        # completion sits within the attended (non-pad) region
        assert end <= int(batch["attention_mask"][i].sum().item())


def test_collator_paired_completion_span(tokenizer):
    """Paired (L-first) completion span starts after <LPRED> and includes <HPRED>."""
    ds = FrameworkCDRRewardDataset(_paired_df(2), mode="paired", group_by="source")
    collator = FrameworkCDRRewardCollator(tokenizer=tokenizer, mode="paired", order="L-first")
    batch = collator([ds[i] for i in range(2)])

    lpred_id, hpred_id, eos_id = (
        _ids(tokenizer, TOK_LPRED),
        _ids(tokenizer, TOK_HPRED),
        _ids(tokenizer, TOK_EOS),
    )
    for i in range(2):
        start, end = batch["completion_spans"][i].tolist()
        ids = batch["input_ids"][i].tolist()
        assert ids[start - 1] == lpred_id  # L-first: first pred tag is LPRED
        assert hpred_id in ids[start:end]  # heavy CDR block falls inside the completion
        assert ids[end - 1] == eos_id


def test_collator_rewards_and_padding(tokenizer):
    """Rewards are collated in order and padding is a multiple of pad_to_multiple_of."""
    ds = FrameworkCDRRewardDataset(_single_df(3), mode="single")
    collator = FrameworkCDRRewardCollator(tokenizer=tokenizer, mode="single", pad_to_multiple_of=8)
    batch = collator([ds[i] for i in range(3)])
    assert batch["rewards"].tolist() == pytest.approx([0.0, 1.0 / 3.0, 2.0 / 3.0])
    assert batch["input_ids"].shape[1] % 8 == 0


def _tiny_causal_lm(vocab_size: int):
    """Build a tiny GPT-2 causal LM for fast CPU/GPU training tests."""
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=256,
        n_embd=32,
        n_layer=2,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return GPT2LMHeadModel(cfg)


def _vocab_size(tokenizer) -> int:
    return max(tokenizer.get_vocab().values()) + 1


def test_module_seq_log_ratio_zero_when_policy_equals_ref(tokenizer):
    """With identical policy/reference weights the completion log-ratio is ~0."""
    pytest.importorskip("pytorch_lightning")
    pytest.importorskip("torchmetrics")
    from gencdr.alignment.trainer import WeightedDPOModule

    torch.manual_seed(0)
    policy = _tiny_causal_lm(_vocab_size(tokenizer))
    ref = _tiny_causal_lm(_vocab_size(tokenizer))
    ref.load_state_dict(policy.state_dict())

    ds = FrameworkCDRRewardDataset(_single_df(4), mode="single")
    collator = FrameworkCDRRewardCollator(tokenizer=tokenizer, mode="single")
    batch = collator([ds[i] for i in range(4)])

    module = WeightedDPOModule(policy, ref, beta=0.15)
    assert all(not p.requires_grad for p in module.ref_model.parameters())

    s = module._seq_log_ratio(batch)
    assert s.shape == (4,)
    assert torch.allclose(s, torch.zeros_like(s), atol=1e-4)


def test_module_loss_backward(tokenizer):
    """A training step yields a finite scalar loss with gradients on the policy."""
    pytest.importorskip("pytorch_lightning")
    pytest.importorskip("torchmetrics")
    from gencdr.alignment.trainer import WeightedDPOModule

    torch.manual_seed(0)
    policy = _tiny_causal_lm(_vocab_size(tokenizer))
    ref = _tiny_causal_lm(_vocab_size(tokenizer))
    ref.load_state_dict(policy.state_dict())

    ds = FrameworkCDRRewardDataset(_single_df(4), mode="single")
    collator = FrameworkCDRRewardCollator(tokenizer=tokenizer, mode="single")
    batch = collator([ds[i] for i in range(4)])

    module = WeightedDPOModule(policy, ref, beta=0.15)
    logps = module.compute_completion_logps(
        batch["input_ids"], batch["attention_mask"], batch["completion_spans"], module.model
    )
    assert logps.shape == (4,)
    assert torch.isfinite(logps).all()

    loss = module.compute_loss(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in module.model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_aligner_end_to_end(tmp_path, tokenizer):
    """WeightedDPOAligner trains a tiny model from a reward CSV and saves it."""
    pytest.importorskip("pytorch_lightning")
    pytest.importorskip("torchmetrics")
    from gencdr.alignment.aligner import WeightedDPOAligner

    model_dir = tmp_path / "model"
    save_tokenizer_dir(model_dir)
    torch.manual_seed(0)
    _tiny_causal_lm(_vocab_size(tokenizer)).save_pretrained(str(model_dir))

    csv_path = tmp_path / "rewards.csv"
    _single_df(16).to_csv(csv_path, index=False)

    out_dir = tmp_path / "aligned"
    aligner = WeightedDPOAligner(
        model=str(model_dir),
        output_dir=str(out_dir),
        mode="single",
        max_epochs=1,
        batch_size=4,
        val_size=0.25,
        num_workers=0,
        precision="32-true",
        num_devices=1,
    )
    aligner.run([str(csv_path)])

    assert (out_dir / "config.json").is_file()
    assert (out_dir / "model.safetensors").is_file() or (out_dir / "pytorch_model.bin").is_file()
    assert (out_dir / "tokenizer.json").is_file()
