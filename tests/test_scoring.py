"""Tests for the region-wise scoring helpers in gencdr.scoring.

``segment_sequence_by_scheme`` imports ``abnumber.Chain`` lazily (the optional
``scoring`` extra), so the tests inject a fake ``abnumber`` module into
``sys.modules`` to exercise the reconciliation logic without a live
anarci/HMMER install.
"""

import sys
import types

import pandas as pd
import pytest
import torch
from gencdr import scoring as su

# A valid human VH sequence, split into deterministic FR/CDR segments. The
# segmentation here is illustrative (not biologically exact); the tests below
# stub abnumber's ``Chain`` so they exercise scoring's reconciliation logic
# without depending on a live anarci/HMMER install (which is unavailable in CI).
VH_SEGMENTS = {
    "fr1": "QVQLVESGGGLVQPGGSLRLSCAAS",
    "cdr1": "GFTFSSYA",
    "fr2": "MSWVRQAPGKGLEWVS",
    "cdr2": "AISGSGGST",
    "fr3": "YYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYC",
    "cdr3": "AKDYYGSGSYYNAFDI",
    "fr4": "WGQGTLVTVSS",
}
VALID_VH = "".join(VH_SEGMENTS[k] for k in ["fr1", "cdr1", "fr2", "cdr2", "fr3", "cdr3", "fr4"])


class _FakeChain:
    """Minimal stand-in for ``abnumber.Chain`` returning preset FR/CDR segments."""

    def __init__(self, segments):
        """Store the segment dict to expose as ``fr*_seq``/``cdr*_seq`` attributes."""
        for key, value in segments.items():
            setattr(self, f"{key}_seq", value)


def _make_fake_chain_factory(segments):
    """Build a ``Chain`` replacement that always returns ``segments`` (ignores input)."""

    def _factory(sequence, scheme):  # noqa: ARG001 - signature mirrors abnumber.Chain
        return _FakeChain(segments)

    return _factory


def _install_fake_abnumber(monkeypatch, chain_impl):
    """Inject a stub ``abnumber`` module so the lazy ``from abnumber import Chain`` resolves."""
    fake = types.ModuleType("abnumber")
    fake.Chain = chain_impl
    monkeypatch.setitem(sys.modules, "abnumber", fake)


# --- segment_sequence_by_scheme ---------------------------------------------


def test_segment_sequence_by_scheme_reconstructs_input(monkeypatch):
    """Segments concatenate back to the original sequence."""
    _install_fake_abnumber(monkeypatch, _make_fake_chain_factory(VH_SEGMENTS))
    seg = su.segment_sequence_by_scheme(VALID_VH, "imgt")
    assert seg is not None
    order = ["fr1", "cdr1", "fr2", "cdr2", "fr3", "cdr3", "fr4"]
    assert "".join(seg[k] for k in order) == VALID_VH
    assert set(seg.keys()) == set(order)


def test_segment_sequence_by_scheme_returns_none_for_invalid(monkeypatch):
    """Unannotatable sequences yield None when Chain raises."""

    def _raises(sequence, scheme):  # noqa: ARG001 - signature mirrors abnumber.Chain
        raise ValueError("cannot annotate")

    _install_fake_abnumber(monkeypatch, _raises)
    assert su.segment_sequence_by_scheme("NOTAREALSEQUENCE", "imgt") is None


def test_segment_sequence_by_scheme_recovers_extra_prefix_suffix(monkeypatch):
    """Extra residues outside the annotatable region fold into fr1/fr4."""
    # Chain trims the padding (returns segments for the core only); scoring
    # must fold the leading/trailing residues back into fr1/fr4.
    _install_fake_abnumber(monkeypatch, _make_fake_chain_factory(VH_SEGMENTS))
    padded = "AA" + VALID_VH + "GG"
    seg = su.segment_sequence_by_scheme(padded, "imgt")
    assert seg is not None
    order = ["fr1", "cdr1", "fr2", "cdr2", "fr3", "cdr3", "fr4"]
    assert "".join(seg[k] for k in order) == padded
    assert seg["fr1"].startswith("AA")
    assert seg["fr4"].endswith("GG")


def test_segment_sequence_by_scheme_raises_without_abnumber(monkeypatch):
    """A helpful ImportError is raised when abnumber is unavailable."""
    # Remove any cached module and force the import to fail.
    monkeypatch.setitem(sys.modules, "abnumber", None)
    with pytest.raises(ImportError, match="gencdr\\[scoring\\]"):
        su.segment_sequence_by_scheme(VALID_VH, "imgt")


# --- normalize_sequence ------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("  abc ", "ABC"),
        ("DefG", "DEFG"),
        (None, ""),
        ("", ""),
        (123, "123"),
    ],
)
def test_normalize_sequence(value, expected):
    """Values are stringified, stripped, and uppercased."""
    assert su.normalize_sequence(value) == expected


# --- spearman_corr / pearson_corr -------------------------------------------


def test_spearman_corr_perfect_monotonic():
    """Perfectly monotonic data gives Spearman 1.0."""
    x = pd.Series([1, 2, 3, 4])
    y = pd.Series([10, 20, 30, 40])
    assert su.spearman_corr(x, y) == pytest.approx(1.0)


def test_spearman_corr_monotonic_nonlinear_is_one():
    """Monotonic but nonlinear data still gives Spearman 1.0."""
    x = pd.Series([1, 2, 3, 4])
    y = pd.Series([1, 4, 9, 16])  # monotonic but not linear
    assert su.spearman_corr(x, y) == pytest.approx(1.0)


def test_spearman_corr_returns_none_when_too_few_points():
    """Fewer than two paired points returns None."""
    assert su.spearman_corr(pd.Series([1.0]), pd.Series([2.0])) is None


def test_spearman_corr_drops_nan_pairs():
    """NaN pairs are dropped before computing the correlation."""
    x = pd.Series([1.0, 2.0, float("nan"), 4.0])
    y = pd.Series([1.0, 2.0, 3.0, float("nan")])
    assert su.spearman_corr(x, y) == pytest.approx(1.0)


def test_pearson_corr_perfect_linear():
    """Perfectly linear data gives Pearson 1.0."""
    x = pd.Series([1, 2, 3, 4])
    y = pd.Series([2, 4, 6, 8])
    assert su.pearson_corr(x, y) == pytest.approx(1.0)


def test_pearson_corr_returns_none_when_too_few_points():
    """Fewer than two paired points returns None."""
    assert su.pearson_corr(pd.Series([1.0]), pd.Series([2.0])) is None


def test_pearson_corr_returns_none_for_constant_series():
    """Zero-variance input yields NaN correlation, which maps to None."""
    # Zero variance -> correlation is NaN -> helper returns None.
    assert su.pearson_corr(pd.Series([5.0, 5.0, 5.0]), pd.Series([1.0, 2.0, 3.0])) is None


# --- compute_region_ll_from_loss --------------------------------------------


def _segment_lengths():
    """Single-sequence batch with one residue per FR/CDR region (7 amino acids)."""
    # fr1=1, cdr1=1, fr2=1, cdr2=1, fr3=1, cdr3=1, fr4=1 => 7 amino acids.
    return [[1, 1, 1, 1, 1, 1, 1]]


def test_compute_region_ll_offset_one():
    """offset=1 skips a leading control token before the first amino acid."""
    # Shifted layout has a leading control token at index 0, aa0 at index 1.
    # L-1 = 8 positions: [ctrl, fr1, cdr1, fr2, cdr2, fr3, cdr3, fr4].
    per_pos_loss = torch.tensor([[5.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0]])
    aa_mask = torch.tensor([[False, True, True, True, True, True, True, True]])

    ll_full, ll_fr, ll_cdr = su.compute_region_ll_from_loss(per_pos_loss, aa_mask, _segment_lengths(), offset=1)
    # FR positions (1,3,5,7) loss=1 each -> mean 1 -> ll_fr=-1.
    # CDR positions (2,4,6) loss=2 each -> mean 2 -> ll_cdr=-2.
    # Full over 7 aa: (1+2+1+2+1+2+1)/7 = 10/7 -> ll_full=-10/7.
    assert ll_fr[0] == pytest.approx(-1.0)
    assert ll_cdr[0] == pytest.approx(-2.0)
    assert ll_full[0] == pytest.approx(-10.0 / 7.0)


def test_compute_region_ll_offset_zero():
    """offset=0 places the first amino acid at index 0 (no control token)."""
    # No leading control token: aa0 at index 0. L-1 = 7 positions.
    per_pos_loss = torch.tensor([[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0]])
    aa_mask = torch.ones(1, 7, dtype=torch.bool)

    ll_full, ll_fr, ll_cdr = su.compute_region_ll_from_loss(per_pos_loss, aa_mask, _segment_lengths(), offset=0)
    assert ll_fr[0] == pytest.approx(-1.0)
    assert ll_cdr[0] == pytest.approx(-2.0)
    assert ll_full[0] == pytest.approx(-10.0 / 7.0)


def test_compute_region_ll_default_offset_is_one():
    """The default offset matches an explicit offset=1."""
    per_pos_loss = torch.tensor([[5.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0]])
    aa_mask = torch.tensor([[False, True, True, True, True, True, True, True]])

    explicit = su.compute_region_ll_from_loss(per_pos_loss, aa_mask, _segment_lengths(), offset=1)
    default = su.compute_region_ll_from_loss(per_pos_loss, aa_mask, _segment_lengths())
    assert default == explicit


def test_compute_region_ll_handles_empty_regions_without_divzero():
    """An all-False mask clamps counts to 1 so log-likelihoods are 0, not NaN."""
    # No amino acids selected by the mask -> counts clamp to 1, lls are 0.
    per_pos_loss = torch.zeros(1, 7)
    aa_mask = torch.zeros(1, 7, dtype=torch.bool)
    ll_full, ll_fr, ll_cdr = su.compute_region_ll_from_loss(per_pos_loss, aa_mask, _segment_lengths(), offset=0)
    assert ll_full[0] == pytest.approx(0.0)
    assert ll_fr[0] == pytest.approx(0.0)
    assert ll_cdr[0] == pytest.approx(0.0)
