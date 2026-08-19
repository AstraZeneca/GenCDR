"""String rendering utilities for GenCDR.

- render_single: build a single-chain string from one sample
- render_paired: build a paired heavy-light string from two samples (frameworks-first layout)

All strings embed special tokens; callers should pass add_special_tokens=False to the tokenizer.
"""

import random
import warnings
from typing import Any, Dict, List, Optional, Tuple

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


def render_single(sample: Dict[str, Any], include_scheme: bool = False) -> str:
    """Build a single-chain string representation with special tokens for framework regions and CDRs.

    The output format is: <BOS>[scheme?]<H|L> FRs <HPRED|LPRED> CDRs <EOS>
    where framework regions (FRs) and CDRs are separated by separator tokens.

    Parameters
    ----------
    sample : Dict[str, Any]
        A dictionary containing antibody chain information with the following keys:
        - "segments" : Dict[str, str]
            Dictionary with keys "fr1", "fr2", "fr3", "fr4" (framework regions)
            and "cdr1", "cdr2", "cdr3" (complementarity-determining regions).
            Each value is a string sequence.
        - "chain_type" : str
            Chain type identifier, either "H" (heavy) or "L" (light).
        - "scheme" : str, optional
            Numbering scheme name (e.g., "imgt", "kabat"). Only used if
            include_scheme is True.
    include_scheme : bool, optional
        If True, include the numbering scheme token after <BOS>. Default is False.

    Returns
    -------
    str
        A tokenized string representation of the antibody chain with special tokens:
        - <BOS>: Beginning of sequence
        - [scheme]: Optional scheme token (if include_scheme=True)
        - <H|L>: Chain type token
        - Framework regions (fr1-fr4) separated by <SEP> tokens
        - <HPRED|LPRED>: CDR prediction delimiter token
        - CDR regions (cdr1-cdr3) separated by <SEP> tokens
        - <EOS>: End of sequence
    """
    segs = sample["segments"]
    chain = sample["chain_type"]
    if chain not in ("H", "L"):
        raise KeyError(f"Invalid chain type: {chain}. Expected 'H' or 'L'.")
    parts: List[str] = [TOK_BOS]
    if include_scheme:
        scheme_tok = SCHEME_TOK_BY_NAME[sample["scheme"]]
        parts.append(scheme_tok)
    parts.append(TOK_H if chain == "H" else TOK_L)
    parts += [segs["fr1"], TOK_SEP, segs["fr2"], TOK_SEP, segs["fr3"], TOK_SEP, segs["fr4"]]
    parts.append(TOK_HPRED if chain == "H" else TOK_LPRED)
    parts += [segs["cdr1"], TOK_SEP, segs["cdr2"], TOK_SEP, segs["cdr3"], TOK_EOS]
    return "".join(parts)


def render_single_ltr(sample: Dict[str, Any], include_scheme: bool = False) -> str:
    """Build a single-chain string in left-to-right (canonical) order.

    The output format is: <BOS>[scheme?]<H|L> FR1 CDR1 FR2 CDR2 FR3 CDR3 FR4 <EOS>
    No separator tokens between regions — the model sees the raw sequence in natural order.
    """
    segs = sample["segments"]
    chain = sample["chain_type"]
    if chain not in ("H", "L"):
        raise KeyError(f"Invalid chain type: {chain}. Expected 'H' or 'L'.")
    parts: List[str] = [TOK_BOS]
    if include_scheme:
        scheme_tok = SCHEME_TOK_BY_NAME[sample["scheme"]]
        parts.append(scheme_tok)
    parts.append(TOK_H if chain == "H" else TOK_L)
    parts += [
        segs["fr1"],
        segs["cdr1"],
        segs["fr2"],
        segs["cdr2"],
        segs["fr3"],
        segs["cdr3"],
        segs["fr4"],
        TOK_EOS,
    ]
    return "".join(parts)


def render_paired(
    h_sample: Dict[str, Any],
    l_sample: Dict[str, Any],
    order: str = "L-first",
    include_scheme: bool = False,
) -> str:
    """
    Paired layout with both frameworks first, then both CDR blocks.

    Assumes a single shared scheme for both chains and, if include_scheme=True,
    inserts that scheme token once after <BOS>.

    If order == "L-first":
      <BOS> [scheme?]
        <L> LFR1 <SEP> LFR2 <SEP> LFR3 <SEP> LFR4
        <H> HFR1 <SEP> HFR2 <SEP> HFR3 <SEP> HFR4
        <LPRED> LCDR1 <SEP> LCDR2 <SEP> LCDR3
        <HPRED> HCDR1 <SEP> HCDR2 <SEP> HCDR3
      <EOS>

    If order == "H-first", frameworks swap and CDR blocks follow in H then L order.

    Parameters
    ----------
    h_sample : Dict[str, Any]
        Heavy chain sample dictionary with keys:
        - "segments" : Dict[str, str]
            Dictionary with keys "fr1", "fr2", "fr3", "fr4" (framework regions)
            and "cdr1", "cdr2", "cdr3" (complementarity-determining regions).
            Each value is a string sequence.
        - "chain_type" : str
            Chain type identifier, should be "H".
        - "scheme" : str
            Numbering scheme name (e.g., "imgt", "kabat").
    l_sample : Dict[str, Any]
        Light chain sample dictionary with same keys as h_sample, with "chain_type" == "L".
    order : str, optional
        Order of chains in output string. Either "L-first" or "H-first". Default is "L-first".
    include_scheme : bool, optional
        If True, include the numbering scheme token after <BOS>. Default is False.
    """
    hs: Dict[str, str] = h_sample["segments"]
    ls: Dict[str, str] = l_sample["segments"]

    h_fw_parts: List[str] = [TOK_H, hs["fr1"], TOK_SEP, hs["fr2"], TOK_SEP, hs["fr3"], TOK_SEP, hs["fr4"]]
    l_fw_parts: List[str] = [TOK_L, ls["fr1"], TOK_SEP, ls["fr2"], TOK_SEP, ls["fr3"], TOK_SEP, ls["fr4"]]

    h_cdr_parts: List[str] = [TOK_HPRED, hs["cdr1"], TOK_SEP, hs["cdr2"], TOK_SEP, hs["cdr3"]]
    l_cdr_parts: List[str] = [TOK_LPRED, ls["cdr1"], TOK_SEP, ls["cdr2"], TOK_SEP, ls["cdr3"]]

    parts: List[str] = [TOK_BOS]
    if include_scheme:
        h_scheme = h_sample.get("scheme")
        l_scheme = l_sample.get("scheme")
        if h_scheme != l_scheme:
            raise ValueError(f"Paired render requires shared scheme; got H={h_scheme}, L={l_scheme}")
        scheme_tok = SCHEME_TOK_BY_NAME[h_scheme]
        parts.append(scheme_tok)

    if order == "H-first":
        parts.extend(h_fw_parts)
        parts.extend(l_fw_parts)
        parts.extend(h_cdr_parts)
        parts.extend(l_cdr_parts)
    else:
        parts.extend(l_fw_parts)
        parts.extend(h_fw_parts)
        parts.extend(l_cdr_parts)
        parts.extend(h_cdr_parts)

    parts.append(TOK_EOS)
    return "".join(parts)


def strip_bos_eos(text: str) -> str:
    """Remove BOS/EOS from generated text, preserving content and SEP."""
    t = text
    if t.startswith(TOK_BOS):
        t = t[len(TOK_BOS) :]
    eos_pos = t.find(TOK_EOS)
    if eos_pos >= 0:
        t = t[:eos_pos]
    return t


def jitter_boundary_lens(lens: List[int], max_jitter: int) -> List[int]:
    """Apply random boundary jitter to region lengths for data augmentation.

    At each of the 6 boundaries between the 7 regions (fr1|cdr1, cdr1|fr2, ...),
    uniformly sample a shift in [-max_jitter, +max_jitter].  A positive shift moves
    the boundary right (left region grows, right region shrinks) and vice versa.
    Shifts are clamped so no region becomes negative.

    Parameters
    ----------
    lens : list of int
        Seven region lengths [fr1, cdr1, fr2, cdr2, fr3, cdr3, fr4].
    max_jitter : int
        Maximum absolute shift per boundary.  0 means no jitter.

    Returns
    -------
    list of int
        Jittered lengths that still sum to the same total.
    """
    if max_jitter <= 0:
        return list(lens)
    out = list(lens)
    for i in range(len(out) - 1):
        delta = random.randint(-max_jitter, max_jitter)
        # Clamp so neither neighbour goes negative
        delta = max(delta, -out[i])
        delta = min(delta, out[i + 1])
        out[i] += delta
        out[i + 1] -= delta
    return out


def slice_segments(sequence: str, lens: List[int]) -> Dict[str, str]:
    """Slice sequence into FR/CDR segments given seven lengths."""
    assert len(lens) == 7, "Expected seven lengths for FR1, CDR1, FR2, CDR2, FR3, CDR3, FR4"
    idxs: List[int] = [0]
    for L in lens:
        idxs.append(idxs[-1] + int(L))
    if idxs[-1] != len(sequence):
        raise RuntimeError(f"Lengths do not sum to sequence length: {idxs[-1]} != {len(sequence)}")

    return {
        "fr1": sequence[idxs[0] : idxs[1]],
        "cdr1": sequence[idxs[1] : idxs[2]],
        "fr2": sequence[idxs[2] : idxs[3]],
        "cdr2": sequence[idxs[3] : idxs[4]],
        "fr3": sequence[idxs[4] : idxs[5]],
        "cdr3": sequence[idxs[5] : idxs[6]],
        "fr4": sequence[idxs[6] : idxs[7]],
    }


Span = Optional[Tuple[int, int]]


def clamp_span(start: int, end: int, max_len_tokens: int) -> Span:
    """Clamp (start,end) to [0,max_len_tokens] and return None for empty spans."""
    s = min(start, max_len_tokens)
    e = min(end, max_len_tokens)
    return (s, e) if e > s else None


def count_and_first_pos(ids: List[int], token_id: int) -> Tuple[int, Optional[int]]:
    """Return (count, first_position) of token_id in ids."""
    count = 0
    first: Optional[int] = None

    for i, t in enumerate(ids):
        if t == token_id:
            count += 1
            if first is None:
                first = i
    return count, first


def find_block_end(ids: List[int], start: int, stop_ids: "set[int]") -> int:
    """Return the first index >= start that is a stop token, or len(ids) if none."""
    for j in range(start, len(ids)):
        if ids[j] in stop_ids:
            return j
    return len(ids)


def find_first_n_positions(ids: List[int], token_id: int, start: int, end: int, n: int) -> List[int]:
    """Return up to n positions of token_id in ids[start:end]."""
    out: List[int] = []
    for j in range(start, end):
        if ids[j] == token_id:
            out.append(j)
            if len(out) == n:
                break
    return out


def validate_pred_tag_counts(hp_count: int, lp_count: int, paired: bool) -> None:
    """Validate counts of <HPRED>/<LPRED> for paired vs single samples."""
    if paired:
        if hp_count != 1 or lp_count != 1:
            raise ValueError(f"Paired sample must contain exactly 1 <HPRED> and 1 <LPRED> (got {hp_count}, {lp_count})")
        return

    if (hp_count == 0 and lp_count == 0) or (hp_count > 0 and lp_count > 0):
        raise ValueError("Single sample must contain exactly one of <HPRED> or <LPRED>")
    if hp_count > 1 or lp_count > 1:
        raise ValueError(f"Single sample must contain exactly 1 pred tag (got HPRED={hp_count}, LPRED={lp_count})")


def validate_pred_block_has_two_seps(
    ids: List[int],
    pred_pos: Optional[int],
    sep_id: int,
    stop_ids: "set[int]",
    tokenizer: Any,
) -> None:
    """Validate that the pred block contains at least two <SEP> tokens before the next stop tag."""
    if pred_pos is None:
        return
    start = pred_pos + 1
    end = find_block_end(ids, start, stop_ids)
    seps = find_first_n_positions(ids, sep_id, start, end, n=2)
    if len(seps) < 2:
        lo = max(0, pred_pos - 10)
        hi = min(len(ids), pred_pos + 80)
        snippet = tokenizer.decode(ids[lo:hi])
        raise ValueError(f"Malformed CDR block: expected 2 <SEP> after pred tag. Context: {snippet}")


def parse_pred_block_spans(
    ids: List[int],
    pred_pos: Optional[int],
    sep_id: int,
    stop_ids: "set[int]",
    max_len_tokens: int,
) -> List[Span]:
    """Parse CDR1/2/3 spans for a single <*PRED> block in token-id space."""
    if pred_pos is None:
        return [None, None, None]

    start = pred_pos + 1
    end = find_block_end(ids, start, stop_ids)
    seps = find_first_n_positions(ids, sep_id, start, end, n=2)

    if not seps:
        warnings.warn("No <SEP> tokens found in CDR block; returning single clamped span for CDR1")
        return [clamp_span(start, end, max_len_tokens), None, None]

    cdr1 = clamp_span(start, seps[0], max_len_tokens)

    s2 = seps[0] + 1
    if len(seps) == 1:
        warnings.warn("Only one <SEP> token found in CDR block; returning clamped spans for CDR1 and CDR2")
        return [cdr1, clamp_span(s2, end, max_len_tokens), None]

    cdr2 = clamp_span(s2, seps[1], max_len_tokens)

    s3 = seps[1] + 1
    cdr3 = clamp_span(s3, end, max_len_tokens)

    return [cdr1, cdr2, cdr3]


def compute_cdr_spans_from_token_ids(
    input_ids: List[int],
    tokenizer: Any,
    max_len_tokens: int,
    strict: bool = True,
) -> Dict[str, List[Span]]:
    """Compute token spans for CDR1/2/3 for heavy and light chains directly from token IDs."""
    ids = input_ids[:max_len_tokens]
    vocab = tokenizer.get_vocab()

    sep_id = vocab[TOK_SEP]
    hp_id = vocab[TOK_HPRED]
    lp_id = vocab[TOK_LPRED]

    stop_ids = {hp_id, lp_id, vocab[TOK_H], vocab[TOK_L], vocab[TOK_EOS], vocab[TOK_BOS]}

    hp_count, hp_pos = count_and_first_pos(ids, hp_id)
    lp_count, lp_pos = count_and_first_pos(ids, lp_id)

    h_count, _ = count_and_first_pos(ids, vocab[TOK_H])
    l_count, _ = count_and_first_pos(ids, vocab[TOK_L])

    paired = (h_count > 0) and (l_count > 0)

    if strict:
        if (h_count > 0) != (hp_count > 0):
            raise ValueError(f"Inconsistent tags: <H> present={h_count>0} but <HPRED> present={hp_count>0}")
        if (l_count > 0) != (lp_count > 0):
            raise ValueError(f"Inconsistent tags: <L> present={l_count>0} but <LPRED> present={lp_count>0}")

        validate_pred_tag_counts(hp_count, lp_count, paired)
        validate_pred_block_has_two_seps(ids, hp_pos, sep_id, stop_ids, tokenizer)
        validate_pred_block_has_two_seps(ids, lp_pos, sep_id, stop_ids, tokenizer)

    return {
        "H": parse_pred_block_spans(ids, hp_pos, sep_id, stop_ids, max_len_tokens),
        "L": parse_pred_block_spans(ids, lp_pos, sep_id, stop_ids, max_len_tokens),
    }


def compute_cdr_spans_ltr(
    input_ids: List[int],
    tokenizer: Any,
    max_len_tokens: int,
    segment_lengths: Optional[List[int]] = None,
) -> Dict[str, List[Span]]:
    """Compute CDR spans for a left-to-right rendered sequence.

    LTR format: <BOS>[scheme?]<H|L> FR1 CDR1 FR2 CDR2 FR3 CDR3 FR4 <EOS>
    No separator tokens — spans are computed from known segment lengths.

    Parameters
    ----------
    segment_lengths : list of 7 ints
        [fr1_len, cdr1_len, fr2_len, cdr2_len, fr3_len, cdr3_len, fr4_len] in residues.
        Required for positional span computation.
    """
    if segment_lengths is None:
        return {"H": [None, None, None], "L": [None, None, None]}

    ids = input_ids[:max_len_tokens]
    vocab = tokenizer.get_vocab()
    h_id = vocab[TOK_H]
    l_id = vocab[TOK_L]

    chain_type: Optional[str] = None
    chain_pos: Optional[int] = None
    for i, t in enumerate(ids):
        if t == h_id:
            chain_type = "H"
            chain_pos = i
            break
        if t == l_id:
            chain_type = "L"
            chain_pos = i
            break

    if chain_type is None or chain_pos is None:
        return {"H": [None, None, None], "L": [None, None, None]}

    # Positions start after the chain tag token
    seq_start = chain_pos + 1
    fr1_len, cdr1_len, fr2_len, cdr2_len, fr3_len, cdr3_len, _ = segment_lengths

    cdr1_start = seq_start + fr1_len
    cdr1_end = cdr1_start + cdr1_len
    cdr2_start = cdr1_end + fr2_len
    cdr2_end = cdr2_start + cdr2_len
    cdr3_start = cdr2_end + fr3_len
    cdr3_end = cdr3_start + cdr3_len

    cdr1 = clamp_span(cdr1_start, cdr1_end, max_len_tokens)
    cdr2 = clamp_span(cdr2_start, cdr2_end, max_len_tokens)
    cdr3 = clamp_span(cdr3_start, cdr3_end, max_len_tokens)

    empty: List[Span] = [None, None, None]
    result: Dict[str, List[Span]] = {"H": list(empty), "L": list(empty)}
    result[chain_type] = [cdr1, cdr2, cdr3]
    return result
