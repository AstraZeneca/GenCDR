"""Unified GenCDR inference utilities: generation (single-chain CDRs, paired CDRs, whole chain) and log-likelihood.

Supports three model families sharing one tokenizer/rendering scheme:
  - IgGenCDR    : single-chain heavy or light (H/L)
  - NanoGenCDR  : nanobody / VHH (heavy-only, H)
  - p-IgGenCDR  : cognate paired heavy+light, jointly generated

Weights and tokenizer are loaded from a local directory (see gencdr.checkpoints).
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LogitsProcessor, PreTrainedTokenizerFast, set_seed

from gencdr.checkpoints import resolve_checkpoint
from gencdr.rendering import render_paired, render_single, slice_segments, strip_bos_eos
from gencdr.tokenizer import (
    AA_ALPHABET,
    CANONICAL_AA_ALPHABET,
    SCHEME_TOK_BY_NAME,
    TOK_BOS,
    TOK_EOS,
    TOK_H,
    TOK_HPRED,
    TOK_L,
    TOK_LPRED,
    TOK_SEP,
)

log = logging.getLogger("gencdr.generator")

_REDUCTION_ERROR = "reduction must be one of {'mean', 'sum'}"


class PerRegionTemperatureProcessor(LogitsProcessor):
    """Apply per-CDR temperatures during generation by scaling logits region-wise.

    The completion after the prompt is ``CDR1 <SEP> CDR2 <SEP> CDR3 <EOS>``. We track,
    per sequence in the batch, how many ``<SEP>`` tokens have been generated SINCE the
    prompt: 0 SEPs => emitting CDR1, 1 => CDR2, >=2 => CDR3. Each region's tokens are
    scaled by its temperature (default = ``base_temperature`` when a region temp is None).
    Implemented as logit scaling (logits / T) so it composes with the top_p warper;
    generate() itself is called with temperature=1.0 so HF's own warper is a no-op and
    this processor owns all temperature.

    Only the framework SEPs are BEFORE ``prompt_len`` (excluded by construction), so the
    count correctly reflects completion SEPs only.
    """

    def __init__(
        self,
        sep_id: int,
        prompt_len: int,
        base_temperature: float,
        cdr1_temperature: Optional[float] = None,
        cdr2_temperature: Optional[float] = None,
        cdr3_temperature: Optional[float] = None,
    ):
        self.sep_id = int(sep_id)
        self.prompt_len = int(prompt_len)
        self.base_t = float(base_temperature)
        # region temps default to base when unset
        self.t1 = float(cdr1_temperature) if cdr1_temperature is not None else self.base_t
        self.t2 = float(cdr2_temperature) if cdr2_temperature is not None else self.base_t
        self.t3 = float(cdr3_temperature) if cdr3_temperature is not None else self.base_t
        if self.base_t <= 0 or self.t1 <= 0 or self.t2 <= 0 or self.t3 <= 0:
            raise ValueError("Temperatures must be > 0 for PerRegionTemperatureProcessor.")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Scale logits by the temperature of the CDR region currently being emitted."""
        # completion tokens generated so far (exclude the prompt)
        gen = input_ids[:, self.prompt_len :]
        # per-sequence SEP count since the prompt: 0 => CDR1, 1 => CDR2, >=2 => CDR3
        sep_counts = (gen == self.sep_id).sum(dim=1).to(scores.device)
        # Build the per-sequence temperature vector without per-step scalar-tensor allocations
        # (this runs every generation step): start at CDR3, then fill CDR1/CDR2 by SEP count.
        temps = scores.new_full((scores.shape[0],), self.t3)
        temps = temps.masked_fill(sep_counts == 0, self.t1)
        temps = temps.masked_fill(sep_counts == 1, self.t2)
        return scores / temps.unsqueeze(1)


def _convert_token_to_id(tok: PreTrainedTokenizerFast, token: str) -> Optional[int]:
    """Return the integer ID of `token`, raising if the tokenizer does not map it to an int."""
    tid = tok.convert_tokens_to_ids(token)
    if isinstance(tid, int):
        return int(tid)
    else:
        raise RuntimeError(f"Token ID for token '{token}' is not an integer: {tid}")


def _build_prompt_from_rendered(rendered: str, known_cdr1: Optional[str], known_cdr2: Optional[str]) -> str:
    """Create a generation prompt: prefix up to pred tag + known CDR1[/SEP + CDR2] + final SEP if both known."""
    pred_pos_h = rendered.find(TOK_HPRED)
    pred_pos_l = rendered.find(TOK_LPRED)
    if pred_pos_h >= 0:
        head = rendered[: pred_pos_h + len(TOK_HPRED)]
    elif pred_pos_l >= 0:
        head = rendered[: pred_pos_l + len(TOK_LPRED)]
    else:
        raise RuntimeError("Prediction tag not found in rendered text.")

    tail = ""
    if known_cdr1 is not None:
        tail += known_cdr1
        tail += TOK_SEP
        if known_cdr2 is not None:
            tail += known_cdr2
            tail += TOK_SEP
    else:
        if known_cdr2 is not None:
            log.warning("You provided CDR2 but not CDR1; ignoring CDR2 conditioning.")
    return head + tail


def decode_cdrs_with_flags_single_chain(
    decoded_text: str,
    known_cdr1: Optional[str] = None,
    known_cdr2: Optional[str] = None,
) -> Dict[str, Any]:
    """Decode CDR1/2/3 from generated single-chain text, flagging malformed structure.

    Behavior:
    - If the structure is correct (exactly 3 SEP-delimited parts after the first pred tag)
      and no unexpected special tokens appear inside any CDR, returns the CDRs
      (respecting known_cdr1/known_cdr2 if provided).
    - If the structure is incorrect (under/over SEP) or unexpected specials appear,
      returns None for CDRs and flags the issue.

    Parameters
    ----------
    decoded_text : str
        Text after tokenizer decoding in the form of full rendered single-chain text:
        <BOS><scheme?><H|L><fr1><fr2><fr3><fr4><HPRED|LPRED><cdr1?><SEP><cdr2?><SEP>
        <cdr3?><EOS>

    Returns
    -------
    dict with keys:
      - text: original clean_text
      - cdr1, cdr2, cdr3: str or None
      - parsed_ok: bool
      - sep_count: int (number of SEP splits after the pred tag)
      - issues: List[str] from
      {"ok","no_pred","under_sep","over_sep","special_between_pred_eos","special_in_cdr1","special_in_cdr2","special_in_cdr3"}
    """
    issues: List[str] = []
    clean_text = strip_bos_eos(decoded_text)

    pred_pos_h = clean_text.find(TOK_HPRED)
    pred_pos_l = clean_text.find(TOK_LPRED)
    if pred_pos_h >= 0 and (pred_pos_l < 0 or pred_pos_h <= pred_pos_l):
        pos = pred_pos_h + len(TOK_HPRED)
    elif pred_pos_l >= 0:
        pos = pred_pos_l + len(TOK_LPRED)
    else:
        issues.append("no_pred")
        return {
            "text": clean_text,
            "cdr1": None if known_cdr1 is None else known_cdr1,
            "cdr2": None if known_cdr2 is None else known_cdr2,
            "cdr3": None,
            "parsed_ok": False,
            "sep_count": 0,
            "issues": issues,
        }

    after = clean_text[pos:]
    parts = after.split(TOK_SEP)
    sep_count = len(parts)

    if sep_count < 3:
        issues.append("under_sep")
    elif sep_count > 3:
        issues.append("over_sep")

    if sep_count != 3:
        return {
            "text": clean_text,
            "cdr1": None,
            "cdr2": None,
            "cdr3": None,
            "parsed_ok": False,
            "sep_count": sep_count,
            "issues": issues,
        }

    p1, p2, p3 = parts
    c1, c2, c3, cdr_issues = _decode_cdr_parts_with_flags(p1, p2, p3, known_cdr1, known_cdr2)
    issues.extend(cdr_issues)

    return {
        "text": clean_text,
        "cdr1": c1,
        "cdr2": c2,
        "cdr3": c3,
        "parsed_ok": len(issues) == 0,
        "sep_count": sep_count,
        "issues": issues,
    }


def _decode_cdr_parts_with_flags(
    p1: str,
    p2: str,
    p3: str,
    known_cdr1: Optional[str],
    known_cdr2: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    """Decode CDR1/2/3 from 3 SEP-delimited parts and return (cdr1,cdr2,cdr3,issues)."""
    issues: List[str] = []

    parts = [
        ("cdr1", p1, known_cdr1),
        ("cdr2", p2, known_cdr2),
        ("cdr3", p3, None),
    ]

    out: Dict[str, Optional[str]] = {}

    for name, part, known in parts:
        has_special = "<" in part
        if known is not None:
            cdr = known
        elif has_special:
            cdr = None
        else:
            cdr = part

        out[name] = cdr

        if has_special and known is None:
            issues.append(f"special_in_{name}")

        if cdr is not None and any(aa not in CANONICAL_AA_ALPHABET for aa in cdr):
            issues.append(f"non_canonical_aa_in_{name}")

    return out["cdr1"], out["cdr2"], out["cdr3"], issues


def _extract_block_after_tag(text: str, tag: str, stop_tags: List[str]) -> Optional[str]:
    """Return the substring from just after `tag` to the next occurrence of any stop tag (exclusive).

    Returns None if `tag` is absent. Used to isolate a single chain's CDR block from a
    paired generation output, independent of chain order.
    """
    pos = text.find(tag)
    if pos < 0:
        return None
    start = pos + len(tag)
    end = len(text)
    for stop in stop_tags:
        sp = text.find(stop, start)
        if sp >= 0:
            end = min(end, sp)
    return text[start:end]


def _decode_paired_block(block: Optional[str], chain_label: str) -> Dict[str, Any]:
    """Decode one chain's CDR block (``cdr1 <SEP> cdr2 <SEP> cdr3``) into flagged CDRs.

    Mirrors the single-chain decoder but without known-CDR conditioning. `chain_label`
    ("H" or "L") is prefixed onto issue codes so paired issues are unambiguous.
    """
    issues: List[str] = []
    if block is None:
        issues.append(f"{chain_label}_no_pred")
        return {"cdr1": None, "cdr2": None, "cdr3": None, "parsed_ok": False, "sep_count": 0, "issues": issues}

    parts = block.split(TOK_SEP)
    sep_count = len(parts)
    if sep_count < 3:
        issues.append(f"{chain_label}_under_sep")
    elif sep_count > 3:
        issues.append(f"{chain_label}_over_sep")

    if sep_count != 3:
        return {"cdr1": None, "cdr2": None, "cdr3": None, "parsed_ok": False, "sep_count": sep_count, "issues": issues}

    out: Dict[str, Optional[str]] = {}
    for name, part in zip(("cdr1", "cdr2", "cdr3"), parts):
        has_special = "<" in part
        if has_special:
            out[name] = None
            issues.append(f"{chain_label}_special_in_{name}")
        else:
            out[name] = part
            if any(aa not in CANONICAL_AA_ALPHABET for aa in part):
                issues.append(f"{chain_label}_non_canonical_aa_in_{name}")

    return {
        "cdr1": out["cdr1"],
        "cdr2": out["cdr2"],
        "cdr3": out["cdr3"],
        "parsed_ok": len(issues) == 0,
        "sep_count": sep_count,
        "issues": issues,
    }


def decode_paired_cdrs_with_flags(decoded_text: str, order: str = "L-first") -> Dict[str, Any]:
    """Decode heavy- and light-chain CDRs from a paired generation output.

    The paired output has both a ``<HPRED>`` and an ``<LPRED>`` block, e.g. (L-first):
      <BOS>[scheme?]<L>LFRs<H>HFRs<LPRED>LCDR1<SEP>LCDR2<SEP>LCDR3<HPRED>HCDR1<SEP>HCDR2<SEP>HCDR3<EOS>

    Parsing is order-agnostic: each chain's CDR block runs from its ``<*PRED>`` tag to the
    next control token. `order` is recorded in the output for reference only.

    Returns
    -------
    dict with keys:
      - text: cleaned decoded text (BOS/EOS stripped)
      - order: the order string passed in
      - "H", "L": per-chain dicts, each {cdr1, cdr2, cdr3, parsed_ok, sep_count, issues}
      - parsed_ok: True iff both chains parsed cleanly
      - issues: aggregated issue codes across both chains
    """
    stop_tags = [TOK_HPRED, TOK_LPRED, TOK_H, TOK_L, TOK_EOS, TOK_BOS]
    h_block = _extract_block_after_tag(decoded_text, TOK_HPRED, stop_tags)
    l_block = _extract_block_after_tag(decoded_text, TOK_LPRED, stop_tags)

    heavy = _decode_paired_block(h_block, "H")
    light = _decode_paired_block(l_block, "L")

    issues = list(heavy["issues"]) + list(light["issues"])
    return {
        "text": strip_bos_eos(decoded_text),
        "order": order,
        "H": heavy,
        "L": light,
        "parsed_ok": bool(heavy["parsed_ok"] and light["parsed_ok"]),
        "issues": issues,
    }


class GenCDRGenerator:
    """
    Inference class for GenCDR-style checkpoints (IgGenCDR / NanoGenCDR / p-IgGenCDR).

    - Loads model/tokenizer from a local folder
    - Generates single-chain CDRs given frameworks (optionally known CDR1, CDR2)
    - Jointly generates paired heavy+light CDRs given both frameworks (p-IgGenCDR)
    - Batch generation across multiple inputs
    - Generates whole chains from chain type only (fragments + CDRs)
    - Evaluates log-likelihood for raw sequences (with lens_by_scheme) or segmented samples.
    """

    def __init__(
        self, model_dir: str, device: Optional[str] = None, fp: int = 16, include_scheme_token: bool = False
    ) -> None:
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Tokenizer/model
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir, local_files_only=True)
        self.include_scheme_token = include_scheme_token
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
        self.model.eval()
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True  # enable KV cache for generation

        # Device/precision
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        self.model.to(self.device)

        self.use_fp16 = (int(fp) == 16) and (self.device == "cuda")

        # Special IDs
        self.bos_id = _convert_token_to_id(self.tokenizer, TOK_BOS)
        self.eos_id = _convert_token_to_id(self.tokenizer, TOK_EOS)
        self.sep_id = _convert_token_to_id(self.tokenizer, TOK_SEP)
        if self.tokenizer.pad_token_id is None:
            raise RuntimeError("Tokenizer must define a pad token (pad_token_id) to enable padding.")
        self.pad_id = int(self.tokenizer.pad_token_id)
        self.h_id = _convert_token_to_id(self.tokenizer, TOK_H)
        self.l_id = _convert_token_to_id(self.tokenizer, TOK_L)
        self.hpred_id = _convert_token_to_id(self.tokenizer, TOK_HPRED)
        self.lpred_id = _convert_token_to_id(self.tokenizer, TOK_LPRED)

        self.scheme_to_id: Dict[str, Optional[int]] = {
            "imgt": _convert_token_to_id(self.tokenizer, SCHEME_TOK_BY_NAME["imgt"]),
            "kabat": _convert_token_to_id(self.tokenizer, SCHEME_TOK_BY_NAME["kabat"]),
            "chothia": _convert_token_to_id(self.tokenizer, SCHEME_TOK_BY_NAME["chothia"]),
        }
        self.aa_token_ids: List[int] = [_convert_token_to_id(self.tokenizer, aa) for aa in AA_ALPHABET]

        log.info("Loaded model from %s on %s (fp=%s).", model_dir, self.device, fp)

    # ----------------------------
    # Alternate constructors
    # ----------------------------

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        device: Optional[str] = None,
        fp: int = 16,
        include_scheme_token: bool = False,
    ) -> "GenCDRGenerator":
        """Load a generator from a local directory or a named checkpoint.

        `name_or_path` may be an existing directory, or a short name (e.g. "iggencdr",
        "nanogencdr", "p-iggencdr") resolved against the local checkpoint cache. See
        gencdr.checkpoints.resolve_checkpoint for resolution rules.
        """
        model_dir = resolve_checkpoint(name_or_path)
        return cls(model_dir=str(model_dir), device=device, fp=fp, include_scheme_token=include_scheme_token)

    # ----------------------------
    # Prompt building utilities
    # ----------------------------

    def _render_single_text(
        self,
        chain_type: str,
        fr1: str,
        fr2: str,
        fr3: str,
        fr4: str,
        cdr1: str,
        cdr2: str,
        cdr3: str,
        scheme: Optional[str] = None,
    ) -> str:
        """Render full single-chain text using encoder."""
        sample = {
            "chain_type": chain_type,
            "segments": {"fr1": fr1, "fr2": fr2, "fr3": fr3, "fr4": fr4, "cdr1": cdr1, "cdr2": cdr2, "cdr3": cdr3},
            "scheme": scheme,
        }
        text = render_single(sample, include_scheme=self.include_scheme_token)
        return text

    def _prompt_for_cdr_generation(
        self,
        chain_type: str,
        frameworks: Dict[str, str],
        known_cdr1: Optional[str] = None,
        known_cdr2: Optional[str] = None,
        scheme: Optional[str] = None,
    ) -> str:
        """
        Build a generation prompt from full rendered text and inject known CDRs after <HPRED|LPRED>.

        Unknown CDRs are left to be generated.
        """
        fr1, fr2, fr3, fr4 = frameworks["fr1"], frameworks["fr2"], frameworks["fr3"], frameworks["fr4"]
        rendered = self._render_single_text(
            chain_type=chain_type,
            fr1=fr1,
            fr2=fr2,
            fr3=fr3,
            fr4=fr4,
            cdr1="",
            cdr2="",
            cdr3="",
            scheme=scheme,
        )
        prompt = _build_prompt_from_rendered(rendered, known_cdr1, known_cdr2)
        return prompt

    def _prompt_for_paired_cdr_generation(
        self,
        h_frameworks: Dict[str, str],
        l_frameworks: Dict[str, str],
        order: str = "L-first",
        scheme: Optional[str] = None,
    ) -> str:
        """Build a paired generation prompt: full rendered paired text truncated at the first pred tag.

        For ``order="L-first"`` the prompt ends at ``<LPRED>`` and the model generates the light
        CDRs, then emits ``<HPRED>`` and the heavy CDRs. For ``order="H-first"`` the prompt ends at
        ``<HPRED>``. Both chains' frameworks are included in the prompt.
        """
        empty = {"cdr1": "", "cdr2": "", "cdr3": ""}
        h_segs = {
            "fr1": h_frameworks["fr1"],
            "fr2": h_frameworks["fr2"],
            "fr3": h_frameworks["fr3"],
            "fr4": h_frameworks["fr4"],
            **empty,
        }
        l_segs = {
            "fr1": l_frameworks["fr1"],
            "fr2": l_frameworks["fr2"],
            "fr3": l_frameworks["fr3"],
            "fr4": l_frameworks["fr4"],
            **empty,
        }
        rendered = self._render_paired_text(h_segs=h_segs, l_segs=l_segs, scheme=scheme, order=order)
        first_pred = TOK_HPRED if order == "H-first" else TOK_LPRED
        pos = rendered.find(first_pred)
        if pos < 0:
            raise RuntimeError(f"Prediction tag {first_pred} not found in rendered paired text.")
        return rendered[: pos + len(first_pred)]

    def _prompt_for_whole_chain(self, chain_type: str, scheme: Optional[str] = None) -> str:
        """Minimal prompt to let the model generate frameworks and CDRs from scratch.

        Note: this prompt has distribution shift vs training (frameworks were given explicitly).
        """
        bos = TOK_BOS
        scheme_tok = SCHEME_TOK_BY_NAME[scheme] if (self.include_scheme_token and scheme) else ""
        ctag = TOK_H if chain_type == "H" else TOK_L
        return f"{bos}{scheme_tok}{ctag}"

    # ----------------------------
    # Generation APIs
    # ----------------------------

    def generate_cdrs_from_frameworks(
        self,
        chain_type: str,
        frameworks: Dict[str, str],
        n_samples: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        known_cdr1: Optional[str] = None,
        known_cdr2: Optional[str] = None,
        scheme: Optional[str] = None,
        seed: Optional[int] = None,
        cdr1_temperature: Optional[float] = None,
        cdr2_temperature: Optional[float] = None,
        cdr3_temperature: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Generate CDR sequences given framework regions and optionally known CDR1/CDR2 sequences.

        When known_cdr1/known_cdr2 are provided they are injected into the prompt and the model
        only completes the remaining CDR(s); otherwise all three CDRs are generated.

        Parameters
        ----------
        chain_type : str
            The type of antibody chain (i.e., 'H' or 'L').
        frameworks : Dict[str, str]
            Dictionary containing framework region sequences. Keys should specify framework
            regions (e.g., {'fr1': '...', 'fr2': '...', 'fr3': '...', 'fr4': '...'}).
        n_samples : int
            Number of CDR sequence samples to generate.
        temperature : float, optional
            Sampling temperature for generation. Default is 1.0.
        top_p : float, optional
            Nucleus sampling parameter. Only tokens with cumulative probability up to
            top_p are considered. Default is 0.95.
        max_new_tokens : int, optional
            Maximum number of new tokens to generate. Default is 256.
        known_cdr1 : Optional[str], optional
            Pre-specified CDR1 sequence to use instead of generating. Default is None.
        known_cdr2 : Optional[str], optional
            Pre-specified CDR2 sequence to use instead of generating. Default is None.
        scheme : Optional[str], optional
            Numbering scheme to use (e.g., 'imgt', 'kabat', 'chothia'). Default is None.
        seed : Optional[int], optional
            Random seed for reproducibility. Default is None.
        cdr1_temperature, cdr2_temperature, cdr3_temperature : Optional[float], optional
            Per-region sampling temperatures. When any is set, that region's tokens are
            sampled at its own temperature and `temperature` is used for the remaining
            regions. Default is None (single temperature for all regions).

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries, one per generated sample. Each dictionary contains:
            - 'text' : str
                The complete decoded text output from the model.
            - 'cdr1' : str or None
                The CDR1 sequence (either known or generated; None if structure invalid).
            - 'cdr2' : str or None
                The CDR2 sequence (either known or generated; None if structure invalid).
            - 'cdr3' : str or None
                The generated CDR3 sequence (None if structure invalid).
            - 'parsed_ok': bool
            - 'sep_count': int
            - 'issues': List[str]
        """
        prompt = self._prompt_for_cdr_generation(
            chain_type=chain_type,
            frameworks=frameworks,
            known_cdr1=known_cdr1,
            known_cdr2=known_cdr2,
            scheme=scheme,
        )
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
        if seed is not None:
            set_seed(seed)

        # Per-CDR temperature: when any of cdr1/2/3_temperature is set, that region's tokens
        # are sampled at its own temperature via a logits processor and generate() runs at
        # temperature=1.0 (HF warper no-op) so the processor owns all temperature. Otherwise
        # behaviour is unchanged (single temp).
        gen_temperature = temperature
        logits_processor = None
        if any(t is not None for t in (cdr1_temperature, cdr2_temperature, cdr3_temperature)):
            logits_processor = [
                PerRegionTemperatureProcessor(
                    sep_id=self.sep_id,
                    prompt_len=int(input_ids.shape[1]),
                    base_temperature=temperature,
                    cdr1_temperature=cdr1_temperature,
                    cdr2_temperature=cdr2_temperature,
                    cdr3_temperature=cdr3_temperature,
                )
            ]
            gen_temperature = 1.0

        with torch.inference_mode():
            with torch.amp.autocast("cuda" if str(self.device).startswith("cuda") else "cpu", enabled=self.use_fp16):
                outputs = self.model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    temperature=gen_temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=int(n_samples),
                    pad_token_id=self.pad_id,
                    eos_token_id=self.eos_id,
                    logits_processor=logits_processor,
                )

        as_lists = [outputs[i, ...].detach().cpu().tolist() for i in range(outputs.shape[0])]
        decoded = self.tokenizer.batch_decode(as_lists, skip_special_tokens=False)

        results: List[Dict[str, Any]] = []
        for txt in decoded:
            dec = decode_cdrs_with_flags_single_chain(txt, known_cdr1, known_cdr2)
            results.append(dec)
        return results

    def generate_paired_cdrs_from_frameworks(
        self,
        h_frameworks: Dict[str, str],
        l_frameworks: Dict[str, str],
        n_samples: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        order: str = "L-first",
        scheme: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Jointly generate heavy- and light-chain CDRs from both frameworks (p-IgGenCDR).

        The paired model conditions on both chains' frameworks and generates all six CDRs
        autoregressively in a single pass, so the second chain's CDRs are drawn conditional
        on the first chain's generated CDRs.

        Parameters
        ----------
        h_frameworks : Dict[str, str]
            Heavy-chain framework regions: {'fr1','fr2','fr3','fr4'}.
        l_frameworks : Dict[str, str]
            Light-chain framework regions: {'fr1','fr2','fr3','fr4'}.
        n_samples : int
            Number of paired CDR samples to generate.
        temperature : float, optional
            Sampling temperature. Default is 1.0.
        top_p : float, optional
            Nucleus sampling parameter. Default is 0.95.
        max_new_tokens : int, optional
            Maximum number of new tokens to generate (covers both chains' CDR blocks).
            Default is 256.
        order : str, optional
            Chain order in the rendered sequence, "L-first" (default) or "H-first". The
            p-IgGenCDR checkpoint is trained order-agnostically; "L-first" is the default
            used at inference.
        scheme : Optional[str], optional
            Numbering scheme ('imgt', 'kabat', 'chothia'). Only used when the model was
            trained with scheme tokens (include_scheme_token=True). Default is None.
        seed : Optional[int], optional
            Random seed for reproducibility. Default is None.

        Returns
        -------
        List[Dict[str, Any]]
            One dict per sample, as returned by decode_paired_cdrs_with_flags:
            {'text', 'order', 'H': {...}, 'L': {...}, 'parsed_ok', 'issues'}.
        """
        prompt = self._prompt_for_paired_cdr_generation(
            h_frameworks=h_frameworks,
            l_frameworks=l_frameworks,
            order=order,
            scheme=scheme,
        )
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
        if seed is not None:
            set_seed(seed)

        with torch.inference_mode():
            with torch.amp.autocast("cuda" if str(self.device).startswith("cuda") else "cpu", enabled=self.use_fp16):
                outputs = self.model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=int(n_samples),
                    pad_token_id=self.pad_id,
                    eos_token_id=self.eos_id,
                )

        as_lists = [outputs[i, ...].detach().cpu().tolist() for i in range(outputs.shape[0])]
        decoded = self.tokenizer.batch_decode(as_lists, skip_special_tokens=False)
        return [decode_paired_cdrs_with_flags(txt, order=order) for txt in decoded]

    def generate_cdrs_batch(
        self,
        inputs: List[Dict[str, Any]],
        n_samples_per_input: int,
        temperature: float = 1.2,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
    ) -> List[List[Dict[str, Any]]]:
        """Batch version of generate_cdrs_from_frameworks.

        Each input is a dict:
        {"chain_type": "H"/"L", "frameworks": {fr1,fr2,fr3,fr4}, "cdr1": Optional[str],
         "cdr2": Optional[str], "scheme": Optional[str]}
        Returns a list of per-input result lists with n_samples_per_input items each.
        """
        outputs_all: List[List[Dict[str, Any]]] = []
        for inp in tqdm(inputs):
            res = self.generate_cdrs_from_frameworks(
                chain_type=inp["chain_type"],
                frameworks=inp["frameworks"],
                n_samples=n_samples_per_input,
                known_cdr1=inp.get("cdr1"),
                known_cdr2=inp.get("cdr2"),
                scheme=inp.get("scheme"),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            outputs_all.append(res)
        return outputs_all

    def generate_whole_chains(
        self,
        chain_type: str,
        n_samples: int,
        scheme: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Generate complete chains (frameworks + CDRs) from starting chain type only."""
        prompt = self._prompt_for_whole_chain(chain_type, scheme=scheme)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
        pad_id = self.pad_id
        if seed is not None:
            set_seed(seed)
        with torch.inference_mode():
            with torch.amp.autocast("cuda" if str(self.device).startswith("cuda") else "cpu", enabled=self.use_fp16):
                outputs = self.model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=n_samples,
                    pad_token_id=pad_id,
                    eos_token_id=self.eos_id,
                )
        as_lists = [outputs[i, ...].detach().cpu().tolist() for i in range(outputs.shape[0])]
        decoded = self.tokenizer.batch_decode(as_lists, skip_special_tokens=False)
        decoded = [strip_bos_eos(t) for t in decoded]

        return decoded

    # ----------------------------
    # Likelihood APIs
    # ----------------------------

    def _validate_segment_item(self, item: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
        """Validate a single item dict and return (chain_type, segments dict)."""
        required = ["chain_type", "fr1", "cdr1", "fr2", "cdr2", "fr3", "cdr3", "fr4"]
        missing = [k for k in required if k not in item]
        if missing:
            raise KeyError(f"Missing required keys in item: {missing}")
        chain_type = str(item["chain_type"])
        if chain_type not in ("H", "L"):
            raise ValueError("chain_type must be 'H' or 'L'")
        segments = {
            "fr1": str(item["fr1"]),
            "cdr1": str(item["cdr1"]),
            "fr2": str(item["fr2"]),
            "cdr2": str(item["cdr2"]),
            "fr3": str(item["fr3"]),
            "cdr3": str(item["cdr3"]),
            "fr4": str(item["fr4"]),
        }
        return chain_type, segments

    def _encode_texts(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """HF tokenizer batch-encode with padding and attention masks on self.device."""
        if self.tokenizer.pad_token_id is None:
            raise RuntimeError("Tokenizer must define a pad token (pad_token_id) to enable padded batch encoding.")
        enc = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        return input_ids, attention_mask

    def _masked_ll_for_batch(
        self,
        input_ids: torch.Tensor,  # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
        reduction: str = "mean",
    ) -> List[float]:
        """Compute masked log-likelihoods for a batch using the same exclusion mask as the single-sample path."""
        with (
            torch.inference_mode(),
            torch.amp.autocast("cuda" if str(self.device).startswith("cuda") else "cpu", enabled=self.use_fp16),
        ):
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, L, V]

        # Shift
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]  # pads excluded

        include_mask = shift_mask.bool()
        exclude_tokens = {
            tid
            for tid in (
                self.scheme_to_id.get("imgt"),
                self.scheme_to_id.get("kabat"),
                self.scheme_to_id.get("chothia"),
                self.pad_id,
            )
            if tid is not None
        }
        for tid in exclude_tokens:
            include_mask = include_mask & (shift_labels != tid)

        V = shift_logits.size(-1)
        per_pos_loss = torch.nn.functional.cross_entropy(
            shift_logits.contiguous().view(-1, V),
            shift_labels.contiguous().view(-1),
            reduction="none",
        ).view(
            shift_logits.size(0), -1
        )  # [B, L-1]

        valid_counts = include_mask.sum(dim=1)
        if (valid_counts == 0).any():
            idxs = (valid_counts == 0).nonzero(as_tuple=False).view(-1).tolist()
            raise RuntimeError(f"Some samples have zero valid tokens after masking (indices={idxs}).")

        masked_sum = (per_pos_loss * include_mask.float()).sum(dim=1)
        if reduction == "sum":
            ll = -masked_sum
        else:  # "mean"
            ll = -(masked_sum / valid_counts.float())

        return ll.detach().cpu().tolist()

    def _masked_ll_from_include_mask(
        self,
        per_pos_loss: torch.Tensor,
        include_mask: torch.Tensor,
        reduction: str,
        mask_name: str,
    ) -> torch.Tensor:
        """Compute masked log-likelihood tensor [B] from precomputed per-position CE loss and a boolean mask."""
        valid_counts = include_mask.sum(dim=1)
        if (valid_counts == 0).any():
            idxs = (valid_counts == 0).nonzero(as_tuple=False).view(-1).tolist()
            raise RuntimeError(f"Some samples have zero valid tokens for mask '{mask_name}' (indices={idxs}).")

        masked_sum = (per_pos_loss * include_mask.float()).sum(dim=1)
        if reduction == "sum":
            return -masked_sum
        return -(masked_sum / valid_counts.float())

    def log_likelihood_from_segments(
        self,
        item: Dict[str, Any],
        scheme: Optional[str] = None,
        reduction: str = "mean",
    ) -> float:
        """Log-likelihood from fully segmented single-chain sample.

        Excludes all special tokens from likelihood computation.

        Parameters
        ----------
        item : dict
            Must contain:
              - chain_type: "H" or "L"
              - fr1, cdr1, fr2, cdr2, fr3, cdr3, fr4: str segments
        scheme : str, optional
            Scheme name to use for segmentation ("imgt", "kabat", "chothia"), by default None
        reduction : str, optional
            Reduction method for log-likelihood ("mean", "sum"), by default "mean"

        Returns
        -------
        float
            Log-likelihood of the sequence segments under the model.
        """
        chain_type, segs = self._validate_segment_item(item)
        text = self._render_single_text(
            chain_type=chain_type,
            fr1=segs["fr1"],
            fr2=segs["fr2"],
            fr3=segs["fr3"],
            fr4=segs["fr4"],
            cdr1=segs["cdr1"],
            cdr2=segs["cdr2"],
            cdr3=segs["cdr3"],
            scheme=scheme,
        )
        input_ids, attention_mask = self._encode_texts([text])  # [1, L], [1, L]
        return self._masked_ll_for_batch(input_ids, attention_mask, reduction=reduction)[0]

    def batch_log_likelihood_from_segments(
        self,
        items: List[Dict[str, Any]],
        scheme: Optional[str] = None,
        reduction: str = "mean",
        batch_size: int = 256,
    ) -> List[float]:
        """Batched log-likelihood from fully segmented single-chain samples.

        Each item in `items` must be a dict with:
          - chain_type: "H" or "L"
          - fr1, cdr1, fr2, cdr2, fr3, cdr3, fr4: strings

        scheme apply to all items uniformly.

        Returns a list of floats (one per input), computed with the same masking
        as log_likelihood_from_segments (excludes BOS/EOS/H/L/HPRED/LPRED/scheme tokens/SEP/PAD).
        """
        if not items:
            return []

        texts: List[str] = []
        for it in items:
            chain_type, segs = self._validate_segment_item(it)
            text = self._render_single_text(
                chain_type=chain_type,
                fr1=segs["fr1"],
                fr2=segs["fr2"],
                fr3=segs["fr3"],
                fr4=segs["fr4"],
                cdr1=segs["cdr1"],
                cdr2=segs["cdr2"],
                cdr3=segs["cdr3"],
                scheme=scheme,
            )
            texts.append(text)

        results: List[float] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            input_ids, attention_mask = self._encode_texts(chunk)
            results.extend(self._masked_ll_for_batch(input_ids, attention_mask, reduction=reduction))
        return results

    def _render_segment_texts(self, items: List[Dict[str, Any]], scheme: Optional[str]) -> List[str]:
        """Validate and render a list of segmented single-chain items to model input texts."""
        texts: List[str] = []
        for it in items:
            chain_type, segs = self._validate_segment_item(it)
            text = self._render_single_text(
                chain_type=chain_type,
                fr1=segs["fr1"],
                fr2=segs["fr2"],
                fr3=segs["fr3"],
                fr4=segs["fr4"],
                cdr1=segs["cdr1"],
                cdr2=segs["cdr2"],
                cdr3=segs["cdr3"],
                scheme=scheme,
            )
            texts.append(text)
        return texts

    def _batch_forward_aa_loss(
        self, chunk: List[str]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Forward pass over rendered texts; return tensors shared by all region scorers.

        Returns
        -------
        per_pos_loss : [B, L-1]  per-position cross-entropy loss
        aa_mask      : [B, L-1]  amino-acid token mask (post-shift, pad-excluded)
        input_ids    : [B, L]
        tgt_pos      : [B, L-1]  unshifted-input position of each label, in [1, L-1]
        """
        input_ids, attention_mask = self._encode_texts(chunk)

        with (
            torch.inference_mode(),
            torch.amp.autocast("cuda" if str(self.device).startswith("cuda") else "cpu", enabled=self.use_fp16),
        ):
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:].bool()

        aa_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        for aa_tid in self.aa_token_ids:
            aa_mask = aa_mask | (shift_labels == aa_tid)
        aa_mask = aa_mask & shift_mask

        B, L = input_ids.shape
        tgt_pos = torch.arange(1, L, device=input_ids.device).unsqueeze(0).expand(B, -1)

        V = shift_logits.size(-1)
        per_pos_loss = torch.nn.functional.cross_entropy(
            shift_logits.contiguous().view(-1, V),
            shift_labels.contiguous().view(-1),
            reduction="none",
        ).view(shift_logits.size(0), -1)

        return per_pos_loss, aa_mask, input_ids, tgt_pos

    def _exactly_one_token_pos(self, input_ids: "torch.Tensor", token_id: int, name: str) -> "torch.Tensor":
        """Return [B] indices of token_id in input_ids; raise if any row has != 1 occurrence."""
        mask = input_ids == token_id
        counts = mask.sum(dim=1)
        if (counts != 1).any():
            bad = (counts != 1).nonzero(as_tuple=False).view(-1).tolist()
            raise RuntimeError(f"Expected exactly one {name} token per sample (indices={bad}).")
        return mask.float().argmax(dim=1).long()

    def _batch_forward_region_tensors(
        self, chunk: List[str]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Single-chain forward: returns (per_pos_loss, aa_mask, pred_pos, input_ids, tgt_pos)."""
        per_pos_loss, aa_mask, input_ids, tgt_pos = self._batch_forward_aa_loss(chunk)

        pred_mask = (input_ids == self.hpred_id) | (input_ids == self.lpred_id)
        pred_counts = pred_mask.sum(dim=1)
        if (pred_counts != 1).any():
            bad = (pred_counts != 1).nonzero(as_tuple=False).view(-1).tolist()
            raise RuntimeError(f"Expected exactly one prediction tag token per sample (indices={bad}).")
        pred_pos = pred_mask.float().argmax(dim=1).long()

        return per_pos_loss, aa_mask, pred_pos, input_ids, tgt_pos

    def batch_log_likelihood_regions_from_segments(
        self,
        items: List[Dict[str, Any]],
        scheme: Optional[str] = None,
        reduction: str = "mean",
        batch_size: int = 256,
    ) -> Dict[str, List[float]]:
        """Batched AA-only log-likelihood by region for segmented single-chain samples.

        Returns dict with keys:
          - "full": all amino-acid tokens in FR+CDR
          - "framework": amino-acid tokens before <HPRED|LPRED>
          - "cdr": amino-acid tokens after <HPRED|LPRED>
        """
        if reduction not in {"mean", "sum"}:
            raise ValueError(_REDUCTION_ERROR)
        if not items:
            return {"full": [], "framework": [], "cdr": []}

        texts = self._render_segment_texts(items, scheme)

        out_full: List[float] = []
        out_fr: List[float] = []
        out_cdr: List[float] = []

        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            per_pos_loss, aa_mask, pred_pos, _input_ids, tgt_pos = self._batch_forward_region_tensors(chunk)

            fr_mask = aa_mask & (tgt_pos < pred_pos.unsqueeze(1))
            cdr_mask = aa_mask & (tgt_pos > pred_pos.unsqueeze(1))

            ll_full = self._masked_ll_from_include_mask(
                per_pos_loss=per_pos_loss,
                include_mask=aa_mask,
                reduction=reduction,
                mask_name="full",
            )
            ll_fr = self._masked_ll_from_include_mask(
                per_pos_loss=per_pos_loss,
                include_mask=fr_mask,
                reduction=reduction,
                mask_name="framework",
            )
            ll_cdr = self._masked_ll_from_include_mask(
                per_pos_loss=per_pos_loss,
                include_mask=cdr_mask,
                reduction=reduction,
                mask_name="cdr",
            )

            out_full.extend(ll_full.detach().cpu().tolist())
            out_fr.extend(ll_fr.detach().cpu().tolist())
            out_cdr.extend(ll_cdr.detach().cpu().tolist())

        return {
            "full": out_full,
            "framework": out_fr,
            "cdr": out_cdr,
        }

    def batch_log_likelihood_cdr3_split_from_segments(
        self,
        items: List[Dict[str, Any]],
        scheme: Optional[str] = None,
        reduction: str = "mean",
        batch_size: int = 256,
    ) -> Dict[str, List[float]]:
        """Batched AA-only log-likelihood split around CDR3 for segmented single-chain samples.

        The post-prediction region is rendered as ``cdr1 <SEP> cdr2 <SEP> cdr3`` so the number
        of <SEP> tokens between the prediction tag and a given amino-acid token identifies the CDR:
        0 -> CDR1, 1 -> CDR2, 2 -> CDR3.

        Returns dict with keys:
          - "no_cdr3": amino-acid tokens in framework + CDR1 + CDR2 (full without CDR3)
          - "cdr3": amino-acid tokens in CDR3 only
        """
        if reduction not in {"mean", "sum"}:
            raise ValueError(_REDUCTION_ERROR)
        if not items:
            return {"no_cdr3": [], "cdr3": []}

        texts = self._render_segment_texts(items, scheme)

        out_no_cdr3: List[float] = []
        out_cdr3: List[float] = []

        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            per_pos_loss, aa_mask, pred_pos, input_ids, tgt_pos = self._batch_forward_region_tensors(chunk)

            # Number of <SEP> tokens between the prediction tag and each label position.
            sep_indicator = (input_ids == self.sep_id).long()  # [B, L]
            cumsep_excl = torch.cumsum(sep_indicator, dim=1) - sep_indicator  # seps strictly before each position
            seps_before_pred = cumsep_excl.gather(1, pred_pos.unsqueeze(1))  # [B, 1]

            seps_excl_at_tgt = cumsep_excl.gather(1, tgt_pos)  # [B, L-1]
            seps_after_pred = seps_excl_at_tgt - seps_before_pred  # [B, L-1]

            cdr_mask = aa_mask & (tgt_pos > pred_pos.unsqueeze(1))

            cdr3_mask = cdr_mask & (seps_after_pred >= 2)
            no_cdr3_mask = aa_mask & ~cdr3_mask

            ll_no_cdr3 = self._masked_ll_from_include_mask(
                per_pos_loss=per_pos_loss,
                include_mask=no_cdr3_mask,
                reduction=reduction,
                mask_name="no_cdr3",
            )
            ll_cdr3 = self._masked_ll_from_include_mask(
                per_pos_loss=per_pos_loss,
                include_mask=cdr3_mask,
                reduction=reduction,
                mask_name="cdr3",
            )

            out_no_cdr3.extend(ll_no_cdr3.detach().cpu().tolist())
            out_cdr3.extend(ll_cdr3.detach().cpu().tolist())

        return {
            "no_cdr3": out_no_cdr3,
            "cdr3": out_cdr3,
        }

    def _validate_paired_segment_item(self, item: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Validate a paired item with H+L segments and return (h_segs, l_segs)."""
        required_h = ("h_fr1", "h_cdr1", "h_fr2", "h_cdr2", "h_fr3", "h_cdr3", "h_fr4")
        required_l = ("l_fr1", "l_cdr1", "l_fr2", "l_cdr2", "l_fr3", "l_cdr3", "l_fr4")
        for k in required_h + required_l:
            if k not in item:
                raise KeyError(f"Paired item missing required key: {k}")
        h_segs = {k.split("_", 1)[1]: str(item[k]) for k in required_h}
        l_segs = {k.split("_", 1)[1]: str(item[k]) for k in required_l}
        return h_segs, l_segs

    def _render_paired_text(
        self,
        h_segs: Dict[str, str],
        l_segs: Dict[str, str],
        scheme: Optional[str],
        order: str = "L-first",
    ) -> str:
        """Render a single paired text using shared scheme; uses include_scheme_token from init."""
        h_sample = {"chain_type": "H", "segments": h_segs, "scheme": scheme}
        l_sample = {"chain_type": "L", "segments": l_segs, "scheme": scheme}
        return render_paired(
            h_sample=h_sample, l_sample=l_sample, order=order, include_scheme=self.include_scheme_token
        )

    def _batch_forward_paired_region_tensors(
        self, chunk: List[str]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Paired forward: returns (per_pos_loss, aa_mask, hpred_pos, lpred_pos, input_ids, tgt_pos)."""
        per_pos_loss, aa_mask, input_ids, tgt_pos = self._batch_forward_aa_loss(chunk)
        hpred_pos = self._exactly_one_token_pos(input_ids, self.hpred_id, "<HPRED>")
        lpred_pos = self._exactly_one_token_pos(input_ids, self.lpred_id, "<LPRED>")
        return per_pos_loss, aa_mask, hpred_pos, lpred_pos, input_ids, tgt_pos

    def batch_log_likelihood_paired_regions_from_segments(
        self,
        items: List[Dict[str, Any]],
        scheme: Optional[str] = None,
        reduction: str = "mean",
        batch_size: int = 256,
        order: str = "L-first",
    ) -> Dict[str, List[float]]:
        """Batched AA-only log-likelihood by region for cognate-paired H+L samples.

        Each item must contain h_{fr1,cdr1,fr2,cdr2,fr3,cdr3,fr4} and l_{...}.
        Rendered as a single paired sequence (frameworks-first layout). Within that
        sequence:
          - heavy_framework  : amino acids inside the H-frameworks block
          - light_framework  : amino acids inside the L-frameworks block
          - heavy_cdr        : amino acids after <HPRED>
          - light_cdr        : amino acids after <LPRED>
          - heavy_cdr3       : amino acids in CDR3 segment of <HPRED> block
          - light_cdr3       : amino acids in CDR3 segment of <LPRED> block
          - heavy_no_cdr3    : H_framework + H_cdr1 + H_cdr2 (CDR3 excluded)
          - light_no_cdr3    : L_framework + L_cdr1 + L_cdr2 (CDR3 excluded)
          - heavy_full       : H_framework + H_cdr (all H aa)
          - light_full       : L_framework + L_cdr (all L aa)
          - paired_full      : all amino acids
          - paired_framework : H_framework + L_framework
          - paired_cdr       : H_cdr + L_cdr
        """
        if reduction not in {"mean", "sum"}:
            raise ValueError(_REDUCTION_ERROR)
        if not items:
            keys = (
                "heavy_full",
                "heavy_framework",
                "heavy_cdr",
                "heavy_no_cdr3",
                "heavy_cdr3",
                "light_full",
                "light_framework",
                "light_cdr",
                "light_no_cdr3",
                "light_cdr3",
                "paired_full",
                "paired_framework",
                "paired_cdr",
            )
            return {k: [] for k in keys}

        texts: List[str] = []
        for it in items:
            h_segs, l_segs = self._validate_paired_segment_item(it)
            texts.append(self._render_paired_text(h_segs=h_segs, l_segs=l_segs, scheme=scheme, order=order))

        outputs: Dict[str, List[float]] = {
            "heavy_full": [],
            "heavy_framework": [],
            "heavy_cdr": [],
            "heavy_no_cdr3": [],
            "heavy_cdr3": [],
            "light_full": [],
            "light_framework": [],
            "light_cdr": [],
            "light_no_cdr3": [],
            "light_cdr3": [],
            "paired_full": [],
            "paired_framework": [],
            "paired_cdr": [],
        }

        # In the paired (frameworks-first) layout, the sequence looks like:
        #   <BOS>[scheme?] <X1> X1FRs <X2> X2FRs <X1PRED> X1CDRs <X2PRED> X2CDRs <EOS>
        # where (X1, X2) = (L, H) for L-first or (H, L) for H-first.
        # Frameworks block for each chain: amino acids between its <chain> tag and the next stop tag.
        # CDR block: amino acids between <chainPRED> and the next stop tag.

        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            (
                per_pos_loss,
                aa_mask,
                hpred_pos,
                lpred_pos,
                input_ids,
                tgt_pos,
            ) = self._batch_forward_paired_region_tensors(chunk)

            B, L = input_ids.shape

            # H/L tag positions: framework block of each chain runs from its tag to the next stop tag.
            h_tag_pos = self._exactly_one_token_pos(input_ids, self.h_id, "<H>")
            l_tag_pos = self._exactly_one_token_pos(input_ids, self.l_id, "<L>")

            # Compute boundaries (next stop after each chain tag)
            stop_token_ids = (self.h_id, self.l_id, self.hpred_id, self.lpred_id, self.eos_id)

            def next_stop(after_pos: torch.Tensor) -> torch.Tensor:
                """For each row, return the smallest position > after_pos where input_ids is a stop tag."""
                pos_idx = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)  # [B, L]
                is_stop = torch.zeros_like(input_ids, dtype=torch.bool)
                for tid in stop_token_ids:
                    is_stop = is_stop | (input_ids == tid)
                # Mask positions <= after_pos
                gt_after = pos_idx > after_pos.unsqueeze(1)
                candidate = is_stop & gt_after
                # If no stop after, fall back to L (so range is empty downstream)
                large_fill = torch.full_like(pos_idx, L)
                pos_or_fill = torch.where(candidate, pos_idx, large_fill)
                return pos_or_fill.min(dim=1).values  # [B]

            h_fr_end = next_stop(h_tag_pos)  # exclusive end of H framework block (token-id index)
            l_fr_end = next_stop(l_tag_pos)
            h_cdr_end = next_stop(hpred_pos)
            l_cdr_end = next_stop(lpred_pos)

            # tgt_pos values are in [1, L-1]; corresponds to label position in unshifted seq
            in_h_framework = aa_mask & (tgt_pos > h_tag_pos.unsqueeze(1)) & (tgt_pos < h_fr_end.unsqueeze(1))
            in_l_framework = aa_mask & (tgt_pos > l_tag_pos.unsqueeze(1)) & (tgt_pos < l_fr_end.unsqueeze(1))
            in_h_cdr = aa_mask & (tgt_pos > hpred_pos.unsqueeze(1)) & (tgt_pos < h_cdr_end.unsqueeze(1))
            in_l_cdr = aa_mask & (tgt_pos > lpred_pos.unsqueeze(1)) & (tgt_pos < l_cdr_end.unsqueeze(1))

            in_h_full = in_h_framework | in_h_cdr
            in_l_full = in_l_framework | in_l_cdr
            in_paired_full = in_h_full | in_l_full
            in_paired_framework = in_h_framework | in_l_framework
            in_paired_cdr = in_h_cdr | in_l_cdr

            # CDR3 split via SEP-count after each pred tag (same logic as cdr3_split single-chain)
            sep_indicator = (input_ids == self.sep_id).long()
            cumsep_excl = torch.cumsum(sep_indicator, dim=1) - sep_indicator  # seps strictly before each position

            # Restrict cumsep to the H-cdr / L-cdr block to count seps relative to the pred tag.
            seps_before_hpred = cumsep_excl.gather(1, hpred_pos.unsqueeze(1))
            seps_before_lpred = cumsep_excl.gather(1, lpred_pos.unsqueeze(1))
            seps_excl_at_tgt = cumsep_excl.gather(1, tgt_pos)
            seps_after_hpred = seps_excl_at_tgt - seps_before_hpred
            seps_after_lpred = seps_excl_at_tgt - seps_before_lpred

            in_h_cdr3 = in_h_cdr & (seps_after_hpred >= 2)
            in_l_cdr3 = in_l_cdr & (seps_after_lpred >= 2)
            in_h_no_cdr3 = in_h_full & ~in_h_cdr3
            in_l_no_cdr3 = in_l_full & ~in_l_cdr3

            region_masks = {
                "heavy_full": in_h_full,
                "heavy_framework": in_h_framework,
                "heavy_cdr": in_h_cdr,
                "heavy_no_cdr3": in_h_no_cdr3,
                "heavy_cdr3": in_h_cdr3,
                "light_full": in_l_full,
                "light_framework": in_l_framework,
                "light_cdr": in_l_cdr,
                "light_no_cdr3": in_l_no_cdr3,
                "light_cdr3": in_l_cdr3,
                "paired_full": in_paired_full,
                "paired_framework": in_paired_framework,
                "paired_cdr": in_paired_cdr,
            }
            for name, mask in region_masks.items():
                ll = self._masked_ll_from_include_mask(
                    per_pos_loss=per_pos_loss,
                    include_mask=mask,
                    reduction=reduction,
                    mask_name=name,
                )
                outputs[name].extend(ll.detach().cpu().tolist())

        return outputs

    def log_likelihood_from_lengths(
        self,
        chain_type: str,
        sequence: str,
        lens_by_scheme: Dict[str, List[int]],
        scheme: str,
        reduction: str = "mean",
    ) -> float:
        """Compute log-likelihood from raw sequence and lengths by scheme.

        Parameters
        ----------
        chain_type : str
            "H" or "L"
        sequence : str
            Full amino acid sequence
        lens_by_scheme : Dict[str, List[int]]
            Mapping from scheme name to segment lengths. e.g.,
             {"imgt": [len_fr1, len_cdr1, len_fr2, len_cdr2, len_fr3, len_cdr3, len_fr4]}
        scheme : str
            Scheme used for segmentation: ("imgt", "kabat", "chothia")
        reduction : str, optional
            Reduction method for log-likelihood ("mean", "sum"), by default "mean"
        """
        lens = lens_by_scheme[scheme]
        segs = slice_segments(sequence, lens)
        item = {
            "chain_type": chain_type,
            "fr1": segs["fr1"],
            "cdr1": segs["cdr1"],
            "fr2": segs["fr2"],
            "cdr2": segs["cdr2"],
            "fr3": segs["fr3"],
            "cdr3": segs["cdr3"],
            "fr4": segs["fr4"],
        }
        return self.log_likelihood_from_segments(item=item, scheme=scheme, reduction=reduction)


# Backwards-compatible alias for the original class name.
IgGenCDRGenerator = GenCDRGenerator
