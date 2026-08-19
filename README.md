![Python version](https://img.shields.io/badge/python-3.10–3.11-blue.svg)

# GenCDR

**GenCDR is a family of generative antibody language models for autoregressive
complementarity-determining region (CDR) design and posterior alignment through reinforcement learning.** It conditions generation on
**framework prompts** and jointly generates compatible, variable-length CDR1, CDR2, and CDR3
sequences as the **CDR response**. The family supports conventional antibody chains, nanobodies,
and cognate heavy-light chain pairs.

For a single heavy chain, tokens are rendered as follows (light chains use `<L>` and `<LPRED>`):

```math
\underbrace{
  \mathtt{\langle BOS\rangle}\;\mathtt{\langle H\rangle}
}_{\text{chain}}
\quad
\underbrace{
  \mathtt{FR1}\;\mathtt{\langle SEP\rangle}\;
  \mathtt{FR2}\;\mathtt{\langle SEP\rangle}\;
  \mathtt{FR3}\;\mathtt{\langle SEP\rangle}\;
  \mathtt{FR4}
}_{\text{framework prompt}}
\quad
\underbrace{
  \mathtt{\langle HPRED\rangle}
}_{\substack{\text{generation} \\ \text{boundary}}}
\quad
\underbrace{
  \mathtt{CDR1}\;\mathtt{\langle SEP\rangle}\;
  \mathtt{CDR2}\;\mathtt{\langle SEP\rangle}\;
  \mathtt{CDR3}
}_{\text{CDR response}}
\quad
\mathtt{\langle EOS\rangle}
```

For paired generation, both framework prompts precede both CDR responses. This framework-first
layout gives every CDR the complete framework context, supports variable-length design, and
separates framework and CDR likelihoods for scoring and posterior [alignment](#alignment-weighted-dpo).

All models use a decoder-only LLaMA architecture:

| Model | Setting | Training corpus |
|---|---|---|
| **IgGenCDR** | Heavy or light chain | ~254M unpaired human OAS sequences |
| **p-IgGenCDR** | Paired heavy-light chains | ~1.8M paired OAS sequences; fine-tuned from IgGenCDR |
| **NanoGenCDR** | Nanobody/VHH | ~6.5M INDI VHH sequences; fine-tuned from IgGenCDR with a 50% human OAS heavy-chain batch mix |

See [Usage](#usage) for the Python and command-line interfaces.

## Setup

Clone the repository if you want a local checkout:

```bash
git clone https://github.com/AstraZeneca/GenCDR.git
cd GenCDR
```

Then install with either [Poetry](https://python-poetry.org/):

```bash
# Core: generation and scoring of pre-segmented sequences
poetry install

# Optional: raw-sequence segmentation and plots
poetry install --extras scoring

# Optional: weighted DPO alignment
poetry install --extras align
```

or pip:

```bash
# Core
pip install .

# Optional: raw-sequence segmentation and plots
pip install ".[scoring]"

# Optional: weighted DPO alignment
pip install ".[align]"
```

Alternatively, install directly from GitHub without a local clone:

```bash
# Core
pip install "gencdr@git+https://github.com/AstraZeneca/GenCDR.git"

# Optional: raw-sequence segmentation and plots
pip install "gencdr[scoring]@git+https://github.com/AstraZeneca/GenCDR.git"

# Optional: weighted DPO alignment
pip install "gencdr[align]@git+https://github.com/AstraZeneca/GenCDR.git"
```

> Model weights are not included in this repository. Release details will be added under
> [Obtaining model weights](#obtaining-model-weights).

## Usage

```python
from gencdr import GenCDRGenerator

gen = GenCDRGenerator.from_pretrained("iggencdr", device="cuda")

# Generate CDRs for a heavy chain given its four frameworks
results = gen.generate_cdrs_from_frameworks(
    chain_type="H",
    frameworks={"fr1": "EVQLVESGGGLVQPGGSLRLSCAAS",
                "fr2": "WVRQAPGKGLEWVS",
                "fr3": "RFTISRDNSKNTLYLQMNSLRAEDTAVYYCAK",
                "fr4": "WGQGTLVTVSS"},
    n_samples=10,
    temperature=1.0,
    scheme="imgt",
)
for r in results:
    if r["parsed_ok"]:
        print(r["cdr1"], r["cdr2"], r["cdr3"])
```

Joint paired (p-IgGenCDR) generation:

```python
results = gen.generate_paired_cdrs_from_frameworks(
    h_frameworks={"fr1": "...", "fr2": "...", "fr3": "...", "fr4": "..."},
    l_frameworks={"fr1": "...", "fr2": "...", "fr3": "...", "fr4": "..."},
    n_samples=10,
    order="L-first",
)
for r in results:
    print(r["H"]["cdr3"], r["L"]["cdr3"])
```

Region log-likelihood of a segmented sample:

```python
item = {"chain_type": "H", "fr1": "...", "cdr1": "...", "fr2": "...", "cdr2": "...",
        "fr3": "...", "cdr3": "...", "fr4": "..."}
ll = gen.log_likelihood_from_segments(item, reduction="mean")
regions = gen.batch_log_likelihood_regions_from_segments([item])  # {'full','framework','cdr'}
```

### Command line

Installing the package provides a `gencdr` command:

```bash
# Generate single-chain CDRs from a frameworks JSON
gencdr generate --model iggencdr --frameworks examples/frameworks/heavy_vh.json \
    --chain-type H --n-samples 20 --out-csv designs.csv

# Joint paired generation
gencdr generate-paired --model p-iggencdr --frameworks examples/frameworks/paired_hl.json \
    --n-samples 20 --out-csv paired_designs.csv

# Score raw sequences in a CSV (requires the 'scoring' extra for segmentation)
gencdr score --model iggencdr --csv-path sequences.csv --sequence-column heavy_chain
```

Runnable scripts are in [`examples/`](examples/).

## Alignment (weighted DPO)

GenCDR generation can be steered towards user objectives using a **scalar reward** with the scaffold-grouped weighted DPO
objective described in the paper. To run alignment one exclusively needs a
[**reward CSV**](examples/rewards/single_rewards.csv) with pre-computed rewards per sequence and framework-cdr splits. You bring the rewards from whatever downstream objective you care about (binding, developability,
a structural score, an assay readout), and GenCDR does the preference optimization.

Alignment compares samples within scaffold groups using their scalar rewards and optimizes the
policy against a frozen reference over CDR-response tokens only. The best checkpoint is selected
by validation Spearman correlation between CDR likelihood and reward.

Install the extra and run:

```bash
poetry install --extras align

gencdr align --model iggencdr \
    --reward-csv examples/rewards/single_rewards.csv \
    --output-dir /tmp/gencdr_aligned \
    --beta 0.15 --max-epochs 5 --batch-size 16
```

Or from Python:

```python
from gencdr.alignment.aligner import WeightedDPOAligner

aligner = WeightedDPOAligner(model="iggencdr", output_dir="/tmp/gencdr_aligned", beta=0.15)
best_ckpt = aligner.run(["examples/rewards/single_rewards.csv"])
```

The reference policy defaults to the model being aligned. The aligned model and tokenizer are
saved to `--output-dir` (weights are not shipped with this package).

**Reward CSV schema (single-chain).** Required columns:

| Column | Meaning |
|-----------------------------------|-----------------------------------------------|
| `meta_framework_source`           | Scaffold group label |
| `meta_fr1`, `meta_fr2`, `meta_fr3`, `meta_fr4` | Framework regions (the prompt) |
| `meta_cdr1`, `meta_cdr2`, `meta_cdr3` | CDR regions (the completion) |
| `reward`                          | Scalar reward (higher is better) |

Optional: `meta_chain_type` (`H` default / `L`), `meta_scheme`. A worked example CSV is at
[`examples/rewards/single_rewards.csv`](examples/rewards/single_rewards.csv).

**Paired alignment.** Passing `--paired` aligns p-IgGenCDR from
`meta_h_*` / `meta_l_*` columns (`meta_h_fr1..4`, `meta_l_fr1..4`, `meta_h_cdr1..3`,
`meta_l_cdr1..3`, plus `meta_framework_source` and `reward`). Paired alignment reuses the same
objective and generalizes the single-chain method.

## Obtaining model weights

Model weights and download instructions will be added here when they are released.

## License

Apache-2.0. See [LICENSE](LICENSE).
