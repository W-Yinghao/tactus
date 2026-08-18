# TACTUS

**T**ouch **A**lignment by **C**ontrastive **T**ransfer to **U**nseen **S**ubjects — EEG–video
contrastive learning on [OpenNeuro ds005662](https://openneuro.org/datasets/ds005662)
("A comprehensive EEG dataset for investigating visual touch perception", *Scientific Data* 13:381, 2026).

The dataset's distinguishing feature is its dense shared design: **80 subjects × the same 360
conditions (90 videos × 4 orientations) × 8 repeats = 230,400 trials**. TACTUS uses that to train
a subject-invariant EEG–video embedding and to test the claim that matters — retrieval for an
**unseen subject watching an unseen video**.

## Headline result

Primary endpoint is pre-registered: `test/video/g18/top1_pseudo` — 18-way retrieval against a
source-video gallery, pseudo-trial k=4, eeg→video, aggregated to n=80 subject-level scores.
Chance = 5.56%.

| Regime / arm | Folds | Primary | 95% CI | Perm p | Frac. of ceiling |
|---|---|---|---|---|---|
| within_subject · NICE+InfoNCE | 5 | 10.99% | 10.51–11.47 | 0.0002 | — |
| within_subject · FHMC (flagship) | 5 | 10.88% | 10.36–11.41 | 0.0002 | — |
| within_subject · NICE+ProtoNCE | 5 | **11.93%** | 11.32–12.55 | 0.0002 | 0.814 |
| within_subject · EA+ridge (linear floor) | 5 | 9.89% | 9.47–10.29 | 0.001 | — |
| **double_disjoint · NICE+ProtoNCE** | **40** | **10.45%** | 9.98–10.92 | **0.0002** | **0.835** |

The double-disjoint cell (5 video folds × 8 subject folds, every subject held out once per video
fold) is the headline. It sits 1.5 points below the within-subject number while recovering a slightly **larger** share of
what the data supports — 83.5% of the split-half noise ceiling versus 81.4%.

Both ceiling fractions are computed against a gallery pinned at 10 subjects. Until that was pinned
the two regimes were divided by different denominators and the gap read 16 points rather than 2; see
`STATUS.md` §10 (D15), which also states why only the raw accuracies and CIs should leave this
repository for now.

Permutation inference uses the **source video** as the exchangeable unit. The trial-level null is
computed only to show it is 3.5–4.1× too narrow; it never supplies a reported p-value.

Full numbers, the defect ledger, and open decisions: **[`tactus/STATUS.md`](tactus/STATUS.md)**.

## Layout

```
BLUEPRINT_v2.md          scientific design, statistical protocol, risk register
tactus/
  MISSION.md             execution brief: gates, frozen decisions D1-D8
  STATUS.md              canonical status document (English; STATUS.zh.md is archived)
  tactus/
    data/                download, event parsing, preprocessing, splits, QC
    models/              EEG encoders, subject conditioning, video towers, FM wrappers
    losses/              the swap point -- 10 contrastive objectives behind one registry
    baselines/           MVPA, CorrCA/ISC, SRM, EA+ridge
    eval/                retrieval, permutation, noise ceilings, RSA, probes, reports
    train/               trainer and fold runner
  configs/               one YAML per arm; `loss.name` selects the objective
  slurm/                 pool.py worker-pool scheduler, cluster.conf
  tests/
server_handoff/          standalone Phase-0 audit scripts
```

## Adding a contrastive objective

This is what the repository is built for. Two edits:

```python
# tactus/losses/my_loss.py
from .base import ContrastiveLoss, register_loss

@register_loss("my_loss")
class MyLoss(ContrastiveLoss):
    requires_meta = ("condition_id", "video_id")   # missing keys fail in second 1, not hour 3
    def forward(self, z_eeg, z_vid, meta):
        ...
        return {"loss": loss, "logs": {"n_valid": float(n)}}
```

```python
# tactus/losses/__init__.py  -- the decorator only runs if the module is imported
from .my_loss import MyLoss
```

That second edit is the one that gets forgotten, and forgetting it is silent: the
file exists, the class is written, and `loss.name: my_loss` raises "unknown loss"
only when a job starts. It is how this repository's own flagship objective went
unrun. `tests/test_loss_registry_complete.py` fails if any loss file declares
`@register_loss` without a matching import line, and names the line to add.

Then a config with one changed key (`loss.name: my_loss`) and:

```bash
python slurm/pool.py submit --name train_myloss --tasks 0-4 --workers 5 --gpus 1 \
  --partition V100,P100,A30,A40,3090,L40S,A100 --time 08:00:00 --cpus 4 --mem 48G \
  --cmd 'python -u -m tactus.train.run -c configs/my_loss.yaml --regime within_subject --folds {task}'
```

Three semantics worth knowing before you do: `loss.name` **replaces** the whole loss block rather
than merging into it; **batch composition is a correctness property, not a hyperparameter**
(`batch.mode=distinct_video` is what makes plain InfoNCE safe against a 360-entry frozen codebook);
and every loss reports `n_valid` and `degenerate` — plot them, because a term silently contributing
zero looks identical to a working one in the logs. See
[`tactus/tactus/losses/README.md`](tactus/tactus/losses/README.md).

## Running the pipeline

```bash
bash slurm/setup_env.sh                     # conda env `tactus`
python -m tactus.losses                     # 10 losses x 8 adversarial batches, then term shares
python -m tactus.models.selftest --all-registered
pytest tests/ -q

python -m tactus.data.download --what all --dest $TACTUS_BIDS_ROOT
python -m tactus.data.events --bids-root $TACTUS_BIDS_ROOT --out $TACTUS_DATA_ROOT/derived/trials.parquet
python slurm/pool.py submit --name epochs --tasks 1-80 --workers 14 ...
```

`slurm/pool.py` exists because this cluster caps a user at **30 queued jobs counted in array
elements** and **8 concurrent GPUs**, so the natural `--array=1-80` cannot be submitted at all. It
runs N tasks through W ≪ N worker jobs claiming from a shared directory.

## What this repository does not contain

No data. The 108 GB of raw BDF, 17 GB of epoch memmaps, video-embedding caches, checkpoints and
run artefacts live under `$TACTUS_WORK` and are reproducible from the code here.

## Notes on the results

Read `STATUS.md` §7 and §10 before trusting any number in a fork. Every module executed for the
first time had at least one real defect, and almost none of them crashed — they produced plausible
numbers. The four that would have invalidated a published result:

- the prototype objective was solvable **without any EEG–video relationship** (100% training
  accuracy on pure-noise EEG) until `live_positive` was pinned off;
- a composite-loss warmup counter was never advanced, so every warmup-gated term had weight
  exactly 0 for entire runs;
- `transformers` 5.15 loads **any** published VideoMAE checkpoint with its trained q/v attention
  biases silently zero-filled (no `_checkpoint_conversion_mapping`) — repairing it changed the
  top-1 nearest video for 61% of the 90 stimuli;
- the flagship objective's disentanglement penalty was a cross-covariance between L2-normalized
  heads, so it scaled as 1/(d_a·d_b) and read 5.8e-07 while the coordinates it was meant to
  separate were correlated at 0.788 — the term the architecture is named for never applied.

Three of those four are the same shape: a term or a cache that is present, exercised, logged, and
inert. `tactus.losses.term_contributions()` and the cache fingerprints exist because noticing them
by hand worked only by luck.

## License and data

Code: see `LICENSE` if present; otherwise all rights reserved pending publication.
ds005662 is CC0. This repository redistributes none of it.
