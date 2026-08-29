# D23 pre-registration — three-space time-resolved partial RSA (FROZEN)

**Frozen: 2026-08-29, before any EEG-to-space correlation was computed.** The git commit
carrying this file is the timestamp. Nothing below may change after that commit; deviations
require a dated appendum committed BEFORE viewing the affected aggregate. Internal
pre-registration only — nothing here authorizes any external posting (user ruling: no arXiv,
no OSF at this stage).

## Question

Does the EEG response to *seen* touch carry a tactile-semantic geometry beyond what visual
semantics, material category, affect ratings and low-level image statistics already explain —
and if so, when does it emerge (H1) and is its strength gated by the vicarious-touch
phenotype (H2)?

## Frozen inputs

| Input | Path (under `$TACTUS_WORK` / `$TACTUS_BIDS`) | Note |
|---|---|---|
| Visual space A | `derived/video_emb/siglip2-base.npz` `base_emb (90, 768)` | frozen cache |
| Frame embeddings (for B1 scoring) | `derived/video_emb/siglip2-base-frames.npz` `frame_emb (360, 15, 768)` | frozen cache |
| Affect space C v1 | `ds005662/code/analysis/VTD.csv` columns valence, arousal, threat, pain | continuous means; the 90x4 rater-count table is NOT on disk (D25) |
| EEG | epochs, window `w0600` (0–600 ms, 200 Hz, T=120), all 80 subjects | post-D11 (sub-17 interpolated) |
| Phenotype | `ds005662/phenotype/VT_data.tsv` via `eval/covariates.py` | binarised VT>0: 33 vs 47 |
| SNR covariate | per-subject split-half reliability (D13 primary covariate) | |
| Orientation specificity covariate | `orientation_peak` from D13 covariate table | |
| Stimuli (low-level control) | `ds005662/code/experiment_files/stimuli/*600ms*/*.mp4` | 90x4 grid |

## Space definitions (exact)

- **A — visual-semantic**: RDM = 1 − cosine over `base_emb` (90×90).
- **B1 — tactile-adjective projection**: the 30 adjectives below, each embedded with SigLIP2
  text tower under 3 templates (`"{adj}"`, `"a {adj} surface"`, `"something that feels {adj}
  to touch"`), L2-normalised then averaged then re-normalised. Score = cosine(frame emb,
  adjective emb), mean over 15 frames, mean over the 4 orientations → (90, 30) profile;
  RDM = 1 − Pearson over profiles. **Declared limitation: B1 reweights directions of the same
  SigLIP2 space as A; it is not an independent modality tower.**
  Adjectives (frozen): rough, smooth, soft, hard, sharp, blunt, sticky, slippery, wet, dry,
  warm, cold, fuzzy, furry, silky, prickly, spiky, bumpy, grainy, slimy, greasy, rubbery,
  firm, squishy, spongy, coarse, springy, heavy, light, textured.
  Excluded as affective (belong to C, listed for transparency): painful, ticklish, pleasant,
  unpleasant, soothing, irritating.
- **B2 — ImageBind vision tower** (independent-tower test): imagebind_huge image encoder on
  the same 15 frames, pooled as in B1, out-of-band `.npz` per `encode.py`'s documented
  mechanism. Conditional: if the install/weights fail, that is reported and every claim
  carries the "within-tower projection" qualifier from B1's limitation.
- **C — affect v1**: RDM = Euclidean over z-scored (valence, arousal, threat, pain).
  C v2 (rater-count distributions, Bhattacharyya) only if D25 locates the table; v2 would be
  an appendum, not a swap.
- **Controls (partialled in every primary statistic)**: `material` (categorical RDM,
  MANDATORY — without it a tactile result is presumed to be material decoding renamed),
  `lowlevel` (per-video mean frame luminance, RMS contrast, and mean absolute inter-frame
  difference (motion energy), z-scored, Euclidean RDM).

## EEG RDMs

Per subject: `time_resolved_rdms(x, base_video_id, n_conditions=90, method="crossnobis",
n_folds=4, window=5, step=2, whiten_method="ledoit-wolf", seed=0)` — 25 ms patterns, 10 ms
stride, Ledoit-Wolf whitening, orientation folded into the 8+ repeats per base video.
Group EEG RDM = mean over the 80 subject RDM stacks (equal weight; each subject's RDM stack
z-scored across cells first to stop any subject owning the average — the D11 lesson).

## Inference

`rsa_time_course` with `model_rdms = {A, B1, B2?, C, material, lowlevel}`,
`control_models` per contrast below, **n_perm=5000** (the 1000 default is below this
project's own standard; 100→1000 once flipped a conclusion), `alpha_pointwise=0.05`,
video-level base permutation (exchangeable unit = source video), max-cluster-mass null,
seed=0.

Primary contrasts:
- **B1 partial** = B1 | {A, C, material, lowlevel} — the tactile-beyond-everything curve.
- **A partial** = A | {B1, C, material, lowlevel} — the visual reference curve.
- **B2 partial** (if built) = B2 | {A, C, material, lowlevel}.

## H1 (timing) — operational

Supported iff BOTH: (i) B1-partial has ≥1 significant cluster (p_cluster<0.05) with onset
> 150 ms; (ii) A-partial has a significant cluster with onset ≤ 150 ms. Onset = first
timepoint of the earliest significant cluster. Onset uncertainty: 1000 bootstrap resamples
of the 80 subjects (rebuild group RDM, re-run against the SAME permutation null threshold),
report the onset CI. No cluster for B1-partial → grid cell 3 below, H1 not evaluable.

## H2 (phenotype gating) — operational

Per-subject tactile alignment = mean partial Spearman of B1 | {A, material, lowlevel} over
the FIXED window **150–600 ms** (frozen now, NOT derived from H1's observed clusters, to
keep H2 non-circular) on that subject's own RDM stack.
- Primary: one-sided Mann-Whitney U, VT>0 (n=33) > VT=0 (n=47), alpha=0.05. MDD d=0.64 is
  quoted with any null.
- Specificity (both must hold for a gating claim): (a) the same test on `orientation_peak`
  is null; (b) the VT effect survives rank-regression adjustment for split-half reliability.

## QC sentinels — gate BEFORE any aggregate is viewed

1. Determinism: two independent builds of every space RDM and of one subject's EEG RDM stack
   are bit-identical.
2. Manipulation check on B1 scoring: the (sharp+prickly+spiky)/3 composite score correlates
   positively with VTD `threat` over the 90 videos, video-permutation p<0.05. Fails → B1 is
   noise → STOP, report, run nothing downstream.
3. Control effectiveness: after partialling material, residual Spearman with the material
   RDM itself is |r|<0.01; `partial_spearman(a, b, controls=a)` ≈ 0.
4. Coverage: exactly 80 subjects × 90 valid conditions; every drop carries a reason code and
   is counted in the report. No silent except→continue.
5. Probe before fleet: full pipeline on ONE subject + group path on 3 timepoints passes
   gates 1–4 before the 80-subject run is submitted.

## Pre-committed interpretation grid

| Cell | Observation | Reading (frozen) |
|---|---|---|
| H1-1 | B1-partial cluster >150 ms AND A-partial ≤150 ms | tactile-semantic geometry beyond visual/material/affect/low-level, late emergence; training version (D23b) goes ahead |
| H1-2 | B1-partial cluster exists, timing condition fails | tactile axis present, timing claim unsupported; D23b goes ahead, H1 reported as partial |
| H1-3 | no B1-partial cluster | no tactile-semantic variance beyond the controls; D23b cancelled, D26 loses its premise; reported as a boundary result |
| H1-4 | B1 cluster only WITHOUT material partialled | "tactile alignment" was material decoding renamed — design-lesson section, NOT an alignment claim |
| H2-1 | VT effect, orientation null, survives SNR adjustment | phenotype-gating claim, exploratory-tier wording |
| H2-2 | VT effect but orientation also predicts | non-specific (global SNR/attention), no gating claim |
| H2-3 | no VT effect | null with the MDD statement (33/47 design caveat) |
| B2-absent | ImageBind not built | all tactile claims carry the within-tower qualifier |

## Forbidden / required wording

- Forbidden: "the brain represents touch", any mechanism wording; "tactile representation"
  for B1 results (required: "SigLIP2's tactile-adjective directions"); any cross-dataset
  claim (D26 deferred); any external quote of ceiling fractions (D15 rule stands).
- Required: B1's within-tower limitation wherever B2 is absent; MDD next to every null;
  n_perm and exchangeable unit named in every table caption.

## Compute

CPU only (space builds, RSA, permutations) — submitted via sbatch, not the login node,
per-subject checkpointing, skip-if-done. B2 embedding is the only GPU step: one job,
A100/L40S/H100 partitions, via slurm.
