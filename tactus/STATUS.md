# TACTUS STATUS  (updated 2026-08-18T16:20Z)

## Stage: Phase 0 and Phase 1 complete. Baseline ladder complete through rung 5.
## Gates: **G0 G1 G2 G3 G4 G5 G6a** passed. **G6b (ocular) deliberately not declared** — see §6.

> Canonical status document required by `MISSION.md` §0. Every number below is traceable to a
> named file under `/projects/EEG-foundation-model/tactus_work/`.
> The earlier Chinese edition is kept verbatim at `STATUS.zh.md`; this file is authoritative.

---

# 1. Headline result

**The blueprint's main claim holds: zero-shot retrieval for an unseen subject watching an
unseen video.**

Primary endpoint throughout is the pre-registered `test/video/g18/top1_pseudo` (decision D4):
18-way, source-video gallery, pseudo-trial k=4, eeg→video, aggregated to n=80 subject-level
scores. Chance = 5.56%.

| Regime / arm | Folds | Primary | 95% CI | Perm p | z | Frac. of ceiling | Beyond material |
|---|---|---|---|---|---|---|---|
| within_subject · NICE+InfoNCE | 5 | 10.99% | 10.51–11.47 | 0.0002 | 11.5 | 0.614 | 74.8% |
| within_subject · NICE+ProtoNCE | 5 | **11.93%** | 11.32–12.55 | 0.0002 | 12.6 | 0.674 | 75.0% |
| within_subject · EA+ridge (linear floor) | 5 | 10.09% | 9.69–10.48 | 0.001 | 10.3 | ≈0.57 | — |
| **double_disjoint · NICE+ProtoNCE** | **40** | **10.45%** | 9.98–10.92 | **0.0002** | **12.6** | **0.835** | **76.9%** |

The double-disjoint cell is the headline: 5 video folds × 8 subject folds, every subject held
out exactly once per video fold. It sits only 1.5 points below the within-subject number while
recovering a **larger** share of what the data can support — 83.5% of the split-half noise
ceiling versus 67.4% — because the cross-subject ceiling is itself lower (0.125 vs 0.177).

Permutation inference uses the **source video** as the exchangeable unit, 5000 permutations;
p is at the 1/5001 floor in every arm. The deliberately wrong trial-level null is reported only
as a contrast: its SD is **4.12×** narrower (within_subject) / **3.50×** (double_disjoint),
inside the 4–9× band the blueprint predicted. It never supplies a reported p-value.

**Loss-family comparison (decision D7).** ProtoNCE − InfoNCE = +0.94 points, paired Wilcoxon
p = 7.4e-8, 59/80 subjects favour ProtoNCE, bootstrap CI [+0.67, +1.22]. This is a
**fixed-stimulus** claim — "better on these 90 videos". It does **not** license a
stimulus-generalising claim; the video-level minimum detectable difference is ≈8 points.

**λ₁ redundancy ablation (ATM + composite, 5 folds).** CLISA at weight 0.2 vs 0:
10.55% vs 10.48%, per-fold differences +0.06 / +0.23 / +0.24 / −0.31 / +0.18. The cross-subject
term's net contribution is **indistinguishable from zero**, confirming the blueprint's suspicion
that it is redundant with the cross-modal term.

**Attribute-shortcut control.** 90 same-hand, same-scene videos make "zero-shot video retrieval"
and "material decoding" overlap by construction, so every arm is also scored against a null that
shuffles the gallery *within material*. Queries whose target is the only gallery video of its
material are excluded from both terms — a within-material permutation is the identity on those,
so they would inflate the null for a non-material reason (26 of 216 gallery slots, 12%).

---

# 2. Foundation-model rung (ladder rung 5)

All three load their published checkpoints, verified tensor-by-tensor by an independent agent
rather than by trusting `from_pretrained` (which uses `strict=False` and silently skips
mismatches).

| Model | Pretrained | Params | Padding | Channel mapping |
|---|---|---|---|---|
| LaBraM-base | 221/221 tensors · 100% | 5.8 M | 40% | 64/64 names matched, no interpolation |
| CBraMod | 211/211 tensors · 100% | 5.7 M | 40% | channel-agnostic; ordering mismatch unquantified |
| EEGPT | 103/103 tensors · 100% | 25.3 M | 6.25% | **P9, P10, Iz, AFz effectively dropped** |

Fold-matched comparison (same three video folds, within_subject, chance 5.56%):

| Arm | Trainable params | Primary | Per fold |
|---|---|---|---|
| NICE + ProtoNCE | **0.33 M** | **11.40%** | 9.71 · 12.00 · 12.48 |
| NICE + InfoNCE | 0.33 M | 10.56% | 9.92 · 10.65 · 11.11 |
| LaBraM-base | 5.8 M | 10.53% | 10.30 · 10.68 · 10.62 |
| CBraMod | 5.7 M | 9.87% | 8.99 · 10.25 · 10.37 |
| EEGPT | 25.3 M | 9.38% | 8.87 · 9.38 · 9.90 |

All three pretrained models land **below** the 0.33 M convolution, and the ordering runs against
parameter count. **Read this as a statement about the interface before treating it as one about
the models**: 40% padding for two of three, untuned hyperparameters (no GPU was free to sweep),
and 3 folds means no permutation p and no fraction-of-ceiling.

**The interface mismatch is the result here, not a footnote.** LaBraM's patcher output width *is*
the transformer's embedding width, so shrinking `patch_size` to fit our 120-sample epoch leaves
only **12 of 221 tensors (0.03%)** loadable — the whole model, transformer included, becomes
randomly initialised, and braindecode does this behind a single `UserWarning`. The wrapper now
promotes it to an error. CBraMod at 200 samples has exactly **one patch**, which degenerates the
temporal half of its criss-cross attention. EEGPT was pretrained at 250 Hz against our 200 Hz, so
each patch spans 320 ms instead of 256 ms.

---

# 3. Are the dataset's two distinguishing assets usable?

The dataset is worth using for two specific reasons: **230,400 single trials**, and stimuli that
are **continuous video** rather than static images. Both were tested directly.

## 3.1 Video temporality — not reachable with frozen clip-level towers

Cosine between a clip's embedding and the embedding of the same frames re-ordered, all 90 base
videos. `d/d_video` expresses that as a fraction of how far swapping in a *different* video moves
the embedding.

| Encoder | cos(native, shuffled) | cos(native, reversed) | d/d_video shuf / rev | NN flips (rev) |
|---|---|---|---|---|
| videomae-v2-base | 0.8816 | 0.9746 | 0.311 / 0.065 | 0 / 90 |
| videomae-base-k400 (repaired) | 0.9367 | 0.9942 | 0.478 / 0.043 | 0 / 90 |
| xclip-base-32 (nominally temporal) | 0.9999 | 1.0000 | 0.000 / 0.000 | 0 / 90 |
| **siglip2-base** (used for every reported number) | **1.00000** | **1.00000** | 0.000 / 0.000 | 0 / 90 |
| clip-vit-l14 | 1.00000 | 1.00000 | 0.000 / 0.000 | 0 / 90 |

Three findings stack and together close the question:

1. **The target every reported number was trained against is order-blind by arithmetic.**
   SigLIP2 mean-pools per-frame embeddings, so a touch video and its frames in random order give
   a *bit-identical* vector.
2. **Reversal is invisible to all five encoders**, including the two that react to scrambling.
   An approaching hand and a retreating hand sit at cos 0.994 in VideoMAE — closer than two
   different videos (0.865) — and 0 of 90 nearest neighbours change. Scrambling destroys temporal
   *smoothness*; reversal preserves the multiset of adjacent-frame transitions, so masked-video
   pretraining never had to encode the arrow of time. This replicates across two independent
   encoders, so it is a property of the pretraining objective, not a bug.
3. **Contract C's pooling deletes the time axis by arithmetic anyway**, for the video-SSL family
   too: the stored vector is the mean over tubelet steps, so the deviation-from-step-mean
   contributes exactly zero. **Swapping encoders cannot fix this.**

Independent confirmation from the re-run census: the frame-**shuffled** cache scores LOVO top-5 =
0.222 against the native 0.233. Destroying frame order costs essentially nothing on the metric
that decides whether video-level zero-shot is possible at all.

A temporal read-out head was built and tested. On SigLIP2 it buys nothing nameable (LOVO at or
below chance; `approaching` AUC 0.494 vs the pooled vector's 0.312). On repaired VideoMAE its
apparent gain is **reproduced by a random mean-orthogonal contrast**, so it is not established as
temporal-specific.

**Implication.** Using the video-ness would need motion-specific features (optical flow, motion
energy) or frame sequences aligned to the EEG time course — not a temporal read-out bolted onto a
pooled clip vector. Artefacts are on disk (`video_emb/*__shuffled`, `*__reversed`,
`video_emb_seq/`) so the training arm is one command away if wanted.

## 3.2 Trial count — measurable for the first time, and it exposed a fact about existing numbers

| Stage | Trials | Kept |
|---|---|---|
| Raw non-target rows (80 × 2880) | 230,400 | 100% |
| after dropping post-target (button-press artefact) | 216,049 | 93.8% |
| within_subject training pool (D1 adjacency embargo applied) | 113,985 | 49.5% |
| within_subject test pool | 43,210 | 18.8% |
| **evaluation units at the pre-registered k=4** | **10,802** | **4.7%** |
| CorrCA / SRM operate on condition averages | 28,800 | discards 87.5% of single trials |

A trial-subsampling knob now exists (exact, nested, balanced, step-matched, with a byte-identical
evaluation set proven across 12 arms on real data). Building it produced two findings:

- **The pseudo-trial curriculum collapses under subsampling.** Realised k across frac 1 → ½ → ¼ →
  ⅛ is **3.90 → 2.71 → 1.46 → 1.00**. At frac ≤ ¼ the k=4 curriculum is already dead, so a naive
  trial-scaling curve would have measured "we switched the curriculum off" and reported it as "we
  removed data". Empirical proof: at ⅛ data, curriculum on and curriculum off produce the
  *identical* number (0.06399). Any usable curve needs the k=1 companion arm, which is wired.
- **Realised k is 3.90 at full data, not 4.** The D1 embargo plus the inner validation split leave
  ~4.9 repeats per (subject, condition) cell, not 8. This is a **pre-existing property of the
  reported 10.99% / 11.93%**, not something the new knob introduced.

---

# 4. Data and preprocessing

| Group | Verified |
|---|---|
| raw BDF | **80/80 = 107.7 GB** |
| authors' MNE derivatives | 80/80 = 10.5 GB |
| events.tsv | 80/80 = 62.9 MB |
| stimuli mp4 | 384 (4 orientations × 90 + 4 target dirs × 6) |
| VTD.csv / participants.tsv / phenotype | 90 rows / 80 rows / 4 tables |

Epochs: 160/160 memmaps (80 subjects × {`w0600` (2880,64,120), `wm100_800` (2880,64,180)}), 17 GB,
`baseline: null`, `n_dropped: 0`, 1:1 row alignment with the trial table, and a **single config
fingerprint** `7d219f48f433` across all 80.

- **EXG/EOG question settled: none exist.** The BDF header is 64 EEG (A1–A32, B1–B32) + Status.
  The blueprint's "0 EOG, use frontal proxies" premise holds. ICA found 2–5 ocular components per
  subject (median 3, **none with zero**), median max |HEOG| r = 0.86, |VEOG| r = 0.96.
- Robust scale median 17.3 µV IQR-sigma (7.8–58.0); **no dead channels**.
- autoreject 80/80: median bad-epoch fraction 8.7% (range 0–42.7%); **10 subjects above 20%**
  (worst sub-42 at 42.7%).

---

# 5. Phase-0 audits

- **Audit A = INTERLEAVED — the blueprint's #1 statistical risk is cleared.** Pure-orientation
  sequence fraction **0.000**, dominant-orientation fraction 0.297 (≈ chance 0.25), Cramér's V
  median 0.092, normalised mutual information orientation↔sequence 0.009. The §6.1-1 fallback is
  **not** triggered and the Q1a equivariance design stands as written.
- 80/80 subjects design-complete (2880 non-target trials, 360 conditions × 8 repeats), SOA median
  0.7998 s, `onset_index_base` unanimously 0 with an exactly-zero residual.
- **Audit C — three attributes are one variable.** Across the 90 stimuli `toucher↔object` = 1.000,
  `toucher↔material` = 1.000, `object↔material` = 0.993; the affective axis is nearly collinear
  too (`threat↔pain` = 0.971). 23 pairs exceed 0.5, and the material × touch_type table has 61 of
  96 cells empty. Any claim separating those three is unsupportable here, and the caveat is
  hard-coded into the census report and the "cannot be certified" list of every generated
  `REPORT.md`.
- **Material stratification arithmetic differs from the blueprint.** Actual counts are
  `{skin 28, metal 27, plastic 13, wood 10, cotton 4, fabric 3, sponge 3, hair 2}`, not "8 classes
  × ~11 videos", so each 5-fold test set covers only 6–7 of 8 classes. `splits.py` degrades
  gracefully; the empty-cell table is a publishable artefact.
- Splits verified leakage-free: within_subject 5 / loso 8 / double_disjoint 40 folds. **D1's cost
  measured**: `adjacency_side="train"` removes 34% of training trials and **0** test trials.
- Phenotype (n=80, no missing): VT 1.84 ± 2.97 (0–10, **zero-inflated**), EQ 16.06 ± 5.14, IRI
  24.82 ± 4.35, MTS self-report Yes **17/80** (the blueprint expected 1–2 true positives; the
  self-report rate is far higher and must be stated as such). No questionnaire correlates with age
  (|r| ≤ 0.124).

**G4 encoder census (re-run on repaired weights).** No encoder collapsed (effective rank
23.7–28.1); RDM–attribute correlations significant for object / material / valence across all
towers; leave-one-video-out attribute→embedding retrieval top-5 = 0.20–0.28 against 0.056 chance,
p = 0.001 for every tower. **Video-level zero-shot remains the primary endpoint**; the
attribute-level fallback is not triggered.

---

# 6. G5 and G6

**G5 — time-resolved MVPA, the pipeline-correctness check.** Only two targets have a clean
pre-stimulus baseline, and those two are exactly the ones that reproduce the published time
course.

| Target | Uniform chance | Majority rate | t₀ | Peak (ms) | Published | Verdict |
|---|---|---|---|---|---|---|
| Orientation | 0.250 | 0.250 | 0.2483 | 100 | 120–130 | reproduces |
| Valence | 0 | — | −0.011 | 320 | 300 | reproduces |
| Material | 0.125 | **0.311** | 0.2996 | 235 | 110–120 | majority-class null |
| Toucher | 0.500 | **0.689** | 0.6856 | 35 | — | majority-class null |
| Touch type | 0.083 | **0.356** | 0.3413 | 50 | 165 | majority-class null |

Scored with plain accuracy against a uniform `1/n_classes`, the bottom three sit 2.4–4.6× "above
chance" at *t = 0 ms*, before any evoked response can exist. They are sitting exactly on their
majority-class rate. Under balanced accuracy only orientation decodes robustly (+4.8 pp); the
others reach +0.1 to +0.4 pp. The report now prints the majority rate and an evoked fraction
beside every latency.

**G6a — primary-endpoint inference: passed** (see §1).

**G6b — ocular control: deliberately NOT declared.** Two blockers:
- The EEG arm carries 64 × 20 = **1280** ridge features against the EOG surrogate's **2**, so
  "EEG beats the surrogate" is confounded with feature dimensionality. A dimension-matched control
  is required.
- The `wm100_800` pre-saccadic selector is `t < 150 ms` on a window that begins at −100 ms, so it
  swallows the baseline.

**Both blockers are now resolved and all three criteria are met — see §10 (D14) for the numbers.**
Over 5 folds at 20 subjects: ablated 0.1068 vs full 0.1009 (ablation costs nothing); the EOG
surrogate sits at the 3rd percentile of 100 random 2-filter EEG projections while full EEG sits at
the 100th (so the margin is not feature count); and the pre-saccadic window carries EEG signal
(+0.0103, p = 0.0069, 5/5 folds) that survives ablation while both surrogates sit at or below chance
there. G6b is declarable at n = 5 folds over a fixed pool of 20 subjects; repeat at 80 before it
carries a paper. The arXiv placeholder G6 gates is deliberately **not** triggered here.

---

# 7. Defect ledger

Every module executed for the first time this session had at least one real defect. The pattern
worth recording: **almost none of them crashed. They produced plausible numbers.**

## 7.1 Would have invalidated a reported result

| Where | What | Evidence |
|---|---|---|
| `losses/protonce.py` | The blueprint's **main objective** had a solvable shortcut. `live_positive=True` (the shipped default) makes the positive logit live and differentiable while every negative is a stale detached EMA prototype, so the video projector wins by outrunning its own lagging bank. | With EEG replaced by **pure noise**, training accuracy reaches 100%; with the flag off it stays at chance (0.018 ≈ 1/90). First real run: `condition_acc` 0.9999 with validation at chance. Pinned off in config, guarded by a `RuntimeWarning`, regression test added. |
| `losses/composite.py` + `train/trainer.py` | `CompositeLoss.step()` exists but the trainer never called it, so the warmup counter stayed at 0 forever and **every component with a warmup had effective weight exactly 0**. | CLISA logged `weight: 0.0` at every epoch, nulling the λ₁ ablation that config exists to run. Fixed; ramp verified 0 → 0.1 → 0.2. |
| `eval/run_report.py` | Run directories are keyed on `run.name`, not regime, so the double-disjoint grid wrote into the same folder as the within-subject folds and the report averaged both. | ProtoNCE read 11.93% (5 within-subject folds) before the grid started and 11.49% after. Now filtered on each fold's own recorded regime, with mixing logged loudly. |
| `data/splits.py` | Double-disjoint folds are emitted video-fold-major, so the first five cells hold out the **same 18 videos** — zero stimulus variance, silently. | Caught before launch; the 40-fold grid was run complete. `fold_run_order()` added so any truncation spans video folds. |
| `baselines/corrca.py` | The headline ISC measured the condition-*invariant* stimulus-onset response, not between-subject coupling to the same stimulus. | Permuting each subject's condition-averages independently leaves c1 at **91.4%** of its value. Genuine stimulus-specific ISC is ~8× smaller; margin against the correct null 1.09×, not the claimed 15×. After repair: per-pair c1 0.1522 → 0.0325, margin 1.09× → 3.50×. |
| `eval/run_ocular.py` | Gallery was all 90 videos (~14 of them trained on) instead of the fold's 18 held-out, and only the intercept-inclusive prediction was scored — pinning every arm below chance. | The module printed "UNINFORMATIVE, there is no signal to ablate". Corrected numbers in §6. |
| transformers 5.15 (upstream) | VideoMAE self-attention was rewritten from bare `q_bias`/`v_bias` parameters to `nn.Linear(bias=…)` with **no checkpoint conversion mapping**, so *any* published VideoMAE checkpoint loads with its trained q and v attention biases zero-filled (158/194 tensors). | Repair moved our cache by cos 0.911, dropped RDM rank correlation to 0.60, and changed the top-1 nearest video for **61% of the 90 stimuli**. Six census numbers moved. Canonical tag now points at repaired weights; the broken cache is quarantined. |

## 7.2 Would have biased or misled

| Where | What |
|---|---|
| `models/heads.py` | Per-window heads ran *outside* `subject_context` while `__init__` attached the SuLoRA adapters to exactly those heads — that arm's subject conditioning was identically zero. |
| `train/trainer.py` | An `n_subjects+1` row 0 that no data ever trains; held-out subjects avoided it only by the grace of `strict_unseen=True`. Now index −1 and the computed rule (decision D2). |
| `eval/probes.py` | The "ocular ablation" used prefix matching `("Fp","AF","F")`, which removes **26 of 64 channels** including all FC*/FT*/Fz — turning "is this eye movement?" into "is this frontal cortex?" (decision D6). |
| `baselines/linear_mvpa.py` | Plain accuracy scored against uniform `1/n_classes` on severely imbalanced labels, making a majority-class predictor look like 2.4–4.6× decoding. |
| `eval/retrieval.py` (reading, not code) | `cross_group` was being read as a material control. Its gallery makes the target the only item of its material, so a pure material classifier scores 100% — an upper bound *inflated by* the code it claims to control for. |
| `data/preprocess.py` | A blown PO4 electrode on sub-17 (80× the median channel SD) reached the frozen epochs. The sidecar had already recorded `abs_max_after_scaling = 33297.9` — nothing read it. IQR-based robust scaling is blind to spikes by construction (sub-17 PO4 `robust_sd_ratio` = 1.00, `sd_ratio` = 82.7). |
| `models/selftest.py` | The architecture list was hardcoded to `[tsconv, atm]`, so any encoder added later was never contract-tested yet the suite stayed green. Now walks the registry: **105/105** with `--all-registered`. |
| `baselines/srm.py` | Cache key omitted the subject list, so an 80-subject run silently reused folds from a 6-subject run; and `subjects=` was never passed to `make_folds`, so subset runs tested on their own training subjects. |
| `train/trainer.py` | `backbone_lr_scale` was a dead key — the trainer built its own parameter groups and never called `encoder.param_groups()`. AdamW is invariant to gradient scale, so only a per-group `lr` works. Now wired. |
| `models/video/encode.py` | transformers ≥5 returns `BaseModelOutputWithPooling` from `get_*_features`, not a tensor; 3 of 4 encoders crashed. |
| `data/download.py` | `--what auto --subject-index` was called by `submit.py` but did not exist; 80 concurrent writers shared one JSON log; and the metadata check asserted 80 per-subject sidecars while ds005662 uses BIDS inheritance (one top-level sidecar, no `channels.tsv` anywhere). |
| `slurm/submit.py` | All seven stage command lines disagreed with the modules' real `argparse`. |

## 7.3 One consequence chain worth following

sub-17's dead electrode carried **91% of the pooled within-subject covariance** in CorrCA — giving
that "80-subject" fit a Kish effective sample size of **1.2 subjects** — and **91% of the total
Frobenius energy** of all 80 condition-average matrices in SRM, which is why the SRM objective was
reconstructing one dead channel and scoring below a no-SRM control. The deep models were
unaffected: their loader clamps at 20σ, sub-17's primary endpoint is 0.0997 (z = −0.69), and
dropping it moves the cohort mean by +0.025 points.

---

# 8. Infrastructure built this session

- **`slurm/pool.py`** — this cluster enforces `QOSMaxSubmitJobPerUserLimit = 30` **counted in array
  elements** (probed: `--array=1-29` accepted, `1-30` rejected) and a separate
  `QOSMaxGRESPerUser = 8` concurrent-GPU cap. The blueprint's 80-way preprocess array and 40-fold
  training array cannot be submitted at all. The pool submits W worker jobs that claim tasks from a
  shared directory: constant queue footprint, automatic load balancing, preempted workers rejoin.
  Claims use `O_CREAT|O_EXCL`; the **heartbeat lives inside the file rather than in `st_mtime`**,
  because this filesystem returns `st_mtime == 0` for freshly created files and every worker then
  believed a one-second-old claim was 53 years stale and stole it.
- **`tactus/eval/census.py`** — the G4 three-way judgment, permutation unit always the source video,
  report carries the Audit-C collinearity caveat.
- **`tactus/eval/run_report.py`** — the G6 driver (`report.py` had rendering but no driver):
  per-subject retrieval → video-level permutation + trial-level narrowing contrast → split-half
  ceiling → material-matched null → `REPORT.md`.
- **`tactus/eval/run_ocular.py`** — the G6b driver.
- **`tactus/data/qc.py`** — per-channel QC over the frozen epochs, with the cohort-relative
  statistic that discriminates a broken electrode from a heavy blinker.
- **`tactus/models/eeg/fm_{labram,cbramod,eegpt}.py`** — the foundation-model wrappers, each with a
  raising pretrained-weight assertion.
- Write-time channel QC gate in `preprocess.py`; `fold_run_order()` in `splits.py`; encoder
  `param_groups` hand-off and `describe()` provenance logging in `trainer.py`.

Cluster facts recorded in `slurm/cluster.conf`: no slurmdbd (no `--account/--qos`); the 3090
partition caps 4 CPUs per GPU and slurm validates a multi-partition request against the *strictest*
member, so every GPU job runs at `--cpus-per-task=4`; observed start latencies V100/P100/A30
immediate, A40 +4.5 h, A100 +9 h, H100 +12 h, L40S +27 h.

Environment: conda env `tactus` (py3.11 / torch 2.6.0+cu124 / mne 1.12 / braindecode 1.7 /
transformers 5.15). **Install order matters**: braindecode pulls torchaudio 2.11, which links
`libcudart.so.13` and dies on import against torch 2.6 — reinstall `torchaudio==2.6.0+cu124`
immediately after. Recorded in `slurm/setup_env.sh`.

Self-tests at time of writing: **72/72** loss scenarios, **33/33** model contract checks
(**105/105** with `--all-registered`), **5/5** pytest. 63 modules, 37,168 lines.

---

# 9. Report index

| Report | Path | Language |
|---|---|---|
| This status document | `tactus/STATUS.md` | English |
| Chinese edition (archived, superseded) | `tactus/STATUS.zh.md` | Chinese |
| Double-disjoint headline | `tactus_work/results/report_dd/REPORT.md` | English |
| Within-subject InfoNCE | `tactus_work/results/report_nice_infonce/REPORT.md` | English |
| Within-subject ProtoNCE | `tactus_work/results/report_nice_protonce/REPORT.md` | English |
| Encoder census (repaired weights) | `tactus_work/phase0_out_v2/census_report.md` | English |
| Phase-0 audits A–D | `tactus_work/phase0_out_en/audit_report.md` | English |
| MVPA, sequence CV | `tactus_work/results/baselines/mvpa/w0600_sequence/report.md` | English |
| MVPA, video-disjoint CV | `tactus_work/results/baselines/mvpa/w0600_video/report.md` | English |
| MVPA, balanced accuracy | `tactus_work/results/baselines/mvpa_balanced/w0600_sequence/report.md` | English |
| Channel QC | `tactus_work/derived/channel_qc.md` | English |

---

# 10. DECISIONS D11-D22 execution log

One line per decision as it closes, with the evidence path. D15 and D19 get their
own paragraphs below because they govern which numbers may leave this repository.

| D | status | one line | evidence |
|---|---|---|---|
| D11 | **done** | sub-17's PO4 interpolated (spherical spline, *before* the average reference); the structural half needed a second fix -- per-feature scaling left one subject owning 20-28% of the SRM objective, so SRM now normalises per subject too | `results/baselines/corrca/w0600{,_pre_D11}`, `results/baselines/srm/w0600_ws_{subjnorm,nosubjnorm}` |
| D12 | **done** | centring unified on "training mean off query *and* gallery", defined once in `tactus/eval/retrieval.py`; linear_align reruns to 0.0989 [0.0947, 0.1029] | `results/baselines/linear_align/within_subject_w0600_siglip2-base_ea1_d4p/summary.json` |
| D13 | **done** | covariate table assembled; two of the four Q3 outcomes are far weaker than n=80 implies, and one specified covariate does not exist | `results/covariates/COVARIATES.md`, `tactus/eval/covariates.py` |
| D14 | **all three criteria met over 5 folds** | surrogate sits at the 3rd percentile of random 2-filter EEG; pre-saccadic p=0.0069 and survives ablation. Supersedes the single-fold null | `results/ocular_d14/` |
| D15 | **answered** | the ceiling was fold-design-dependent; see below | `results/report_*/REPORT.md` |
| D16 | **in force** | video fold 4 (the fifth) is the sealed confirmation fold | -- |
| D17 | **done** | design-lesson section generated as a reproducible module; the unanswerable list stays hard-coded | `results/design_lesson/DESIGN_LESSON.md`, `tactus/eval/design_lesson.py` |
| D18 | **contract extended and sized** | `frame_emb (360, 15, 768)` built; the time axis is 14.4% of the video-side variance and diffuse | `derived/video_emb/siglip2-base-frames.npz` |
| D19 | **answered** | metric difference, not a real one -- the three disputed targets never beat their own majority rate; see below | `results/baselines/mvpa{,_balanced}/w0600_sequence/report.md` |
| D20 | **done, negative** | the flagship arm was unrunnable, then ran, then turned out to have an inoperative disentangler; see below | `results/probes_fhmc_ws/PROBES.md`, `results/report_fhmc_ws/REPORT.md` |
| D21 | **done** | lambda_1 contributes +0.08 pts, indistinguishable from zero; "dual contrast" retired from the contributions | `results/runs/atm_composite_l1_{00,02}` |
| D22 | running | offline line done (D11/D12); training line is FHMC dd (40 folds) + the fixed arm | -- |

## D15 -- the answer, and what may be quoted

The split-half ceiling averages each gallery item over **every subject the caller
passes**, so its cleanliness scales with the fold's subject count. Measured on one
within_subject fold, 10/20/40/80 subjects give 0.1133 / 0.1317 / 0.1497 / 0.1633.
A within_subject fold carries 80 test subjects; a double_disjoint fold carries 10.
The two "fraction of ceiling" figures being compared were therefore divided by
different denominators, and most of the 16-point gap was fold design rather than
model behaviour.

Pinning the *count* at 10 (`--ceiling-subjects`) was the first fix and it was not
enough. **Which** 10 subjects get drawn moves the pooled ceiling from 0.1122 to
0.1539 across eight seeds on one fold -- sd 0.0143 on a mean of 0.1272, which is
larger than the accuracy gaps being compared. Two arms drawing different subsets
were being divided by denominators that differed by more than the effect.

The pooled ceiling is therefore averaged over 20 independent subject draws
(`--ceiling-draws`) and carries its own spread. With that in place the three
within_subject arms finally share a denominator, which they always should have --
a split-half EEG-to-EEG ceiling has nothing to do with the model:

| arm | raw | ceiling | fraction | denominator +-1 sd |
|---|---|---|---|---|
| within_subject NICE+ProtoNCE | 11.93% | 0.1429 | 0.835 | [0.748, 0.945] |
| within_subject NICE+InfoNCE | 10.99% | 0.1403 | 0.784 | [0.705, 0.882] |
| within_subject FHMC | 10.88% | 0.1403 | 0.775 | [0.694, 0.879] |
| double_disjoint NICE+ProtoNCE | 10.45% | 0.1253 | 0.835 | -- (no draw: a fold has exactly 10 subjects) |

**The two regimes recover the same fraction, 0.835 and 0.835.** This quantity has
now given three answers -- 0.674 vs 0.835, then 0.814 vs 0.835, now 0.835 vs
0.835 -- and the first two were both properties of the denominator rather than of
the models. The arms are not separable on it either: every band is about +-0.09
wide, comfortably wider than the gaps between them.

So the standing instruction is unchanged and now better justified: **quote raw
accuracy and its CI externally.** Fraction-of-ceiling is a scale for "how much of
what the data supports is being recovered", not a discriminator between arms or
regimes.

## D13 -- the covariate set, and three things it turned up

`tactus/eval/covariates.py` assembles the whole v2 section-7 set from artefacts
that already exist, so Q3 cannot quietly use a different set than it reports.
The ordering the decision fixed is in the code: primary SNR covariate is
per-subject split-half reliability, secondary is the repaired scale-invariant
ISC ratio, and the pre-repair ISC column is **not exported at all** -- voiding a
column that correlated rho = 0.19 with reliability means removing it, not
annotating it.

Assembling it surfaced three things worth knowing before Q3 runs.

**The behavioural covariate is not what the plan called it.** There is no
per-target hit rate in this dataset. `rt`/`resp` are populated on exactly 32 rows
per subject, none of them target rows, and they sit 3 to 56 events away from the
nearest target. The task is to *count* the targets within each of the 32
sequences and report the count at the end; `cresp` is the true count and `resp`
the reported one. The attention measure is therefore per-sequence counting
accuracy, and it is usable -- mean 0.812, sd 0.190, range 0.06-1.00, only 9 of 80
at ceiling.

**One specified covariate does not exist.** `n_trials_kept` is 2880 for every one
of the 80 subjects and `n_trials_dropped` is 0 for every one. Our pipeline
rejects no trials at all; artefact handling is scaling and clamping, not
rejection. "Retained trial count" has zero variance and is dropped, with
`frac_abs_gt_20` and `abs_max_after_scaling` carrying the artefact axis instead.

**Two of the four Q3 outcomes are much weaker than n = 80 suggests.**

| outcome | test | n effective | MDD |
|---|---|---|---|
| EQ_score | Spearman | 80 | r = 0.31 |
| IRI_score | Spearman | 80 | r = 0.31 |
| VT_score, as continuous | Spearman | 80 | r = 0.31 (optimistic) |
| VT_score, binarised > 0 | two-sample | 33 | d = 0.64 |
| MTS | two-sample | 17 | d = 0.77 |

VT_score is exactly 0 for 47 of 80 subjects, so the rank test over its full range
is mostly one large tie and the honest version is a 33-vs-47 comparison. MTS
splits 17/63. Both are small two-group tests wearing an n = 80 label. This is
printed above the table in the generated report rather than discovered
afterwards: a null on either is a statement about the design.

## D14 -- the ocular evidence chain, now across five folds

w0600, 20 subjects, 5 video folds, held-out 18-video gallery, intercept-free
scorer, chance 0.0556. Fold-level mean +- sd.

| arm | 0-595 ms | 0-150 ms (pre-saccadic) | 150-595 ms |
|---|---|---|---|
| ocular_ablated (D6 list, 8 frontal channels removed) | **0.1068 +- 0.0070** | 0.0643 +- 0.0038 | 0.0954 +- 0.0060 |
| full_eeg | 0.1009 +- 0.0087 | 0.0659 +- 0.0040 | 0.0957 +- 0.0053 |
| eog_surrogate_saved | 0.0586 +- 0.0028 | 0.0522 +- 0.0039 | 0.0595 +- 0.0029 |
| ocular_surrogate | 0.0589 +- 0.0020 | 0.0541 +- 0.0066 | 0.0619 +- 0.0038 |
| **eeg_rand2**, 100 draws (dimension-matched) | 0.0711 +- 0.0063, p95 0.0812 | -- | -- |

**Criterion 1 -- ablated approximately equals full.** 0.1068 against 0.1009, with
the ablated arm marginally *higher*. Removing the channels an eye movement would
dominate costs nothing.

**Criterion 2 -- the dimension-matched control.** Over 100 random 2-filter draws,
full EEG sits at the 100th percentile and the EOG surrogate at the **3rd** -- the
surrogate is not merely unremarkable as a 2-dimensional projection of EEG, it is
worse than 97% of random ones. Both halves of the feature-count confound are
answered.

**Criterion 3 -- the pre-saccadic window, and a correction.** Fold-level
one-sample tests against chance:

| arm, 0-150 ms | delta | t(4) | p | folds above chance |
|---|---|---|---|---|
| full_eeg | +0.0103 | 5.11 | **0.0069** | 5/5 |
| ocular_ablated | +0.0088 | 4.66 | **0.0096** | 5/5 |
| eog_surrogate_saved | -0.0033 | -1.73 | 0.159 | 1/5 |
| ocular_surrogate | -0.0014 | -0.44 | 0.684 | 3/5 |
| paired full_eeg - surrogate | +0.0137 | 4.31 | **0.0125** | -- |

The pre-saccadic window carries EEG signal, that signal survives ocular
ablation, and both ocular surrogates sit at or below chance inside it. That is
the strongest form the criterion could take.

**It also reverses what this file said an hour earlier.** On fold 0 alone every
pre-saccadic arm was at chance, and that was written up as a null that "will not
change with more compute". Four more folds changed it. The error was not the
measurement but the generalisation: one fold of 18 held-out videos was treated as
settling a stimulus-generalising question, and the claim was stated as
compute-proof rather than as what it was, a single noisy estimate.

**All three criteria are met, so G6b is declarable** -- with the size of the
inference honestly stated. These are n = 5 fold-level tests over a fixed pool of
20 subjects, so they generalise over videos and not over subjects, and the
pre-saccadic effect is +0.0103 on a chance of 0.0556. It is consistent (5/5
folds, both EEG arms, surrogates on the other side of chance) but small and
resting on four degrees of freedom. The declaration should be repeated at 80
subjects before it carries a paper.

**Not triggered here:** the arXiv placeholder that G6 gates. That is an
outward-facing action and is the user's to take.

## D17 -- the design lesson, as a section rather than a caveat

`tactus/eval/design_lesson.py` regenerates the whole thing from the trial table,
so the paper section and the enforcement list cannot drift apart. Four blocks:
attribute collinearity, the affect axes, the material x touch_type occupancy
table, and class imbalance.

The claim it exists to license: on these 90 stimuli a hand touches skin and an
object touches nearly everything else, so **"material decoding" and
"hand-versus-object decoding" are two names for one claim** -- not two pieces of
converging evidence. `UNANSWERABLE` stays hard-coded, so making that claim in
future requires deleting a line rather than forgetting a caveat.

Two things had to be handled to make the section honest rather than merely
alarming. Cramer's V is reported both bias-corrected and uncorrected: the
material x touch_type table is 61 of 96 cells empty, where the uncorrected
statistic inflates, and the Phase-0 audit quoted the uncorrected values
(toucher/material 1.000, object/material 0.993) which have to remain
reconcilable rather than silently replaced by the corrected 0.965 and 0.867.
And orientation is scored over the 360 conditions rather than the 90 videos --
it is a property of the condition, so deduplicating on video_id sampled
whichever orientation came first and reported a majority rate of 0.278 for a
factor that is exactly balanced by construction.

That last point is the section's punchline rather than a footnote. Orientation
is the only attribute crossed with video by design, it is the only one whose
`majority_over_uniform` is exactly 1.0, and it is the only one whose MVPA
replicates. The others are sampled naturalistically -- `object` alone reaches
8.4x -- which is what a majority-class predictor collects for free.

## D18 -- the time axis exists, is 14% of the variance, and is diffuse

Contract C now carries `frame_emb (360, 15, 768)`. The first thing it settles is
that "frame order is invisible to SigLIP2" was never a finding about SigLIP2: for
the image_clip family the clip vector *is* the arithmetic mean of these rows, so
the native and the reversed ordering both reproduce the cached pooled vector at
cosine 0.99999988. A mean is permutation-invariant by arithmetic.

The second thing it settles is how much a time-resolved Q2 can possibly gain, on
the 90 native-orientation clips:

| quantity | value |
|---|---|
| between-video variance | 85.6% |
| **within-clip (temporal) variance** | **14.4%** |
| mean cosine between frames of one clip | 0.9793 |
| mean cosine between different clips (pooled) | 0.8817 |

So 14.4% is the entire budget a time axis can spend, and only the part of it the
EEG can read is reachable. It is not nothing -- but it is diffuse rather than
structured. After removing each clip's own mean, which is exactly what the pooled
vector discards, the leading component of the remainder correlates with frame
index at mean |r| = 0.536 and exceeds 0.8 for only 16 of 90 clips, and that
component carries just 6.2% of the temporal variance. There is no single "time
direction" to project onto; a time-resolved analysis has to work with a spread-out
signal.

One provenance note that travels with the file. The canonical cache was built on
CUDA with fp16 and this one on CPU in fp32, so they agree at cosine 0.9999982 and
give the same top-1 nearest video for 89 of 90 stimuli rather than 90. Use
`frame_emb` with the `cond_emb` from its own file -- inside one file the pooled
vector reproduces the frame mean at cosine 0.99999985.

## D19 -- the MVPA null is a metric difference, and the design says which one

The decision asked whether material / toucher / touch_type are null because our
metric differs from the companion paper's, or because the effect is not there.
Both metrics had already been run under the companion's protocol
(`--cv sequence`, per-class Ledoit-Wolf shrinkage), so the answer needed no new
compute -- only reading the majority-rate column that was added after the first
MVPA pass looked good.

| target | uniform chance | majority rate | plain-accuracy peak | peak - majority | peak / uniform |
|---|---|---|---|---|---|
| material | 0.125 | 0.311 | 0.309 | **-0.002** | 2.47x |
| toucher | 0.500 | 0.689 | 0.686 | **-0.003** | 1.37x |
| touch_type | 0.083 | 0.356 | 0.342 | **-0.014** | 4.10x |
| orientation | 0.250 | **0.250** | 0.298 | **+0.048** | 1.19x |

Every disputed target sits *below* the accuracy of a decoder that always answers
with the most common class, while looking like 1.4-4.1x decoding when scored
against uniform chance. The one target that replicates is the one target whose
classes are exactly balanced by construction -- 90 videos x 4 orientations -- so
its uniform chance and its majority rate are the same number and the two ways of
scoring cannot diverge.

Under balanced accuracy the disputed targets collapse as expected: material 0.129
against 0.125, touch_type 0.0835 against 0.083, toucher 0.502 against 0.500.
Material's small residue peaks at 180-250 ms rather than the companion's
110-120 ms, so it does not support the landmark either.

**Verdict: metric difference.** Report both metrics, and quote the majority rate
beside every accuracy on this dataset. This is a stimulus-set property -- skin is
31% of the 90 videos and "touch" is 36% -- so it belongs with D17's design-lesson
section rather than being filed as a decoding failure.

**Boundary of this claim.** It is a statement about *our* pipeline: our own two
metrics differ by exactly the majority rate. Confirming that the published
numbers carry the same property needs the companion's paper, which is not in this
repository; their exact trial handling, baseline correction and Bayesian
statistics are not replicated here. The claim is that these targets are not
decodable above their majority rate in our hands, not that the published result
is wrong.

## D20 -- the flagship arm, and why it does not support contribution 2

Three findings, in the order they were forced out.

1. **It had never run.** `losses/factorized.py` existed but was not imported, so
   `@register_loss` never executed and `loss.name: factorized` raised "unknown loss".
   The evidence had been visible for days: BLUEPRINT_v3 documents an 80/80 regression
   battery and this repository was reporting 72/72. The missing eight were FHMC.

2. **It ran, and lost.** within_subject, 5 folds, primary endpoint: **10.88%**
   (95% CI 10.36-11.41), video-level permutation p = 0.0002, z = 12.45. That is below
   NICE+ProtoNCE (11.93%) and below plain NICE+InfoNCE (10.99%).

3. **The term the architecture is named for was inoperative.** `disent` was a squared
   cross-*covariance* between two L2-normalized heads, so it scaled as
   1/(d_content x d_geometry). It read 5.8e-07 at epoch 0 and fell from there;
   weighted by lambda=0.1 that is 6e-08 of a loss of magnitude 9. On the same trained
   checkpoint the largest cross-correlation between a content and a geometry
   coordinate was **0.788**.

The probe table is what an inoperative constraint predicts (subject-grouped CV for
stimulus attributes, video-grouped CV for subject identity, probing trial averages):

| subspace | video (18) | orientation (4) | subject (80) | 18-way retrieval retained |
|---|---|---|---|---|
| trunk | 0.231 | 0.577 | 0.969 | 0.122 |
| content | 0.255 | **0.595** | 0.887 | 0.126 |
| geometry | 0.223 | **0.604** | 0.639 | 0.065 |
| semantic | 0.244 | 0.596 | 0.836 | 0.105 |

The flip-**invariant** content head decodes orientation as well as the
flip-**equivariant** geometry head. The three factors are near-redundant projections
of the trunk. Subject identity is decodable at 0.887 from the "content" subspace
under video-grouped CV -- though note this is the within_subject regime, where the
encoder has seen all 80 subjects and carries subject-conditioned parameters by
design, so the number that settles the subject-invariance claim is the
double_disjoint one, which is still running.

`configs/factorized_fhmc_disent.yaml` re-runs the objective with a scale-free
cross-correlation penalty. Both arms will be reported: this is the before and after
of a defect, not a hyperparameter search.

## The recurring shape

Three defects in this project have now been the same thing -- a loss term that is
present, differentiable, logged every epoch, and contributing nothing: ProtoNCE's
live-positive shortcut, the composite warmup counter that never advanced, and this
disentangler. None crashed; none showed up in a loss curve.
`tactus.losses.term_contributions()` now reports each term's weighted share of the
total and flags anything below 1e-3, and `tests/test_disentangler_scale.py` carries
a test asserting the old formulation fails the file, so the guard cannot quietly
stop biting.

Two cache defects of the same family surfaced alongside it. The condition-average
cache was keyed on window/split/decim but not on the epochs it was built from, so a
re-epoched sub-17 kept two-day-old averages and CorrCA returned byte-identical
results; and running folds in parallel made every `linear_align` process write
`summary.json` from only its own fold, last writer wins, once reporting a headline
from one fold while five sat on disk. Both now carry a fingerprint and a warning.

---

## DECISIONS_NEEDED

1. ~~**sub-17.**~~ **Resolved by D11** — interpolated (spherical spline, before the reference).
   Post-fix the subject is unremarkable: worst channel SD ratio 80.10 -> 1.54, peak |x| after
   scaling 33297.9 -> 18.6, samples beyond 20 sigma 3.75e-03 -> 0. It now sits below the clamp
   threshold, so the "clamping destroys its pattern" dilemma is gone rather than traded off.
2. ~~**Centring convention.**~~ **Resolved by D12** — query *and* gallery, defined once in
   `tactus/eval/retrieval.py`. Worth 0.002 against a CI of width 0.008; adopted on principle
   (training-only statistics), not for accuracy.
3. **Which SNR regressor Q3 uses.** The repaired scale-invariant per-subject ISC correlates only
   ρ = 0.61 with the old one, and ρ = 0.19 with split-half reliability once the onset ERP is
   removed. Any phenotype analysis built on the old column has to be redone.

## Next

1. **Swap in the new contrastive loss** — what the repository was built for. Two edits:
   `tactus/losses/my_loss.py` plus one import line, and a config with one `loss.name` key. Then
   `python slurm/pool.py submit --name train_myloss --tasks 0-4 --workers 5 --gpus 1 ...`.
   Baselines are in place with CIs, permutation p, ceiling fractions and the design's MDD.
2. G6b: dimension-matched ocular control and the `wm100_800` window fix.
3. Trial-count scaling curve (knob ready, needs the k=1 companion arm).
4. Subject-scaling curve (10/20/40/79), Q1a equivariance, Q2 onset curves, Q3 phenotype.
5. OSF pre-registration.
