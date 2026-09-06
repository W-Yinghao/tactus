# W1 pre-registration — target-space factor and the live projector (FROZEN)

**Frozen: 2026-09-06, before any W1 training unit was submitted.** Executes wave W1 of
`EXPLORATION_PROGRAM_v2.md` (kept in the user's notes repository; the program's §2
statistics constitution applies verbatim). Internal; nothing here authorizes external
posting. Fold 4 stays sealed (D16); folds vf00–vf03 only.

## Calibration from existing data (disclosed; computed by `eval/w1_offline.py` before this freeze)

| Quantity | Value |
|---|---|
| Per-video k=1 paired SD (InfoNCE−ProtoNCE, 72 videos) | 0.047 → SE 0.55 pp, **MDD (paired, 80% power) 1.5 pp** |
| Per-video k=4 paired SD | 0.088 → MDD 2.9 pp |
| E1a-i: InfoNCE (trained projector) − ProtoNCE (frozen projector), k=1 | −0.40 pp, p=0.47 (k=4: −0.78 pp, p=0.46) |
| k calibration (ProtoNCE): k=1/2/4/8 | 0.0874 / 0.1009 / 0.1149 / 0.1328 |
| k=1 retrieval ceiling, pooled 10-subject gallery: k1/k1 · k1/k7 · k4/k4 | 0.0796 · 0.1039 · 0.1393 |

Consequence written down before any arm runs: the k=1 headroom under the k1/k7 ceiling is
**~1.6 pp**, no larger than k=4's 2.4 pp, and equal to the MDD. Single-point k=1 comparisons
therefore resolve only effects at the edge of what the data can hold; every confirmatory
contrast below states this and is read against the MDD, not against zero.

## Arms (same encoder, folds vf00–vf03, seeds 0/1/2 via `run.seed`)

| Arm | Config / override | Role |
|---|---|---|
| T-vid · ProtoNCE | `nice_protonce.yaml` (seed 0 exists as `__subj80`; seeds 1,2 new) | baseline, frozen random projector |
| T-vid · codebook_ce | `w1_codebook_ce.yaml` | **P1**: every prototype live, projector trains |
| T-cap-full / -vis / -blind · ProtoNCE | `-o video.model_tag=target-cap-{full,vis,blind}` | E2 text targets (frozen SigLIP2 text tower) |
| T-mat · SupCon | `w1_supcon_material.yaml` | E2 material-only, multi-positive |
| T-aff · RnC | `w1_rnc_affect.yaml` | E2 affect-only; **retrieval at chance by construction**, probes are its endpoints |
| windows 0–200 / 200–400 / 400–600 ms × {T-cap-vis, T-aff} | `-o data.crop_samples=[0,40]` / `[40,80]` / `[80,120]` | **P2b** double dissociation |

E2 uses ProtoNCE (frozen projector) as its base so a target swap changes only the codebook
the EEG must match — no projector adaptation confound. `T-cap-blind` is material-blind at
the token level only; D17 collinearity means description/toucher tokens still carry material.

## Endpoints

- **Retrieval**: per-video k=1 18-way top-1 on the fold's held-out videos (uncentred cosine
  against `z_vid_video`, exactly the trainer's eval); seeds averaged per video before pairing;
  paired unit = video (n=72). k=4 pseudo as reference row.
- **Affect probe** (for P2): per-(subject, video) EEG prototypes → LOVO ridge (α=1) to
  z-scored valence/arousal/threat means; statistic = Spearman(pred, actual) over the 18
  held-out videos per fold pooled to 72, **after residualizing both sides on material
  dummies and the low-level RDM features** (D23's mandatory control). Same for a
  **visual probe**: LOVO ridge to the SigLIP2 base embedding's top-5 PCs, Spearman of the
  predicted-vs-actual cosine structure.
- text→EEG top-1 (D24 recipe, centred) per arm — descriptive only.

## Confirmatory contrasts (the only two; everything else exploratory, uncorrected)

**P1** — codebook_ce − ProtoNCE at T-vid, per-video paired k=1, seeds averaged.
Grid: Δ > +1.5 pp with 95% CI excluding 0 → the frozen projector costs retrieval;
|Δ| ≤ 1.5 pp → no detectable cost at MDD 1.5 pp; Δ < −1.5 pp → the frozen codebook
acts as a regularizer. The E1a-i figure (−0.4 pp, loss-family confounded) is the prior.

**P2** — target-space factor, two parts, both required for the program's §6 positive sentence:
(a) affect-probe ρ under T-aff minus under T-cap-vis, material/low-level partialled, video-level
permutation p<0.05 → the affect target puts affect geometry into EEG beyond what visual
targets do; (b) double dissociation across windows: T-cap-vis per-video k=1 (early − late)
> 0 AND T-aff affect-probe ρ (late − early) > 0, each with 95% CI excluding 0. Grid:
both → "visual early, affect late"; one → partial, named; neither → target type does not
reshape EEG geometry (the §6 null sentence, quoted with MDD and the k=1 ceiling).

## QC gates before the fleet (probe = seed 0, fold vf00, one unit per new mechanism)

1. codebook_ce unit: training accuracy rises above its first epoch, validation retrieval finite,
   no non-finite loss, projector weights differ from init after training.
2. Crop unit: log line `time crop: samples [0, 40)` and encoder `n_times=40`.
3. Target-swap unit: `video.model_tag` logged as the target tag; `d_video=768`.
4. Coverage: every arm 4 folds × 3 seeds; `code_commit` homogeneous within the wave.
Probe accuracies are not read before the fleet is complete.

## Forbidden wording

No k=1 number is compared with the k=4 headline; "semantic" is not used; T-aff/T-mat
retrieval numbers are not evidence of anything (at chance by construction for T-aff);
no claim about material beyond the D17 bundle; W2–W5 items are out of scope here.
