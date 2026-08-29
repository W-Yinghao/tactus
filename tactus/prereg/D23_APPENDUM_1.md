# D23 appendum 1 — sentinel-2 failure, diagnosis, and two frozen revisions

**Date: 2026-08-29. Frozen BEFORE any EEG-space statistic was computed.** The probe chain
stopped exactly where the prereg said it must: QC sentinel 2 failed and nothing downstream ran.
No subject-level or group EEG RDM has been correlated with any space to date; the only EEG
artifact in existence is one subject's RDM stack, never compared with a model RDM.

## What happened

Sentinel 2 v1 — Spearman between the raw-score composite (sharp+prickly+spiky)/3 and VTD
threat — read r = 0.030, p = 0.396. FAIL.

## What was viewed during diagnosis (stimulus side only; full disclosure)

1. Per-adjective raw-score correlations with VTD threat: sharp ρ=0.47, blunt 0.295, hard 0.147;
   spongy −0.403, dry −0.33, squishy −0.328, soft −0.315, fuzzy −0.305. prickly and spiky do
   not track threat in this stimulus set (ρ < 0.05).
2. Raw scores live in a narrow band (0.043–0.126) with per-adjective baselines that dominate
   any raw-score average — v1's composite failed for this mechanical reason while individual
   adjectives demonstrably carry the intended structure.
3. The z-scored profile separates materials strongly (mean within-material distance 28.0 vs
   between 69.7).
4. The frozen v1 B1 RDM (1 − Pearson over RAW profiles) has median row correlation 0.76 and is
   ρ = 0.817 redundant with the visual RDM A; per-adjective z-scoring first drops the median row
   correlation to ~0 and the redundancy with A to 0.478.

## Revision 1 — sentinel 2 v2 (mechanism-level, no hand-picked words)

Leave-one-video-out ridge (alpha = 1.0, features standardized on the training folds) from the
full 30-adjective profile to VTD threat; test statistic = Spearman(predicted, actual) over the
90 held-out predictions; null = 5000 video-level permutations of threat; pass iff one-sided
p < 0.05. Chosen because it tests whether the SCORING MECHANISM carries tactile-relevant
information at all — the sentinel's actual purpose — rather than whether one pre-guessed
3-word composite does. It is not chosen for its observed outcome: it has not been run when
this appendum is frozen. v1's failed result stays in the manifest as the audit trail.

## Revision 2 — B1 RDM built on per-adjective z-scored profiles

B1 = 1 − Pearson over profiles z-scored per adjective across the 90 videos. Rationale from
the diagnosis: the raw version is 0.82-redundant with A, so v1's "B1 partial of A" would have
tested a nearly-empty contrast; z-scoring is the standard feature-RDM construction and
removes exactly the per-adjective baseline artifact identified above. All other definitions,
contrasts, thresholds, windows, and the interpretation grid stand unchanged.

## Honest accounting

The v1 composite and the raw-Pearson RDM were my design errors, frozen without a stimulus-side
pilot that was always legitimate to run (VTD is public metadata; no EEG involved). The
sentinel-before-aggregate ordering caught both. Had v1 passed by luck, the primary contrast
would have been run against a B1 that mostly duplicates A.
