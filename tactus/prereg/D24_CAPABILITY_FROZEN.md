# D24 pre-registration — text→EEG capability table (FROZEN)

**Frozen: 2026-08-29, before any text-EEG number was computed.** Per D27. Offline evaluation
against already-trained arms; no training. Internal, exploration phase; nothing here triggers
any external posting.

## What is being demonstrated

A bimodal EEG↔video system cannot answer a text query. The trained arms align EEG to the
SigLIP2 *image* tower's video embeddings; SigLIP2's text tower is aligned to its image tower
by pretraining, so a caption can address the shared space transitively — through the arm's
own frozen video projector (768→1024→256, `checkpoints/best.pt: projector`).

## Frozen inputs

- Arm: `nice_protonce__subj80`, within_subject folds vf00–vf03 (**vf04 sealed, D16**);
  per fold `test_embeddings.npz` (z_eeg, ids) + `checkpoints/best.pt` projector.
- Captions: `derived/text_emb/siglip2-base-captions.npz` `text_emb (90, 768)` (D24 build,
  90/90 unique).
- Video side for the tower gate: `derived/video_emb/siglip2-base.npz` `base_emb (90, 768)`.
- Attribute labels: VTD via `load_vtd`.

## Space and centring

Captions pass through the fold's video projector, then L2. **Primary variant: label-free
per-side centring** (text side: mean of the 90 projected captions; EEG side: per-subject mean
over that subject's test-trial embeddings; video side for the gate: mean of the 90 projected
videos) — the standard modality-gap correction, computable without labels. **Sensitivity
variant: raw (no centring).** Both reported; the primary is named in every table.

## Table A — text→EEG retrieval

Per fold, per subject: query = projected caption of test video v (18 per fold); gallery =
that subject's 18 per-video EEG prototypes (mean of test-trial z_eeg per video). Score = 
top-1 (correct video's prototype ranked first), chance 1/18. Aggregate per subject over the
subject's folds → n=80 subject scores → mean and 95% bootstrap CI. Inference: caption↔video
assignment permuted at the video level within fold, 5000 permutations, subject-aggregated
statistic. Mirror direction (EEG prototype query, 18-caption gallery) reported in the same
table. k=1 companion: gallery = single test trials; a query's prediction is the video of the
top-1 trial; same aggregation (reference row, per D27 the capability grain here is the
prototype — stated in the caption).

## Table B — zero-shot attribute prompts

Frozen prompt sets (verbatim):
- material (8): "a video of a hand being touched by something made of {material}",
  material ∈ {metal, wood, plastic, cotton, fabric, hair, skin, sponge};
- approaching (2): "a video of a hand with something approaching it" /
  "a video of a hand with something moving away from it";
- toucher (2): "a video of a hand touched by another person's hand" /
  "a video of a hand touched by a handheld object";
- valence (3): "a video of a pleasant touch" / "a video of a neutral touch" /
  "a video of an unpleasant touch" (labels = the caption build's valence terciles).

Prediction: per-(subject, video) EEG prototype → argmax cosine over the attribute's prompt
embeddings (projected, primary centring). Score: balanced accuracy beside **majority rate**
(D19) and chance; video-level permutation p (5000). **D17 stands: material/toucher/object are
one bundle on these stimuli — material and toucher rows carry the bundle qualifier and no
separate object row exists.**

## QC gates — before any aggregate is viewed

1. **Tower gate (stimulus side, no EEG):** projected captions vs projected 90 video
   embeddings, text→video top-1 over 90. PASS iff top-1 ≥ 0.15 AND video-permutation
   p < 0.05 in the primary variant. FAIL → Tables A/B are reported as *blocked at the tower*
   (no brain-level claim in either direction), grid cell A-3.
2. Determinism: two builds bit-identical.
3. Coverage: 4 folds × 80 subjects × 18 videos; every drop reason-coded.
4. Probe: fold vf00, 2 subjects, gates 1–3 machinery, before the full run.

## Interpretation grid (pre-committed)

| Cell | Observation | Reading |
|---|---|---|
| A-1 | tower gate passes ∧ text→EEG above chance (p<0.05) | capability demonstrated; quoted with the tower top-1 as its scale |
| A-2 | tower gate passes ∧ text→EEG at chance | text reaches the video cluster, EEG-side signal insufficient; boundary result |
| A-3 | tower gate fails | blocked at the tower; no brain-level conclusion |
| B-per-attribute | above majority ∧ p<0.05 | attribute queryable by prompt (bundle qualifier where D17 applies) |
| B-null | otherwise | null with majority rate quoted |

## Forbidden wording

"semantic understanding", any separated object/toucher claim, any comparison presenting
Table A as the same task as the primary endpoint. λ_text training arm is out of scope for
this document; if run later it gets its own freeze.
