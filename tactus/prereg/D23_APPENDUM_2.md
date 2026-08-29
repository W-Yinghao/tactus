# D23 appendum 2 — contrast ceiling (D29, report layer; dated addition)

**Date: 2026-08-29, added AFTER the H1/H2 aggregates were viewed**, on the instruction of
D29_NOTE.md. This is disclosed openly: it changes NO inference — H1/H2 statistics, grid
cells and thresholds stand exactly as frozen — it adds the report-layer scale that decides
which of two wordings the observed H1 null may carry:

- null AND ample ceiling  -> "no tactile-specific signal" (substantive null);
- null AND floor ceiling  -> "no sensitivity survives the partialling" (design limitation).

## Frozen definitions

1. **Residual-EEG split-half reliability curve** (the ceiling for ANY partial statistic
   under the B1 contrast's controls): 20 seeded random 40/40 subject splits; each half's
   group RDM built exactly as the machinery builds it (per-subject z-scored stacks, mean);
   per timepoint, rank both halves' 4005 cells, residualize each against the ranked
   controls {A, C, material, lowlevel}, Pearson between the two residual vectors;
   average over the 20 splits, then Spearman-Brown x2 to the full-group scale.
   The bound on any observable partial correlation at t is sqrt(SB reliability).
2. **Model unique-variance fractions** (stimulus side): 1 - R^2 of the ranked B1 (and B2)
   RDM cells regressed on the ranked controls — how much of the model direction survives
   the partialling at all.
3. **Reported beside H1/H2**: the ceiling curve, its 150-600 ms mean, sqrt-bounds, and the
   unique-variance fractions; the H1 null wording in STATUS follows the D29 rule using the
   150-600 ms mean ceiling.

Seeds: splits seeded 0..19. No EEG quantity beyond the already-built subj_rdms is used.
