"""Classical baselines -- the ladder the deep model has to beat.

Four modules, in ascending order of ambition:

``linear_mvpa``
    Time-resolved decoding with shrinkage LDA and ridge regression, the exact
    analysis of the companion paper (Imaging Neuroscience 2025), re-run under
    *our* cross-validation so the two are commensurable.  This is the pipeline
    correctness check: if the published onset latencies do not reproduce, the
    preprocessing is wrong and nothing downstream means anything.
``linear_align``
    Ridge from EEG to the frozen video embedding, retrieval by cosine.  The
    "is the deep encoder doing anything a linear map cannot?" control.
``corrca``
    Correlated Component Analysis / inter-subject correlation.  Supplies the
    per-subject ISC magnitude that the phenotype analysis needs as an SNR
    nuisance regressor.
``srm``
    Shared Response Model with cheap new-subject enrollment -- the classical
    competitor to deep cross-subject alignment, and the source of the
    "k calibration trials" curve.

Every module is runnable as ``python -m tactus.baselines.<name> --help`` and
writes into ``results/baselines/<name>/``.  All four are resumable: per-subject
outputs are cached and skipped on a re-run.
"""

from __future__ import annotations

__all__ = ["linear_mvpa", "linear_align", "corrca", "srm"]
