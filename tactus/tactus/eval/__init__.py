"""Evaluation, statistics and confound probes for TACTUS.

Module map
----------
``retrieval``      contract G: N-way zero-shot retrieval, both directions,
                   single-trial and pseudo-trial, within-material distractors.
``permutation``    permutation nulls whose exchangeable unit is the BASE VIDEO,
                   plus the deliberately-wrong trial-level null kept for the
                   narrowing-factor table.
``noise_ceiling``  split-half reliability from the 8 repeats -> per-subject and
                   per-condition ceilings, Spearman-Brown, fraction-of-ceiling.
``rsa``            time-resolved crossnobis/correlation RDMs over the 360
                   conditions, model RDMs, partial Spearman, cluster inference.
``probes``         the confound battery (identity, trial index, ocular
                   surrogate, frontal ablation, low-level video features).
``stats``          inference-target-aware statistics: subject-level paired
                   Wilcoxon vs crossed subject x video mixed model, hierarchical
                   FDR, cluster permutation, minimal detectable difference.
``report``         fold aggregation, bootstrap CIs, results tables, REPORT.md.

Three rules this package enforces rather than documents
-------------------------------------------------------
1. The exchangeable unit for stimulus inference is the base video (90), not the
   trial (2880 per subject).  ``permutation`` will not silently give you a
   trial-level p-value.
2. Subject-identity accuracy is never returned without the alignment score it
   must be traded against (:func:`probes.subject_identity_probe`).
3. Every ablation is read against the minimal detectable difference of the
   inference target it claims (:func:`stats.print_mdd_table`,
   :func:`stats.flag_below_mdd`); the report prints the resolution before it
   prints any ablation.
"""

from __future__ import annotations

from . import noise_ceiling, permutation, probes, report, retrieval, rsa, stats
from .noise_ceiling import (
    attenuation_correct,
    fraction_of_ceiling,
    retrieval_noise_ceiling,
    spearman_brown,
    split_half_reliability,
    subject_noise_ceiling,
)
from .permutation import (
    PermutationResult,
    maxstat_correction,
    null_narrowing_report,
    retrieval_permutation_test,
    trial_level_null_diagnostic,
    video_level_permutation_test,
)
from .probes import (
    OCULAR_CLAIM_LIMITATION,
    RidgeAlignmentProbe,
    frontal_ablation_sensitivity,
    lowlevel_control_analysis,
    ocular_control,
    run_confound_battery,
    subject_identity_probe,
    trial_index_control,
)
from .report import ReportInputs, aggregate_folds, compare_arms, emit_report, write_table
from .retrieval import (
    DEFAULT_GALLERY_SIZES,
    RetrievalConfig,
    build_pseudo_trials,
    evaluate_retrieval,
    primary_endpoint,
    retrieval_metrics,
)
from .rsa import (
    RSAResult,
    build_model_rdms,
    partial_spearman,
    rsa_noise_ceiling,
    rsa_time_course,
    time_resolved_rdms,
)
from .stats import (
    bootstrap_ci,
    cluster_permutation_1d,
    cluster_permutation_from_null,
    crossed_mixed_model,
    flag_below_mdd,
    hierarchical_fdr,
    mdd_table,
    paired_wilcoxon,
    print_mdd_table,
    recommend_test,
)

__all__ = [
    # submodules
    "retrieval", "permutation", "noise_ceiling", "rsa", "probes", "stats", "report",
    # retrieval
    "retrieval_metrics", "evaluate_retrieval", "RetrievalConfig",
    "DEFAULT_GALLERY_SIZES", "build_pseudo_trials", "primary_endpoint",
    # permutation
    "PermutationResult", "video_level_permutation_test",
    "trial_level_null_diagnostic", "null_narrowing_report",
    "retrieval_permutation_test", "maxstat_correction",
    # ceilings
    "spearman_brown", "attenuation_correct", "fraction_of_ceiling",
    "split_half_reliability", "subject_noise_ceiling", "retrieval_noise_ceiling",
    # rsa
    "RSAResult", "time_resolved_rdms", "build_model_rdms", "rsa_time_course",
    "partial_spearman", "rsa_noise_ceiling",
    # probes
    "OCULAR_CLAIM_LIMITATION", "RidgeAlignmentProbe", "subject_identity_probe",
    "trial_index_control", "ocular_control", "frontal_ablation_sensitivity",
    "lowlevel_control_analysis", "run_confound_battery",
    # stats
    "bootstrap_ci", "paired_wilcoxon", "crossed_mixed_model", "hierarchical_fdr",
    "cluster_permutation_1d", "cluster_permutation_from_null", "mdd_table",
    "print_mdd_table", "flag_below_mdd", "recommend_test",
    # report
    "aggregate_folds", "compare_arms", "ReportInputs", "emit_report", "write_table",
]
