"""The noise ceiling is a property of the data, so it must not depend on the arm.

A split-half EEG-to-EEG ceiling never sees the model. Two arms evaluated on the
same folds must therefore be divided by the same number, and for a while they
were not: pooling every available subject made a within-subject fold (80 test
subjects) and a double-disjoint fold (10) incomparable, and pinning only the
*count* left the draw random, which moved the ceiling from 0.1122 to 0.1539
across seeds -- more than the accuracy gaps it was being used to compare
(DECISIONS D15).

These tests pin the two properties that keep the denominator honest.
"""

import numpy as np
import pytest

from tactus.eval.noise_ceiling import retrieval_noise_ceiling


def _synthetic(n_subjects=40, n_items=18, k=8, dim=32, snr=0.10, seed=0):
    """Items with a shared signal plus per-trial noise; subjects are exchangeable."""
    rng = np.random.default_rng(seed)
    item_sig = rng.standard_normal((n_items, dim))
    z, items, subs = [], [], []
    for s in range(n_subjects):
        for i in range(n_items):
            x = snr * item_sig[i] + (1 - snr) * rng.standard_normal((k, dim))
            z.append(x)
            items.extend([i] * k)
            subs.extend([s] * k)
    z = np.concatenate(z)
    return z / np.linalg.norm(z, axis=1, keepdims=True), np.array(items), np.array(subs)


def _pooled(z, items, subs, **kw):
    t = retrieval_noise_ceiling(z, items, subs, k=4, gallery_sizes=(18,),
                                n_resamples=6, per_subject=False, **kw)
    row = t[(t.subject_id == "pooled") & (t.endpoint == "nway18_top1")]
    assert len(row) == 1, f"expected exactly one pooled row, got {len(row)}"
    return row.iloc[0]


def test_single_draw_denominator_is_seed_dependent():
    """Guard the guard: without averaging, the seed alone must move the ceiling.

    If this stops failing, the averaging test below has become vacuous.
    """
    z, items, subs = _synthetic()
    vals = [float(_pooled(z, items, subs, n_gallery_subjects=8,
                          n_gallery_draws=1, seed=s).ceiling) for s in range(6)]
    assert np.std(vals) > 0, (
        f"a single subject draw gave identical ceilings across seeds ({vals}); "
        "the subsample is not actually varying and this file tests nothing"
    )


def test_averaging_over_draws_stabilises_the_denominator():
    """Many draws must be markedly more seed-stable than one."""
    z, items, subs = _synthetic()
    one = [float(_pooled(z, items, subs, n_gallery_subjects=8,
                         n_gallery_draws=1, seed=s).ceiling) for s in range(6)]
    many = [float(_pooled(z, items, subs, n_gallery_subjects=8,
                          n_gallery_draws=20, seed=s).ceiling) for s in range(6)]
    assert np.std(many) < np.std(one), (
        f"averaging did not stabilise the ceiling: sd {np.std(many):.5f} over 20 "
        f"draws vs {np.std(one):.5f} over 1"
    )


def test_pooled_row_carries_its_own_spread():
    """A denominator without an uncertainty invites a fraction quoted as exact."""
    z, items, subs = _synthetic()
    row = _pooled(z, items, subs, n_gallery_subjects=8, n_gallery_draws=20)
    for col in ("ceiling_sd", "ceiling_lo", "ceiling_hi", "n_gallery_draws"):
        assert col in row.index, f"pooled row is missing {col}"
    assert row.ceiling_sd > 0
    assert row.ceiling_lo <= row.ceiling <= row.ceiling_hi
    assert int(row.n_gallery_draws) == 20


def test_no_subsampling_when_the_fold_has_exactly_the_pinned_count():
    """A double-disjoint fold has exactly 10 subjects: use them, do not resample."""
    z, items, subs = _synthetic(n_subjects=10)
    row = _pooled(z, items, subs, n_gallery_subjects=10, n_gallery_draws=20)
    assert "n_gallery_draws" not in row.index or np.isnan(row.get("n_gallery_draws", np.nan)), (
        "the fold already has exactly the pinned number of subjects, so there is "
        "nothing to draw and the ceiling must be deterministic"
    )


def test_ceiling_rises_with_the_number_of_pooled_subjects():
    """The effect that started all of this, as a property rather than an anecdote."""
    z, items, subs = _synthetic(n_subjects=40)
    vals = [float(_pooled(z, items, subs, n_gallery_subjects=n,
                          n_gallery_draws=8, seed=0).ceiling) for n in (5, 10, 20)]
    assert vals[0] < vals[1] < vals[2], (
        f"pooling more subjects should clean the gallery and raise the ceiling; got {vals}"
    )
