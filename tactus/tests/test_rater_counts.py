"""The VTD rater-count rebuild must fail loudly when the mapping is wrong.

The loader replicates VTD_analysis.Rmd's slices and maps questionnaire order
onto video_id through Video_overview.csv, then verifies every video against
the published percentage table.  A silent off-by-one in any of those steps
would poison the SoftCLIP targets while looking perfectly healthy, so the
verification step is the actual product here -- these tests prove it trips.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tactus.data.rater_counts import (
    CATEGORIES,
    N_RATERS_PER_VIDEO,
    N_VIDEOS,
    load_rater_counts,
    verify_against_published,
)

RNG = np.random.default_rng(20260829)


def _write_fixture(tmp_path, *, shuffle_overview: bool = False,
                   corrupt_published: bool = False):
    """Synthetic DATA1/DATA2/overview/published quadruple with known counts."""
    # questionnaire video q gets a distinctive distribution
    probs = RNG.dirichlet(np.ones(4), size=N_VIDEOS)
    responses = {}
    for q in range(1, N_VIDEOS + 1):
        responses[q] = RNG.choice(list(CATEGORIES), size=N_RATERS_PER_VIDEO,
                                  p=probs[q - 1])

    def data_file(qs, n_raters_raw):
        cols = {}
        for j in range(10):  # the 10 demographic columns the Rmd skips
            cols[f"meta{j}"] = np.arange(n_raters_raw)
        for local_v, q in enumerate(qs, start=1):
            block = np.array(["-"] * n_raters_raw, dtype=object)
            block[:N_RATERS_PER_VIDEO] = responses[q]
            cols[f"V{local_v}_category"] = block
            for metric in ("pleasant", "unpleasant", "painful", "threat", "arousal"):
                cols[f"V{local_v}_{metric}"] = RNG.integers(0, 10, n_raters_raw)
        # the 46th block the Rmd's 11:280 slice drops
        cols["V46_category"] = np.array(["Neutral"] * n_raters_raw, dtype=object)
        for metric in ("pleasant", "unpleasant", "painful", "threat", "arousal"):
            cols[f"V46_{metric}"] = RNG.integers(0, 10, n_raters_raw)
        return pd.DataFrame(cols)

    data_file(range(1, 46), 179).to_csv(tmp_path / "DATA1.csv", index=False)
    data_file(range(46, 91), 176).to_csv(tmp_path / "DATA2.csv", index=False)

    yt = np.arange(1, N_VIDEOS + 1)
    if shuffle_overview:
        yt = RNG.permutation(yt)
    pd.DataFrame({"Video_YouTube": yt,
                  "Video_Questionnaire": np.arange(1, N_VIDEOS + 1)}
                 ).to_csv(tmp_path / "Video_overview.csv", index=False)

    rows = []
    for q in range(1, N_VIDEOS + 1):
        vc = pd.Series(responses[q]).value_counts()
        pct = {c: round(float(vc.get(c, 0)) / N_RATERS_PER_VIDEO * 100.0)
               for c in CATEGORIES}
        rows.append({"Video_YouTube": int(yt[q - 1]), **pct})
    pub = pd.DataFrame(rows)
    if corrupt_published:
        pub.loc[10, "Neutral"] += 7
    pub.to_csv(tmp_path / "table_all_videos.csv", index=False)
    return responses, yt


def test_roundtrip_and_verification(tmp_path):
    responses, yt = _write_fixture(tmp_path)
    counts = load_rater_counts(tmp_path)
    assert counts.shape == (N_VIDEOS, len(CATEGORIES))
    assert (counts.sum(axis=1) == N_RATERS_PER_VIDEO).all()
    # spot-check: questionnaire video 1 landed on video_id yt[0]
    vc = pd.Series(responses[1]).value_counts()
    for c in CATEGORIES:
        assert counts.loc[int(yt[0]), c] == int(vc.get(c, 0))
    assert verify_against_published(counts, tmp_path) == N_VIDEOS


def test_shuffled_overview_still_verifies(tmp_path):
    """A permuted questionnaire->video_id map is fine as long as the published
    table was built through the same map -- the pair must stay consistent."""
    _write_fixture(tmp_path, shuffle_overview=True)
    counts = load_rater_counts(tmp_path)
    assert verify_against_published(counts, tmp_path) == N_VIDEOS


def test_corrupted_published_table_trips_verification(tmp_path):
    _write_fixture(tmp_path, corrupt_published=True)
    counts = load_rater_counts(tmp_path)
    with pytest.raises(RuntimeError, match="refusing to write targets"):
        verify_against_published(counts, tmp_path)


def test_wrong_mapping_trips_verification(tmp_path):
    """Simulate the real failure mode: counts assembled under a different
    questionnaire->video_id map than the published table used."""
    _write_fixture(tmp_path)
    counts = load_rater_counts(tmp_path)
    swapped = counts.copy()
    swapped.iloc[[0, 1]] = counts.iloc[[1, 0]].to_numpy()
    if (counts.iloc[0].to_numpy() == counts.iloc[1].to_numpy()).all():
        pytest.skip("degenerate fixture: two identical rows")
    with pytest.raises(RuntimeError, match="refusing to write targets"):
        verify_against_published(swapped, tmp_path)
