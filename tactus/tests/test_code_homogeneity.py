"""An arm's folds must have been trained by one version of the code.

Folds run for hours and repositories get edited meanwhile. Live example: the
FHMC disentangler was changed from a cross-covariance to a cross-correlation
while a 40-fold double-disjoint grid was in flight. Three folds finished with
that term reading 1.2e-06 and the rest 4.1e-02 -- a 33,000x difference -- inside
one run directory, with every artefact looking normal. Aggregating them averages
two objectives and reports the result as one arm.

The check is a warning rather than an error: most mid-run edits touch nothing
relevant, and only the reader knows which is which. What must not happen is
silence.
"""

import json

import pytest

from tactus.eval.run_report import fold_dirs


def _fold(root, name, regime="within_subject", commit=None, dirty=False):
    d = root / "folds" / name
    d.mkdir(parents=True)
    m = {"regime": regime, "fold_key": name}
    if commit is not None:
        m["code_commit"] = commit
        m["code_dirty"] = dirty
    (d / "metrics.json").write_text(json.dumps(m))
    return d


def test_mixed_commits_are_reported(tmp_path, capsys):
    run = tmp_path / "arm"
    for i, c in enumerate(["aaa1111", "aaa1111", "bbb2222"]):
        _fold(run, f"vf{i:02d}", commit=c)
    fold_dirs(run, "within_subject")
    err = capsys.readouterr().err
    assert "WARNING" in err and "2 code versions" in err, err
    assert "aaa1111" in err and "bbb2222" in err, err


def test_a_dirty_tree_is_a_different_version(tmp_path, capsys):
    """The same commit with uncommitted edits does not pin the code."""
    run = tmp_path / "arm"
    _fold(run, "vf00", commit="aaa1111", dirty=False)
    _fold(run, "vf01", commit="aaa1111", dirty=True)
    fold_dirs(run, "within_subject")
    err = capsys.readouterr().err
    assert "WARNING" in err and "+dirty" in err, err


def test_uniform_commits_are_silent(tmp_path, capsys):
    run = tmp_path / "arm"
    for i in range(3):
        _fold(run, f"vf{i:02d}", commit="aaa1111")
    fold_dirs(run, "within_subject")
    err = capsys.readouterr().err
    assert "WARNING" not in err, err
    assert "code versions" not in err, err


def test_folds_without_fingerprints_say_so(tmp_path, capsys):
    """Absence of the field must read as "unchecked", never as "checked and fine"."""
    run = tmp_path / "arm"
    for i in range(3):
        _fold(run, f"vf{i:02d}")
    fold_dirs(run, "within_subject")
    err = capsys.readouterr().err
    assert "predate per-fold code fingerprints" in err, err


def test_a_single_fold_is_not_flagged(tmp_path, capsys):
    """One fold cannot be heterogeneous with itself."""
    run = tmp_path / "arm"
    _fold(run, "vf00")
    fold_dirs(run, "within_subject")
    err = capsys.readouterr().err
    assert "predate" not in err and "WARNING" not in err, err


def test_the_trainer_records_what_the_check_reads():
    """Guard the guard: the writer and the reader must agree on the field names."""
    from dataclasses import fields

    from tactus.train.trainer import TrainResult, _git_commit, _git_dirty

    names = {f.name for f in fields(TrainResult)}
    assert {"code_commit", "code_dirty"} <= names, sorted(names)
    assert isinstance(_git_commit(), str)
    assert isinstance(_git_dirty(), bool)
