"""A repeated ``-o`` must accumulate, never replace.

The trap: argparse ``nargs="*"`` without ``action="extend"`` keeps only the last
occurrence.  ``-o train.n_train_trials=10953 -o train.curriculum.enabled=false``
then trains on the full set under a tag that says otherwise -- which is exactly
what happened once, for twelve GPU tasks, with four "budgets" coming back
bit-identical.  These tests pin the parser behaviour and prove the old form fails.
"""
from __future__ import annotations

import argparse
import inspect


def test_run_py_declares_extend():
    import tactus.train.run as run

    assert 'action="extend"' in inspect.getsource(run), "the -o flag lost its extend action"


def test_repeated_override_accumulates():
    p = argparse.ArgumentParser()
    p.add_argument("--override", "-o", nargs="*", action="extend", default=[])
    ns = p.parse_args(["-o", "a=1", "-o", "b=2", "c=3"])
    assert ns.override == ["a=1", "b=2", "c=3"]


def test_the_old_declaration_loses_the_first_flag():
    """Guard the guard: the shipped-before form really drops overrides."""
    p = argparse.ArgumentParser()
    p.add_argument("--override", "-o", nargs="*", default=[])
    ns = p.parse_args(["-o", "a=1", "-o", "b=2"])
    assert ns.override == ["b=2"]
