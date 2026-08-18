"""Every loss file on disk must actually be registered.

`@register_loss` runs as an import side effect, so a loss whose module is never
imported in `tactus/losses/__init__.py` exists on disk, passes review, is listed
in the blueprint -- and raises "unknown loss" the moment a config selects it.
That is not hypothetical: it is why the flagship FHMC objective had never been
run (DECISIONS D20). The only signal was that the blueprint documented an 80/80
regression battery while the repository reported 72/72, and nobody chased the
missing eight.

This is the guard for the workflow the repository is built around -- add one
file, add one import line, change one config key. It fails on the step that is
easy to forget.
"""

import ast
import re
from pathlib import Path

import pytest

from tactus.losses import list_losses

LOSS_DIR = Path(__file__).resolve().parents[1] / "tactus" / "losses"

#: Modules that legitimately register nothing.
INFRASTRUCTURE = {"__init__", "base"}


def _registered_names(path: Path):
    """Names passed to @register_loss in this file, read statically.

    Static parsing rather than import: importing is the very thing whose absence
    is under test, so using it here would make the check vacuous.
    """
    tree = ast.parse(path.read_text())
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and getattr(dec.func, "id", "") == "register_loss":
                for arg in dec.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        names.append((arg.value, node.name))
    return names


def _loss_files():
    return [p for p in sorted(LOSS_DIR.glob("*.py")) if p.stem not in INFRASTRUCTURE]


def test_there_are_loss_files_to_check():
    """Guard the guard: a bad glob would make every test below pass silently."""
    assert len(_loss_files()) >= 5, f"only found {_loss_files()} under {LOSS_DIR}"


@pytest.mark.parametrize("path", _loss_files(), ids=lambda p: p.stem)
def test_every_registered_loss_is_reachable(path):
    declared = _registered_names(path)
    if not declared:
        pytest.skip(f"{path.name} registers no loss")
    available = set(list_losses())
    for name, cls in declared:
        assert name in available, (
            f"{path.name} declares @register_loss({name!r}) on class {cls}, but "
            f"{name!r} is not in list_losses(). The decorator only runs when the "
            f"module is imported: add `from .{path.stem} import {cls}` to "
            f"tactus/losses/__init__.py. Until then `loss.name: {name}` raises "
            f"'unknown loss' and that arm cannot run at all."
        )


def test_init_imports_every_loss_module():
    """The import list is the mechanism; check it directly, not just its effect.

    A module could be reachable transitively -- imported by another loss file --
    which would satisfy the test above while leaving the documented invariant
    ("keep every loss file listed") quietly false.
    """
    init = (LOSS_DIR / "__init__.py").read_text()
    for path in _loss_files():
        if not _registered_names(path):
            continue
        assert re.search(rf"^from \.{re.escape(path.stem)} import ", init, re.M), (
            f"tactus/losses/__init__.py has no `from .{path.stem} import ...` line. "
            "Registration is an import side effect and this file is the import list."
        )


def test_registry_and_files_agree_in_both_directions():
    """No registered name without a file, no file without a registered name."""
    from_files = {n for p in _loss_files() for n, _ in _registered_names(p)}
    from_registry = set(list_losses())
    assert from_registry - from_files == set(), (
        f"registered but not declared in any loss file: "
        f"{sorted(from_registry - from_files)}"
    )
    assert from_files - from_registry == set(), (
        f"declared on disk but not registered: {sorted(from_files - from_registry)}"
    )
