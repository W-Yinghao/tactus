"""TACTUS pluggable contrastive losses.

This package is the project's algorithm swap point.  Adding a new objective
means writing exactly one file here and decorating its class -- no edits to the
trainer, the config schema, or this file's consumers.

Quick start
-----------
>>> from tactus.losses import build_loss, list_losses
>>> list_losses()
['clisa', 'composite', 'infonce', 'masked_infonce', 'protonce', 'rnc',
 'siglip', 'softclip', 'supcon']
>>> loss_fn = build_loss({"name": "protonce", "dim": 256})
>>> out = loss_fn(z_eeg, z_vid, meta)
>>> out["loss"].backward()

See ``README.md`` in this directory for the 15-line template, what ``meta``
contains, and the three gotchas that bite on this dataset.

Smoke test
----------
``python -m tactus.losses`` runs every registered loss over a battery of
adversarial batches (duplicate conditions, single-condition batches,
single-subject batches, B=1) and asserts that none of them produce NaN, a
non-scalar loss, or a failed backward pass.  Run it first after any edit.
"""

from __future__ import annotations

from .base import (
    CONTINUOUS_META_KEYS,
    ID_META_KEYS,
    LOSS_REGISTRY,
    META_KEYS,
    ContrastiveLoss,
    TemperatureMixin,
    build_loss,
    get_loss,
    get_meta,
    list_losses,
    make_dummy_batch,
    mask_value,
    masked_log_softmax,
    masked_logsumexp,
    pairwise_eq,
    register_loss,
    safe_normalize,
    safe_pdist,
    to_float,
)

# --- importing these modules is what runs the @register_loss decorators ----- #
# Keep this list alphabetical and keep every loss file listed, or build_loss
# will raise "unknown loss" for a file that exists on disk.
from .clisa import CLISA
from .composite import CompositeLoss
# Flagship objective (BLUEPRINT_v3). Registration is an import side effect, so
# without this line `loss.name: factorized` raises "unknown loss" and the arm
# cannot run at all -- which is exactly why it had never been run (DECISIONS D20).
from .factorized import FactorizedFHMC
from .infonce import InfoNCE
from .masked_infonce import MaskedInfoNCE
from .protonce import ProtoNCE
from .rnc import RankNContrast
from .siglip import SigLIP
from .softclip import SoftCLIP
from .supcon import SupCon

__all__ = [
    # contract + registry
    "ContrastiveLoss",
    "TemperatureMixin",
    "LOSS_REGISTRY",
    "register_loss",
    "build_loss",
    "get_loss",
    "list_losses",
    # meta contract
    "META_KEYS",
    "ID_META_KEYS",
    "CONTINUOUS_META_KEYS",
    "get_meta",
    # numerics helpers for new losses
    "safe_normalize",
    "safe_pdist",
    "mask_value",
    "masked_log_softmax",
    "masked_logsumexp",
    "pairwise_eq",
    "to_float",
    "make_dummy_batch",
    # concrete losses
    "InfoNCE",
    "MaskedInfoNCE",
    "SupCon",
    "ProtoNCE",
    "SoftCLIP",
    "SigLIP",
    "RankNContrast",
    "CLISA",
    "CompositeLoss",
    "FactorizedFHMC",
    # smoke test
    "selftest",
]


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #

#: Constructor kwargs used by :func:`selftest` for losses that cannot be built
#: with no arguments at all (or whose defaults would be slow in a smoke test).
_SELFTEST_KWARGS = {
    "protonce": {"dim": 256},
    "softclip": {"target_matrix": "__random_90__"},
    "composite": {
        "components": {
            "protonce": {"weight": 1.0, "dim": 256},
            "clisa": {"weight": 0.3},
            "rnc": {"weight": 0.1},
        }
    },
}

#: Batch scenarios every loss must survive.  These are not hypothetical: the
#: dense 80 x 360 x 8 design makes duplicate conditions the normal case, and a
#: stratified sampler can easily emit a single-subject or single-condition batch
#: at the tail of an epoch.
_SELFTEST_SCENARIOS = [
    ("random", {}),
    ("many_dups", {"n_unique_conditions": 4}),
    ("all_distinct", {"n_unique_conditions": 32}),
    ("single_condition", {"single_condition": True}),
    ("single_subject", {"single_subject": True}),
    ("single_subj_seq", {"single_subject": True, "single_sequence": True}),
    ("batch_of_2", {"batch_size": 2}),
    ("batch_of_1", {"batch_size": 1}),
]


def _build_for_selftest(name: str, dim: int):
    import copy

    import torch

    kwargs = copy.deepcopy(_SELFTEST_KWARGS.get(name, {}))
    if kwargs.get("target_matrix") == "__random_90__":
        # A dedicated generator, not the global RNG: the smoke-test numbers must
        # be byte-reproducible across runs so a real regression is visible.
        gen = torch.Generator().manual_seed(20260816)
        counts = torch.randint(0, 200, (90, 4), generator=gen).float()
        kwargs["target_matrix"] = SoftCLIP.from_rater_counts(counts)
        kwargs["row_weights"] = SoftCLIP.disagreement_weights(counts)
    if name in ("protonce", "factorized"):
        # Both size internal parameters from the embedding width, so the battery
        # has to hand it over; the rest take it from the batch.
        kwargs["dim"] = dim
    if name == "composite":
        kwargs["components"]["protonce"]["dim"] = dim
    return build_loss({"name": name, **kwargs})


def selftest(
    batch_size: int = 32,
    dim: int = 256,
    names=None,
    strict: bool = True,
    verbose: bool = True,
) -> dict:
    """Run every registered loss over the adversarial batch battery.

    Checks, for each (loss, scenario) pair:

    * the returned ``loss`` is a 0-dim tensor,
    * it is finite (no NaN, no inf),
    * ``logs`` contains only plain floats (a live tensor here leaks the graph),
    * ``.backward()`` succeeds and leaves finite gradients on the inputs.

    Parameters
    ----------
    strict
        Re-raise the first failure instead of only recording it.

    Returns
    -------
    dict
        ``{loss_name: {scenario: value_or_error_string}}``.
    """
    import torch

    results: dict = {}
    targets = list(names) if names else list_losses()

    for name in targets:
        results[name] = {}
        for tag, kw in _SELFTEST_SCENARIOS:
            spec = {"batch_size": batch_size, "dim": dim, **kw}
            try:
                # fresh instance per scenario: ProtoNCE carries bank state
                fn = _build_for_selftest(name, dim)
                z_eeg, z_vid, meta = make_dummy_batch(seed=0, **spec)
                out = fn(z_eeg, z_vid, meta)

                if not isinstance(out, dict) or "loss" not in out:
                    raise TypeError(f"forward returned {type(out)!r}, expected dict")
                loss = out["loss"]
                if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
                    raise TypeError(
                        f"loss must be a 0-dim tensor, got "
                        f"{type(loss)!r} shape "
                        f"{tuple(loss.shape) if isinstance(loss, torch.Tensor) else '-'}"
                    )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"loss is not finite: {loss}")
                for k, v in out.get("logs", {}).items():
                    if isinstance(v, torch.Tensor):
                        raise TypeError(
                            f"logs['{k}'] is a Tensor; logs must hold plain "
                            f"floats or the training graph leaks"
                        )
                    float(v)

                loss.backward()
                for tensor_name, t in (("z_eeg", z_eeg), ("z_vid", z_vid)):
                    if t.grad is not None and not bool(torch.isfinite(t.grad).all()):
                        raise FloatingPointError(
                            f"non-finite gradient on {tensor_name}"
                        )
                results[name][tag] = float(loss.item())
            except Exception as exc:  # noqa: BLE001
                results[name][tag] = f"FAIL: {type(exc).__name__}: {exc}"
                if strict:
                    raise

    if verbose:
        scen = [t for t, _ in _SELFTEST_SCENARIOS]
        width = max(len(n) for n in targets) + 2
        header = "loss".ljust(width) + "".join(s[:15].rjust(17) for s in scen)
        print(header)
        print("-" * len(header))
        for name in targets:
            row = name.ljust(width)
            for tag in scen:
                v = results[name][tag]
                row += (f"{v:17.4f}" if isinstance(v, float) else str(v)[:17].rjust(17))
            print(row)
        print("-" * len(header))
        n_fail = sum(
            1
            for name in targets
            for tag in scen
            if not isinstance(results[name][tag], float)
        )
        print(
            f"{len(targets)} losses x {len(scen)} scenarios: "
            f"{len(targets) * len(scen) - n_fail} passed, {n_fail} failed"
        )
    return results


if __name__ == "__main__":  # pragma: no cover
    selftest()
