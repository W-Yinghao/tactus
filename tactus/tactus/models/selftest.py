#!/usr/bin/env python
"""Self-test for the TACTUS model zoo.  Run this first on the server.

    python -m tactus.models.selftest            # all checks
    python -m tactus.models.selftest --quick    # skip the ~3M-parameter ATM checks

It asserts the parts of the interface contract that are easy to break silently:

* contract F -- ``forward(x, subject_id)`` returns ``(B, D)`` unit vectors;
* the **pre-registered unseen-subject rules** actually fire (mean token / identity
  layer / zero adapter), including for a subject that has parameters but was not in
  ``set_train_subjects`` -- the LOSO failure mode that would otherwise read noise;
* SuLoRA adapters survive ``deepcopy`` inside ``TimeWindowHeads(share_trunk=False)``;
* time-window slicing lands on the right samples and every head emits unit vectors;
* the torch EA applier agrees with the numpy transform it was built from.

Every check prints PASS/FAIL; the exit code is non-zero if anything failed.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Callable, List, Tuple

import numpy as np
import torch

from .ea import EAApplier, EuclideanAlignment
from .eeg.atm import ATMEncoder
from .eeg.base import count_parameters
from .eeg.tsconv import TSConvEncoder
from .heads import DEFAULT_TIME_WINDOWS_MS, TimeWindowHeads, VideoProjector
from . import build_eeg_encoder, build_video_projector

_RESULTS: List[Tuple[str, bool, str]] = []


def check(name: str) -> Callable:
    """Decorator turning a function into a reported check."""

    def deco(fn: Callable) -> Callable:
        def wrapped(*a, **k):
            try:
                detail = fn(*a, **k) or ""
                _RESULTS.append((name, True, str(detail)))
                print(f"  PASS  {name}  {detail}")
            except Exception as exc:  # noqa: BLE001
                _RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
                print(f"  FAIL  {name}  {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=3)

        return wrapped

    return deco


def _mk(cls, cond: str, **kw):
    torch.manual_seed(0)
    kw.setdefault("n_times", 120)
    kw.setdefault("embed_dim", 256)
    kw.setdefault("n_subjects", 80)
    enc = cls(subject_cond=cond, **kw)
    return enc.eval()  # eval: no dropout / BN update, so comparisons are deterministic


# ======================================================================================


@check("forward contract: (B,64,120) -> (B,256), unit norm")
def t_forward(cls, cond):
    enc = _mk(cls, cond)
    x = torch.randn(6, 64, 120)
    sid = torch.tensor([1, 2, 3, 4, 5, -1])
    with torch.no_grad():
        z = enc(x, sid)
    assert z.shape == (6, 256), z.shape
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), norms
    assert torch.isfinite(z).all()
    return f"params={count_parameters(enc):,}"


@check("subject_id=None is legal and equals the all-unseen batch")
def t_none_subject(cls, cond):
    enc = _mk(cls, cond)
    x = torch.randn(4, 64, 120)
    with torch.no_grad():
        za = enc(x, None)
        zb = enc(x, torch.full((4,), -1))
    assert torch.allclose(za, zb, atol=1e-5), (za - zb).abs().max().item()


@check("unseen rule: subject_layer -> identity layer")
def t_unseen_layer():
    enc = _mk(TSConvEncoder, "subject_layer")
    x = torch.randn(3, 64, 120)
    known, unseen = torch.tensor([1, 2, 3]), torch.full((3,), -1)
    with torch.no_grad():
        z_unseen_0 = enc(x, unseen)
        assert torch.allclose(enc(x, known), z_unseen_0, atol=1e-6), "delta must start at zero"
        enc.conditioner.delta.normal_(0, 0.1)  # "train" the layer
        z_known_1, z_unseen_1 = enc(x, known), enc(x, unseen)
    assert not torch.allclose(z_known_1, z_unseen_0, atol=1e-4), "known subject did not change"
    assert torch.allclose(z_unseen_1, z_unseen_0, atol=1e-6), "unseen subject was affected"
    st = enc.unseen_subject_state()
    assert st["rule"] == "identity_layer" and st["fixed_a_priori"]
    assert torch.allclose(st["state"]["weight"], torch.eye(64))


@check("unseen rule: subject_token -> mean of TRAIN tokens only")
def t_unseen_token():
    enc = _mk(TSConvEncoder, "subject_token")
    emb = enc.conditioner.embedding.weight
    emb.data.normal_(0, 1.0)
    assert torch.allclose(enc.conditioner.unseen_token(), emb.data.mean(0), atol=1e-6)
    enc.set_train_subjects(list(range(1, 41)))  # fold: subjects 1..40 train
    assert torch.allclose(enc.conditioner.unseen_token(), emb.data[:40].mean(0), atol=1e-6), (
        "unseen token must average TRAIN rows only, not all 80"
    )
    x = torch.randn(2, 64, 120)
    with torch.no_grad():
        z_heldout = enc(x, torch.tensor([50, 60]))  # have rows, but are NOT train subjects
        z_unseen = enc(x, torch.tensor([-1, -1]))
    assert torch.allclose(z_heldout, z_unseen, atol=1e-6), (
        "strict_unseen failed: a held-out subject read its own untrained token"
    )
    return "held-out subjects correctly routed to the mean token"


@check("unseen rule: sulora -> zero adapter")
def t_unseen_sulora():
    enc = _mk(TSConvEncoder, "sulora", subject_cond_kwargs={"rank": 4})
    cond = enc.conditioner
    assert cond.attached_targets, "SuLoRA attached to nothing"
    assert cond.adapter_parameter_count() > 0
    x = torch.randn(3, 64, 120)
    known, unseen = torch.tensor([1, 2, 3]), torch.full((3,), -1)
    with torch.no_grad():
        z0 = enc(x, unseen)
        assert torch.allclose(enc(x, known), z0, atol=1e-6), "adapter must start at zero"
        for p in cond.adapter_parameters():
            p.normal_(0, 0.2)
        z_known, z_unseen = enc(x, known), enc(x, unseen)
    assert not torch.allclose(z_known, z0, atol=1e-4), "adapter had no effect on known subjects"
    assert torch.allclose(z_unseen, z0, atol=1e-6), "unseen subject picked up an adapter"
    st = enc.unseen_subject_state()
    assert st["rule"] == "zero_adapter" and st["fixed_a_priori"]
    return f"targets={cond.attached_targets} adapter_params={cond.adapter_parameter_count():,}"


@check("sulora raises rather than silently matching nothing")
def t_sulora_empty():
    try:
        _mk(TSConvEncoder, "sulora", subject_cond_kwargs={"target_patterns": ("*nonexistent*",)})
    except RuntimeError as exc:
        assert "matched no modules" in str(exc)
        return "guard fired"
    raise AssertionError("a no-op SuLoRA arm was allowed through")


@check("gradients reach the conditioner and the trunk")
def t_grad(cond):
    enc = _mk(TSConvEncoder, cond).train()
    x = torch.randn(8, 64, 120)
    sid = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
    z = enc(x, sid)
    z.pow(2).sum().backward()
    named = dict(enc.named_parameters())
    got = [n for n, p in named.items() if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0]
    assert any("trunk" in n for n in got), "no gradient reached the conv trunk"
    if cond == "subject_layer":
        assert any("conditioner" in n for n in got), "no gradient reached the subject layer"
    if cond == "sulora":
        assert any("lora_" in n for n in got), "no gradient reached the SuLoRA adapters"
    return f"{len(got)}/{len(named)} tensors received gradient"


@check("TimeWindowHeads: slices, keys, unit norms")
def t_twh():
    enc = _mk(TSConvEncoder, "subject_layer")
    twh = TimeWindowHeads(enc, window="w0600").eval()
    assert twh.window_slices == {"early": (0, 30), "mid": (30, 70), "late": (70, 120)}, twh.window_slices
    x = torch.randn(5, 64, 120)
    with torch.no_grad():
        out = twh(x, torch.tensor([1, 2, 3, 4, -1]))
    assert set(out) == {"early", "mid", "late", "full"}, set(out)
    for k, v in out.items():
        assert v.shape == (5, 256), (k, v.shape)
        assert torch.allclose(v.norm(dim=-1), torch.ones(5), atol=1e-4), k
    return f"windows={ {k: twh.window_slices[k] for k in ('early','mid','late')} }"


@check("TimeWindowHeads on the sensitivity window wm100_800")
def t_twh_sensitivity():
    enc = _mk(TSConvEncoder, "none", n_times=180)
    twh = TimeWindowHeads(enc, window="wm100_800").eval()
    # tmin = -100 ms, so 0 ms sits at sample 20
    assert twh.window_slices == {"early": (20, 50), "mid": (50, 90), "late": (90, 140)}, twh.window_slices
    with torch.no_grad():
        out = twh(torch.randn(3, 64, 180), torch.tensor([1, 2, 3]))
    assert all(v.shape == (3, 256) for v in out.values())


@check("TimeWindowHeads(share_trunk=False) relinks SuLoRA adapters after deepcopy")
def t_twh_deepcopy():
    enc = _mk(TSConvEncoder, "sulora", subject_cond_kwargs={"rank": 4})
    twh = TimeWindowHeads(enc, share_trunk=False).eval()
    assert all(v > 0 for v in twh.adapted_head_modules.values()), (
        f"per-window heads received no SuLoRA adapters: {twh.adapted_head_modules}"
    )
    for name, trunk in twh._trunks.items():
        assert trunk.conditioner.attached_targets, f"copy {name} lost its adapters"
        for ref in trunk.conditioner._adapter_refs:
            mod = ref()
            assert mod is not None and mod.conditioner is trunk.conditioner, (
                f"copy {name} adapter still points at another conditioner"
            )
        for p in trunk.conditioner.adapter_parameters():
            p.data.normal_(0, 0.2)
    x = torch.randn(2, 64, 120)
    with torch.no_grad():
        a = twh(x, torch.tensor([1, 1]))["early"]
        b = twh(x, torch.tensor([7, 7]))["early"]
        c = twh(x, torch.tensor([-1, -1]))["early"]
    assert not torch.allclose(a, b, atol=1e-5), "per-subject adapters are inert in the copies"
    assert not torch.allclose(a, c, atol=1e-5)
    return "adapters live and subject-specific in every copy"


@check("sliding windows generate a dense onset curve")
def t_sliding():
    enc = _mk(TSConvEncoder, "none")
    wins = TimeWindowHeads.sliding_windows(width_ms=100, step_ms=50, t_stop_ms=600)
    twh = TimeWindowHeads(enc, windows_ms=wins, include_full=False).eval()
    with torch.no_grad():
        out = twh(torch.randn(2, 64, 120), torch.tensor([1, 2]))
    assert len(out) == len(wins) == 11, (len(out), len(wins))
    return f"{len(wins)} windows, 100 ms wide, 50 ms step"


@check("VideoProjector contract: (B,D_vid) -> (B,D) unit norm")
def t_video_projector():
    proj = VideoProjector(in_dim=768, out_dim=256).eval()
    v = torch.nn.functional.normalize(torch.randn(16, 768), dim=-1)
    with torch.no_grad():
        z = proj(v)
    assert z.shape == (16, 256)
    assert torch.allclose(z.norm(dim=-1), torch.ones(16), atol=1e-4)
    try:
        proj(torch.randn(4, 1024))
    except ValueError:
        return f"params={count_parameters(proj):,}; dim mismatch rejected"
    raise AssertionError("wrong D_vid was not rejected")


@check("build_eeg_encoder / build_video_projector from a config dict")
def t_builders():
    cfg = {
        "model": {
            "name": "tsconv",
            "n_times": 120,
            "embed_dim": 128,
            "subject_cond": {"name": "subject_layer", "rank": 8},
            "n_filters": 20,
        },
        "video_projector": {"in_dim": 512, "out_dim": 128},
    }
    enc = build_eeg_encoder(cfg).eval()
    proj = build_video_projector(cfg).eval()
    with torch.no_grad():
        z_eeg = enc(torch.randn(4, 64, 120), torch.tensor([1, 2, 3, -1]))
        z_vid = proj(torch.randn(4, 512))
    assert z_eeg.shape == z_vid.shape == (4, 128)
    assert enc.conditioner.rank == 8 and enc.trunk.n_filters == 20
    return f"eeg={count_parameters(enc):,} vid={count_parameters(proj):,}"


@check("trainer calling convention (build_model in tactus/train/trainer.py)")
def t_trainer_compat():
    # exactly what trainer.build_model assembles and passes through _flex_call
    eeg_params = {
        "n_channels": 64,
        "n_times": 120,
        "d_embed": 256,
        "d_out": 256,
        "n_subjects": 81,          # trainer reserves index 0 for the unseen subject
        "subject_conditioning": "token",
    }
    enc = build_eeg_encoder("nice", **eeg_params).eval()
    assert enc.embed_dim == 256 and enc.n_times == 120
    p_params = {"d_in": 768, "d_video": 768, "d_out": 256, "d_embed": 256}
    proj = build_video_projector("mlp", **p_params).eval()
    with torch.no_grad():
        z_eeg = enc(torch.randn(4, 64, 120), torch.tensor([1, 2, 3, 0]))
        z_vid = proj(torch.randn(4, 768))
        # the trainer's "row 0 = mean/unseen subject" must equal our unseen rule
        z_zero = enc(torch.randn(1, 64, 120) * 0 + 1.0, torch.tensor([0]))
        z_neg = enc(torch.randn(1, 64, 120) * 0 + 1.0, torch.tensor([-1]))
    assert z_eeg.shape == z_vid.shape == (4, 256)
    assert torch.allclose(z_zero, z_neg, atol=1e-6), (
        "subject_id=0 must resolve to the same unseen rule as -1"
    )
    return "d_embed/d_out/subject_conditioning aliases and subject_id=0 all resolve"


@check("EA: torch applier matches the numpy transform")
def t_ea():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 64, 120)).astype(np.float32)
    x = x - x.mean(axis=1, keepdims=True)  # common average reference -> rank 63
    alg = EuclideanAlignment(min_trials=100)
    st = alg.fit_subject(x, subject_id=3)
    assert st.rank == 63, f"expected rank 63 for CAR data, got {st.rank}"
    ref = alg.transform(x[:8], 3)
    app = EAApplier(alg, n_channels=64)
    got = app(torch.from_numpy(x[:8]), torch.tensor([3] * 8)).numpy()
    assert np.allclose(ref, got, atol=1e-4), np.abs(ref - got).max()
    unseen = app(torch.from_numpy(x[:4]), torch.tensor([99, 99, 99, 99])).numpy()
    assert np.allclose(unseen, x[:4], atol=1e-6), "unseen policy 'identity' did not pass data through"
    return f"rank={st.rank} cond#={st.condition_number:.3g}"


@check("EA fitted on train trials only leaves test trials untouched by test statistics")
def t_ea_fold_aware():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 64, 120)).astype(np.float32)
    train, test = x[:300], x[300:]
    a = EuclideanAlignment(min_trials=100).fit_subject(train, 1).whitener
    b = EuclideanAlignment(min_trials=100).fit_subject(x, 1).whitener
    assert not np.allclose(a, b, atol=1e-8), "fitting on train-only vs all data gave identical whiteners"
    return "train-only fit is distinguishable from a leaky all-data fit"


def main(argv=None) -> int:  # noqa: D103
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="skip the ATM (~3M param) checks")
    ap.add_argument("--all-registered", action="store_true",
                    help="also run the forward contract over every encoder in the registry "
                         "(includes the foundation-model wrappers; downloads checkpoints)")
    args = ap.parse_args(argv)
    torch.manual_seed(0)

    print("\n== forward contract ==")
    archs = [("tsconv", TSConvEncoder)] + ([] if args.quick else [("atm", ATMEncoder)])
    if args.all_registered:
        # Contract F is a claim about EVERY registered encoder, but this list was
        # hardcoded, so an encoder added later (e.g. the foundation-model wrappers
        # in tactus/models/eeg/fm_*.py) stayed green here without ever being
        # exercised.  --all-registered walks the registry instead.  It is opt-in
        # because the FM backbones are 5-25M params and download checkpoints.
        from .eeg.base import get_eeg_encoder, list_eeg_encoders
        seen = {n for n, _ in archs}
        for name in list_eeg_encoders():
            if name in seen or name in ("nice", "shallow"):   # aliases of tsconv
                continue
            try:
                archs.append((name, get_eeg_encoder(name)))
            except Exception as exc:  # noqa: BLE001
                print(f"  SKIP  {name}: {type(exc).__name__}: {exc}")
    for arch_name, cls in archs:
        for cond in ("none", "subject_token", "subject_layer", "sulora"):
            print(f" [{arch_name}/{cond}]")
            t_forward(cls, cond)
            t_none_subject(cls, cond)

    print("\n== pre-registered unseen-subject rules ==")
    t_unseen_layer()
    t_unseen_token()
    t_unseen_sulora()
    t_sulora_empty()

    print("\n== optimisation ==")
    for cond in ("none", "subject_token", "subject_layer", "sulora"):
        t_grad(cond)

    print("\n== time-resolved heads ==")
    t_twh()
    t_twh_sensitivity()
    t_twh_deepcopy()
    t_sliding()

    print("\n== video side / builders ==")
    t_video_projector()
    t_builders()
    t_trainer_compat()

    print("\n== Euclidean Alignment ==")
    t_ea()
    t_ea_fold_aware()

    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print(f"\n{'=' * 70}\n{len(_RESULTS) - n_fail}/{len(_RESULTS)} checks passed")
    if n_fail:
        print("FAILURES:")
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f"  - {name}: {detail}")
    if not args.quick:
        print("\nParameter budgets:")
        for label, enc in (
            ("tsconv", TSConvEncoder(n_times=120, embed_dim=256)),
            ("atm", ATMEncoder(n_times=120, embed_dim=256)),
        ):
            print(f"  {label:<8s} {count_parameters(enc):>10,d}")
    return 1 if n_fail else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
