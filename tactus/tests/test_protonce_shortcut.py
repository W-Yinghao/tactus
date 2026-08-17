"""ProtoNCE must not be solvable without any EEG-video relationship.

Regression guard for a real failure found on 2026-08-16: with the shipped
default ``live_positive=True``, ``configs/nice_protonce.yaml`` reached
``condition_acc = 0.9999`` on the training split within two epochs while the
inner-validation retrieval sat exactly at chance.

Mechanism.  ``live_positive=True`` replaces the positive logit with the *live*
differentiable ``scale * <z_eeg, z_vid>``, while every negative is a *stale*,
detached EMA entry of the prototype bank.  The video projector can therefore
drive the loss to zero by rotating the current batch's embeddings away from
their own lagging bank copies, which requires no information in ``z_eeg`` at
all.  The training curve looks excellent and means nothing.

The probe below makes that unambiguous by feeding the EEG tower **pure noise**,
independent of the video identity by construction.  Any accuracy above chance is
therefore a shortcut, not learning.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from tactus.losses.protonce import ProtoNCE  # noqa: E402

DIM, N_VIDEOS, BATCH, STEPS = 64, 90, 56, 300
CHANCE = 1.0 / N_VIDEOS


def _train_on_noise(live_positive: bool) -> float:
    """Final training accuracy when the EEG side carries zero information."""
    torch.manual_seed(0)
    codebook = F.normalize(torch.randn(N_VIDEOS, DIM), dim=-1)  # frozen video tower
    eeg = nn.Sequential(nn.Linear(DIM, DIM), nn.GELU(), nn.Linear(DIM, DIM))
    vid = nn.Sequential(nn.Linear(DIM, DIM), nn.GELU(), nn.Linear(DIM, DIM))
    loss_fn = ProtoNCE(
        dim=DIM, granularities=("video",), n_videos=N_VIDEOS, source="video",
        live_positive=live_positive, momentum=0.99,
    )
    params = [*eeg.parameters(), *vid.parameters(), *loss_fn.parameters()]
    opt = torch.optim.AdamW(params, lr=3e-4)

    acc = 0.0
    for _ in range(STEPS):
        vids = torch.randperm(N_VIDEOS)[:BATCH]
        z_eeg = F.normalize(eeg(torch.randn(BATCH, DIM)), dim=-1)   # <- pure noise
        z_vid = F.normalize(vid(codebook[vids]), dim=-1)
        out = loss_fn(z_eeg, z_vid, {"video_id": vids + 1, "condition_id": vids * 4})
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        acc = out["logs"]["video_acc"]
    return acc


def test_live_positive_false_cannot_beat_chance_on_noise():
    """The configuration TACTUS actually trains with must be honest."""
    acc = _train_on_noise(live_positive=False)
    assert acc < 10 * CHANCE, (
        f"ProtoNCE(live_positive=False) reached {acc:.3f} on noise EEG "
        f"(chance {CHANCE:.3f}); the objective has a shortcut"
    )


def test_live_positive_true_is_the_known_shortcut():
    """Documents *why* the config pins live_positive: false.

    If this ever stops holding, the stale-negative/live-positive asymmetry has
    been fixed upstream and the config comment should be revisited.
    """
    acc = _train_on_noise(live_positive=True)
    assert acc > 0.5, (
        "live_positive=True no longer trivially solves the noise probe "
        f"(acc={acc:.3f}); re-examine configs/nice_protonce.yaml"
    )
