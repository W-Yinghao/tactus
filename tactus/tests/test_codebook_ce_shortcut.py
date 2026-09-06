"""The live-codebook loss must be unsolvable without an EEG-video relationship.

Same probe as ``test_protonce_shortcut.py``: the EEG tower receives pure noise,
independent of video identity by construction, so any training accuracy above
chance is a shortcut.  ProtoNCE with ``live_positive=True`` fails this probe
(stale negatives, live positive).  ``codebook_ce`` makes every prototype live,
so there is no asymmetry to exploit -- this test pins that, and also checks the
loss actually learns when the EEG carries the signal.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from tactus.losses.codebook_ce import LiveCodebookCE  # noqa: E402

DIM, N_COND, N_ORI, BATCH, STEPS = 64, 360, 4, 64, 300
N_VID = N_COND // N_ORI


def _run(signal: bool) -> float:
    torch.manual_seed(0)
    # frozen "video tower": the four orientation siblings of a video are near-
    # identical embeddings (cos ~0.96 on the real cache), so the video prototype
    # formed by averaging them is meaningful -- the synthetic codebook keeps that.
    video_vec = torch.randn(N_VID, DIM)
    codebook_raw = F.normalize(video_vec.repeat_interleave(N_ORI, dim=0) + 0.2 * torch.randn(N_COND, DIM), dim=-1)
    eeg = nn.Sequential(nn.Linear(DIM, DIM), nn.GELU(), nn.Linear(DIM, DIM))
    proj = nn.Sequential(nn.Linear(DIM, DIM), nn.GELU(), nn.Linear(DIM, DIM))
    loss_fn = LiveCodebookCE(n_conditions=N_COND, n_orientations=N_ORI, temperature=0.05)
    opt = torch.optim.Adam([*eeg.parameters(), *proj.parameters(), *loss_fn.parameters()], lr=3e-3)
    acc = 0.0
    for _ in range(STEPS):
        cond = torch.randint(0, N_COND, (BATCH,))
        x = codebook_raw[cond] + 0.3 * torch.randn(BATCH, DIM) if signal else torch.randn(BATCH, DIM)
        z_eeg = F.normalize(eeg(x), dim=-1)
        cb = F.normalize(proj(codebook_raw), dim=-1)                 # ALL prototypes live
        out = loss_fn(z_eeg, None, {"condition_id": cond, "codebook": cb})
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        acc = float(out["cb/video_acc"])
    return acc


def test_noise_eeg_stays_at_chance():
    acc = _run(signal=False)
    assert acc < 3.0 / N_VID, f"noise EEG reached video acc {acc:.3f}; a shortcut exists"


def test_signal_eeg_learns():
    acc = _run(signal=True)
    assert acc > 0.5, f"signal EEG only reached {acc:.3f}; the loss does not learn"


def test_requires_codebook():
    loss_fn = LiveCodebookCE(n_conditions=N_COND)
    with pytest.raises(KeyError):
        loss_fn(torch.randn(4, DIM), None, {"condition_id": torch.zeros(4, dtype=torch.long)})
