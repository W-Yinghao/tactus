"""The FHMC disentangler must measure entanglement, not embedding width.

Shipped as a cross-*covariance* between L2-normalized heads, the penalty scaled
as 1/(d_content * d_geometry).  On the trained checkpoint of the first FHMC run
it read 5.8e-07 -- weighted by lambda_disentangle=0.1, six parts in a hundred
million of a loss of magnitude 9 -- while the largest cross-correlation between
a content and a geometry coordinate was 0.788.  The constraint was inoperative
for the entire run, and the probe table showed it: the flip-invariant content
head decoded orientation at 0.595, the flip-equivariant geometry head at 0.604.

These tests pin the two properties that would have caught it.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tactus.losses import build_loss


def _correlated(n: int, d_a: int, d_b: int, r: float, seed: int = 0):
    """Two blocks sharing a common factor of strength ``r``, L2-normalized.

    The normalization is the whole point and must not be dropped: it is what
    makes a coordinate's variance ~1/d, and therefore what made the covariance
    form shrink with head width.  On unnormalized unit-variance blocks
    covariance and correlation coincide and this file tests nothing.
    """
    g = torch.Generator().manual_seed(seed)
    shared = torch.randn(n, 1, generator=g)
    a = r * shared + (1 - r) * torch.randn(n, d_a, generator=g)
    b = r * shared + (1 - r) * torch.randn(n, d_b, generator=g)
    return (a / a.norm(dim=1, keepdim=True).clamp_min(1e-12),
            b / b.norm(dim=1, keepdim=True).clamp_min(1e-12))


def _old_covariance_form(a: torch.Tensor, b: torch.Tensor) -> float:
    """The shipped penalty, kept here so the tests below can prove they bite."""
    a_c = a - a.mean(dim=0, keepdim=True)
    b_c = b - b.mean(dim=0, keepdim=True)
    return float(((a_c.t() @ b_c / max(len(a) - 1, 1)) ** 2).mean())


@pytest.mark.parametrize("d_a,d_b", [(128, 32), (512, 128)])
def test_the_shipped_covariance_form_fails_this_file(d_a, d_b):
    """Guard against the guard: these tests must reject the defect they describe."""
    ref = _old_covariance_form(*_correlated(1024, 32, 8, r=0.7))
    got = _old_covariance_form(*_correlated(1024, d_a, d_b, r=0.7))
    assert not (0.5 < got / ref < 2.0), (
        f"the covariance form survived the width check at {d_a}x{d_b} "
        f"(ratio {got / ref:.3f}); this test file would not have caught the bug"
    )


@pytest.mark.parametrize("d_a,d_b", [(32, 8), (128, 32), (512, 128)])
def test_penalty_is_scale_free_in_head_width(d_a, d_b):
    """Same entanglement, different widths -> same penalty (within a factor of 2).

    The covariance form fails this by roughly (512*128)/(32*8) = 256x.
    """
    fn = build_loss({"name": "factorized", "dim": 256})
    a, b = _correlated(1024, d_a, d_b, r=0.7)
    pen, _ = fn._cross_correlation(a, b)
    ref_a, ref_b = _correlated(1024, 32, 8, r=0.7)
    ref, _ = fn._cross_correlation(ref_a, ref_b)
    assert 0.5 < float(pen) / float(ref) < 2.0, (
        f"penalty moved {float(pen) / float(ref):.1f}x when only the head widths "
        f"changed ({d_a}x{d_b} vs 32x8); it is measuring width, not entanglement"
    )


def test_penalty_tracks_actual_entanglement():
    """Independent blocks read ~0; strongly shared blocks read far above them."""
    fn = build_loss({"name": "factorized", "dim": 256})
    lo, lo_max = fn._cross_correlation(*_correlated(2048, 128, 32, r=0.0))
    hi, hi_max = fn._cross_correlation(*_correlated(2048, 128, 32, r=0.7))
    assert float(hi) > 20 * float(lo), (
        f"entangled blocks scored {float(hi):.3e} vs independent {float(lo):.3e}"
    )
    assert float(hi_max) > 0.5 and float(lo_max) < 0.35


def test_disent_term_is_a_visible_share_of_the_total():
    """A term worth keeping has to be able to reach a readable share of the loss.

    Not a claim about the tuned lambda -- a claim that the *units* allow one to
    exist.  At the entanglement the first run actually reached (mean squared
    cross-correlation 5.2e-02) the shipped lambda of 0.1 buys 0.06% of the
    total, which is why this checks the penalty's own magnitude rather than the
    config.
    """
    fn = build_loss({"name": "factorized", "dim": 256})
    pen, _ = fn._cross_correlation(*_correlated(2048, 128, 32, r=0.7))
    assert float(pen) > 1e-2, (
        f"penalty is {float(pen):.3e} on heavily entangled heads; no sane lambda "
        "can turn that into a constraint without swamping the numerics"
    )
