"""Regression tests for tactus.data.qc -- the channel QC that would have caught sub-17.

The load-bearing pin is the *third* block: an IQR-based robust spread estimator
is blind to a blown electrode by construction.  That is exactly why the robust
scaling in ``tactus.data.preprocess`` let sub-17's ``PO4`` (max|x| = 33298
robust sigmas) through, and why the detector here keys on the **ordinary** sd.
If someone ever "improves" ``channel_stats`` to flag on ``robust_sd_ratio``,
that block fails.

Run with ``pytest tests/test_channel_qc.py`` or ``python tests/test_channel_qc.py``.
The real-data block is skipped when the frozen memmaps are not mounted.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

fails = []


def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("  " + str(extra) if extra != "" else ""))
    if not cond:
        fails.append(name)


from tactus.data.qc import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    QCThresholds,
    channel_stats,
    clamp_impact,  # noqa: F401  (imported to pin the public surface)
    clamped_fraction,
    robust_clamp,
    run_qc,
    subject_qc,
    verify_qc,
)
from tactus.common import SubjectTransform  # noqa: E402
from tactus.data import EPOCH_DIR, N_CHANNELS, epoch_path  # noqa: E402


# ------------------------------------------------------------------ robust_clamp
rng = np.random.default_rng(0)
x = (rng.standard_normal((16, N_CHANNELS, 40)) * 3.0).astype(np.float32)
x[3, 7, 11] = 5000.0
x[9, 2, 4] = -900.0

y = robust_clamp(x, 20.0)
check("robust_clamp bounds", float(np.abs(y).max()) <= 20.0 + 1e-6, float(np.abs(y).max()))
check("robust_clamp does not mutate its input", float(np.abs(x).max()) == 5000.0)
check("robust_clamp leaves in-range values alone",
      np.allclose(y[np.abs(x) <= 20.0], x[np.abs(x) <= 20.0]))
check("robust_clamp keeps shape/dtype", y.shape == x.shape and y.dtype == np.float32)

# the whole point of the helper: byte-identical to what the trainer dataloader
# already applies via SubjectTransform (identity whitener/center/scale, clamp=20)
tf = SubjectTransform(
    subject_id=0,
    whitener=None,
    center=np.zeros((N_CHANNELS, 1), dtype=np.float32),
    scale=np.ones((N_CHANNELS, 1), dtype=np.float32),
    clamp=20.0,
)
check("robust_clamp == SubjectTransform.apply(clamp=20)",
      np.array_equal(robust_clamp(x, 20.0), tf.apply(x.copy())))

# ... and to EpochDataset's np.clip(x, -clip, clip)
ref = x.copy()
np.clip(ref, -20.0, 20.0, out=ref)
check("robust_clamp == EpochDataset clip path", np.array_equal(robust_clamp(x, 20.0), ref))

check("robust_clamp(sigma=inf) is a no-op", np.array_equal(robust_clamp(x, float("inf")), x))
check("robust_clamp(sigma=None) is a no-op", np.array_equal(robust_clamp(x, None), x))

ro = np.zeros((4, 4), dtype=np.float32)
ro.flags.writeable = False
try:
    robust_clamp(ro, 20.0, inplace=True)
    ok = False
except ValueError:
    ok = True
check("robust_clamp(inplace=True) refuses a read-only array", ok)

inp = x.copy()
out = robust_clamp(inp, 20.0, inplace=True)
check("robust_clamp(inplace=True) writes through", out is inp and float(np.abs(inp).max()) <= 20.0)

check("clamped_fraction", abs(clamped_fraction(x, 20.0) - float((np.abs(x) > 20.0).mean())) < 1e-12,
      clamped_fraction(x, 20.0))
check("clamped_fraction of empty", clamped_fraction(np.zeros((0, 4, 4)), 20.0) == 0.0)


# ------------------------------------------------------------------ detection
names = [f"ch{i:02d}" for i in range(N_CHANNELS)]
clean = rng.standard_normal((200, N_CHANNELS, 60)).astype(np.float32)
st = channel_stats(clean, ch_names=names)
check("clean data -> no outliers", not any(c["outlier"] for c in st),
      [c["channel"] for c in st if c["outlier"]])
check("clean data -> no source channels", not any(c["source"] for c in st))
check("channel_stats returns one entry per channel", len(st) == N_CHANNELS)
check("channel_stats preserves names", [c["channel"] for c in st] == names)

# a blown electrode: rare, enormous spikes on one channel only
blown = clean.copy()
spike_rows = rng.choice(200, size=6, replace=False)
blown[spike_rows, 13, 30] = 3.0e4
sb = channel_stats(blown, ch_names=names)
ch13 = sb[13]
check("blown channel flagged as source", ch13["source"] and ch13["outlier"], ch13["flags"])
check("blown channel flagged by sd_ratio", "sd_ratio" in ch13["flags"], ch13["sd_ratio"])
check("only the blown channel is a source",
      [c["channel"] for c in sb if c["source"]] == ["ch13"])

# THE PIN: the IQR-based robust sd cannot see the spike; the plain sd can.
check("IQR robust sd is blind to the spike (this is why preprocess missed sub-17)",
      abs(ch13["robust_sd_ratio"] - 1.0) < 0.25 and ch13["sd_ratio"] > 50.0,
      f"robust_sd_ratio={ch13['robust_sd_ratio']:.3f} sd_ratio={ch13['sd_ratio']:.1f}")

# a dead electrode
dead = clean.copy()
dead[:, 40, :] = 0.0
sd_ = channel_stats(dead, ch_names=names)
check("dead channel flagged flat", sd_[40]["flat"] and sd_[40]["outlier"], sd_[40]["flags"])
check("dead channel frac_zero == 1", sd_[40]["frac_zero"] == 1.0)
check("only the dead channel is flat", [c["channel"] for c in sd_ if c["flat"]] == ["ch40"])

# thresholds are honoured, not hard-coded
loose = QCThresholds(sd_ratio_bad=1e9, max_abs_bad=1e9, frac_gt_clamp_bad=1.1, flat_ratio=0.0)
check("loosening the thresholds clears the flags",
      not any(c["outlier"] for c in channel_stats(blown, ch_names=names, thresholds=loose)))


# ------------------------------------------------------------------ the gate
def _fake_report(sd_ratio, sid=17):
    ch = [
        {"index": i, "channel": names[i], "sd": 1.0, "sd_ratio": (sd_ratio if i == 13 else 1.0),
         "robust_sd": 1.0, "robust_sd_ratio": 1.0, "max_abs": 5.0, "p9999_abs": 4.0,
         "frac_gt_clamp": 0.0, "frac_zero": 0.0, "n_nonfinite": 0, "flat": False,
         "outlier": (i == 13 and sd_ratio > 5.0), "source": (i == 13 and sd_ratio > 5.0),
         "flags": (["sd_ratio"] if (i == 13 and sd_ratio > 5.0) else [])}
        for i in range(N_CHANNELS)
    ]
    bad = [c for c in ch if c["outlier"]]
    return {
        "n_subjects": 1,
        "per_subject": {str(sid): {
            "subject_id": sid, "channels": ch, "severity": sd_ratio,
            "n_bad_channels": len(bad), "n_source_channels": len(bad),
        }},
    }


ok, msgs = verify_qc(_fake_report(82.7), thresholds=DEFAULT_THRESHOLDS, mode="source")
check("verify_qc fails on a bad subject", ok is False and any("FAIL" in m for m in msgs), msgs[:1])
ok2, msgs2 = verify_qc(_fake_report(82.7), allow_subjects=[17], mode="source")
check("verify_qc honours an explicit waiver",
      ok2 is True and any("WAIVED" in m for m in msgs2), msgs2[:1])
ok3, _ = verify_qc(_fake_report(1.2), mode="source")
check("verify_qc passes a clean cohort", ok3 is True)
try:
    verify_qc(_fake_report(82.7), mode="cohort")
    ok4 = False
except ValueError:
    ok4 = True
check("verify_qc(mode='cohort') refuses a report without cohort stats", ok4)
try:
    verify_qc(_fake_report(1.2), mode="nonsense")
    ok5 = False
except ValueError:
    ok5 = True
check("verify_qc rejects an unknown gate mode", ok5)


# ------------------------------------------------------------------ real data
p17 = epoch_path(17, "w0600", EPOCH_DIR)
if p17.exists():
    b17 = subject_qc(17, window="w0600")
    po4 = next(c for c in b17["channels"] if c["channel"] == "PO4")
    check("sub-17 PO4 is the source channel", b17["source_channels"] == ["PO4"],
          b17["source_channels"])
    check("sub-17 PO4 max|x| ~ 33298", abs(po4["max_abs"] - 33297.94) < 1.0, po4["max_abs"])
    check("sub-17 PO4 plain sd ratio > 50", po4["sd_ratio"] > 50.0, round(po4["sd_ratio"], 2))
    check("sub-17 PO4 IQR robust sd ratio is unremarkable",
          abs(po4["robust_sd_ratio"] - 1.0) < 0.2, round(po4["robust_sd_ratio"], 3))
    check("sub-17 severity beats every other subject's by >5x",
          b17["severity"] > 50.0, round(b17["severity"], 2))
    b1 = subject_qc(1, window="w0600")
    check("sub-01 is clean", b1["n_bad_channels"] == 0 and b1["n_source_channels"] == 0,
          b1["bad_channels"])
    check("sub-01 max|x| ~ 40.9", abs(b1["max_abs"] - 40.9) < 0.5, b1["max_abs"])
    rep = run_qc([1, 17, 54], progress=False)
    check("run_qc roll-up names sub-17 and sub-54",
          rep["cohort"]["subjects_with_source_channel"] == [17, 54],
          rep["cohort"]["subjects_with_source_channel"])
    okr, _ = verify_qc(rep, mode="source")
    check("verify_qc gates the real cohort", okr is False)
    check("run_qc computes cohort-relative excess",
          rep["per_subject"]["17"]["cohort_worst_channel"] == "PO4"
          and rep["per_subject"]["17"]["cohort_severity"] > 5.0,
          round(rep["per_subject"]["17"]["cohort_severity"], 2))
    okc, _ = verify_qc(rep, mode="cohort")
    check("cohort gate fails on the real cohort", okc is False)
else:
    print("SKIP  real-data checks (frozen memmaps not mounted at %s)" % EPOCH_DIR)


print()
print("=" * 60)
print(f"{len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else " -- all green"))


def test_no_failures():
    """pytest entry point; the checks above run at import."""
    assert not fails, f"{len(fails)} check(s) failed: {fails}"


if __name__ == "__main__":
    sys.exit(1 if fails else 0)
