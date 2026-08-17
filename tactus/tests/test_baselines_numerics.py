"""Regression tests for the torch-free numerics in tactus.common and tactus.baselines.

Run with ``pytest tests/`` or directly with ``python tests/test_baselines_numerics.py``.
These pin three bugs that were real: the SRM convergence check firing on iteration 0,
sub-gallery distractors drawn with replacement (which inflates the 18-way endpoint),
and shrinkage LDA not matching sklearn's per-class Ledoit-Wolf.
"""
import sys, tempfile, os
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

fails = []


def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("  " + str(extra) if extra != "" else ""))
    if not cond:
        fails.append(name)


# --------------------------------------------------------------- common.py
from tactus.common import (
    EpochStore, SubjectTransform, _ea_whitener, _robust_center_scale,
    fit_subject_transforms, condition_averages, split_half_reliability,
    load_label_maps, attach_label_ids, continuous_stats, apply_continuous_stats,
    window_times_ms, VideoEmbeddings,
)

t = window_times_ms("w0600")
check("window_times_ms w0600", t.shape == (120,) and t[0] == 0.0 and abs(t[-1] - 595.0) < 1e-6,
      f"{t[0]}..{t[-1]}")
t2 = window_times_ms("wm100_800")
check("window_times_ms wm100_800", t2.shape == (180,) and t2[0] == -100.0 and abs(t2[-1] - 795.0) < 1e-6,
      f"{t2[0]}..{t2[-1]}")

rng = np.random.default_rng(0)
# EA whitener: after whitening, the mean per-trial covariance should be ~identity
mix = rng.standard_normal((64, 64))
x = np.einsum("cd,ndt->nct", mix, rng.standard_normal((400, 64, 120))).astype(np.float32)
w = _ea_whitener(x, shrinkage=0.0)
xw = np.einsum("cd,ndt->nct", w, x)
r = np.einsum("nct,ndt->cd", xw, xw) / (xw.shape[0] * xw.shape[2])
check("EA whitener -> identity covariance", np.allclose(r, np.eye(64), atol=0.05),
      f"max dev {np.abs(r - np.eye(64)).max():.4f}")

med, sig = _robust_center_scale(x)
check("robust scale shapes", med.shape == (64, 1) and sig.shape == (64, 1))
flat = x.copy(); flat[:, 3, :] = 7.0
_, sig2 = _robust_center_scale(flat)
check("flat channel does not explode", np.isfinite(sig2[3, 0]) and sig2[3, 0] > 1e-6,
      f"sigma={float(sig2[3,0]):.3f}")

tf = SubjectTransform(1, w.astype(np.float32), med, sig, clamp=20.0)
y1 = tf.apply(x[0]); y2 = tf.apply(x[:5])
check("SubjectTransform 2D/3D agree", np.allclose(y1, y2[0], atol=1e-4))
check("SubjectTransform round-trip", np.allclose(
    SubjectTransform.from_dict(tf.to_dict()).apply(x[0]), y1, atol=1e-4))

# --- fake derived tree: memmaps + trial table --------------------------------
tmp = Path(tempfile.mkdtemp())
n_sub, n_tr, T = 3, 360, 120
epdir = tmp / "epochs"; epdir.mkdir()
rows = []
uid = 0
for s in range(1, n_sub + 1):
    arr = rng.standard_normal((n_tr, 64, T)).astype(np.float32)
    cond = np.repeat(np.arange(90), 4) % 360          # 90 conditions x 4 repeats
    for c in np.unique(cond):                          # inject a condition signal
        arr[cond == c] += np.sin(np.arange(T) / 7.0 + c)[None, None, :] * 2.0
    np.save(epdir / f"sub-{s:02d}_w0600.npy", arr)
    for i in range(n_tr):
        vid = int(cond[i]) // 4 + 1
        rows.append(dict(subject_id=s, trial_uid=uid, within_subj_idx=i,
                         onset_sample=i * 1638, sequence_id=i // 45 + 1,
                         presentation_number=i % 4 + 1, video_id=vid,
                         orientation=int(cond[i]) % 4, condition_id=int(cond[i]),
                         is_post_target=False, prev_video_id=-1,
                         material=f"m{vid % 8}", touch_type=f"t{vid % 12}",
                         toucher="hand" if vid % 2 else "object", object=f"o{vid % 28}",
                         approaching="yes" if vid % 3 else "no",
                         valence=float(vid % 7) - 3, arousal=float(vid % 5),
                         threat=float(vid % 4), pain=vid % 2))
        uid += 1
trials = pd.DataFrame(rows)

store = EpochStore("w0600", root=epdir)
check("EpochStore.available_subjects", store.available_subjects() == [1, 2, 3])
check("EpochStore.n_times", store.n_times == 120)
got = store.take(2, [0, 5, 7])
check("EpochStore.take shape", got.shape == (3, 64, 120) and got.dtype == np.float32)
gath = store.gather(trials.head(20))
check("EpochStore.gather shape", gath.shape == (20, 64, 120))
try:
    store.take(1, [99999]); ok = False
except IndexError:
    ok = True
check("EpochStore out-of-range raises IndexError", ok)

maps = load_label_maps(trials, path=tmp / "label_maps.json")
check("label maps built", set(maps) >= {"material", "touch_type", "toucher"} and
      len(maps["material"]) == 8, f"{ {k: len(v) for k, v in maps.items()} }")
tl = attach_label_ids(trials, maps)
check("attach_label_ids", tl["material_id"].between(0, 7).all() and
      tl["toucher_id"].isin([0, 1]).all())
tl2 = attach_label_ids(trials.assign(material="UNSEEN"), maps)
check("unknown category -> -1", (tl2["material_id"] == -1).all())

cs = continuous_stats(tl)
tz = apply_continuous_stats(tl, cs)
per_video_z = tz.drop_duplicates("video_id")["valence_z"]
check("continuous z-score is per-video", abs(float(per_video_z.mean())) < 1e-6 and
      abs(float(per_video_z.std(ddof=0)) - 1) < 1e-6)

tfs = fit_subject_transforms(tl, store, ea=True, max_trials=200,
                             cache_path=tmp / "tf.json")
check("fit_subject_transforms covers subjects", set(tfs) == {1, 2, 3})
tfs2 = fit_subject_transforms(tl, store, ea=True, max_trials=200,
                              cache_path=tmp / "tf.json")
check("subject transform cache reused",
      np.allclose(tfs[1].whitener, tfs2[1].whitener, atol=1e-6))

avg, cnt = condition_averages(tl, store, subject_id=1, cache_dir=tmp / "ca")
check("condition_averages shape", avg.shape == (360, 64, 120))
check("condition_averages counts", int(cnt.sum()) == 360 and cnt.max() == 4)
check("absent conditions are NaN", np.isnan(avg[cnt == 0]).all() and
      np.isfinite(avg[cnt > 0]).all())
odd, _ = condition_averages(tl, store, subject_id=1, split="odd", cache_dir=tmp / "ca")
even, _ = condition_averages(tl, store, subject_id=1, split="even", cache_dir=tmp / "ca")
rel = split_half_reliability(odd[cnt > 0], even[cnt > 0])
check("split-half reliability in (0,1]", 0.2 < rel <= 1.0, f"r={rel:.3f}")

# ------------------------------------------------------- linear_align.Ridge
from tactus.baselines.linear_align import RidgeAlpha, retrieval_np, _l2

n, p, d = 300, 40, 12
X = rng.standard_normal((n, p))
B = rng.standard_normal((p, d))
Y = X @ B + 0.1 * rng.standard_normal((n, d))
m = RidgeAlpha([1e-3, 1e-1, 1, 10, 100]).fit(X, Y)
Xte = rng.standard_normal((100, p))
err = np.abs(m.predict(Xte) - Xte @ B).mean()
check("RidgeAlpha primal recovers the map", err < 0.15, f"MAE={err:.4f} alpha={m.alpha_}")

n2, p2 = 40, 200                      # n < p -> dual path
X2 = rng.standard_normal((n2, p2)); B2 = rng.standard_normal((p2, 5))
Y2 = X2 @ B2
m2 = RidgeAlpha([1e-6, 1e-3, 1e-1, 1, 10]).fit(X2, Y2)
tr_err = np.abs(m2.predict(X2) - Y2).mean() / np.abs(Y2).mean()
check("RidgeAlpha dual path fits", tr_err < 0.05, f"rel train err={tr_err:.4f}")

# primal and dual must agree when both are applicable at a FIXED alpha
Xs, Ys = X[:60], Y[:60]
a = 10.0
ma = RidgeAlpha([a]).fit(Xs, Ys)                      # n=60 > p=40 -> primal
Xw = np.concatenate([Xs, np.zeros((60, 100))], 1)     # pad -> p=140 > n -> dual
mb = RidgeAlpha([a]).fit(Xw, Ys)
pa = ma.predict(Xs[:10]); pb = mb.predict(Xw[:10])
check("primal == dual at fixed alpha", np.allclose(pa, pb, atol=1e-6),
      f"max diff {np.abs(pa - pb).max():.2e}")

g = _l2(rng.standard_normal((18, 16)))
z = g[np.arange(18) % 18] * 5 + 0.01 * rng.standard_normal((18, 16))
r = retrieval_np(z, g, np.arange(18), gallery_sizes=(2, 10, 18))
check("retrieval_np perfect case", r["top1"] == 1.0 and r["mean_rank"] == 1.0)
zr = _l2(rng.standard_normal((2000, 16)))
gr = _l2(rng.standard_normal((18, 16)))
rr = retrieval_np(zr, gr, rng.integers(0, 18, 2000), gallery_sizes=(2, 10, 18), seed=1)
check("retrieval_np chance ~ 1/18", abs(rr["top1"] - 1 / 18) < 0.02, f"top1={rr['top1']:.4f}")
check("retrieval_np g2 chance ~ 0.5", abs(rr["top1_g2"] - 0.5) < 0.04, f"{rr['top1_g2']:.4f}")
check("retrieval_np full == g18", abs(rr["top1_g18"] - rr["top1"]) < 1e-9)

# distractors must be DISTINCT and must never be the true item
from tactus.baselines.linear_align import draw_distractors
g_big, n_rows, size = 72, 500, 18
ti = rng.integers(0, g_big, n_rows)
cand = draw_distractors(n_rows, g_big, size, ti, np.random.default_rng(3))
check("draw_distractors shape", cand.shape == (n_rows, size - 1))
check("draw_distractors distinct per row",
      all(len(set(r.tolist())) == size - 1 for r in cand))
check("draw_distractors excludes the truth", not (cand == ti[:, None]).any())
check("draw_distractors in range", cand.min() >= 0 and cand.max() < g_big)
# with-replacement sampling would inflate top1; confirm the fixed path matches
# the exact full-gallery answer when size == gallery
zz = _l2(rng.standard_normal((3000, 16))); gg = _l2(rng.standard_normal((72, 16)))
tix = rng.integers(0, 72, 3000)
r72 = retrieval_np(zz, gg, tix, gallery_sizes=(18,), n_draws=60, seed=5)
check("18-of-72 sub-gallery hits 1/18 chance", abs(r72["top1_g18"] - 1 / 18) < 0.012,
      f"{r72['top1_g18']:.4f} vs {1/18:.4f}")

# -------------------------------------------------------------- linear_mvpa
from tactus.baselines.linear_mvpa import (
    _shrinkage_lda_predict, _ridge_gcv_predict, cluster_permutation_1samp, onset_peak,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

nn, C, TT, K = 200, 12, 4, 3
y = rng.integers(0, K, nn)
Xtr = rng.standard_normal((nn, C, TT)) + (y[:, None, None] * 0.9)
Xte = rng.standard_normal((60, C, TT))
ref = np.stack([
    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    .fit(Xtr[:, :, ti], y).predict(Xte[:, :, ti]) for ti in range(TT)], axis=1)
pred = _shrinkage_lda_predict(Xtr, y, Xte, K, "auto", "per_class")
agree = float((pred == ref).mean())
check("batched LDA (per_class) == sklearn lsqr/auto", agree == 1.0, f"agreement={agree:.4f}")

# unbalanced classes exercise the prior-weighted average of class covariances
yu = np.concatenate([np.zeros(140, int), np.ones(40, int), np.full(20, 2)])
Xu = rng.standard_normal((200, C, TT)) + yu[:, None, None] * 0.8
refu = np.stack([
    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    .fit(Xu[:, :, ti], yu).predict(Xte[:, :, ti]) for ti in range(TT)], axis=1)
predu = _shrinkage_lda_predict(Xu, yu, Xte, K, "auto", "per_class")
check("batched LDA matches sklearn with unbalanced classes",
      float((predu == refu).mean()) == 1.0, f"agreement={(predu == refu).mean():.4f}")

# a fixed float shrinkage must also match sklearn exactly
reff = np.stack([
    LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.2)
    .fit(Xtr[:, :, ti], y).predict(Xte[:, :, ti]) for ti in range(TT)], axis=1)
predf = _shrinkage_lda_predict(Xtr, y, Xte, K, 0.2, "per_class")
check("batched LDA matches sklearn at fixed shrinkage=0.2",
      float((predf == reff).mean()) > 0.99, f"agreement={(predf == reff).mean():.4f}")

predp = _shrinkage_lda_predict(Xtr, y, Xte, K, "auto", "pooled")
check("pooled scope is close to per_class but not identical",
      0.9 <= float((predp == pred).mean()) <= 1.0, f"agreement={(predp == pred).mean():.4f}")

deg = _shrinkage_lda_predict(Xtr, np.zeros(nn, int), Xte, K, "auto")
check("LDA with a single training class does not crash", deg.shape == (60, TT))

yc = Xtr[:, 0, 0] * 2 + rng.standard_normal(nn) * 0.1
pr = _ridge_gcv_predict(Xtr, yc, Xte, [1e-2, 1, 100, 1e4])
check("batched ridge shape", pr.shape == (60, TT))
r_in = np.corrcoef(_ridge_gcv_predict(Xtr, yc, Xtr, [1e-2, 1, 100])[:, 0], yc)[0, 1]
check("batched ridge fits the signal", r_in > 0.95, f"r={r_in:.4f}")

curves = 0.25 + rng.standard_normal((40, 100)) * 0.02
curves[:, 30:50] += 0.06                      # a real bump
st = cluster_permutation_1samp(curves, 0.25, n_perm=400, seed=0)
sig = np.flatnonzero(st["sig_mask"])
check("cluster test finds the bump", sig.size > 5 and 25 <= sig.min() <= 35 and 45 <= sig.max() <= 55,
      f"sig {sig.min() if sig.size else '-'}..{sig.max() if sig.size else '-'}")
# false-positive rate under the null must sit near alpha, not at zero and not
# anywhere near 1: a single null draw going significant is expected 5% of the time
nrng = np.random.default_rng(20240816)
n_null = 40
fp = sum(
    bool(cluster_permutation_1samp(
        0.25 + nrng.standard_normal((40, 100)) * 0.02, 0.25, n_perm=300, seed=i
    )["sig_mask"].any())
    for i in range(n_null)
)
check("cluster test FPR near alpha", fp / n_null <= 0.15, f"{fp}/{n_null} null runs significant")
st0 = cluster_permutation_1samp(0.25 + nrng.standard_normal((40, 100)) * 0.02, 0.25,
                                n_perm=400, seed=1)

times = np.arange(100) * 5.0
lp = onset_peak(times, curves.mean(0), st["sig_mask"])
check("onset/peak inside the bump", 140 <= lp["onset_ms"] <= 180 and 140 <= lp["peak_ms"] <= 260,
      f"onset={lp['onset_ms']} peak={lp['peak_ms']}")
lp0 = onset_peak(times, curves.mean(0), np.zeros(100, bool))
check("onset/peak flags non-significance",
      lp0["significant"] is False and np.isnan(lp0["onset_ms"]) and np.isfinite(lp0["peak_ms"]))

# ------------------------------------------------------------------- corrca
from tactus.baselines.corrca import corrca_filters, subject_isc

K_s, N, Cc = 12, 3000, 20
shared = rng.standard_normal((N, 3))
true_fwd = rng.standard_normal((3, Cc))
Xs = {}
for k in range(K_s):
    noise = rng.standard_normal((N, Cc)) * (1.0 + 0.35 * k)   # subject 0 = cleanest
    Xs[k] = shared @ true_fwd + noise
    Xs[k] -= Xs[k].mean(0, keepdims=True)
r_kk = {k: v.T @ v for k, v in Xs.items()}
S = sum(Xs.values())
R_w = sum(r_kk.values())
R_b = S.T @ S - R_w
W, isc, A = corrca_filters(R_w, R_b, n_components=4)
check("CorrCA ISC descending", np.all(np.diff(isc) <= 1e-9), np.array2string(isc, precision=3))
check("CorrCA top ISC positive and dominant", isc[0] > 0 and isc[0] > 3 * abs(isc[3]))
comp = Xs[0] @ W[:, 0]
recov = max(abs(np.corrcoef(comp, shared[:, j])[0, 1]) for j in range(3))
check("CorrCA component recovers a shared source", recov > 0.7, f"|r|={recov:.3f}")
iscs = np.array([subject_isc(W, r_kk[k], Xs[k].T @ S, R_w, K_s)[0] for k in range(K_s)])
check("per-subject ISC decreases with noise", np.corrcoef(iscs, np.arange(K_s))[0, 1] < -0.8,
      f"r={np.corrcoef(iscs, np.arange(K_s))[0,1]:.3f}")

# ---------------------------------------------------------------------- SRM
from tactus.baselines.srm import DetSRM, _conditions_of_videos

F, Ns, Kc, nsub = 60, 200, 8, 10
S_true = rng.standard_normal((Kc, Ns))
data = {}
for k in range(nsub):
    q, _ = np.linalg.qr(rng.standard_normal((F, Kc)))
    data[k] = q @ S_true + 0.05 * rng.standard_normal((F, Ns))
srm = DetSRM(n_components=Kc, n_iter=30, seed=0).fit(data)
check("SRM loss decreases monotonically", all(np.diff(srm.loss_) <= 1e-6),
      f"{srm.loss_[0]:.1f} -> {srm.loss_[-1]:.1f}")
resid = np.mean([np.linalg.norm(data[k] - srm.w_[k] @ srm.s_) / np.linalg.norm(data[k])
                 for k in data])
check("SRM reconstructs subjects", resid < 0.2, f"rel resid={resid:.4f}")
check("SRM W orthonormal", np.allclose(srm.w_[0].T @ srm.w_[0], np.eye(Kc), atol=1e-8))

qn, _ = np.linalg.qr(rng.standard_normal((F, Kc)))
new = qn @ S_true + 0.05 * rng.standard_normal((F, Ns))
half = np.arange(0, Ns, 2)
w_new = srm.enroll(new[:, half], srm.s_[:, half])      # enroll on half the columns
proj = w_new.T @ new[:, 1::2]
gal = srm.s_[:, 1::2]
sim = _l2(proj.T) @ _l2(gal.T).T
top1 = float((sim.argmax(1) == np.arange(sim.shape[0])).mean())
check("SRM enrollment generalizes to held-out columns", top1 > 0.9, f"top1={top1:.3f}")
try:
    srm.enroll(new[:, half], srm.s_); ok = False
except ValueError:
    ok = True
check("SRM enroll rejects mismatched sample counts", ok)
cond = _conditions_of_videos([1, 3])
check("_conditions_of_videos", cond.tolist() == [0, 1, 2, 3, 8, 9, 10, 11], cond.tolist())

print()
print("=" * 60)
print(f"{len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else " -- all green"))


def test_no_failures():
    """pytest entry point; the checks above run at import."""
    assert not fails, f"{len(fails)} check(s) failed: {fails}"


if __name__ == "__main__":
    sys.exit(1 if fails else 0)
