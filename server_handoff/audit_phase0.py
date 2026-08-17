#!/usr/bin/env python
"""TACTUS Phase 0 audits A-D on ds005662 metadata (no EEG data needed).

Usage: python audit_phase0.py --bids ds005662 --out phase0_out

Outputs: audit_report.md, seq_orient_crosstab.csv, attr_association_matrix.csv,
per_subject_summary.csv, material_touchtype_contingency.csv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mutual_info_score

ORIENT_TOKENS = ["horvertflip", "horflip", "vertflip"]  # longest/most specific first


def parse_stim(stim: str):
    s = str(stim).replace("\\", "/").lower()
    orient = "original"
    for tok in ORIENT_TOKENS:
        if tok in s:
            orient = tok
            break
    base = s.split("/")[-1]
    m = re.match(r"^(\d+)\.", base)
    vid = int(m.group(1)) if m else -1
    return orient, vid


def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    if ct.size == 0 or min(ct.shape) < 2:
        return np.nan
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.values.sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


def corr_ratio(cats, vals):
    # eta: sqrt(SS_between / SS_total) for categorical -> continuous
    df = pd.DataFrame({"c": cats, "v": pd.to_numeric(vals, errors="coerce")}).dropna()
    if df.empty or df["c"].nunique() < 2:
        return np.nan
    grand = df["v"].mean()
    ss_tot = ((df["v"] - grand) ** 2).sum()
    if ss_tot == 0:
        return np.nan
    ss_between = sum(len(g) * (g["v"].mean() - grand) ** 2 for _, g in df.groupby("c"))
    return float(np.sqrt(ss_between / ss_tot))


def norm_mi(x, y):
    mi = mutual_info_score(x, y)
    hx = stats.entropy(pd.Series(x).value_counts(normalize=True))
    hy = stats.entropy(pd.Series(y).value_counts(normalize=True))
    denom = min(hx, hy)
    return float(mi / denom) if denom > 0 else np.nan


def audit_events(bids: Path, out: Path, report: list):
    files = sorted(bids.glob("sub-*/eeg/*_task-video_events.tsv"))
    report.append(f"\n## Audit A/B -- events.tsv ({len(files)} subjects)\n")
    if not files:
        report.append("**No events.tsv found -- check the download.**")
        return

    per_subj = []
    crosstab_rows = []
    seq_orient_maps = {}  # subj -> {seq: dominant orientation} for pure sequences

    for f in files:
        subj = f.name.split("_")[0]
        ev = pd.read_csv(f, sep="\t", na_values=["n/a", "NA", ""])
        for col in ("istarget", "sequencenumber"):
            if col not in ev.columns:
                report.append(f"- **{subj}: missing column {col}; actual columns {list(ev.columns)} -- needs manual adaptation.**")
                return
        parsed = ev["stim"].map(parse_stim)
        ev["orient"] = [p[0] for p in parsed]
        ev["vid"] = [p[1] for p in parsed]

        nt = ev[ev["istarget"] == 0].copy()

        # --- A: sequence x orientation composition ---
        seq_stats, pure_map = [], {}
        for seq, g in nt.groupby("sequencenumber"):
            n_or = g["orient"].nunique()
            dom = g["orient"].value_counts(normalize=True).iloc[0]
            seq_stats.append((seq, len(g), n_or, dom, g["vid"].nunique()))
            if n_or == 1:
                pure_map[seq] = g["orient"].iloc[0]
            crosstab_rows.append(
                dict(subject=subj, sequence=seq, n_trials=len(g), n_orients=n_or,
                     dominant_frac=round(dom, 3), n_unique_videos=g["vid"].nunique(),
                     orient_counts=g["orient"].value_counts().to_dict()))
        seq_df = pd.DataFrame(seq_stats, columns=["seq", "n", "n_or", "dom", "nvid"])
        seq_orient_maps[subj] = pure_map

        # --- B: timing / contamination ---
        ev_sorted = ev.sort_values("onset").reset_index(drop=True)
        soa = ev_sorted.groupby("sequencenumber")["onset"].diff().dropna()
        post_target = int((ev_sorted["istarget"].shift(1) == 1).sum())
        # presentationnumber semantics: per (vid,orient) should count repeats 1..8
        pn_ok = np.nan
        if "presentationnumber" in nt.columns:
            counts = nt.groupby(["vid", "orient"])["presentationnumber"].nunique()
            pn_ok = float((counts == 8).mean())

        per_subj.append(dict(
            subject=subj, n_nontarget=len(nt), n_seq=nt["sequencenumber"].nunique(),
            frac_pure_seq=round((seq_df["n_or"] == 1).mean(), 3),
            mean_dominant_frac=round(seq_df["dom"].mean(), 3),
            videos_per_seq_min=int(seq_df["nvid"].min()), videos_per_seq_max=int(seq_df["nvid"].max()),
            cramers_v_orient_seq=round(cramers_v(nt["orient"], nt["sequencenumber"]), 3),
            nmi_orient_seq=round(norm_mi(nt["orient"], nt["sequencenumber"]), 3),
            nmi_vid_seq=round(norm_mi(nt["vid"], nt["sequencenumber"]), 3),
            soa_median=round(float(soa.median()), 4),
            soa_p2_5=round(float(soa.quantile(0.025)), 4),
            soa_p97_5=round(float(soa.quantile(0.975)), 4),
            n_post_target=post_target,
            frac_full8_presentations=pn_ok,
        ))

    ps = pd.DataFrame(per_subj)
    ps.to_csv(out / "per_subject_summary.csv", index=False)
    pd.DataFrame(crosstab_rows).to_csv(out / "seq_orient_crosstab.csv", index=False)

    frac_pure = ps["frac_pure_seq"].mean()
    if frac_pure > 0.9:
        verdict = "**BLOCKED: orientation is blocked by sequence.** Triggers the BLUEPRINT_v2 6.1-1 fallback: downgrade the orientation-decoding claim, treat sequence as a crossed random effect, and redesign the equivariance analysis to run within sequence."
    elif frac_pure < 0.1:
        verdict = "**INTERLEAVED: orientation is well mixed within sequences.** The block confound is cleared and the equivariance design stands as written."
    else:
        verdict = f"**MIXED (pure-orientation sequence fraction {frac_pure:.2f}):** partly blocked; model per sequence."
    report.append(f"### Audit A verdict\n{verdict}\n")
    report.append(f"- pure-orientation sequence fraction (mean over subjects): {frac_pure:.3f}")
    report.append(f"- unique videos per sequence, range: {ps['videos_per_seq_min'].min()}-{ps['videos_per_seq_max'].max()} (=90 would mean every sequence shows all videos once)")
    report.append(f"- orientation x sequence Cramer's V (mean over subjects): {ps['cramers_v_orient_seq'].mean():.3f}")

    # cross-subject consistency of pure-sequence orientation assignment
    if any(seq_orient_maps.values()):
        rows = [{"subject": s, **m} for s, m in seq_orient_maps.items() if m]
        wide = pd.DataFrame(rows).set_index("subject")
        n_distinct = wide.nunique(dropna=True)
        report.append(
            f"- cross-subject consistency of the pure-sequence orientation assignment: median distinct orientations per sequence position = {n_distinct.median():.0f} "
            " (=1 -> the mapping is fixed by design; >1 -> randomised across subjects)")

    report.append("\n### Audit B verdict")
    report.append(f"- SOA median {ps['soa_median'].median():.3f}s, 95% interval "
                  f"[{ps['soa_p2_5'].min():.3f}, {ps['soa_p97_5'].max():.3f}] (expected tight around 0.800)")
    report.append(f"- post-target trials (must be excluded; they carry the button-press motor potential): mean per subject {ps['n_post_target'].mean():.1f}, "
                  f"total {ps['n_post_target'].sum()}")
    report.append(f"- fraction of conditions with a complete 1-8 presentationnumber (mean over subjects): {ps['frac_full8_presentations'].mean():.3f}")
    report.append(f"- normalised mutual information orientation<->sequence: {ps['nmi_orient_seq'].mean():.3f}; video<->sequence: {ps['nmi_vid_seq'].mean():.3f}"
                  " (the a priori magnitude of trial-index leakage; only values near 0 are safe)")


def audit_vtd(bids: Path, out: Path, report: list):
    report.append("\n## Audit C -- VTD.csv attribute cross-association (90 videos)\n")
    p = bids / "code" / "analysis" / "VTD.csv"
    if not p.exists():
        report.append(f"**{p} not found.**")
        return
    vtd = pd.read_csv(p)
    vtd.columns = [c.strip().lower().replace(" ", "_") for c in vtd.columns]
    cont = [c for c in ("arousal", "threat", "valence") if c in vtd.columns]
    cat = [c for c in ("pain", "touch_type", "toucher", "object", "material", "approaching") if c in vtd.columns]
    cols = cont + cat
    report.append(f"- continuous columns: {cont}; categorical columns: {cat}; n={len(vtd)}")

    A = pd.DataFrame(np.nan, index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i:]:
            if a == b:
                v = 1.0
            elif a in cont and b in cont:
                v = abs(stats.pearsonr(pd.to_numeric(vtd[a], errors="coerce"),
                                       pd.to_numeric(vtd[b], errors="coerce"))[0])
            elif a in cat and b in cat:
                v = cramers_v(vtd[a], vtd[b])
            else:
                c_, k_ = (a, b) if a in cat else (b, a)
                v = corr_ratio(vtd[c_], vtd[k_])
            A.loc[a, b] = A.loc[b, a] = round(v, 3)
    A.to_csv(out / "attr_association_matrix.csv")

    flagged = [(a, b, A.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:] if A.loc[a, b] > 0.5]
    report.append("\n### Attribute pairs associated above 0.5 (inseparable on these 90 stimuli -> the Q1b do-not-answer list)")
    if flagged:
        for a, b, v in sorted(flagged, key=lambda t: -t[2]):
            report.append(f"- {a} ↔ {b}: {v:.3f}")
    else:
        report.append("- none (the attribute structure is better than expected)")

    if "material" in cat and "touch_type" in cat:
        ct = pd.crosstab(vtd["material"], vtd["touch_type"])
        ct.to_csv(out / "material_touchtype_contingency.csv")
        report.append(f"\n- material x touch_type contingency table: {ct.shape[0]}x{ct.shape[1]}, "
                      f"{(ct.values == 0).sum()}/{ct.size} cells empty (evidence on whether attribute-level stratification is feasible)")


def audit_phenotype(bids: Path, report: list):
    report.append("\n## Audit D -- phenotype table health check\n")
    p = bids / "participants.tsv"
    if not p.exists():
        report.append("**participants.tsv not found.**")
        return
    pt = pd.read_csv(p, sep="\t", na_values=["n/a", "NA", ""])
    pt.columns = [c.strip() for c in pt.columns]
    scores = [c for c in ("VT_score", "EQ_score", "IRI_score") if c in pt.columns]
    report.append(f"- n={len(pt)}; columns: {list(pt.columns)}")
    for c in scores:
        v = pd.to_numeric(pt[c], errors="coerce")
        report.append(f"- {c}: mean={v.mean():.2f} sd={v.std():.2f} range=[{v.min():.0f},{v.max():.0f}] missing={v.isna().sum()}")
        if "age" in pt.columns:
            age = pd.to_numeric(pt["age"], errors="coerce")
            ok = v.notna() & age.notna()
            r, pval = stats.spearmanr(v[ok], age[ok])
            report.append(f"  - Spearman r with age = {r:.3f} (p={pval:.3f}) (evidence for the Q3 covariate set)")
        sex_col = next((c2 for c2 in ("sex", "gender") if c2 in pt.columns), None)
        if sex_col:
            gm = pt.groupby(sex_col)[c].apply(lambda s: pd.to_numeric(s, errors="coerce").mean()).round(2)
            report.append(f"  - group means by {sex_col}: {gm.to_dict()}")
    if "MTS" in pt.columns:
        report.append(f"- MTS counts: {pt['MTS'].value_counts(dropna=False).to_dict()} (1-2 true positives expected -> case description only, no inference)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("phase0_out"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = ["# TACTUS Phase 0 audit report (generated by audit_phase0.py)"]
    audit_events(args.bids, args.out, report)
    audit_vtd(args.bids, args.out, report)
    audit_phenotype(args.bids, report)

    (args.out / "audit_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\n[written] {args.out}/audit_report.md")


if __name__ == "__main__":
    main()
