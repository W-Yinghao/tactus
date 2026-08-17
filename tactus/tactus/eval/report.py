"""Fold aggregation, results tables and the REPORT.md emitter.

The report is deliberately ordered so that the reader meets the *resolution of
the design* before meeting any ablation numbers, and meets the confound battery
and the named limitations before the discussion.  Sections that could not be
computed appear as explicit "NOT RUN" entries rather than being omitted: a
partially-run battery must never look like a clean one.

Aggregation rule everywhere in this module: average across folds **within a
unit of inference** first, then bootstrap **across units**.  Bootstrapping
trials or folds would treat clustered observations as independent and is the
same error as the trial-level permutation null.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .stats import bootstrap_ci, flag_below_mdd, mdd_table

try:  # the shared helper lives in tactus.common; keep this import soft
    from ..common import results_dir as _results_dir  # type: ignore
except Exception:  # pragma: no cover
    def _results_dir(root: Any = None) -> Path:  # type: ignore[misc]
        return Path(root) if root is not None else Path("results")

__all__ = [
    "aggregate_folds",
    "compare_arms",
    "df_to_markdown",
    "write_table",
    "ReportInputs",
    "emit_report",
]

DEFAULT_GROUP_COLS: Tuple[str, ...] = ("direction", "trial_type", "gallery", "metric")


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def aggregate_folds(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    unit_col: str = "subject_id",
    fold_col: Optional[str] = "video_fold_id",
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
    chance_col: Optional[str] = "chance",
    n_boot: int = 10_000,
    ci: float = 95.0,
    seed: int = 0,
    exclude_pooled: bool = True,
) -> pd.DataFrame:
    """Fold-average within unit, then bootstrap across units.

    Parameters
    ----------
    df : long results frame (e.g. from
        :func:`tactus.eval.retrieval.evaluate_retrieval`, concatenated over
        folds with a ``video_fold_id`` column added).
    unit_col : the unit of inference -- ``subject_id`` for fixed-stimulus
        claims, ``video_id`` for stimulus-generalising claims.
    exclude_pooled : drop the ``"pooled"`` pseudo-subject rows, which are a
        diagnostic and must not enter the across-subject bootstrap.

    Returns
    -------
    One row per ``group_cols`` combination with ``estimate``, ``ci_lo``,
    ``ci_hi``, ``se``, ``n_units``, ``chance`` and ``above_chance``.
    """
    work = df.copy()
    missing = [c for c in list(group_cols) + [value_col, unit_col] if c not in work.columns]
    if missing:
        raise KeyError(f"aggregate_folds is missing columns {missing}")
    if exclude_pooled:
        work = work[work[unit_col].astype(str) != "pooled"]
    if work.empty:
        return pd.DataFrame(
            columns=list(group_cols) + ["estimate", "ci_lo", "ci_hi", "se",
                                        "n_units", "chance", "above_chance"]
        )

    keys = list(group_cols) + [unit_col]
    per_unit = work.groupby(keys, dropna=False, observed=True)[value_col].mean().reset_index()
    if fold_col is not None and fold_col in work.columns:
        n_folds = work.groupby(list(group_cols), dropna=False, observed=True)[fold_col].nunique()
    else:
        n_folds = None

    rows: List[Dict[str, Any]] = []
    for key, sub in per_unit.groupby(list(group_cols), dropna=False, observed=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        boot = bootstrap_ci(
            sub[value_col].to_numpy(dtype=float), n_boot=n_boot, ci=ci, seed=seed
        )
        chance = np.nan
        if chance_col and chance_col in work.columns:
            mask = np.ones(len(work), dtype=bool)
            for col, val in zip(group_cols, key_tuple):
                mask &= (work[col] == val).to_numpy()
            chance = float(np.nanmean(work.loc[mask, chance_col].to_numpy(dtype=float)))
        row = dict(zip(group_cols, key_tuple))
        row.update(
            {
                "estimate": boot["estimate"],
                "ci_lo": boot["lo"],
                "ci_hi": boot["hi"],
                "se": boot["se"],
                "n_units": boot["n"],
                "unit": unit_col,
                "chance": chance,
                "above_chance": bool(np.isfinite(boot["lo"]) and np.isfinite(chance)
                                     and boot["lo"] > chance),
            }
        )
        if n_folds is not None:
            try:
                row["n_folds"] = int(n_folds.loc[key])
            except (KeyError, TypeError):
                row["n_folds"] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ("direction", "trial_type", "metric", "gallery") if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out


def compare_arms(
    df: pd.DataFrame,
    *,
    arm_col: str = "probe",
    reference: str,
    value_col: str = "value",
    unit_col: str = "subject_id",
    group_cols: Sequence[str] = DEFAULT_GROUP_COLS,
    mdd: Optional[float] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired per-unit differences of every arm against a reference arm.

    Adds the MDD verdict when ``mdd`` is supplied, so an ablation table cannot
    be read without its resolution.  Used for the confound-battery arms
    (full EEG vs ocular surrogate vs frontal-ablated) and for loss ablations.
    """
    from .stats import paired_wilcoxon

    work = df.copy()
    work = work[work[unit_col].astype(str) != "pooled"]
    keys = list(group_cols) + [unit_col, arm_col]
    per_unit = work.groupby(keys, dropna=False, observed=True)[value_col].mean().reset_index()

    rows: List[Dict[str, Any]] = []
    for key, sub in per_unit.groupby(list(group_cols), dropna=False, observed=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        piv = sub.pivot_table(index=unit_col, columns=arm_col, values=value_col)
        if reference not in piv.columns:
            continue
        for arm in piv.columns:
            if arm == reference:
                continue
            paired = piv[[arm, reference]].dropna()
            if paired.shape[0] < 3:
                continue
            res = paired_wilcoxon(
                paired[arm].to_numpy(), paired[reference].to_numpy(),
                labels=(str(arm), str(reference)), seed=seed,
            )
            row = dict(zip(group_cols, key_tuple))
            row.update(
                {
                    "arm": arm,
                    "reference": reference,
                    "mean_arm": float(paired[arm].mean()),
                    "mean_reference": float(paired[reference].mean()),
                    "mean_diff": res["mean_diff"],
                    "median_diff": res["median_diff"],
                    "ci_lo": res["ci_lo"],
                    "ci_hi": res["ci_hi"],
                    "p_value": res["p_value"],
                    "effect_size_rb": res["effect_size_rb"],
                    "n_units": res["n_pairs"],
                }
            )
            if mdd is not None:
                row["mdd_verdict"] = flag_below_mdd(res["mean_diff"], mdd)["verdict"]
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def df_to_markdown(df: pd.DataFrame, *, floatfmt: str = "{:.4g}", max_rows: int = 200) -> str:
    """Markdown table with a dependency-free fallback when tabulate is absent."""
    if df is None or len(df) == 0:
        return "_(empty)_"
    shown = df.head(max_rows).copy()
    truncated = len(df) > max_rows
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(
                lambda v: "" if pd.isna(v) else floatfmt.format(v)
            )
        else:
            shown[col] = shown[col].astype(str)
    try:
        text = shown.to_markdown(index=False)
    except Exception:
        header = "| " + " | ".join(map(str, shown.columns)) + " |"
        sep = "| " + " | ".join("---" for _ in shown.columns) + " |"
        body = [
            "| " + " | ".join(str(v) for v in row) + " |"
            for row in shown.itertuples(index=False, name=None)
        ]
        text = "\n".join([header, sep] + body)
    if truncated:
        text += f"\n\n_(showing {max_rows} of {len(df)} rows; full table in the parquet)_"
    return text


def write_table(
    df: pd.DataFrame, out_dir: str | Path, name: str, *, also_markdown: bool = True
) -> Dict[str, str]:
    """Write a table as parquet (+ markdown) and return the written paths."""
    out = Path(out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    pq = out / "tables" / f"{name}.parquet"
    try:
        df.to_parquet(pq, index=False)
        paths["parquet"] = str(pq)
    except Exception as exc:  # pyarrow missing or object columns
        csv = out / "tables" / f"{name}.csv"
        df.to_csv(csv, index=False)
        paths["csv"] = str(csv)
        paths["parquet_error"] = str(exc)
    if also_markdown:
        md = out / "tables" / f"{name}.md"
        md.write_text(df_to_markdown(df), encoding="utf-8")
        paths["markdown"] = str(md)
    return paths


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# report assembly
# --------------------------------------------------------------------------- #
@dataclass
class ReportInputs:
    """Everything the report can show.  Every field is optional."""

    run_name: str = "tactus"
    regime: str = "double_disjoint"
    config: Optional[Mapping[str, Any]] = None

    # retrieval
    retrieval_long: Optional[pd.DataFrame] = None
    primary_endpoint: Tuple[str, str, str, str] = ("eeg2vid", "pseudo4", "nway18", "top1")

    # ceilings
    noise_ceiling_long: Optional[pd.DataFrame] = None
    subject_reliability: Optional[pd.DataFrame] = None

    # permutation
    permutation: Optional[Mapping[str, Any]] = None  # {"video_level":..., "trial_level":..., "narrowing":...}

    # rsa
    rsa_curves: Optional[pd.DataFrame] = None
    rsa_clusters: Optional[pd.DataFrame] = None
    rsa_noise_ceiling: Optional[Tuple[float, float]] = None

    # confounds
    confounds: Optional[Mapping[str, Any]] = None

    # statistics
    ablations: Optional[pd.DataFrame] = None
    fdr_table: Optional[pd.DataFrame] = None
    mdd: Optional[pd.DataFrame] = None
    mdd_kwargs: Mapping[str, Any] = field(default_factory=dict)

    # free-form
    notes: Sequence[str] = ()
    uncertifiable_claims: Sequence[str] = ()


def _section(title: str, body: str) -> str:
    return f"\n## {title}\n\n{body.rstrip()}\n"


def _not_run(what: str, why: str) -> str:
    return f"**NOT RUN** -- {what}. Reason: {why}\n"


def emit_report(
    inputs: ReportInputs,
    out_dir: str | Path | None = None,
    *,
    write_tables: bool = True,
) -> Path:
    """Assemble ``<out_dir>/REPORT.md`` and the accompanying tables.

    ``out_dir`` defaults to :func:`tactus.common.results_dir` (``<repo>/results``
    unless ``TACTUS_RESULTS_DIR`` overrides it).

    Returns the path to the written report.
    """
    out = _results_dir(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Dict[str, str]] = {}
    parts: List[str] = []

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts.append(
        f"# TACTUS results -- {inputs.run_name}\n\n"
        f"- generated: {now}\n"
        f"- git commit: `{_git_commit()}`\n"
        f"- python: {sys.version.split()[0]} on {platform.platform()}\n"
        f"- evaluation regime: `{inputs.regime}`\n"
    )
    if inputs.config:
        cfg_txt = json.dumps(dict(inputs.config), indent=2, default=str)
        parts.append(
            "\n<details><summary>run configuration</summary>\n\n"
            f"```json\n{cfg_txt}\n```\n\n</details>\n"
        )

    # ---------------- 1. design resolution, deliberately first -------------
    mdd_df = inputs.mdd if inputs.mdd is not None else mdd_table(**dict(inputs.mdd_kwargs))
    if write_tables:
        written["mdd"] = write_table(mdd_df, out, "design_resolution")
    def _mdd(unit: str, col: str = "mdd") -> float:
        if mdd_df is None or "unit" not in mdd_df.columns:
            return float("nan")
        sel = mdd_df.loc[mdd_df["unit"] == unit, col]
        return float(sel.iloc[0]) if len(sel) else float("nan")

    subject_mdd = _mdd("subject")
    subject_mdd_floor = _mdd("subject", "mdd_floor")
    video_mdd = _mdd("base video")
    video_mdd_floor = _mdd("base video", "mdd_floor")
    parts.append(
        _section(
            "1. Resolution of this design (read before any table below)",
            df_to_markdown(mdd_df)
            + "\n\n"
            + textwrap.dedent(
                f"""
                Differences smaller than the relevant minimal detectable difference are
                **not resolvable** by this design. They are reported as unresolved --
                never as a win, never as evidence of equivalence. Each target is quoted
                as a range: the measurement-error floor (no between-unit variance, which
                nothing can beat) and the conservative bound (unit SD taken as
                sqrt(p(1-p))). The truth lies between them and is pinned down once the
                folds supply the empirical between-unit SD.

                - fixed-stimulus claims (model ranking, ablations): MDD {subject_mdd_floor*100:.1f}-{subject_mdd*100:.1f} points, n=80 subjects
                - stimulus-generalising claims: MDD {video_mdd_floor*100:.1f}-{video_mdd*100:.1f} points, n=90 base videos

                Effect-size calibration: 18-way chance is 5.56%. A single fold of a single
                subject over 18 video units has a Clopper-Pearson 95% interval of roughly
                0-19% at chance, so no single cell can distinguish 15% from chance. Every
                claim below lives on the fold x subject aggregate.
                """
            ).strip(),
        )
    )

    # ---------------- 2. primary endpoint ----------------------------------
    if inputs.retrieval_long is not None and len(inputs.retrieval_long):
        agg = aggregate_folds(inputs.retrieval_long)
        if write_tables:
            written["retrieval"] = write_table(agg, out, "retrieval_aggregated")
            written["retrieval_long"] = write_table(
                inputs.retrieval_long, out, "retrieval_long", also_markdown=False
            )
        d, tt, gal, met = inputs.primary_endpoint
        wanted = {"direction": d, "trial_type": tt, "gallery": gal, "metric": met}
        mask = pd.Series(True, index=agg.index)
        for col, val in wanted.items():
            if col in agg.columns:
                mask &= agg[col] == val
            else:
                mask &= False
        prim = agg[mask]
        if len(prim):
            r = prim.iloc[0]
            primary_txt = (
                f"**Pre-registered primary endpoint** -- {d}, {tt}, {gal}, {met}, "
                f"video-disjoint, aggregated over folds:\n\n"
                f"- **{r['estimate']*100:.2f}%** "
                f"(95% CI {r['ci_lo']*100:.2f}-{r['ci_hi']*100:.2f}), "
                f"chance {r['chance']*100:.2f}%, n = {int(r['n_units'])} subjects\n"
                f"- above chance by CI: **{'yes' if r['above_chance'] else 'no'}**\n"
            )
        else:
            primary_txt = _not_run(
                "primary endpoint row not found in the aggregated table",
                f"no rows matched {inputs.primary_endpoint}",
            )
        parts.append(_section("2. Primary endpoint", primary_txt))

        # ---------------- 3. gallery ladder --------------------------------
        ladder = agg[agg["metric"].isin(["top1", "top5"])]
        parts.append(
            _section(
                "3. Retrieval ladder (gallery sizes, both directions, "
                "single vs pseudo-trial)",
                df_to_markdown(
                    ladder[
                        [c for c in ("direction", "trial_type", "gallery", "metric",
                                     "estimate", "ci_lo", "ci_hi", "chance",
                                     "n_units", "above_chance") if c in ladder.columns]
                    ]
                ),
            )
        )

        # ---------------- 4. attribute-shortcut ceiling --------------------
        within = agg[agg["gallery"].astype(str).str.startswith("within_group")]
        cross = agg[agg["gallery"].astype(str).str.startswith("cross_group")]
        shortcut_txt = (
            "Retrieval with distractors drawn from the **same material** isolates how "
            "much of the full-gallery result is a repackaged 8-way material code. "
            "Within-material accuracy at chance means the 'zero-shot video retrieval' "
            "claim reduces to attribute decoding.\n\n"
            "**`within_group` is the control. `cross_group` is NOT.** In "
            "`tactus.eval.retrieval` the cross-group gallery is built per query as "
            "{every item of a *different* material} + {the true item}, so the target is "
            "the only item of its own material and a decoder that knows nothing except "
            "the material scores 100%. Verified directly: a synthetic pure-material "
            "classifier scores cross_group top-1 = 1.000 while its within-material 2-way "
            "top-1 is 0.500, i.e. exactly chance. cross_group is therefore an upper bound "
            "*inflated by* the material code, not a control on it, and it is listed below "
            "only so nobody recomputes it and reads it the other way.\n\n"
            "The size-harmonised control is the 2-way pair: within-material 2-way top-1 "
            "against overall 2-way top-1, both with chance 0.500.\n\n"
            + df_to_markdown(pd.concat([within, cross], ignore_index=True))
        )
        parts.append(_section("4. Attribute-shortcut ceiling (within-material distractors)", shortcut_txt))
    else:
        parts.append(_section("2. Primary endpoint", _not_run("retrieval", "no retrieval_long supplied")))

    # ---------------- 5. noise ceilings ------------------------------------
    if inputs.noise_ceiling_long is not None and len(inputs.noise_ceiling_long):
        nc = inputs.noise_ceiling_long
        if write_tables:
            written["ceiling"] = write_table(nc, out, "noise_ceilings")
        body = (
            "Split-half (k vs k repeats, within subject) EEG->EEG retrieval gives a "
            "ceiling **in the endpoint's own units**. Accuracies are therefore quoted "
            "as a fraction of this ceiling, not as a fraction of 100%.\n\n"
            "Three directional caveats travel with the number, and it is a scale rather "
            "than a hard bound: both sides carry k-repeat noise, so it *under*estimates "
            "the ceiling for retrieval against the noiseless frozen video gallery (a "
            "model exceeding it is informative, not paradoxical, and such rows are "
            "flagged `exceeds_ceiling`); it *under*estimates what a model averaging more "
            "than k repeats could reach; and it *over*estimates what a cross-subject "
            "model could reach.\n\n"
            + df_to_markdown(
                nc[nc["subject_id"].astype(str) == "pooled"]
                if "subject_id" in nc.columns else nc
            )
        )
        if inputs.subject_reliability is not None and len(inputs.subject_reliability):
            if write_tables:
                written["reliability"] = write_table(
                    inputs.subject_reliability, out, "subject_reliability"
                )
            rel = inputs.subject_reliability["pattern_r_sb"].to_numpy(dtype=float)
            body += (
                f"\n\nPer-subject split-half reliability (Spearman-Brown corrected): "
                f"median {np.nanmedian(rel):.3f}, range "
                f"{np.nanmin(rel):.3f}-{np.nanmax(rel):.3f} over "
                f"{len(rel)} subjects.\n"
            )
        parts.append(_section("5. Noise ceilings and fraction-of-ceiling", body))
    else:
        parts.append(
            _section("5. Noise ceilings", _not_run("noise ceilings", "no ceiling table supplied"))
        )

    # ---------------- 6. permutation --------------------------------------
    if inputs.permutation:
        perm = inputs.permutation
        vid = perm.get("video_level")
        nar = perm.get("narrowing")
        lines = [
            "Exchangeable unit = **base video**. The video -> embedding assignment is "
            "shuffled (all four orientations of a video move together); trial labels "
            "are never shuffled.\n",
        ]
        if vid is not None:
            summ = vid.summary() if hasattr(vid, "summary") else dict(vid)
            lines.append(df_to_markdown(pd.DataFrame([summ])))
        if nar:
            lines.append(
                "\n**Why the unit matters.** The same observed statistic evaluated "
                "against a (wrong) trial-level null:\n\n"
                + df_to_markdown(pd.DataFrame([dict(nar)]))
                + f"\n\nThe trial-level null is {nar.get('narrowing_factor', float('nan')):.2f}x "
                "narrower in SD than the video-level null; using it would have turned "
                f"p = {nar.get('p_value_video_level', float('nan')):.4f} into "
                f"p = {nar.get('p_value_if_trial_level_null', float('nan')):.4f}. "
                "This table is reported, not asserted."
            )
        mm = perm.get("material_matched")
        mmv = perm.get("video_level_material_matched")
        if mm:
            frac = mm.get("frac_of_effect_beyond_material")
            lines.append(
                "\n**Material-matched null (the attribute-shortcut quantification, "
                "blueprint 5.2).** The same statistic against a null that shuffles the "
                "gallery only *within material*, so a model that has learned nothing "
                "except the 8-way material code scores at the null rather than at "
                "chance:\n\n"
                + df_to_markdown(pd.DataFrame([dict(mm)]))
                + (
                    f"\n\nOnly **{100 * frac:.0f}%** of the above-chance endpoint "
                    "survives a null that already knows the material; the remainder is "
                    "what the 8-way material code alone buys. Report the endpoint with "
                    "this number attached -- 90 same-hand, same-scene videos make "
                    "'zero-shot video retrieval' and 'material decoding' overlap by "
                    "construction, and this is the honest split."
                    if isinstance(frac, (int, float)) else ""
                )
            )
        if mmv is not None:
            lines.append("\n" + df_to_markdown(pd.DataFrame([dict(mmv)])))
        parts.append(_section("6. Permutation inference", "\n".join(lines)))
    else:
        parts.append(
            _section("6. Permutation inference", _not_run("permutation", "no results supplied"))
        )

    # ---------------- 7. RSA ----------------------------------------------
    if inputs.rsa_curves is not None and len(inputs.rsa_curves):
        if write_tables:
            written["rsa"] = write_table(inputs.rsa_curves, out, "rsa_time_courses", also_markdown=False)
        body = (
            "Time-resolved RDMs over the held-out conditions (pseudo-trial averaged, "
            "cross-validated distance). Model RDMs are correlated with Spearman and, "
            "for the semantic models, with a **partial** Spearman that removes the "
            "low-level (motion energy / luminance / contrast) RDM. Inference is a "
            "cluster-mass permutation whose exchangeable unit is the base video.\n"
        )
        if inputs.rsa_clusters is not None and len(inputs.rsa_clusters):
            if write_tables:
                written["rsa_clusters"] = write_table(inputs.rsa_clusters, out, "rsa_clusters")
            body += "\n" + df_to_markdown(inputs.rsa_clusters)
        if inputs.rsa_noise_ceiling:
            lo, hi = inputs.rsa_noise_ceiling
            body += (
                f"\n\nRDM noise ceiling: lower {lo:.3f}, upper {hi:.3f}. Model "
                "correlations at or above the lower bound are already at the "
                "group-consistent ceiling."
            )
        parts.append(_section("7. RSA (time-resolved, partial)", body))
    else:
        parts.append(_section("7. RSA", _not_run("RSA", "no curves supplied")))

    # ---------------- 8. confound battery ---------------------------------
    caveats: List[str] = []
    if inputs.confounds:
        cf = inputs.confounds
        chunks: List[str] = []

        ident = cf.get("identity")
        if ident:
            chunks.append(
                "**(a) Subject-identity probe, reported jointly with alignment.** "
                "Identity accuracy alone is unfalsifiable -- a collapsed embedding wins "
                "it outright -- so it is only ever read against the alignment retained "
                "on the same embedding.\n\n"
                + df_to_markdown(
                    pd.DataFrame([{k: v for k, v in ident.items() if not isinstance(v, (dict, list))}])
                )
            )

        for key, label in (
            ("trial_index_subject_split", "across-subject split"),
            ("trial_index_sequence_split", "within-subject, sequence held out"),
        ):
            tic = cf.get(key)
            if tic:
                chunks.append(
                    f"**(b) Trial-index / time control decoder ({label}).** "
                    f"Target `{tic['target']}` from timing metadata alone: "
                    f"{tic['accuracy']*100:.2f}% "
                    f"(95% CI {tic.get('acc_ci_lo', float('nan'))*100:.2f}-"
                    f"{tic.get('acc_ci_hi', float('nan'))*100:.2f}) "
                    f"vs chance {tic['chance']*100:.2f}%, block-permutation "
                    f"p = {tic['p_value']:.4f}. "
                    f"Verdict: {tic['verdict']}"
                )

        oc = cf.get("ocular")
        if oc:
            chunks.append(
                "**(c) Frontal-proxy ocular control.** "
                + f"{oc['verdict']} (margin {oc['margin']*100:.2f} points on the primary endpoint).\n\n"
                + "> " + oc["limitation"].replace("\n", "\n> ")
            )
            caveats.append(oc["limitation"])

        fa = cf.get("frontal_ablation")
        if fa:
            chunks.append(
                "**(d) Frontal-channel-ablated sensitivity model.** "
                f"Dropped {len(fa.get('channels_dropped', []))} frontal channels, kept "
                f"{len(fa.get('channels_kept', []))}. See "
                "`tables/confound_frontal_ablation.*`."
            )
            if write_tables and isinstance(fa.get("table"), pd.DataFrame):
                written["frontal"] = write_table(fa["table"], out, "confound_frontal_ablation", also_markdown=False)

        ll = cf.get("lowlevel")
        if ll:
            chunks.append(
                "**(e) Low-level video feature control.** Optical-flow energy, luminance "
                "and contrast per clip explain "
                f"R^2 = {ll['r2_pooled']:.3f} of the video embedding (pooled over "
                f"dimensions; max per-dimension {ll['r2_per_dim_max']:.3f}). The "
                "`lowlevel_fitted_gallery` arm -- the projection of the embedding onto "
                "the low-level span -- is the accuracy reachable with no semantics at "
                "all; the `residual_gallery` arm is the accuracy attributable to "
                "embedding structure orthogonal to those features."
            )
            if write_tables and isinstance(ll.get("table"), pd.DataFrame):
                written["lowlevel"] = write_table(ll["table"], out, "confound_lowlevel", also_markdown=False)

        skipped = cf.get("skipped") or {}
        for name, why in skipped.items():
            chunks.append(_not_run(f"confound probe `{name}`", why))

        parts.append(_section("8. Confound battery", "\n\n".join(chunks)))
        caveats.extend([c for c in (cf.get("caveats") or []) if c not in caveats])
    else:
        parts.append(_section("8. Confound battery", _not_run("confound battery", "no results supplied")))

    # ---------------- 9. ablations with MDD verdicts ----------------------
    if inputs.ablations is not None and len(inputs.ablations):
        abl = inputs.ablations.copy()
        if "mean_diff" in abl.columns and "mdd_verdict" not in abl.columns:
            abl["mdd_verdict"] = [
                flag_below_mdd(v, subject_mdd)["verdict"] for v in abl["mean_diff"]
            ]
        if write_tables:
            written["ablations"] = write_table(abl, out, "ablations")
        n_unres = int(abl["mdd_verdict"].astype(str).str.startswith("BELOW").sum()) \
            if "mdd_verdict" in abl.columns else 0
        parts.append(
            _section(
                "9. Ablations and arm comparisons",
                df_to_markdown(abl)
                + f"\n\n{n_unres} of {len(abl)} comparisons fall below the design's "
                "resolution and are reported as unresolved.",
            )
        )

    # ---------------- 10. multiplicity ------------------------------------
    if inputs.fdr_table is not None and len(inputs.fdr_table):
        fdr = inputs.fdr_table
        if write_tables:
            written["fdr"] = write_table(fdr, out, "hierarchical_fdr")
        meta = (
            f"Families: {fdr.attrs.get('n_families', '?')}, selected: "
            f"{fdr.attrs.get('n_families_selected', '?')}, level-1 q = "
            f"{fdr.attrs.get('q_level1', '?')}, level-2 q = "
            f"{fdr.attrs.get('q_level2', '?')}.\n\n"
        )
        parts.append(
            _section(
                "10. Multiplicity (hierarchical FDR over the endpoint zoo)",
                meta + df_to_markdown(fdr),
            )
        )

    # ---------------- 11. what this cannot certify ------------------------
    standing = [
        "Ocular contribution outside the pre-saccadic window (< ~150 ms) is not "
        "certifiable: ds005662 has no EOG channels and the surrogate is built from "
        "frontal EEG.",
        "Adjacent-trial overlap at 800 ms SOA is handled by forbidding temporally "
        "adjacent train/test epochs, but the 0-600 ms window still contains the "
        "physical tail of the preceding trial's response; this is bounded by the "
        "post-target exclusion and the rERP sensitivity analysis, not eliminated.",
        "Orientation-level and repeat-level effects are conditional on the "
        "sequence x orientation crosstab: if orientation is blocked by sequence, "
        "orientation decoding is partly block/time decoding.",
        "Differences below the resolutions quoted in section 1 are unresolved, "
        "in both directions.",
    ]
    all_claims = list(inputs.uncertifiable_claims) + standing + caveats
    parts.append(
        _section(
            "11. Claims this design cannot certify",
            "\n".join(f"- {c}" for c in all_claims),
        )
    )

    if inputs.notes:
        parts.append(_section("12. Notes", "\n".join(f"- {n}" for n in inputs.notes)))

    if written:
        parts.append(
            _section(
                "Artifacts",
                "\n".join(
                    f"- `{name}`: " + ", ".join(f"{k} -> `{v}`" for k, v in paths.items())
                    for name, paths in written.items()
                ),
            )
        )

    report_path = out / "REPORT.md"
    report_path.write_text("\n".join(parts), encoding="utf-8")
    return report_path
