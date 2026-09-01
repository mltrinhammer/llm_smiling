"""Analysis 3 — BLRI predicted from facial synchrony, per participant.

    BLRI ~ 1 + average_sync + diff_sync + (1 | interviewer_id)

average_sync is the mean of the three sentiment means, diff_sync their
positive-minus-negative difference.
Uncentered BLRI, uncorrected p; lme4 + lmerTest via pymer4.
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from utils import ranef_variances
from common import BLRI_TARGET

warnings.filterwarnings("ignore", category=FutureWarning)

SENTS = ("positive", "neutral", "negative")
PREDICTORS = ("average_sync", "diff_sync")

#   (level, measure, transform, per-turn column, drop the sign)
MEASURE_SPECS = [
    ("synchrony", "simul_smile",    "identity", "simul_smile_freq", False),
    ("synchrony", "duchenne_smile", "fisherz",  "cc_r_peak",        True),
]


def load_blri_scores(data_model_path: Path, target: str) -> pd.DataFrame:
    """Per-participant BLRI from data_model.yaml. Each participant carries the
    score under exactly one interview type, so the first finite value is
    unambiguous (verified against the data model)."""
    with open(data_model_path, "r", encoding="utf-8") as f:
        dm = yaml.safe_load(f)

    rows = []
    for iv in dm.get("interviews", []):
        pid = str(iv.get("patient", {}).get("patient_id", "")).strip()
        tid = str(iv.get("therapist", {}).get("therapist_id", "")).strip()
        for idata in iv.get("types", {}).values():
            try:
                fval = float(idata.get("labels", {}).get(target))
            except (TypeError, ValueError):
                continue
            if np.isfinite(fval):
                rows.append({"participant_id": pid, "interviewer_id_dm": tid,
                             target: fval})
                break

    df = pd.DataFrame(rows).drop_duplicates(subset=["participant_id"])
    print(f"BLRI scores loaded: {len(df)} participants (target={target})")
    return df


def fisher_z(r: pd.Series, cap: float = 0.999) -> pd.Series:
    """Fisher z; correlations are not additive on the r scale."""
    return np.arctanh(r.clip(lower=-cap, upper=cap))


def build_participant_synchrony(
    clc: pd.DataFrame,
    level: str,
    measure: str,
    transform: str,
    value_col: str,
    absolute: bool,
) -> pd.DataFrame:
    """Aggregate per-turn synchrony into the two predictors. Measures share the
    same rows and differ only by ``value_col``, so the filter is on ``level``."""
    if value_col not in clc.columns:
        print(f"  [skip] {measure}: column '{value_col}' absent from synchrony turns")
        return pd.DataFrame()
    sub = clc[clc["level"] == level].copy()
    sub = sub[np.isfinite(sub[value_col])]
    if sub.empty:
        return pd.DataFrame()

    vals = sub[value_col].abs() if absolute else sub[value_col]
    sub["dv"] = fisher_z(vals) if transform == "fisherz" else vals

    records = []
    for pid, grp in sub.groupby("participant_id", sort=False):
        per_sent = grp.groupby("sentiment")["dv"].mean()
        mean_pos = per_sent.get("positive", np.nan)
        mean_neg = per_sent.get("negative", np.nan)
        if not (np.isfinite(mean_pos) and np.isfinite(mean_neg)):
            continue        # diff_sync undefined without both categories
        # Mean of the three sentiment means, so the categories weigh equally
        # regardless of how many turns of each kind the participant produced.
        bucket_means = [m for m in (per_sent.get(s, np.nan) for s in SENTS)
                        if np.isfinite(m)]
        average = float(np.mean(bucket_means))
        records.append({
            "participant_id": pid,
            "interviewer_id": grp["interviewer_id"].iloc[0],
            "measure": measure,
            "average_sync": average,
            "diff_sync": float(mean_pos - mean_neg),
            "n_turns": int(len(grp)),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _blank_result(design: pd.DataFrame, measure: str, target: str) -> dict:
    out = {
        "measure": measure, "target": target,
        "n_obs": int(len(design)),
        "n_interviewers": int(design["interviewer_id"].nunique()) if len(design) else 0,
        "intercept": math.nan, "intercept_se": math.nan,
        "sigma2_interviewer": math.nan, "sigma2_resid": math.nan,
        "icc": math.nan, "r2_marginal": math.nan, "r2_conditional": math.nan,
        "converged": False,
        "engine": "", "method": "",
    }
    for p in PREDICTORS:
        out.update({f"beta_{p}": math.nan, f"se_{p}": math.nan,
                    f"ci_lo_{p}": math.nan, f"ci_hi_{p}": math.nan,
                    f"t_{p}": math.nan, f"p_{p}": math.nan, f"std_beta_{p}": math.nan})
    return out


def _fittable(design: pd.DataFrame, target: str) -> bool:
    return not (len(design) < 10
                or design["interviewer_id"].nunique() < 2
                or design[target].std(ddof=1) < 1e-10)


def _r2_and_icc(design: pd.DataFrame, coefs: dict, sigma2_re: float,
                sigma2_resid: float) -> tuple[float, float, float]:
    """Nakagawa-Schielzeth marginal/conditional R2 and the ICC."""
    fe_pred = (coefs["intercept"]
               + coefs["beta_average_sync"] * design["average_sync"].values
               + coefs["beta_diff_sync"] * design["diff_sync"].values)
    var_fixed = float(np.var(fe_pred))
    sigma2_re = max(float(sigma2_re), 0.0) if np.isfinite(sigma2_re) else 0.0
    if not np.isfinite(sigma2_resid):
        return math.nan, math.nan, math.nan
    denom = var_fixed + sigma2_re + float(sigma2_resid)
    if denom < 1e-15:
        return math.nan, math.nan, math.nan
    return (var_fixed / denom,
            (var_fixed + sigma2_re) / denom,
            sigma2_re / (sigma2_re + sigma2_resid) if (sigma2_re + sigma2_resid) > 0 else math.nan)


def _std_betas(design: pd.DataFrame, target: str, betas: dict) -> dict:
    """b * SD(x)/SD(y). The predictors span a small fraction of [0, 1], so the
    raw coefficients are not directly interpretable."""
    sd_y = design[target].std(ddof=1)
    return {f"std_beta_{p}": (float(betas[f"beta_{p}"] * design[p].std(ddof=1) / sd_y)
                              if (np.isfinite(betas.get(f"beta_{p}", math.nan)) and sd_y > 1e-12)
                              else math.nan)
            for p in PREDICTORS}



def fit_model(design: pd.DataFrame, measure: str, target: str) -> dict:
    """Fit the model with pymer4 (REML + Satterthwaite, both fit() defaults)."""
    import polars as pl
    from pymer4.models import lmer
    out = _blank_result(design, measure, target)
    out["engine"] = "pymer4"
    if not _fittable(design, target):
        out["method"] = "not_fittable"
        return out

    d = design.rename(columns={target: "blri"}).copy()
    d["interviewer_id"] = d["interviewer_id"].astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = lmer("blri ~ average_sync + diff_sync + (1|interviewer_id)",
                    data=pl.from_pandas(d))
        full.fit()
    out["converged"] = True
    out["method"] = "lmer"

    rf = full.result_fit
    rf = (rf.to_pandas() if hasattr(rf, "to_pandas") else rf).set_index("term")

    def _cell(term, *names, default=math.nan):
        if term not in rf.index:
            return default
        row = rf.loc[term]
        for n in names:
            if n in row.index and pd.notna(row[n]):
                return float(row[n])
        return default

    for iname in ("(Intercept)", "Intercept"):
        if iname in rf.index:
            out["intercept"] = _cell(iname, "estimate", "Estimate")
            out["intercept_se"] = _cell(iname, "std_error", "SE")
            break

    coefs = {"intercept": out["intercept"]}
    for p in PREDICTORS:
        coefs[f"beta_{p}"] = math.nan
        if p in rf.index:
            out[f"beta_{p}"] = coefs[f"beta_{p}"] = _cell(p, "estimate", "Estimate")
            out[f"se_{p}"] = _cell(p, "std_error", "SE")
            out[f"t_{p}"] = _cell(p, "t_stat", "T-stat", "T_stat")
            out[f"p_{p}"] = _cell(p, "p_value", "P-val", "Pr(>|t|)")
            out[f"ci_lo_{p}"] = _cell(p, "conf_low", "2.5_ci")
            out[f"ci_hi_{p}"] = _cell(p, "conf_high", "97.5_ci")

    sigma2_re = math.nan
    try:
        v = ranef_variances(full)
        sigma2_re = v.get("interviewer_id", math.nan)
        if np.isfinite(v.get("residual", math.nan)):
            out["sigma2_resid"] = v["residual"]
    except Exception:
        pass
    out["sigma2_interviewer"] = sigma2_re
    out["r2_marginal"], out["r2_conditional"], out["icc"] = _r2_and_icc(
        design, coefs, sigma2_re, out["sigma2_resid"])
    out.update(_std_betas(design, target, out))

    if np.isfinite(sigma2_re) and sigma2_re < 1e-8:
        out["method"] = "lmer_singular"
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_analysis(clc_turns: Path, data_model: Path, out_dir: Path,
                 target: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading turn-level synchrony …")
    clc = pd.read_csv(clc_turns)
    clc["sentiment"] = clc["sentiment"].astype(str).str.strip().str.lower()
    print(f"  {len(clc)} rows, {clc['participant_id'].nunique()} participants, "
          f"{clc['interviewer_id'].nunique()} interviewers")

    blri = load_blri_scores(data_model, target)

    designs: dict[str, pd.DataFrame] = {}
    results: list[dict] = []
    all_design_rows: list[pd.DataFrame] = []

    for level, measure, transform, value_col, absolute in MEASURE_SPECS:
        part = build_participant_synchrony(clc, level, measure, transform,
                                           value_col, absolute)
        if part.empty:
            print(f"  [skip] {measure}: no participants with both pos & neg turns")
            designs[measure] = part
            results.append(_blank_result(part, measure, target))
            continue
        design = part.merge(blri, on="participant_id", how="inner")
        design = design.dropna(subset=["average_sync", "diff_sync", target])
        designs[measure] = design
        all_design_rows.append(design.assign(measure=measure))
        print(f"  {measure:16s}: {len(design)} participants, "
              f"{design['interviewer_id'].nunique()} interviewers")
        results.append(fit_model(design, measure, target))

    # Uncorrected: a tab_model layout, not a family of tests.
    for r in results:
        for pred in PREDICTORS:
            r[f"q_{pred}"] = r.get(f"p_{pred}", np.nan)

    if all_design_rows:
        dpath = out_dir / "analysis3_participant_synchrony.csv"
        design_all = pd.concat(all_design_rows, ignore_index=True)
        design_all.to_csv(dpath, index=False)
        print(f"WROTE {dpath} ({len(design_all)} rows)")

    res_df = pd.DataFrame(results)
    lead = ["measure", "target", "n_obs", "n_interviewers"]
    res_path = out_dir / "analysis3_blri_synchrony_lmm.csv"
    res_df.reindex(columns=lead + [c for c in res_df.columns if c not in lead]
                   ).to_csv(res_path, index=False)
    print(f"WROTE {res_path} ({len(res_df)} rows)")

    print("\n" + "=" * 78)
    print(f"ANALYSIS 3 — BLRI ({target}) ~ average_sync + diff_sync + (1|interviewer)")
    print("=" * 78)
    print(f"  {'measure':16s} {'n':>4s} {'b_avg':>8s} {'p_avg':>8s} "
          f"{'b_diff':>8s} {'p_diff':>8s} {'R2m':>6s} {'R2c':>6s}")
    for r in results:
        print(f"  {r['measure']:16s} {r['n_obs']:>4d} "
              f"{r.get('beta_average_sync', float('nan')):>8.3f} "
              f"{r.get('p_average_sync', float('nan')):>8.3f} "
              f"{r.get('beta_diff_sync', float('nan')):>8.3f} "
              f"{r.get('p_diff_sync', float('nan')):>8.3f} "
              f"{r.get('r2_marginal', float('nan')):>6.3f} "
              f"{r.get('r2_conditional', float('nan')):>6.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analysis 3 — BLRI regressed on facial synchrony (LMM)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    p.add_argument("--sync-turns", "--clc-turns", dest="clc_turns", type=Path, required=True,
                   help="bindung_synchrony_turns.csv (output of analysis2.py)")
    p.add_argument("--data-model", type=Path, default=Path("data_model.yaml"),
                   help="data_model.yaml (source of BLRI scores)")
    p.add_argument("--out-dir", dest="out_dir", type=Path, required=True)
    p.add_argument("--blri-target", type=str, default=BLRI_TARGET,
                   help=f"BLRI column to predict (default: {BLRI_TARGET})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args.clc_turns, args.data_model, args.out_dir,
                 target=args.blri_target)
