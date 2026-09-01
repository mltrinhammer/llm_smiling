"""Analysis 1 — individual Duchenne smiling by speech-turn sentiment.

    smile ~ sentiment + (1 | patient_id)          neutral = reference

Four tests: participant and interviewer x frequency and intensity. Omnibus LRT
(df = 2) with a within-patient permutation null, because the residuals are far
from Gaussian. BH-FDR within each operationalisation across the two persons.
"""
from __future__ import annotations

import argparse
import math
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2 as _chi2_dist

import statsmodels.formula.api as smf

from utils import (
    SENTIMENTS,
    setup_logger,
    build_labeled_index,
    write_csv,
    bh_fdr,
    residual_normality,
    cohens_d_pooled,
    ranef_variances,
)
from common import (
    add_common_args,
    build_jobs,
    run_parallel_jobs,
    process_turns_job,
    apply_inclusion_filter,
    write_turns_csv,
)

# (person label, output measure name, per-turn column, BH-FDR family)
SMILE_MEASURES: list[tuple[str, str, str, str]] = [
    ("client",    "duchenne_smile_freq",      "client_smile_freq",         "frequency"),
    ("therapist", "duchenne_smile_freq",      "therapist_smile_freq",      "frequency"),
    ("client",    "duchenne_smile_intensity", "client_smile_intensity",    "intensity"),
    ("therapist", "duchenne_smile_intensity", "therapist_smile_intensity", "intensity"),
]

FIT_FIELDS = [
    "lr_stat", "lr_pval",
    "coef_positive", "se_positive", "z_positive", "p_positive",
    "coef_negative", "se_negative", "z_negative", "p_negative",
    "coef_pos_vs_neg", "se_pos_vs_neg", "z_pos_vs_neg", "p_pos_vs_neg",
    "sigma2_patient", "sigma2_resid", "icc",
    "resid_shapiro_w", "resid_shapiro_p", "resid_skewness", "resid_kurtosis",
]


def _nan_fit() -> dict:
    out: dict = {k: math.nan for k in FIT_FIELDS}
    out.update({"engine": "", "method": ""})
    return out


def _nan_result() -> dict:
    out = _nan_fit()
    out.update({"n_turns": 0, "n_patients": 0, "n_therapists": 0,
                "grand_mean_raw": math.nan,
                "d_positive": math.nan, "d_negative": math.nan,
                "d_pos_vs_neg": math.nan})
    for s in SENTIMENTS:
        out[f"mean_{s}"] = math.nan
        out[f"n_turns_{s}"] = 0
    return out


def lmm_sentiment_test(all_turn_rows: list[dict], col: str) -> dict:
    """Omnibus LRT, the three contrasts, variance components and descriptives."""
    triples = [
        (row["patient_id"], str(row.get("therapist_id", "")), row["sentiment"], float(row[col]))
        for row in all_turn_rows
        if row.get("sentiment") in SENTIMENTS
        and row.get(col) is not None and not math.isnan(row.get(col, math.nan))
    ]
    if len(triples) < 6:
        return _nan_result()

    df = pd.DataFrame(triples, columns=["patient_id", "therapist_id", "sentiment", "au_value"])
    if df["patient_id"].nunique() < 3 or df["sentiment"].nunique() < 2:
        return _nan_result()

    by_sent = {s: df.loc[df["sentiment"] == s, "au_value"].to_numpy() for s in SENTIMENTS}
    fit = _fit_sentiment_pymer4(df)
    fit["lr_stat"], fit["lr_pval"] = _omnibus_lrt(df)

    return {
        "n_turns": len(triples),
        "n_patients": int(df["patient_id"].nunique()),
        "n_therapists": int(df["therapist_id"].nunique()),
        "grand_mean_raw": float(df["au_value"].mean()),
        "d_positive": cohens_d_pooled(by_sent["positive"], by_sent["neutral"]),
        "d_negative": cohens_d_pooled(by_sent["negative"], by_sent["neutral"]),
        "d_pos_vs_neg": cohens_d_pooled(by_sent["positive"], by_sent["negative"]),
        **{f"mean_{s}": float(by_sent[s].mean()) if by_sent[s].size else math.nan
           for s in SENTIMENTS},
        **{f"n_turns_{s}": int(by_sent[s].size) for s in SENTIMENTS},
        **fit,
    }


ML_OPTIMISERS = ("lbfgs", "bfgs", "powell")

FULL_FORMULA = "au_value ~ C(sentiment, Treatment('neutral'))"
NULL_FORMULA = "au_value ~ 1"


def _ml_llf(formula: str, data: pd.DataFrame) -> float:
    """ML log-likelihood, or NaN if no optimiser converges."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in ML_OPTIMISERS:
            try:
                llf = smf.mixedlm(formula, data=data,
                                  groups=data["patient_id"]).fit(reml=False,
                                                                 method=method).llf
            except Exception:
                continue
            if np.isfinite(llf):
                return float(llf)
    return math.nan


def _omnibus_lrt(df: pd.DataFrame) -> tuple[float, float]:
    """Omnibus sentiment LRT (df = 2), statsmodels MixedLM at ML.

    Must be the same estimator that builds the permutation null, or the
    permutation p is invalid. Raises rather than dropping the random effect.
    """
    full_llf = _ml_llf(FULL_FORMULA, df)
    null_llf = _ml_llf(NULL_FORMULA, df)
    if not (np.isfinite(full_llf) and np.isfinite(null_llf)):
        raise RuntimeError(
            f"MixedLM failed to converge with any of {ML_OPTIMISERS} "
            f"(full llf={full_llf}, null llf={null_llf})")
    lr = max(-2.0 * (null_llf - full_llf), 0.0)
    return float(lr), float(_chi2_dist.sf(lr, df=2))



def _fit_sentiment_pymer4(df: pd.DataFrame) -> dict:
    """Two lmer fits; the negative-reference one gives an exact
    positive-vs-negative contrast rather than a hand-built one."""
    import polars as pl
    from pymer4.models import lmer
    out = _nan_fit()
    out["engine"] = "pymer4"

    d = df.copy()
    d["patient_id"] = d["patient_id"].astype(str)
    d["sentiment"] = d["sentiment"].astype(str)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = lmer("au_value ~ sentiment + (1|patient_id)", data=pl.from_pandas(d))
        full.set_factors({"sentiment": ["neutral", "positive", "negative"]})
        full.fit()
        neg_ref = lmer("au_value ~ sentiment + (1|patient_id)", data=pl.from_pandas(d))
        neg_ref.set_factors({"sentiment": ["negative", "positive", "neutral"]})
        neg_ref.fit()
    out["method"] = "lmer"

    def _tbl(model):
        rf = model.result_fit
        return (rf.to_pandas() if hasattr(rf, "to_pandas") else rf).set_index("term")

    def _cell(rf, sub, *names, default=math.nan):
        term = next((t for t in rf.index if sub in str(t).lower()), None)
        if term is None:
            return default
        row = rf.loc[term]
        for n in names:
            if n in row.index and pd.notna(row[n]):
                return float(row[n])
        return default

    rf = _tbl(full)
    for sent in ("positive", "negative"):
        out[f"coef_{sent}"] = _cell(rf, sent, "estimate", "Estimate")
        out[f"se_{sent}"] = _cell(rf, sent, "std_error", "SE")
        out[f"z_{sent}"] = _cell(rf, sent, "t_stat", "T-stat", "T_stat")
        out[f"p_{sent}"] = _cell(rf, sent, "p_value", "P-val", "Pr(>|t|)")

    rfn = _tbl(neg_ref)
    out["coef_pos_vs_neg"] = _cell(rfn, "positive", "estimate", "Estimate")
    out["se_pos_vs_neg"] = _cell(rfn, "positive", "std_error", "SE")
    out["z_pos_vs_neg"] = _cell(rfn, "positive", "t_stat", "T-stat", "T_stat")
    out["p_pos_vs_neg"] = _cell(rfn, "positive", "p_value", "P-val", "Pr(>|t|)")

    try:
        v = ranef_variances(full)
        sre, sres = v.get("patient_id", math.nan), v.get("residual", math.nan)
        out["sigma2_patient"], out["sigma2_resid"] = sre, sres
        out["icc"] = sre / (sre + sres) if np.isfinite(sre + sres) and sre + sres > 0 else math.nan
    except Exception:
        pass

    try:
        dd = getattr(full, "data", None)
        dd = dd.to_pandas() if hasattr(dd, "to_pandas") else dd
        if dd is not None and "resid" in getattr(dd, "columns", []):
            out.update(residual_normality(np.asarray(dd["resid"], dtype=float)))
    except Exception:
        pass

    if np.isfinite(out["sigma2_patient"]) and out["sigma2_patient"] < 1e-8:
        out["method"] = "lmer_singular"
    return out


def _lmm_perm_worker(args: tuple) -> tuple[float, int]:
    """Within-patient permutation p for the omnibus LRT.

    Non-converging permutations are excluded from both the count and the
    denominator, and n_valid is returned so the loss can be reported.
    args = (col, all_turn_rows, observed_lr, n_perm, seed) -> (p, n_valid).
    """
    col, all_turn_rows, observed_lr, n_perm, seed = args
    if math.isnan(observed_lr) or n_perm <= 0:
        return math.nan, 0

    rows = [(row["patient_id"], row["sentiment"], float(row[col]))
            for row in all_turn_rows
            if row.get("sentiment") in SENTIMENTS
            and row.get(col) is not None and not math.isnan(row.get(col, math.nan))]
    if len(rows) < 6:
        return math.nan, 0

    df = pd.DataFrame(rows, columns=["patient_id", "sentiment", "au_value"])
    rng = np.random.default_rng(seed)
    patient_indices = [idx.to_numpy() for idx in df.groupby("patient_id").groups.values()]

    extremes = n_valid = 0
    for _ in range(n_perm):
        perm_sent = df["sentiment"].values.copy()
        for indices in patient_indices:
            perm_sent[indices] = rng.permutation(perm_sent[indices])
        df_perm = df.assign(sentiment=perm_sent)
        f_llf = _ml_llf(FULL_FORMULA, df_perm)
        n_llf = _ml_llf(NULL_FORMULA, df_perm)
        if not (np.isfinite(f_llf) and np.isfinite(n_llf)):
            continue
        n_valid += 1
        if max(-2.0 * (n_llf - f_llf), 0.0) >= observed_lr:
            extremes += 1

    if n_valid == 0:
        return math.nan, 0
    return (extremes + 1) / (n_valid + 1), n_valid   # Phipson & Smyth (2016)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Individual Duchenne smiling by sentiment (LMM, patient random intercept)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    parser.add_argument("--n-ftest-jobs", type=int, default=4,
                        help="Workers for the parallel permutation tests")
    parser.add_argument("--anova-n-perm", type=int, default=0,
                        help="Permutations for the omnibus LRT (0 = off; the "
                             "reported analyses use 3000)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "bindung_ftest.log", verbose=args.verbose)
    logger.info("Starting — individual smiling by sentiment")
    logger.info("Args: %s", vars(args))

    with Path(args.data_model).open("r", encoding="utf-8") as f:
        dm = yaml.safe_load(f)

    labeled_index = build_labeled_index(Path(args.labeled_dir), suffix=args.labeled_suffix)
    logger.info("Labeled files indexed: %d patient IDs", len(labeled_index))

    jobs = build_jobs(dm, labeled_index, args, repo_root, logger)
    if not jobs:
        logger.warning("No jobs. Exiting.")
        return

    all_results = run_parallel_jobs(jobs, process_turns_job, args.n_jobs, logger)
    if not all_results:
        logger.error("No successful results. Exiting.")
        return

    all_turn_rows: list[dict] = []
    for res in all_results:
        all_turn_rows.extend(res.get("turn_rows", []))
    logger.info("Total labeled client turns: %d across %d patients",
                len(all_turn_rows), len(all_results))

    # No BLRI restriction here; that enters in Analysis 3.
    all_turn_rows, all_results = apply_inclusion_filter(
        all_turn_rows, all_results, logger, min_bucket_turns=args.min_bucket_turns,
    )
    write_turns_csv(all_turn_rows, output_dir, logger)

    logger.info("Fitting %d LMMs …", len(SMILE_MEASURES))
    results = [lmm_sentiment_test(all_turn_rows, col)
               for _person, _measure, col, _fam in SMILE_MEASURES]

    perm_p: list[float] = [math.nan] * len(SMILE_MEASURES)
    perm_n: list[int] = [0] * len(SMILE_MEASURES)
    if args.anova_n_perm > 0:
        logger.info("Permutation test: %d measures x %d shuffles …",
                    len(SMILE_MEASURES), args.anova_n_perm)
        perm_args = [(col, all_turn_rows, results[i]["lr_stat"],
                      args.anova_n_perm, args.seed + 100_000 + i)
                     for i, (_p, _m, col, _f) in enumerate(SMILE_MEASURES)]
        n_sj = max(1, args.n_ftest_jobs)
        if n_sj == 1:
            pairs = [_lmm_perm_worker(pa) for pa in perm_args]
        else:
            with ProcessPoolExecutor(max_workers=n_sj) as ex:
                pairs = list(ex.map(_lmm_perm_worker, perm_args))
        perm_p = [p for p, _n in pairs]
        perm_n = [n for _p, n in pairs]
        for i, (_pn, measure, col, _f) in enumerate(SMILE_MEASURES):
            if perm_n[i] < args.anova_n_perm:
                logger.warning("%s: only %d/%d permutations converged",
                               col, perm_n[i], args.anova_n_perm)

    p_for_fdr = [perm_p[i] if not math.isnan(perm_p[i]) else results[i]["lr_pval"]
                 for i in range(len(SMILE_MEASURES))]
    # Within each operationalisation, not across them.
    bh_q: list[float] = [math.nan] * len(SMILE_MEASURES)
    for fam in {f for *_rest, f in SMILE_MEASURES}:
        idxs = [i for i, (*_r, f) in enumerate(SMILE_MEASURES) if f == fam]
        fam_q = bh_fdr([p_for_fdr[i] for i in idxs])
        for pos, i in enumerate(idxs):
            bh_q[i] = fam_q[pos]

    out_rows = []
    for idx, (person, measure, _col, fam) in enumerate(SMILE_MEASURES):
        r = results[idx]
        out_rows.append({
            "person": person, "measure": measure, "fdr_family": fam,
            "n_turns": r["n_turns"], "n_patients": r["n_patients"],
            "n_therapists": r["n_therapists"],
            "grand_mean_raw": r["grand_mean_raw"],
            **{f"mean_{s}": r[f"mean_{s}"] for s in SENTIMENTS},
            **{f"n_turns_{s}": r[f"n_turns_{s}"] for s in SENTIMENTS},
            **{k: r[k] for k in FIT_FIELDS},
            "d_positive": r["d_positive"], "d_negative": r["d_negative"],
            "d_pos_vs_neg": r["d_pos_vs_neg"],
            "engine": r["engine"], "method": r["method"],
            "perm_p": perm_p[idx], "perm_n_valid": perm_n[idx],
            "q_value_fdr": bh_q[idx],
            "q_source": "perm" if not math.isnan(perm_p[idx]) else "lr",
        })

    fieldnames = (
        ["person", "measure", "fdr_family", "n_turns", "n_patients", "n_therapists",
         "grand_mean_raw"]
        + [f"mean_{s}" for s in SENTIMENTS]
        + [f"n_turns_{s}" for s in SENTIMENTS]
        + FIT_FIELDS[:2]                       # lr_stat, lr_pval
        + FIT_FIELDS[2:14]                     # the three contrasts
        + ["d_positive", "d_negative", "d_pos_vs_neg"]
        + FIT_FIELDS[14:]                      # variance components + residuals
        + ["engine", "method", "perm_p", "perm_n_valid", "q_value_fdr", "q_source"]
    )
    smile_csv = output_dir / "bindung_smile_sentiment_lmm.csv"
    write_csv(smile_csv, fieldnames, out_rows)
    logger.info("WROTE %s (%d rows)", smile_csv, len(out_rows))
    logger.info("Done in %.1f s", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
