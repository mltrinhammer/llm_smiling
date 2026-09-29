"""Analysis 2, step 2 — models the per-turn synchrony measures by sentiment.

    synchrony ~ 1 + sentiment + (1|interviewer_id) + (1|participant_id)

Fitted on positive vs negative turns; neutral is descriptive. Three rows are
written: simul_smile (binomial GLMM on co-smile counts) and duchenne_smile (the
cross-lagged correlation, Fisher-z of |r|), both reported, plus simul_smile_any.
"""
from __future__ import annotations

import argparse
import itertools
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from utils import cohens_q, cohens_d_pooled, ranef_variances

SENTS_DESC = ("positive", "neutral", "negative")
FOCAL = ("negative", "positive")     # reference first

SYNC_MEASURE = "duchenne_smile"
RE_TERMS = "(1|interviewer_id) + (1|participant_id)"

K_COL = "simul_smile_frames"
N_COL = "simul_smile_n_frames"
FREQ_COL = "simul_smile_freq"


def fisher_z(r: pd.Series, cap: float = 0.999) -> pd.Series:
    return np.arctanh(r.clip(lower=-cap, upper=cap))


def _find_positive_term(names) -> str | None:
    return next((n for n in names if "positive" in str(n)), None)


def _base_out(sub: pd.DataFrame) -> dict:
    return {
        "n_obs": int(len(sub)),
        "n_participants": int(sub["participant_id"].nunique()),
        "n_interviewers": int(sub["interviewer_id"].nunique()),
        "estimate": math.nan, "se": math.nan, "z": math.nan,
        "p_wald": math.nan, "lrt_stat": math.nan, "p_lrt": math.nan,
        "odds_ratio": math.nan, "cohens_d": math.nan, "cohens_q": math.nan,
        "sigma2_interviewer": math.nan, "sigma2_participant": math.nan,
        "converged": False, "engine": "", "method": "", "lrt_source": "",
        "slope_var_participant": math.nan, "slope_lrt_stat": math.nan,
        "slope_p_lrt": math.nan, "slope_source": "",
    }


def _fittable(sub: pd.DataFrame) -> bool:
    return not (sub["sentiment"].nunique() < 2
                or sub["interviewer_id"].nunique() < 2
                or sub["participant_id"].nunique() < 2
                or len(sub) < 10)


def _to_r(df: pd.DataFrame, name: str) -> None:
    """Push a frame into R's global environment."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr
    importr("lme4")
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv[name] = ro.conversion.get_conversion().py2rpy(
            df.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Gaussian LMM (the correlation-valued measures)
# ---------------------------------------------------------------------------
def _focal_lrt_lme4(sub: pd.DataFrame, dv: str) -> tuple[float, float, str]:
    """Focal sentiment LRT (df = 1), refitted in lme4 at ML so the test and the
    estimate come from the same model."""
    import rpy2.robjects as ro

    s = sub[sub["sentiment"].isin(FOCAL)][[dv, "sentiment", "interviewer_id",
                                           "participant_id"]].dropna()
    if s.empty:
        return math.nan, math.nan, "failed"
    s = s.rename(columns={dv: "dvval"})
    for c in ("sentiment", "interviewer_id", "participant_id"):
        s[c] = s[c].astype(str)
    _to_r(s, "dat_lmm")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ro.r('''
            dat_lmm$sentiment      <- relevel(factor(dat_lmm$sentiment), ref="negative")
            dat_lmm$interviewer_id <- factor(dat_lmm$interviewer_id)
            dat_lmm$participant_id <- factor(dat_lmm$participant_id)
            .lfull <- lme4::lmer(dvval ~ sentiment + (1|interviewer_id) + (1|participant_id),
                                 data=dat_lmm, REML=FALSE)
            .lnull <- lme4::lmer(dvval ~ 1 + (1|interviewer_id) + (1|participant_id),
                                 data=dat_lmm, REML=FALSE)
            .lan   <- anova(.lnull, .lfull)
        ''')
    return (float(np.asarray(ro.r('.lan$Chisq[2]'), dtype=float)[0]),
            float(np.asarray(ro.r('.lan[["Pr(>Chisq)"]][2]'), dtype=float)[0]),
            "lrt_lme4_ml")


def _random_slope_check_lme4(sub: pd.DataFrame, dv: str) -> dict:
    """Does the positive-vs-negative difference vary between participants?

    The slope variance IS the between-person variance in Analysis 3's diff_sync
    predictor: near zero would explain a null there better than "no association".
    Returns NaNs rather than raising if the maximal model does not converge.
    """
    out = {"slope_var_participant": math.nan, "slope_lrt_stat": math.nan,
           "slope_p_lrt": math.nan, "slope_source": ""}
    try:
        import rpy2.robjects as ro

        s = sub[sub["sentiment"].isin(FOCAL)][[dv, "sentiment", "interviewer_id",
                                               "participant_id"]].dropna()
        if s.empty:
            out["slope_source"] = "empty"
            return out
        s = s.rename(columns={dv: "dvval"})
        for c in ("sentiment", "interviewer_id", "participant_id"):
            s[c] = s[c].astype(str)
        _to_r(s, "dat_sl")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ro.r('''
                dat_sl$sentiment      <- relevel(factor(dat_sl$sentiment), ref="negative")
                dat_sl$interviewer_id <- factor(dat_sl$interviewer_id)
                dat_sl$participant_id <- factor(dat_sl$participant_id)
                .sint <- lme4::lmer(dvval ~ sentiment + (1|interviewer_id) + (1|participant_id),
                                    data=dat_sl, REML=FALSE)
                .sslo <- lme4::lmer(dvval ~ sentiment + (1|interviewer_id) + (1 + sentiment|participant_id),
                                    data=dat_sl, REML=FALSE)
                .svcm <- as.matrix(lme4::VarCorr(.sslo)$participant_id)
                .svnm <- rownames(.svcm)
                .san  <- anova(.sint, .sslo)
            ''')

        mat = np.asarray(ro.r(".svcm"), dtype=float)
        nms = [str(x) for x in ro.r(".svnm")]
        idx = next((i for i, n in enumerate(nms) if "sentiment" in n.lower()), None)
        if idx is not None and mat.ndim == 2 and mat.shape[0] > idx:
            out["slope_var_participant"] = float(mat[idx, idx])
        out["slope_lrt_stat"] = float(np.asarray(ro.r('.san$Chisq[2]'), dtype=float)[0])
        out["slope_p_lrt"] = float(np.asarray(ro.r('.san[["Pr(>Chisq)"]][2]'), dtype=float)[0])
        out["slope_source"] = "lme4_random_slope"
    except Exception as exc:
        out["slope_source"] = f"unavailable:{type(exc).__name__}"
    return out


def fit_focal(sub: pd.DataFrame, dv: str) -> dict:
    """Gaussian LMM via pymer4 (REML + Satterthwaite); LRT refitted at ML."""
    import polars as pl
    from pymer4.models import lmer
    out = _base_out(sub)
    out["engine"] = "pymer4"
    if not _fittable(sub):
        out["method"] = "not_fittable"
        return out

    sub = sub.copy()
    for c in ("sentiment", "interviewer_id", "participant_id"):
        sub[c] = sub[c].astype(str)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = lmer(f"{dv} ~ sentiment + {RE_TERMS}", data=pl.from_pandas(sub))
        full.set_factors({"sentiment": list(FOCAL)})   # negative = reference
        full.fit()

    rf = full.result_fit
    rf = (rf.to_pandas() if hasattr(rf, "to_pandas") else rf).set_index("term")
    term = _find_positive_term(rf.index)
    if term is None:
        out["method"] = "no_term"
        return out
    row = rf.loc[term]

    def _c(*names, default=math.nan):
        for n in names:
            if n in row.index and pd.notna(row[n]):
                return float(row[n])
        return default

    out["converged"] = True
    out["estimate"] = _c("estimate", "Estimate")
    out["se"] = _c("std_error", "SE")
    out["p_wald"] = _c("p_value", "P-val", "Pr(>|t|)")
    if np.isfinite(out["se"]) and out["se"] > 0:
        out["z"] = out["estimate"] / out["se"]

    out["lrt_stat"], out["p_lrt"], out["lrt_source"] = _focal_lrt_lme4(sub, dv)

    try:
        v = ranef_variances(full)
        out["sigma2_interviewer"] = v.get("interviewer_id", math.nan)
        out["sigma2_participant"] = v.get("participant_id", math.nan)
    except Exception:
        pass

    boundary = [nm for nm, key in (("interviewer", "sigma2_interviewer"),
                                   ("participant", "sigma2_participant"))
                if np.isfinite(out[key]) and out[key] < 1e-8]
    out["method"] = f"lmer_singular({'+'.join(boundary)})" if boundary else "lmer"
    return out


# ---------------------------------------------------------------------------
# Simultaneous smiling — binomial GLMMs via lme4::glmer through rpy2
# ---------------------------------------------------------------------------
def _glmer_readout(out: dict, co_name: str, rn_name: str, vc_name: str,
                   an_name: str) -> dict:
    """Positive term, variance components and LRT from a fitted glmer."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    co = np.asarray(ro.r(co_name), dtype=float)   # Estimate, SE, z, Pr(>|z|)
    rn = [str(x) for x in ro.r(rn_name)]
    idx = next((i for i, n in enumerate(rn) if "positive" in n.lower()), None)
    if idx is None:
        out["method"] = "no_term"
        return out

    out["estimate"] = float(co[idx, 0])
    out["se"] = float(co[idx, 1])
    out["z"] = float(co[idx, 2])
    out["p_wald"] = float(co[idx, 3])
    if np.isfinite(out["estimate"]):
        out["odds_ratio"] = float(np.exp(out["estimate"]))

    try:
        with localconverter(ro.default_converter + pandas2ri.converter):
            vc = ro.conversion.get_conversion().rpy2py(ro.r(vc_name))
        for grp, key in (("interviewer_id", "sigma2_interviewer"),
                         ("participant_id", "sigma2_participant")):
            row = vc[vc["grp"].astype(str) == grp]
            if len(row):
                out[key] = float(row["vcov"].iloc[0])
    except Exception:
        pass

    try:
        out["lrt_stat"] = float(ro.r(f'{an_name}$Chisq[2]')[0])
        out["p_lrt"] = float(ro.r(f'{an_name}[["Pr(>Chisq)"]][2]')[0])
        out["lrt_source"] = "lrt_lme4_glmer"
    except Exception:
        pass

    out["converged"] = True
    return out


def fit_focal_binomial(sub: pd.DataFrame) -> dict:
    """Aggregated-binomial GLMM on the co-smile counts.

     Called through raw rpy2 because pymer4's polars layer cannot marshal a
    two-column cbind response. The null keeps all three random effects.
    """
    import rpy2.robjects as ro

    out = _base_out(sub)
    out["engine"] = "lme4::glmer(rpy2)"
    if not _fittable(sub):
        out["method"] = "not_fittable"
        return out

    s = sub[sub["sentiment"].isin(FOCAL)].copy().reset_index(drop=True)
    s["succ"] = s[K_COL].astype(int)
    s["fail"] = np.clip(s[N_COL].astype(int) - s[K_COL].astype(int), 0, None).astype(int)
    for c in ("sentiment", "interviewer_id", "participant_id"):
        s[c] = s[c].astype(str)
    s["obs_id"] = s.index.astype(str)
    _to_r(s[["succ", "fail", "sentiment", "interviewer_id", "participant_id", "obs_id"]], "dat")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ro.r('''
            dat$sentiment      <- relevel(factor(dat$sentiment), ref="negative")
            dat$interviewer_id <- factor(dat$interviewer_id)
            dat$participant_id <- factor(dat$participant_id)
            dat$obs_id         <- factor(dat$obs_id)
            .ctrl <- lme4::glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=200000))
            .full <- lme4::glmer(cbind(succ, fail) ~ sentiment + (1|interviewer_id) +
                                                     (1|participant_id) + (1|obs_id),
                                 data=dat, family=binomial, control=.ctrl)
            .null <- lme4::glmer(cbind(succ, fail) ~ 1 + (1|interviewer_id) +
                                                     (1|participant_id) + (1|obs_id),
                                 data=dat, family=binomial, control=.ctrl)
            .co <- summary(.full)$coefficients
            .rn <- rownames(.co)
            .vc <- as.data.frame(lme4::VarCorr(.full))
            .an <- anova(.null, .full)
        ''')

    out = _glmer_readout(out, ".co", ".rn", ".vc", ".an")
    if out["method"] == "no_term":
        return out
    boundary = [nm for nm, key in (("interviewer", "sigma2_interviewer"),
                                   ("participant", "sigma2_participant"))
                if np.isfinite(out[key]) and out[key] < 1e-8]
    out["method"] = (f"glmer_olre_singular({'+'.join(boundary)})" if boundary
                     else "glmer_olre")
    return out


def fit_focal_binary(sub: pd.DataFrame) -> dict:
    """Bernoulli GLMM on the per-turn 0/1 indicator: did the dyad co-smile at
    all? No OLRE -- a single-trial Bernoulli cannot be overdispersed."""
    import rpy2.robjects as ro

    out = _base_out(sub)
    out["engine"] = "lme4::glmer(rpy2)"
    if not _fittable(sub):
        out["method"] = "not_fittable"
        return out

    s = sub[sub["sentiment"].isin(FOCAL)].copy().reset_index(drop=True)
    s["anyco"] = (s[K_COL].astype(float) > 0).astype(int)
    for c in ("sentiment", "interviewer_id", "participant_id"):
        s[c] = s[c].astype(str)
    if s["anyco"].nunique() < 2:
        out["method"] = "no_variance_in_binary_dv"
        return out
    _to_r(s[["anyco", "sentiment", "interviewer_id", "participant_id"]], "datb")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ro.r('''
            datb$sentiment      <- relevel(factor(datb$sentiment), ref="negative")
            datb$interviewer_id <- factor(datb$interviewer_id)
            datb$participant_id <- factor(datb$participant_id)
            .bctrl <- lme4::glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=200000))
            .bfull <- lme4::glmer(anyco ~ sentiment + (1|interviewer_id) + (1|participant_id),
                                  data=datb, family=binomial, control=.bctrl)
            .bnull <- lme4::glmer(anyco ~ 1 + (1|interviewer_id) + (1|participant_id),
                                  data=datb, family=binomial, control=.bctrl)
            .bco <- summary(.bfull)$coefficients
            .brn <- rownames(.bco)
            .bvc <- as.data.frame(lme4::VarCorr(.bfull))
            .ban <- anova(.bnull, .bfull)
        ''')

    out = _glmer_readout(out, ".bco", ".brn", ".bvc", ".ban")
    if out["method"] != "no_term":
        out["method"] = "glmer_bernoulli"
    return out


# ---------------------------------------------------------------------------
# Descriptives and robust inference
# ---------------------------------------------------------------------------
def descriptives_freq(sub_all: pd.DataFrame, col: str = FREQ_COL,
                      binarize: bool = False) -> dict:
    """Per-sentiment mean/SD/n over ALL turns, neutral included."""
    d: dict = {}
    for s in SENTS_DESC:
        v = sub_all.loc[sub_all["sentiment"] == s, col].astype(float)
        v = v[np.isfinite(v)]
        if binarize:
            v = (v > 0).astype(float)
        d[f"mean_freq_{s}"] = float(v.mean()) if len(v) else math.nan
        d[f"sd_freq_{s}"] = float(v.std(ddof=1)) if len(v) > 1 else math.nan
        d[f"n_{s}"] = int(len(v))
    return d


def descriptives(sub_all: pd.DataFrame, z_col: str) -> dict:
    """Per-sentiment descriptives on the Fisher-z DV, plus mean r = tanh(mean z)."""
    d: dict = {}
    for s in SENTS_DESC:
        v = sub_all.loc[sub_all["sentiment"] == s, z_col]
        v = v[np.isfinite(v)]
        mz = float(v.mean()) if len(v) else math.nan
        d[f"mean_z_{s}"] = mz
        d[f"mean_r_{s}"] = float(np.tanh(mz)) if np.isfinite(mz) else mz
        d[f"sd_z_{s}"] = float(v.std(ddof=1)) if len(v) > 1 else math.nan
        d[f"n_{s}"] = int(len(v))
    return d


def wild_cluster_bootstrap(y, D, clusters, n_boot: int = 3000,
                           seed: int = 42, alpha: float = 0.05) -> dict:
    """Wild cluster bootstrap for the slope in y ~ 1 + D (Cameron, Gelbach &
    Miller, 2008). With G <= 12 all 2^G sign vectors are enumerated, so p is
    exact but granular: the smallest attainable two-sided p is 2/2^G."""
    y = np.asarray(y, float); D = np.asarray(D, float); cl = np.asarray(clusters)
    ok = np.isfinite(y) & np.isfinite(D)
    y, D, cl = y[ok], D[ok], cl[ok]
    n = len(y)
    out = {"wcb_p": math.nan, "wcb_ci_lo": math.nan, "wcb_ci_hi": math.nan,
           "wcb_beta": math.nan, "wcb_n_clusters": 0, "wcb_exact": False}
    if n < 4 or len(np.unique(D)) < 2:
        return out
    X = np.column_stack([np.ones(n), D])
    uniq = np.unique(cl); G = len(uniq); idx = np.searchsorted(uniq, cl)
    out["wcb_n_clusters"] = int(G)
    if G < 2:
        return out
    try:
        XtXi = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return out
    beta = XtXi @ (X.T @ y)
    out["wcb_beta"] = float(beta[1])

    def crv(u):
        meat = np.zeros((2, 2))
        for g in range(G):
            m = idx == g
            sg = X[m].T @ u[m]
            meat += np.outer(sg, sg)
        c = (G / (G - 1.0)) * ((n - 1.0) / (n - 2.0))
        return c * XtXi @ meat @ XtXi

    se = math.sqrt(crv(y - X @ beta)[1, 1])
    if not (np.isfinite(se) and se > 0):
        return out
    t_obs = beta[1] / se
    ybar = y.mean()
    fit_r, res_r = np.full(n, ybar), y - ybar     # null DGP, for the p-value
    res_u = y - X @ beta                          # unrestricted, for the CI
    if G <= 12:
        combos = np.array(list(itertools.product([1.0, -1.0], repeat=G)))
        out["wcb_exact"] = True
    else:
        combos = np.random.default_rng(seed).choice([1.0, -1.0], size=(n_boot, G))

    cnt, tstar_u = 0, []
    for w in combos:
        ys = fit_r + res_r * w[idx]
        bs = XtXi @ (X.T @ ys)
        ss = crv(ys - X @ bs)[1, 1]
        if ss > 0 and abs(bs[1] / math.sqrt(ss)) >= abs(t_obs) - 1e-12:
            cnt += 1
        yu = X @ beta + res_u * w[idx]
        bu = XtXi @ (X.T @ yu)
        su = crv(yu - X @ bu)[1, 1]
        if su > 0:
            tstar_u.append((bu[1] - beta[1]) / math.sqrt(su))
    out["wcb_p"] = cnt / len(combos)
    if tstar_u:
        qhi, qlo = np.quantile(tstar_u, [1 - alpha / 2, alpha / 2])
        out["wcb_ci_lo"] = float(beta[1] - se * qhi)
        out["wcb_ci_hi"] = float(beta[1] - se * qlo)
    return out


# ---------------------------------------------------------------------------
# Per-measure drivers
# ---------------------------------------------------------------------------
def analyse_simul_smile(df: pd.DataFrame) -> list[dict]:
    """Simultaneous smiling: the binomial GLMM (reported) plus the binary
    reading of the same data (secondary)."""
    if not {FREQ_COL, N_COL, K_COL} <= set(df.columns):
        print(f"[analysis2_stats] {FREQ_COL}/{N_COL}/{K_COL} absent — skipping.")
        return []
    sub_all = df[(df["level"] == "synchrony") & (df["au"] == SYNC_MEASURE)].copy()
    sub_all = sub_all[np.isfinite(sub_all[FREQ_COL]) & (sub_all[N_COL] > 0)]
    if sub_all.empty:
        return []
    focal = sub_all[sub_all["sentiment"].isin(FOCAL)].copy()

    res = fit_focal_binomial(focal)
    # d, not q: the DV is a proportion, i.e. a mean.
    res["cohens_d"] = cohens_d_pooled(
        focal.loc[focal["sentiment"] == "positive", FREQ_COL].to_numpy(),
        focal.loc[focal["sentiment"] == "negative", FREQ_COL].to_numpy())
    res.update(wild_cluster_bootstrap(
        focal[FREQ_COL].to_numpy(),
        (focal["sentiment"] == "positive").to_numpy(float),
        focal["interviewer_id"].to_numpy()))
    res.update(_random_slope_check_lme4(
        focal.assign(dvval=focal[FREQ_COL].astype(float)), "dvval"))

    rows = [{"level": "synchrony", "au": "simul_smile", "measure": "simul_smile",
             "transform": "identity", **descriptives_freq(sub_all), **res}]

    binary = fit_focal_binary(focal)
    binary["cohens_d"] = cohens_d_pooled(
        (focal.loc[focal["sentiment"] == "positive", K_COL].astype(float) > 0).astype(float).to_numpy(),
        (focal.loc[focal["sentiment"] == "negative", K_COL].astype(float) > 0).astype(float).to_numpy())
    binary.update(wild_cluster_bootstrap(
        (focal[K_COL].astype(float) > 0).astype(float).to_numpy(),
        (focal["sentiment"] == "positive").to_numpy(float),
        focal["interviewer_id"].to_numpy()))
    binary.update(_random_slope_check_lme4(
        focal.assign(dvval=(focal[K_COL].astype(float) > 0).astype(float)), "dvval"))
    rows.append({"level": "synchrony", "au": "simul_smile_any",
                 "measure": "simul_smile_any", "transform": "binary",
                 **descriptives_freq(sub_all, col=K_COL, binarize=True), **binary})
    return rows


def analyse_cc(df: pd.DataFrame, value_col: str, measure_name: str) -> list[dict]:
    """Gaussian LMM on Fisher-z of |r| for a correlation-valued per-turn measure."""
    if value_col not in df.columns:
        print(f"[analysis2_stats] '{value_col}' absent — skipping '{measure_name}'.")
        return []
    sub_all = df[(df["level"] == "synchrony") & (df["au"] == SYNC_MEASURE)].copy()
    sub_all = sub_all[np.isfinite(sub_all[value_col])]   # drops flat/undefined turns
    if sub_all.empty:
        return []

    sub_all["dvval"] = fisher_z(sub_all[value_col].astype(float).abs())
    focal = sub_all[sub_all["sentiment"].isin(FOCAL)]

    res = fit_focal(focal, "dvval")
    desc = descriptives(sub_all, "dvval")
    # q = mean(z_pos) - mean(z_neg), the same quantity as the LMM estimate.
    res["cohens_q"] = cohens_q(desc["mean_r_positive"], desc["mean_r_negative"])

    res.update(wild_cluster_bootstrap(
        focal["dvval"].to_numpy(),
        (focal["sentiment"] == "positive").to_numpy(float),
        focal["interviewer_id"].to_numpy()))
    res.update(_random_slope_check_lme4(focal, "dvval"))
    return [{"level": "synchrony", "au": SYNC_MEASURE, "measure": measure_name,
             "transform": "fisherz", **desc, **res}]


# Union across measures; unpopulated cells are left NaN by the reindex.
FIELDNAMES = [
    "measure", "level", "au", "transform",
    "mean_freq_positive", "sd_freq_positive",
    "mean_freq_neutral", "sd_freq_neutral",
    "mean_freq_negative", "sd_freq_negative",
    "mean_r_positive", "mean_z_positive", "sd_z_positive",
    "mean_r_neutral", "mean_z_neutral", "sd_z_neutral",
    "mean_r_negative", "mean_z_negative", "sd_z_negative",
    "n_positive", "n_neutral", "n_negative",
    "n_obs", "n_participants", "n_interviewers",
    "estimate", "se", "z", "p_wald", "lrt_stat", "p_lrt", "lrt_source",
    "odds_ratio", "cohens_d", "cohens_q",
    "sigma2_interviewer", "sigma2_participant",
    "slope_var_participant", "slope_lrt_stat", "slope_p_lrt", "slope_source",
    "wcb_beta", "wcb_p", "wcb_ci_lo", "wcb_ci_hi", "wcb_n_clusters", "wcb_exact",
    "converged", "engine", "method",
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Models the per-turn synchrony measures by sentiment",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    ap.add_argument("--sync-turns", "--clc-turns", dest="sync_turns", required=True,
                    help="bindung_synchrony_turns.csv (output of analysis2.py)")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Do NOT pre-filter here: a turn can have a defined co-smile share while its
    # correlation is undefined. Each analyse_* helper filters for itself.
    df = pd.read_csv(args.sync_turns)
    n_cc = int(np.isfinite(df["cc_r_peak"]).sum()) if "cc_r_peak" in df.columns else 0
    n_ss = int((df[N_COL] > 0).sum()) if N_COL in df.columns else 0
    print(f"Synchrony turns: {len(df)} total ({n_cc} with a defined "
          f"cross-correlation; {n_ss} with co-smile frame counts)")

    rows = analyse_simul_smile(df) + analyse_cc(df, "cc_r_peak", "duchenne_smile")

    sync_lmm = out_dir / "bindung_synchrony_lmm.csv"
    pd.DataFrame(rows).reindex(columns=FIELDNAMES).to_csv(sync_lmm, index=False)
    print(f"WROTE {sync_lmm} ({len(rows)} rows)")



if __name__ == "__main__":
    main()
