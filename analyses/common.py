"""Signal construction and per-turn measures shared by Analyses 1-3.

Two per-frame signals per partner, from the Duchenne pair AU06 and AU12:
    S(t) = mean(AU06_r, AU12_r)   intensity, 0-5
    P(t) = AU06_c AND AU12_c      presence, binary
"""
from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np

from utils import (
    SENTIMENTS,
    load_openface_csv,
    parse_turns,
    choose_labeled_file,
    write_csv,
    resolve_path,
    build_common_grid,
    interpolate_series,
    nearest_series,
)

MIN_BUCKET_TURNS: int = 0
BLRI_TARGET: str = "T5_BLRI_ges_Pr"

SMILE_HZ: float = 30.0                 # common grid for the dyadic measures
SMILE_INTENSITY_COLS: tuple[str, ...] = (
    "client_smile_intensity", "therapist_smile_intensity",
)
SMILE_FREQ_COLS: tuple[str, ...] = (
    "client_smile_freq", "therapist_smile_freq",
)


def smile_intensity_series(of: dict) -> np.ndarray:
    """S(t): nan-aware mean of AU06_r and AU12_r; empty if unavailable."""
    aus = of.get("aus", {})
    a06 = aus.get("AU06_r", np.array([], dtype=float))
    a12 = aus.get("AU12_r", np.array([], dtype=float))
    if a06.size == 0 or a12.size == 0 or a06.size != a12.size:
        return np.array([], dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.vstack([a06, a12]), axis=0)


def duchenne_presence_series(of: dict) -> np.ndarray:
    """P(t): 1.0 where AU06_c and AU12_c are both present, NaN where missing."""
    aus_c = of.get("aus_c", {})
    c06 = aus_c.get("AU06_c", np.array([], dtype=float))
    c12 = aus_c.get("AU12_c", np.array([], dtype=float))
    if c06.size == 0 or c12.size == 0 or c06.size != c12.size:
        return np.array([], dtype=float)
    with np.errstate(invalid="ignore"):
        both = ((c06 >= 0.5) & (c12 >= 0.5)).astype(float)
    both[~(np.isfinite(c06) & np.isfinite(c12))] = math.nan
    return both


# ---------------------------------------------------------------------------
# Analysis 1 — per-turn individual smiling
# ---------------------------------------------------------------------------
def _turn_mean(ts: np.ndarray, sig: np.ndarray, start_s: float, end_s: float,
               min_frames: int) -> float:
    """Mean of ``sig`` over frames with a timestamp in [start_s, end_s)."""
    if ts.size == 0 or sig.size != ts.size:
        return math.nan
    vals = sig[(ts >= start_s) & (ts < end_s)]
    vals = vals[np.isfinite(vals)]
    if vals.size < max(1, min_frames):
        return math.nan
    return float(vals.mean())


def compute_turn_rows(
    pid: str,
    tid: str,
    p_of: dict,
    t_of: dict,
    turns: list[dict],
    min_frames_per_turn: int,
) -> list[dict]:
    """Per-turn mean of S(t) and of P(t) per partner, on each person's own
    frames (the dyadic-grid marginals come from compute_simul_smile_by_turn)."""
    p_S = smile_intensity_series(p_of)
    t_S = smile_intensity_series(t_of)
    p_P = duchenne_presence_series(p_of)
    t_P = duchenne_presence_series(t_of)
    p_ts = p_of.get("timestamps", np.array([], dtype=float))
    t_ts = t_of.get("timestamps", np.array([], dtype=float))

    return [{
        "patient_id": pid, "therapist_id": tid,
        "turn_index": turn["turn_index"],
        "sentiment": turn["sentiment"],
        "start_s": turn["start_s"], "end_s": turn["end_s"],
        "client_smile_intensity": _turn_mean(p_ts, p_S, turn["start_s"], turn["end_s"], min_frames_per_turn),
        "therapist_smile_intensity": _turn_mean(t_ts, t_S, turn["start_s"], turn["end_s"], min_frames_per_turn),
        "client_smile_freq": _turn_mean(p_ts, p_P, turn["start_s"], turn["end_s"], min_frames_per_turn),
        "therapist_smile_freq": _turn_mean(t_ts, t_P, turn["start_s"], turn["end_s"], min_frames_per_turn),
    } for turn in turns]


# ---------------------------------------------------------------------------
# Analysis 2a — cross-lagged correlation of the two intensity signals
# ---------------------------------------------------------------------------
def _signed_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Signed Pearson r on the finite overlap; NaN if < 3 frames or either
    signal is flat (a partner who never smiles). NaN, not dropped, so the
    share of such turns stays reportable."""
    m = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(m) < 3:
        return math.nan
    x, y = a[m], b[m]
    if x.std() < 1e-12 or y.std() < 1e-12:
        return math.nan
    r = float(np.corrcoef(x, y)[0, 1])
    return r if np.isfinite(r) else math.nan


def _cross_corr_peak(pv: np.ndarray, tv: np.ndarray, max_lag_frames: int
                     ) -> tuple[float, int]:
    """Signed r at the |r|-maximising lag in [-K, +K], and that lag.
    k > 0 = participant leads."""
    n = pv.size
    best_r, best_k, best_abs = math.nan, 0, -1.0
    K = max(0, min(int(max_lag_frames), n - 3))
    for k in range(-K, K + 1):
        a, b = (pv[0:n - k], tv[k:n]) if k >= 0 else (pv[-k:n], tv[0:n + k])
        r = _signed_corr(a, b)
        if np.isfinite(r) and abs(r) > best_abs:
            best_abs, best_r, best_k = abs(r), r, k
    return best_r, best_k


MAX_LAG_S: float = 2.5

MIN_TURN_LAG_MULTIPLE: float = 4.0
MIN_TURN_SECONDS: float = MIN_TURN_LAG_MULTIPLE * MAX_LAG_S   # = 10.0 s


def compute_smile_cc_by_turn(
    p_of: dict,
    t_of: dict,
    turns: list[dict],
    min_frames_per_turn: int,
    hz: float = SMILE_HZ,
    max_lag_s: float = MAX_LAG_S,
    min_turn_seconds: float = MIN_TURN_SECONDS,
) -> dict[int, dict[str, float]]:
    """Per-turn cross-lagged correlation of the two S(t) signals, on a common
    grid. Emits cc_r (lag 0), cc_r_peak (the CLC score, modelled in Analyses
    2-3), peak_lag_s and n_frames.

    Stored SIGNED; the sign is removed downstream before averaging, or in-phase
    and anti-phase turns cancel. Turns below min_turn_seconds yield NaN, not
    dropped, so they stay available to the simultaneous-smiling measure.
    """
    nan_row = {"cc_r": math.nan, "peak_lag_s": math.nan,
               "cc_r_peak": math.nan, "n_frames": 0}
    out: dict[int, dict[str, float]] = {t["turn_index"]: dict(nan_row) for t in turns}

    p_S = smile_intensity_series(p_of)
    t_S = smile_intensity_series(t_of)
    if p_S.size == 0 or t_S.size == 0:
        return out

    grid = build_common_grid(p_of["timestamps"], t_of["timestamps"], hz=hz)
    if grid.size < 3:
        return out
    p_grid = interpolate_series(p_of["timestamps"], p_S, grid)
    t_grid = interpolate_series(t_of["timestamps"], t_S, grid)
    if p_grid.size != grid.size or t_grid.size != grid.size:
        return out

    max_lag_frames = int(round(max_lag_s * hz))
    for turn in turns:
        i0 = int(np.searchsorted(grid, turn["start_s"], "left"))
        i1 = int(np.searchsorted(grid, turn["end_s"], "right"))
        pv, tv = p_grid[i0:i1], t_grid[i0:i1]
        n = int(pv.size)
        if n < max(3, min_frames_per_turn) or \
                float(turn["end_s"] - turn["start_s"]) < min_turn_seconds:
            out[turn["turn_index"]] = dict(nan_row, n_frames=n)
            continue
        r0 = _signed_corr(pv, tv)
        r_peak, k_peak = _cross_corr_peak(pv, tv, max_lag_frames)
        out[turn["turn_index"]] = {
            "cc_r": r0,
            "peak_lag_s": float(k_peak / hz) if np.isfinite(r_peak) else math.nan,
            "cc_r_peak": r_peak,
            "n_frames": n,
        }
    return out


# ---------------------------------------------------------------------------
# Analysis 2b — simultaneous smiling (pre-registered primary measure)
# ---------------------------------------------------------------------------
def compute_simul_smile_by_turn(
    p_of: dict,
    t_of: dict,
    turns: list[dict],
    min_frames_per_turn: int,
    hz: float = SMILE_HZ,
) -> dict[int, dict[str, float]]:
    """Per-turn simultaneous smiling from the two binary P(t) signals, resampled
    NEAREST NEIGHBOUR (a binary signal must not be linearly blended). Emits the
    co-smile count, the turn's frame count and their ratio."""
    nan_row = {"simul_smile_frames": 0, "simul_smile_n_frames": 0,
               "simul_smile_freq": math.nan}
    out: dict[int, dict[str, float]] = {t["turn_index"]: dict(nan_row) for t in turns}

    p_P = duchenne_presence_series(p_of)
    t_P = duchenne_presence_series(t_of)
    if p_P.size == 0 or t_P.size == 0:
        return out

    grid = build_common_grid(p_of["timestamps"], t_of["timestamps"], hz=hz)
    if grid.size < 3:
        return out
    p_grid = nearest_series(p_of["timestamps"], p_P, grid)
    t_grid = nearest_series(t_of["timestamps"], t_P, grid)
    if p_grid.size != grid.size or t_grid.size != grid.size:
        return out

    with np.errstate(invalid="ignore"):
        cosmile = ((p_grid >= 0.5) & (t_grid >= 0.5)).astype(float)

    for turn in turns:
        i0 = int(np.searchsorted(grid, turn["start_s"], "left"))
        i1 = int(np.searchsorted(grid, turn["end_s"], "right"))
        seg = cosmile[i0:i1]
        n = int(seg.size)
        if n < max(3, min_frames_per_turn):
            out[turn["turn_index"]] = dict(nan_row, simul_smile_n_frames=n)
            continue
        out[turn["turn_index"]] = {
            "simul_smile_frames": int(np.nansum(seg)),
            "simul_smile_n_frames": n,
            "simul_smile_freq": float(np.nansum(seg)) / n,
        }
    return out


# ---------------------------------------------------------------------------
# CLI, job construction, execution
# ---------------------------------------------------------------------------
def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-model", required=True)
    parser.add_argument("--labeled-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-key", default="sentiment_label")
    parser.add_argument("--labeled-suffix", default="_sentiment")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--min-frames-per-turn", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Workers for processing patients in parallel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-bucket-turns", type=int, default=MIN_BUCKET_TURNS,
                        help="Min turns per sentiment bucket for inclusion; "
                             f"0 disables the filter (default: {MIN_BUCKET_TURNS})")
    parser.add_argument("--blri-target", type=str, default=BLRI_TARGET,
                        help=f"BLRI column to predict (default: {BLRI_TARGET})")
    parser.add_argument("--verbose", action="store_true")


def load_patient_data(job: dict) -> dict:
    """Both OpenFace CSVs and the labelled turns for one interview."""
    repo_root = Path(job["repo_root"])
    patient_of_path = resolve_path(job["patient_openface"], repo_root)
    therapist_of_path = resolve_path(job["therapist_openface"], repo_root)
    labeled_json = Path(job["labeled_json"])

    if patient_of_path is None or therapist_of_path is None or not labeled_json.exists():
        return {"ok": False, "reason": "missing_paths"}
    try:
        p_of = load_openface_csv(patient_of_path, job["confidence_threshold"])
        t_of = load_openface_csv(therapist_of_path, job["confidence_threshold"])
        turns = parse_turns(labeled_json, job["label_key"])
    except Exception as exc:
        return {"ok": False, "reason": f"exception:{type(exc).__name__}:{exc}"}

    if p_of["timestamps"].size == 0 or t_of["timestamps"].size == 0:
        return {"ok": False, "reason": "empty_openface"}
    if not turns:
        return {"ok": False, "reason": "no_labeled_turns"}
    return {"ok": True, "patient_id": job["patient_id"],
            "therapist_id": job["therapist_id"],
            "p_of": p_of, "t_of": t_of, "turns": turns}


def process_turns_job(job: dict) -> dict:
    """Per-patient job for Analysis 1: per-turn individual smiling."""
    data = load_patient_data(job)
    if not data["ok"]:
        return data
    return {
        "ok": True,
        "patient_id": data["patient_id"],
        "therapist_id": data["therapist_id"],
        "turn_rows": compute_turn_rows(
            data["patient_id"], data["therapist_id"],
            data["p_of"], data["t_of"], data["turns"],
            job["min_frames_per_turn"]),
    }


def build_jobs(
    dm: dict,
    labeled_index: dict[str, list],
    args: argparse.Namespace,
    repo_root: Path,
    logger: logging.Logger,
) -> list[dict]:
    """One job per Bindung interview with a transcript and a labelled file
    (a null transcript marks a dyad excluded upstream for failed diarisation)."""
    jobs: list[dict] = []
    missing_labeled = transcript_null = 0
    dm_bindung_pids: set[str] = set()

    for interview in dm.get("interviews", []):
        therapist_id = str(interview.get("therapist", {}).get("therapist_id", "")).strip()
        patient_id = str(interview.get("patient", {}).get("patient_id", "")).strip()
        bindung = (interview.get("types", {}) or {}).get("bindung")
        if not bindung:
            continue
        dm_bindung_pids.add(patient_id.upper())
        if bindung.get("transcript") is None:
            transcript_null += 1
            logger.debug("Transcript null for patient %s — skipped", patient_id)
            continue
        labeled_path = choose_labeled_file(patient_id, labeled_index)
        if labeled_path is None:
            missing_labeled += 1
            logger.debug("No labeled file for patient %s — skipped", patient_id)
            continue
        jobs.append({
            "repo_root": str(repo_root),
            "therapist_id": therapist_id,
            "patient_id": patient_id,
            "patient_openface": str(bindung.get("patient_openface", "") or ""),
            "therapist_openface": str(bindung.get("therapist_openface", "") or ""),
            "labeled_json": str(labeled_path),
            "label_key": args.label_key,
            "confidence_threshold": args.confidence_threshold,
            "min_frames_per_turn": args.min_frames_per_turn,
        })

    orphaned = set(labeled_index.keys()) - dm_bindung_pids
    if orphaned:
        logger.warning("Labeled PIDs absent from data_model (%d): %s",
                       len(orphaned), sorted(orphaned))
    logger.info("Jobs=%d | transcript_null=%d | missing_labeled=%d | orphaned_pids=%d",
                len(jobs), transcript_null, missing_labeled, len(orphaned))
    return jobs


def run_parallel_jobs(
    jobs: list[dict],
    job_fn: Callable[[dict], dict],
    n_jobs: int,
    logger: logging.Logger,
) -> list[dict]:
    """Execute per-patient jobs serially (n_jobs == 1) or in a process pool."""
    all_results: list[dict] = []
    fail_reasons: dict[str, int] = defaultdict(int)

    def record(r: dict) -> None:
        if r.get("ok"):
            all_results.append(r)
        else:
            fail_reasons[r.get("reason", "unknown")] += 1

    if n_jobs == 1:
        for i, job in enumerate(jobs, 1):
            record(job_fn(job))
            if i % 5 == 0 or i == len(jobs):
                logger.info("Progress: %d/%d", i, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(job_fn, job) for job in jobs]
            for done, fut in enumerate(as_completed(futures), 1):
                record(fut.result())
                if done % 5 == 0 or done == len(futures):
                    logger.info("Progress: %d/%d", done, len(futures))

    logger.info("Successful: %d/%d", len(all_results), len(jobs))
    if fail_reasons:
        logger.info("Failures: %s", dict(fail_reasons))
    return all_results


def apply_inclusion_filter(
    all_turn_rows: list[dict],
    all_results: list[dict],
    logger: logging.Logger,
    min_bucket_turns: int = MIN_BUCKET_TURNS,
) -> tuple[list[dict], list[dict]]:
    """Keep participants with >= min_bucket_turns in every sentiment category.
    The reported analyses use 0; the requirement bites only in Analysis 3."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in SENTIMENTS})
    for row in all_turn_rows:
        sent = str(row.get("sentiment", "")).strip().lower()
        if sent in SENTIMENTS:
            counts[str(row.get("patient_id", ""))][sent] += 1

    eligible = {pid for pid, c in counts.items()
                if all(c[s] >= min_bucket_turns for s in SENTIMENTS)}
    n_before = len({r["patient_id"] for r in all_turn_rows})
    all_turn_rows = [r for r in all_turn_rows if r["patient_id"] in eligible]
    all_results = [r for r in all_results if r["patient_id"] in eligible]
    n_after = len({r["patient_id"] for r in all_turn_rows})
    logger.info("Inclusion filter (min_bucket_turns=%d): %d/%d patients kept",
                min_bucket_turns, n_after, n_before)
    return all_turn_rows, all_results


def write_turns_csv(all_turn_rows: list[dict], output_dir: Path,
                    logger: logging.Logger) -> None:
    """Write bindung_au_sentiment_turns.csv."""
    turn_cols = (
        ["patient_id", "therapist_id", "turn_index", "sentiment", "start_s", "end_s"]
        + list(SMILE_INTENSITY_COLS) + list(SMILE_FREQ_COLS)
    )
    turn_csv = output_dir / "bindung_au_sentiment_turns.csv"
    write_csv(turn_csv, turn_cols, all_turn_rows)
    logger.info("WROTE %s (%d rows)", turn_csv, len(all_turn_rows))
