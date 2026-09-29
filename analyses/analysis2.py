"""Analysis 2, step 1 — per-turn dyadic smile synchrony.

Writes bindung_synchrony_turns.csv: one row per (participant, turn) carrying
simultaneous smiling and the
cross-lagged correlation of the two smile-intensity signals.
Models are fitted by analysis2_stats.py.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from utils import setup_logger, build_labeled_index, write_csv
from common import (
    add_common_args,
    build_jobs,
    run_parallel_jobs,
    load_patient_data,
    compute_turn_rows,
    compute_smile_cc_by_turn,
    compute_simul_smile_by_turn,
    apply_inclusion_filter,
    MAX_LAG_S,
    MIN_TURN_SECONDS,
    SMILE_HZ,
)

SYNC_FIELDNAMES = [
    "participant_id", "interviewer_id", "turn_index", "sentiment",
    "duration_s", "n_frames", "level", "au",
    "cc_r", "peak_lag_s", "cc_r_peak",
    "simul_smile_frames", "simul_smile_n_frames", "simul_smile_freq",
]


def process_sync_job(job: dict) -> dict:
    """Both synchrony measures for one interview. The duration floor is applied
    inside the cross-correlation, not by dropping rows, so simultaneous smiling
    keeps every turn."""
    data = load_patient_data(job)
    if not data["ok"]:
        return data

    pid, tid = data["patient_id"], data["therapist_id"]

    # Only for the inclusion filter's per-sentiment bookkeeping.
    turn_rows = compute_turn_rows(
        pid, tid, data["p_of"], data["t_of"], data["turns"], job["min_frames_per_turn"]
    )
    cc_by_turn = compute_smile_cc_by_turn(
        data["p_of"], data["t_of"], data["turns"], job["min_frames_per_turn"],
        hz=job["smile_hz"], max_lag_s=job["max_lag_s"],
        min_turn_seconds=job["clc_min_turn_seconds"],
    )
    ss_by_turn = compute_simul_smile_by_turn(
        data["p_of"], data["t_of"], data["turns"], job["min_frames_per_turn"],
        hz=job["smile_hz"],
    )

    nan = float("nan")
    sync_rows = []
    for turn in data["turns"]:
        cc = cc_by_turn.get(turn["turn_index"], {})
        ss = ss_by_turn.get(turn["turn_index"], {})
        sync_rows.append({
            "participant_id": pid,
            "interviewer_id": tid,
            "turn_index": int(turn["turn_index"]),
            "sentiment": str(turn["sentiment"]).strip().lower(),
            "duration_s": round(float(turn["end_s"] - turn["start_s"]), 3),
            "n_frames": int(cc.get("n_frames", 0)),
            "level": "synchrony",
            "au": "duchenne_smile",
            "cc_r": cc.get("cc_r", nan),
            "peak_lag_s": cc.get("peak_lag_s", nan),
            "cc_r_peak": cc.get("cc_r_peak", nan),
            "simul_smile_frames": int(ss.get("simul_smile_frames", 0)),
            "simul_smile_n_frames": int(ss.get("simul_smile_n_frames", 0)),
            "simul_smile_freq": ss.get("simul_smile_freq", nan),
        })

    return {"ok": True, "patient_id": pid, "therapist_id": tid,
            "turn_rows": turn_rows, "sync_rows": sync_rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-turn dyadic smile synchrony by sentiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_common_args(parser)
    parser.add_argument("--max-lag-s", type=float, default=MAX_LAG_S,
                        help="Half-width (s) of the lag search for the "
                             f"cross-lagged correlation (default: {MAX_LAG_S}; "
                             "see common.MAX_LAG_S for the rationale)")
    parser.add_argument("--clc-min-turn-seconds", type=float, default=MIN_TURN_SECONDS,
                        help="Minimum turn duration (s) for the CROSS-CORRELATION "
                             "only; shorter turns yield NaN there but are kept for "
                             f"simultaneous smiling (default: {MIN_TURN_SECONDS}, "
                             "Cappella's rule of four times the maximum lag)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "bindung_synchrony.log", verbose=args.verbose)
    logger.info("Starting — per-turn dyadic smile synchrony")
    logger.info("Args: %s", vars(args))

    with Path(args.data_model).open("r", encoding="utf-8") as f:
        dm = yaml.safe_load(f)

    labeled_index = build_labeled_index(Path(args.labeled_dir), suffix=args.labeled_suffix)
    logger.info("Labeled files indexed: %d participant IDs", len(labeled_index))

    jobs = build_jobs(dm, labeled_index, args, repo_root, logger)
    for job in jobs:
        job["clc_min_turn_seconds"] = args.clc_min_turn_seconds
        job["max_lag_s"] = args.max_lag_s
        job["smile_hz"] = SMILE_HZ
    if not jobs:
        logger.warning("No jobs. Exiting.")
        return

    all_results = run_parallel_jobs(jobs, process_sync_job, args.n_jobs, logger)
    if not all_results:
        logger.error("No successful results. Exiting.")
        return

    all_turn_rows: list[dict] = []
    for res in all_results:
        all_turn_rows.extend(res.get("turn_rows", []))
    logger.info("Total labeled client turns: %d across %d participants",
                len(all_turn_rows), len(all_results))

    # No BLRI outcome here, so no restriction to participants with questionnaire data.
    all_turn_rows, all_results = apply_inclusion_filter(
        all_turn_rows, all_results, logger, min_bucket_turns=args.min_bucket_turns,
    )

    sync_rows: list[dict] = []
    for res in all_results:
        sync_rows.extend(res.get("sync_rows", []))

    n_total = len(sync_rows)
    n_cc = sum(1 for r in sync_rows if np.isfinite(r["cc_r_peak"]))
    logger.info("Per-turn rows: %d | defined cross-correlation: %d (%.1f%%)",
                n_total, n_cc, 100.0 * n_cc / n_total if n_total else 0.0)

    sync_csv = output_dir / "bindung_synchrony_turns.csv"
    write_csv(sync_csv, SYNC_FIELDNAMES, sync_rows)
    logger.info("WROTE %s (%d rows)", sync_csv, len(sync_rows))
    logger.info("Now run:  python analysis2_stats.py --sync-turns %s --output-dir %s",
                sync_csv, output_dir)
    logger.info("Done in %.1f s", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
