import argparse
import json
import logging
import re
import time
from pathlib import Path

import yaml
from openai import OpenAI

SYSTEM_PROMPT = (
    "You are an expert clinical psychologist coding transcripts. "
    "Your task is to classify the emotional sentiment of a single participant speech turn "
    "from a relationship interview — a structured attachment-relationship reflection where "
    "the client describes the quality of their close personal relationships. "
    "Reply with exactly one word: positive, neutral, negative, or other. "
    "Use 'other' when the turn is not about the relationship itself (e.g. "
    "technical issues with the video call, scheduling, or small talk unrelated to "
    "attachment). Output nothing else."
)

# Few-shot examples shown to the model before every real turn.
# Keep them short and representative of the four categories.
# All relationship examples are tailored to a Bindung (attachment) interview
# in which the client reflects on the quality of close personal bonds.
FEW_SHOT_EXAMPLES = [
    # --- positive: warmth, security, trust in the relationship ---
    (
        "She was always there for me when I came home from school. "
        "I felt really safe and loved in that relationship.",
        "positive",
    ),
    (
        "I remember my partner holding my hand during a really hard time. "
        "It made me feel like I could count on them no matter what.",
        "positive",
    ),
    # --- neutral: factual or ambivalent description of the bond ---
    (
        "We didn't really talk much about feelings. "
        "The relationship was just, you know, normal I guess.",
        "neutral",
    ),
    (
        "I would say we got along okay. There were good days and bad days, "
        "nothing that really stands out.",
        "neutral",
    ),
    # --- negative: distress, insecurity, conflict in the relationship ---
    (
        "He had a terrible temper. When he got angry I would hide in my room. "
        "I never felt safe in that relationship.",
        "negative",
    ),
    (
        "There were a lot of arguments. I felt like I was always walking on "
        "eggshells around her, afraid of being rejected.",
        "negative",
    ),
    # --- other: not about the relationship (technical, scheduling, etc.) ---
    (
        "Sorry, I think my connection dropped for a second. Can you hear me now?",
        "other",
    ),
    (
        "Hold on, let me adjust my camera. The Teams window is a bit glitchy today.",
        "other",
    ),
]

SENTIMENT_LABELS = {"positive", "neutral", "negative", "other"}
CLIENT_TOKENS = {"client", "patient", "pr", "p"}


THERAPIST_TOKENS = {"therapist", "in", "therapeut", "therapeutin", "t"}


def build_prompt(text: str, previous_snippets: list[str]) -> list[dict]:
    """
    Build a list of OpenAI chat messages (system + few-shot + current turn).
    Returning a message list lets the caller pass it directly to
    client.chat.completions.create(messages=...).
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Few-shot examples as alternating user/assistant turns
    for example_text, example_label in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": f"Speech turn: {example_text}"})
        messages.append({"role": "assistant", "content": example_label})

    # Actual turn, with optional preceding context
    context_block = ""
    if previous_snippets:
        lines = [f"Previous snippet {i}: {s}" for i, s in enumerate(previous_snippets, 1)]
        context_block = "\n".join(lines) + "\n\n"

    messages.append({
        "role": "user",
        "content": (
            f"{context_block}"
            f"Speech turn: {text}\n"
        ),
    })
    return messages


def setup_logger(log_file: Path, verbose: bool):
    logger = logging.getLogger("bindung_llm_labeler")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def wsl_to_windows(p: str) -> str:
    if p.startswith("/mnt/") and len(p) > 6 and p[5].isalpha() and p[6] == "/":
        drive = p[5].upper()
        rest = p[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return p


def windows_to_wsl(p: str) -> str:
    if len(p) >= 3 and p[1] == ":" and p[2] in ("\\", "/"):
        drive = p[0].lower()
        rest = p[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p


def resolve_path(raw: str, repo_root: Path):
    if not raw:
        return None

    candidates = [raw]
    if raw.startswith("/mnt/"):
        candidates.append(wsl_to_windows(raw))
    elif len(raw) >= 3 and raw[1] == ":":
        candidates.append(windows_to_wsl(raw))
    else:
        candidates.append(str((repo_root / raw).resolve()))

    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


def canonical_speaker(v):
    s = str(v or "").strip().lower()
    s = s.replace("_", "").replace("-", "").replace(" ", "")
    if s in CLIENT_TOKENS or "client" in s or "patient" in s:
        return "client"
    if s in THERAPIST_TOKENS or "therap" in s:
        return "therapist"
    return "other"


def extract_patient_code_from_filename(name: str):
    m = re.match(r"^translate_([A-Za-z0-9]+)", name)
    if not m:
        return None
    return m.group(1).upper()


def collect_bindung_patient_ids(data_model_path: Path):
    with data_model_path.open("r", encoding="utf-8") as f:
        dm = yaml.safe_load(f)

    patient_ids = set()
    for interview in dm.get("interviews", []):
        types = interview.get("types", {}) or {}
        if "bindung" in types:
            pid = str(interview.get("patient", {}).get("patient_id", "")).strip().upper()
            if pid:
                patient_ids.add(pid)
    return patient_ids


def discover_bindung_files(input_dir: Path, bindung_patient_ids: set[str], strict_patient_match: bool, logger):
    all_files = sorted(input_dir.glob("translate_*.json"))
    bindung_files = []
    skipped_non_bindung = 0
    skipped_patient_mismatch = 0

    for p in all_files:
        lower_name = p.name.lower()
        is_bindung_named = ("bindung" in lower_name) or ("brfi" in lower_name)
        if not is_bindung_named:
            skipped_non_bindung += 1
            continue

        if strict_patient_match:
            pid = extract_patient_code_from_filename(p.name)
            if pid is None or pid not in bindung_patient_ids:
                skipped_patient_mismatch += 1
                continue

        bindung_files.append(p)

    logger.info(
        "Discovered translate files=%d | bindung_candidates=%d | skipped_non_bindung=%d | skipped_patient_mismatch=%d",
        len(all_files),
        len(bindung_files),
        skipped_non_bindung,
        skipped_patient_mismatch,
    )
    return bindung_files


def log_gpu_status(logger, prefix: str):
    """No-op for llama-cpp backend; GPU offload is controlled via --n-gpu-layers."""
    logger.debug("%s llama-cpp backend: GPU info not available via Python", prefix)

def parse_sentiment_output(raw_text: str) -> str:
    txt = str(raw_text or "").strip().lower()
    txt = re.sub(r"[^a-z\s]", " ", txt)
    for token in txt.split():
        if token in SENTIMENT_LABELS:
            return token
    return "other"


def generate_labels_for_payloads(payloads, client: OpenAI, model: str, max_tokens: int = 12):
    """
    Run inference one prompt at a time via the OpenAI-compatible HTTP API
    (served locally by llama-server on port 8080).
    """
    outputs = []
    for p in payloads:
        messages = build_prompt(p["text"], p["previous_snippets"])
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,   # slight randomness stops small models getting stuck on "neutral"
                stop=["\n"],
            )
            text = response.choices[0].message.content or ""
        except Exception:
            text = ""
        outputs.append(parse_sentiment_output(text))
    return outputs


def load_llm(server_url: str, model: str, logger) -> tuple[OpenAI, str]:
    """
    Create an OpenAI client pointed at the local llama-server HTTP endpoint.

    server_url : base URL of the OpenAI-compatible server, e.g. http://localhost:8080/v1
    model      : model name passed in API requests (llama-server ignores it but the
                 field is required by the OpenAI schema).  Use any non-empty string.
    """
    logger.info("Connecting to llama-server at %s (model=%s)", server_url, model)
    openai_client = OpenAI(base_url=server_url, api_key="local")
    return openai_client, model


def process_single_file(in_path: Path, out_path: Path, client: OpenAI, model: str, batch_size: int, label_key: str, log_every: int, context_turns: int, logger):
    t0 = time.perf_counter()
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {in_path}")

    previous_texts = []
    client_indices = []
    client_payloads = []

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        text = str(item.get("text", "")).strip()
        if not text:
            continue

        speaker = canonical_speaker(item.get("speaker_id", ""))
        if speaker == "client":
            context = previous_texts[-max(0, context_turns):]
            client_indices.append(idx)
            client_payloads.append({
                "text": text,
                "previous_snippets": context,
            })

        previous_texts.append(text)

    logger.info(
        "[%s] turns=%d client_turns=%d therapist_or_other=%d",
        in_path.name,
        len(data),
        len(client_indices),
        len(data) - len(client_indices),
    )

    labels = []
    if client_payloads:
        for start in range(0, len(client_payloads), max(1, log_every)):
            end = min(len(client_payloads), start + max(1, log_every))
            part = client_payloads[start:end]
            labels.extend(generate_labels_for_payloads(part, client=client, model=model))
            logger.info("[%s] client_turn_progress=%d/%d", in_path.name, end, len(client_payloads))
            log_gpu_status(logger, prefix=in_path.name)

    out_data = []
    label_ptr = 0
    client_index_set = set(client_indices)
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            out_data.append(item)
            continue

        row = dict(item)
        if idx in client_index_set:
            row[label_key] = labels[label_ptr] if label_ptr < len(labels) else "other"
            label_ptr += 1
        out_data.append(row)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    logger.info("WROTE %s | elapsed=%.1fs", out_path, time.perf_counter() - t0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify bindung client turns into two-word relation/sentiment labels using a GGUF model via llama-cpp-python."
    )
    parser.add_argument("--data-model", type=str, required=True, help="Path to data_model.yaml")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing translate_*.json files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write output JSON files")
    parser.add_argument(
        "--server-url", type=str,
        default="http://localhost:8080/v1",
        help="Base URL of the OpenAI-compatible llama-server (default: http://localhost:8080/v1)",
    )
    parser.add_argument(
        "--model-name", type=str, default="local",
        help="Model name string sent in API requests (llama-server ignores it; default: local)",
    )
    parser.add_argument("--log-every", type=int, default=64, help="Log progress every N client turns")
    parser.add_argument("--context-turns", type=int, default=2, help="Number of previous snippets to include in prompt context")
    parser.add_argument("--label-key", type=str, default="sentiment_label", help="Key added only for client turns")
    parser.add_argument("--strict-patient-match", action="store_true", help="Require translate file patient code to exist in bindung patients from data_model")
    parser.add_argument("--max-files", type=int, default=0, help="Optional cap for number of files processed (0 = all)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_model_path = Path(args.data_model)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    safe_mkdir(output_dir)

    log_file = output_dir / "bindung_sentiment.log"
    logger = setup_logger(log_file=log_file, verbose=args.verbose)

    logger.info("Starting bindung relation/sentiment labeling")
    logger.info("Args: %s", vars(args))

    bindung_patient_ids = collect_bindung_patient_ids(data_model_path)
    logger.info("Bindung patients in data_model=%d", len(bindung_patient_ids))

    files = discover_bindung_files(
        input_dir=input_dir,
        bindung_patient_ids=bindung_patient_ids,
        strict_patient_match=args.strict_patient_match,
        logger=logger,
    )

    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]
        logger.info("Applying max-files cap => processing %d files", len(files))

    if not files:
        logger.warning("No bindung translate files found. Nothing to do.")
        return

    client, model = load_llm(
        server_url=args.server_url,
        model=args.model_name,
        logger=logger,
    )
    log_gpu_status(logger, prefix="[startup]")

    total_start = time.perf_counter()
    processed = 0
    failures = 0

    for idx, in_path in enumerate(files, start=1):
        out_name = in_path.stem + "_sentiment.json"
        out_path = output_dir / out_name
        logger.info("Processing file %d/%d: %s", idx, len(files), in_path)
        try:
            process_single_file(
                in_path=in_path,
                out_path=out_path,
                client=client,
                model=model,
                batch_size=1,
                label_key=args.label_key,
                log_every=max(1, args.log_every),
                context_turns=max(0, args.context_turns),
                logger=logger,
            )
            processed += 1
        except Exception as exc:
            failures += 1
            logger.exception("FAILED %s | %s", in_path, exc)

    elapsed = time.perf_counter() - total_start
    logger.info("DONE processed=%d failures=%d elapsed=%.1fs", processed, failures, elapsed)

    # ---- write summary of label counts across all output files ----
    summary_path = output_dir / "sentiment_label_summary.txt"
    try:
        _write_label_summary(output_dir, args.label_key, summary_path, logger)
    except Exception as exc:
        logger.exception("Failed to write label summary: %s", exc)


def _write_label_summary(output_dir: Path, label_key: str, out_path: Path, logger):
    """Count sentiment labels across all *_sentiment.json files and write a .txt summary."""
    from collections import Counter

    total = Counter()
    per_file: dict[str, Counter] = {}

    for jf in sorted(output_dir.glob("*_sentiment.json")):
        with jf.open("r", encoding="utf-8") as f:
            data = json.load(f)
        fc = Counter()
        for row in data:
            if not isinstance(row, dict):
                continue
            lbl = row.get(label_key)
            if lbl:
                fc[lbl] += 1
        per_file[jf.name] = fc
        total.update(fc)

    lines = ["Sentiment-label summary", "=" * 40, ""]
    lines.append("Overall counts:")
    for lbl in sorted(total):
        lines.append(f"  {lbl:12s} {total[lbl]:>6d}")
    lines.append(f"  {'TOTAL':12s} {sum(total.values()):>6d}")
    lines.append("")
    lines.append("Per-file counts:")
    for fname, fc in per_file.items():
        parts = ", ".join(f"{k}={v}" for k, v in sorted(fc.items()))
        lines.append(f"  {fname}: {parts}")

    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    logger.info("Wrote label summary to %s", out_path)


if __name__ == "__main__":
    main()
