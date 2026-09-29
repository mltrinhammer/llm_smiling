import argparse
import json
import re
from pathlib import Path
from typing import List

import torch
from transformers import pipeline


DEFAULT_MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"


def _coerce_ms(value, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return default
    if text.lstrip("-").isdigit():
        return int(text)
    return default


def group_text_by_speaker_turns(turns: List[dict]) -> List[dict]:
    """Group consecutive turns by the same speaker.

    Mirrors summarize.py grouping so translated snippets align to the same
    start/end windows used for summaries.
    """
    if not turns:
        return []

    speaker_turns: list[dict] = []
    current_speaker = None
    current_texts: list[str] = []
    current_start_ms = None
    current_end_ms = None
    current_turn_indices: list[int] = []

    for idx, turn in enumerate(turns):
        text = str(turn.get("text", "")).strip()
        if not text:
            continue

        speaker_id = str(turn.get("speaker_id", "unknown")).strip().lower()
        start_ms = max(0, _coerce_ms(turn.get("start"), 0))
        end_ms = max(_coerce_ms(turn.get("end"), start_ms), start_ms)

        if speaker_id != current_speaker:
            if current_speaker is not None and current_texts:
                speaker_turns.append(
                    {
                        "speaker_id": current_speaker,
                        "text": " ".join(current_texts),
                        "start_ms": current_start_ms,
                        "end_ms": current_end_ms,
                        "original_turn_indices": current_turn_indices,
                    }
                )

            current_speaker = speaker_id
            current_texts = [text]
            current_start_ms = start_ms
            current_end_ms = end_ms
            current_turn_indices = [idx]
        else:
            current_texts.append(text)
            current_end_ms = end_ms
            current_turn_indices.append(idx)

    if current_speaker is not None and current_texts:
        speaker_turns.append(
            {
                "speaker_id": current_speaker,
                "text": " ".join(current_texts),
                "start_ms": current_start_ms,
                "end_ms": current_end_ms,
                "original_turn_indices": current_turn_indices,
            }
        )

    return speaker_turns


def discover_result_jsons(root: Path) -> List[Path]:
    return sorted(root.rglob("results_*.json"))


def resolve_device(preferred: str | int) -> tuple[str, int]:
    device_pref = str(preferred)
    if device_pref.lower() in ("auto", ""):
        try:
            device_pref = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device_pref = "cpu"
    try:
        device_arg = int(device_pref)
        device_name = f"cuda:{device_arg}" if device_arg >= 0 else "cpu"
        return device_name, device_arg
    except ValueError:
        device_name = device_pref
        device_arg = 0 if device_pref.lower().startswith("cuda") else -1
        return device_name, device_arg


def load_translation_pipeline(device: str = "auto"):
    device_name, device_arg = resolve_device(device)
    print(f"[translate] Loading translation model '{DEFAULT_MODEL_NAME}' on {device_name} (pipeline device arg={device_arg})")

    pipe_kwargs = {
        "task": "translation",
        "model": DEFAULT_MODEL_NAME,
        "device": device_arg
    }
    if device_arg != -1 and torch.cuda.is_available():
        pipe_kwargs["torch_dtype"] = torch.float16

    text_pipe = pipeline(**pipe_kwargs)
    return text_pipe, device_name


def _translate_single_text(
    text: str,
    text_pipe,
    max_new_tokens: int,
    max_input_tokens: int,
) -> str:
    chunks = chunk_text_for_translation(text, text_pipe, max_input_tokens=max_input_tokens)
    translated_chunks: list[str] = []
    for chunk in chunks:
        outputs = text_pipe(chunk, max_new_tokens=max_new_tokens, truncation=True)
        if isinstance(outputs, list) and outputs:
            translated_chunks.append(outputs[0].get("translation_text", chunk).strip())
        else:
            translated_chunks.append(chunk)
    return " ".join(x for x in translated_chunks if x).strip()


def _safe_model_max_length(text_pipe, default: int = 512) -> int:
    tokenizer = getattr(text_pipe, "tokenizer", None)
    if tokenizer is None:
        return default

    model_max = getattr(tokenizer, "model_max_length", default)
    if not isinstance(model_max, int) or model_max <= 0 or model_max > 16384:
        return default
    return model_max


def _token_len(text: str, text_pipe) -> int:
    tokenizer = getattr(text_pipe, "tokenizer", None)
    if tokenizer is None:
        return len(text.split())
    try:
        ids = tokenizer(text, add_special_tokens=False, truncation=False).get("input_ids", [])
        return len(ids)
    except Exception:
        return len(text.split())


def _split_oversized_sentence(sentence: str, max_tokens: int, text_pipe) -> list[str]:
    words = sentence.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []

    for w in words:
        candidate = " ".join(current + [w])
        if not current or _token_len(candidate, text_pipe) <= max_tokens:
            current.append(w)
            continue

        chunks.append(" ".join(current))
        current = [w]

    if current:
        chunks.append(" ".join(current))

    return chunks


def _split_by_token_ids(text: str, text_pipe, max_tokens: int) -> list[str]:
    tokenizer = getattr(text_pipe, "tokenizer", None)
    if tokenizer is None:
        return [text]

    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    token_ids = encoded.get("input_ids", [])
    if not token_ids:
        return [text]

    if len(token_ids) <= max_tokens:
        return [text]

    chunks: list[str] = []
    for i in range(0, len(token_ids), max_tokens):
        sub_ids = token_ids[i:i + max_tokens]
        sub_text = tokenizer.decode(sub_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        if sub_text:
            chunks.append(sub_text)

    return chunks if chunks else [text]


def chunk_text_for_translation(text: str, text_pipe, max_input_tokens: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    if _token_len(text, text_pipe) <= max_input_tokens:
        return [text]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        sentences = [text]

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if _token_len(sentence, text_pipe) > max_input_tokens:
            sub_sentences = _split_oversized_sentence(sentence, max_input_tokens, text_pipe)
        else:
            sub_sentences = [sentence]

        for sub in sub_sentences:
            candidate = f"{current} {sub}".strip() if current else sub
            if current and _token_len(candidate, text_pipe) > max_input_tokens:
                chunks.append(current)
                current = sub
            else:
                current = candidate

    if current:
        chunks.append(current)

    strict_chunks: list[str] = []
    for c in (chunks if chunks else [text]):
        strict_chunks.extend(_split_by_token_ids(c, text_pipe, max_tokens=max_input_tokens))

    return strict_chunks if strict_chunks else [text]


def translate_file(
    input_path: Path,
    output_path: Path,
    text_pipe,
    max_new_tokens: int | None,
    max_input_tokens: int,
) -> None:
    with open(input_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError(f"Expected list in {input_path}, found {type(records).__name__}")

    grouped_records = group_text_by_speaker_turns(records)
    print(
        f"[translate] Grouped {len(records)} raw turns into {len(grouped_records)} speaker turns for {input_path.name}"
    )

    translated_records = []
    model_max_len = _safe_model_max_length(text_pipe, default=512)
    safe_input_tokens = min(max_input_tokens, max(32, model_max_len - 32))
    generation_new_tokens = min(max_new_tokens or 256, 256)

    for idx, entry in enumerate(grouped_records):
        text = (entry.get("text") or "").strip()
        if not text:
            translated_text = ""
        else:
            try:
                translated_text = _translate_single_text(
                    text,
                    text_pipe,
                    max_new_tokens=generation_new_tokens,
                    max_input_tokens=safe_input_tokens,
                )
            except Exception as exc:
                print(f"[translate] Error translating segment {idx} in {input_path}: {exc}")
                translated_text = text

        translated_records.append(
            {
                "turn_index": idx,
                "text": translated_text,
                "start_ms": entry.get("start_ms"),
                "end_ms": entry.get("end_ms"),
                "speaker_id": entry.get("speaker_id"),
                "original_turn_indices": entry.get("original_turn_indices", []),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(translated_records, handle, ensure_ascii=False, indent=4)
    print(f"[translate] Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Translate ASR results from German to English using Helsinki-NLP/opus-mt-de-en.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing results_*.json files (recursively scanned).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write translated JSON files. Defaults to the same location as inputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for translation: 'auto', 'cuda', 'cpu', or device index (default: auto).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Optional cap for generated tokens per segment.",
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=192,
        help="Max source tokens per translation chunk (conservative GPU-safe cap).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing translated files instead of skipping them.",
    )
    args = parser.parse_args()

    text_pipe, device_name = load_translation_pipeline(args.device)

    input_root = Path(args.input_dir).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory {input_root} does not exist")

    output_root = Path(args.output_dir).resolve() if args.output_dir else None

    result_files = discover_result_jsons(input_root)
    if not result_files:
        print(f"[translate] No results_*.json files found in {input_root}")
        return

    print(f"[translate] Found {len(result_files)} files to translate")

    for input_path in result_files:
        # Keep source identifier from results_<identifier>.json
        stem = input_path.stem
        file_id = stem[len("results_"):] if stem.startswith("results_") else stem
        
        # Output translate_<id>.json in the same directory as the source results_*.json
        if output_root:
            relative_parent = input_path.parent.relative_to(input_root)
            target_dir = output_root / relative_parent
        else:
            target_dir = input_path.parent

        output_path = target_dir / f"translate_{file_id}.json"

        if output_path.exists() and not args.overwrite:
            print(f"[translate] Skipping existing {output_path}")
            continue

        translate_file(
            input_path,
            output_path,
            text_pipe,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
        )


if __name__ == "__main__":
    main()

