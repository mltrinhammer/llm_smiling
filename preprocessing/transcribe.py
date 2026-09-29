"""Language preprocessing pipeline: diarization (pyannote) and ASR (faster-whisper).
The script is constructed to execute on a single NVIDIA GeForce RTX 4070 with 12GB VRAM.
It is not tested to run for multi-GPU or CPU-only setups.
Note that there are elaborate checks of device availability and compatibility, which is included because
the script uses multiple libraries that have different device requirements."""

import argparse
import json
import time
from pathlib import Path
import re

import ffmpeg
import torch

from utils import load_config


DIAR_PIPELINE = None  # Lazily loaded diarization pipeline


def _normalise_device_name(device: str) -> str:
    '''I noticed pyannote would sometimes not allocate the device correctly, because it only accepted "cuda", therefore this change'''
    device = str(device)
    if device.startswith("cuda") and ":" not in device:
        return "cuda:0"
    return device


def _load_diarization_pipeline(config: dict, log_prefix: str = "[language_pipeline]"):
    """Load the pyannote diarization pipeline and move it to device."""
    from pyannote.audio import Pipeline

    diary_model = config.get("diary_model", "pyannote/speaker-diarization-3.1")
    hf_token = config.get("hf_token")
    target_device = _normalise_device_name(config.get("device", "cpu"))
    try:
        device_obj = torch.device(target_device)
    except Exception:
        device_obj = torch.device("cpu")
        print(f"{log_prefix} Invalid device '{target_device}', defaulting to CPU")

    start = time.time()
    pipeline = Pipeline.from_pretrained(diary_model, use_auth_token=hf_token)
    load_time = time.time() - start

    try:
        moved = pipeline.to(device_obj)
        if moved is not None:
            pipeline = moved
        print(f"{log_prefix} Loaded diarization pipeline '{diary_model}' on {device_obj} in {load_time:.2f}s")
    except Exception as exc:  
        print(
            f"{log_prefix} Unable to move pipeline to {device_obj}; falling back to CPU ({exc})."
        )
    return pipeline


def _sanitize_uri(text: str) -> str:
    """Make a safe RTTM-friendly URI by replacing whitespace with underscores.

    RTTM is space-separated; URIs must not contain spaces. We keep other characters
    intact and collapse any whitespace runs to a single underscore.
    """
    return re.sub(r"\s+", "_", str(text))


def run_diarization(input_wav, output_dir, config):
    """Run pyannote diarization on the provided audio file."""
    global DIAR_PIPELINE

    start = time.time()
    input_wav_path = Path(input_wav)
    file_subdir = Path(output_dir) / input_wav_path.stem
    file_subdir.mkdir(exist_ok=True)

    target_device = config.get("device", "cpu")
    print(f"[language_pipeline] Running diarization on device: {target_device}")

    if DIAR_PIPELINE is None:
        try:
            DIAR_PIPELINE = _load_diarization_pipeline(config)
        except Exception as e:
            print(
                "Failed to load pyannote pipeline. "
                f"Error: {e}"
            )
            return

    num_speakers = int(config.get("diarization_num_speakers", 2))
    diarization = DIAR_PIPELINE(str(input_wav_path), num_speakers=num_speakers)
    # Ensure RTTM URI contains no spaces (RTTM uses space-separated fields)
    try:
        diarization.uri = _sanitize_uri(input_wav_path.stem)
    except Exception:
        # If attribute assignment fails for some reason, fall back silently; write_rttm may error if spaces remain
        pass
    rttm_path = file_subdir / f"{input_wav_path.stem}.rttm"
    with open(rttm_path, "w") as rttm_file:
        diarization.write_rttm(rttm_file)
    print(f"Diarization for {input_wav_path} done in {time.time() - start:.2f}s. RTTM: {rttm_path}")

def run_asr(input_wav, output_dir, config):
    from pyannote.core import Annotation, Segment
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    from pydub import AudioSegment

    start = time.time()
    input_wav_path = Path(input_wav)
    file_subdir = Path(output_dir) / input_wav_path.stem
    rttm_path = file_subdir / f"{input_wav_path.stem}.rttm"
    if not rttm_path.exists():
        print(f"RTTM file not found for {input_wav_path}. Run diarization first.")
        return

    annotation = Annotation()
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] == "SPEAKER":
                start_t = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segment = Segment(start_t, start_t + duration)
                annotation[segment] = speaker
    mapping = {
        annotation.argmax(): "client",
        [label for label in annotation.labels() if label != annotation.argmax()][0]: "therapist" #The most-speaking individual is inferred to be the client
    }

    model_size = config.get("whisper_model_size", "small")
    device_str = str(config.get("device", "cuda"))
    try:
        torch_device = torch.device(device_str)
    except Exception:
        torch_device = torch.device("cpu")
        print(f"[language_pipeline] Invalid ASR device '{device_str}', defaulting to CPU")

    device_type = torch_device.type
    device_index = torch_device.index if torch_device.index is not None else 0

    compute_type = config.get("whisper_compute_type", "int8")
    print(
        f"[language_pipeline] Running ASR on device: {device_type}"
        f" (index={device_index}, compute_type={compute_type})"
    )
    whisper_model = WhisperModel(
        model_size,
        device=device_type,
        device_index=device_index,
        compute_type=compute_type,
    )
    if device_type == "cuda":
        try:
            if torch.cuda.current_device() != device_index:
                torch.cuda.set_device(device_index)
            free_mem, total_mem = torch.cuda.mem_get_info(device_index)
            used_mem = total_mem - free_mem
            print(
                f"[language_pipeline] ASR GPU memory after model load: "
                f"used={used_mem / 1e9:.2f}GB free={free_mem / 1e9:.2f}GB total={total_mem / 1e9:.2f}GB"
            )
        except Exception as gpu_log_exc:
            print(f"[language_pipeline] Unable to query GPU memory stats: {gpu_log_exc}")
    batched_model = BatchedInferencePipeline(model=whisper_model)
    audio_pydub = AudioSegment.from_file(str(input_wav_path))
    results = []
    export_path = file_subdir / f"{input_wav_path.stem}_chunk.wav"

    for segment in annotation.itersegments():
        speaker_label = mapping[next(iter(annotation.get_labels(segment)))]
        if not config.get("both_speakers", False) and speaker_label == "therapist":
            continue
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000) 
        if end_ms - start_ms < 1500: #if the speach turn is less than 1500 milliseconds, dont transcribe ("uh", "mhmm", "und..")
            continue
        chunk = audio_pydub[start_ms:end_ms]
        chunk.export(export_path, format="wav")
        segments, _ = batched_model.transcribe(
            str(export_path),
            beam_size=3,
            language="de",
            condition_on_previous_text=False,
            word_timestamps=False,
            batch_size=8,
        )
        for t in segments:
            results.append({
                "text": t.text,
                "start": start_ms,
                "end": end_ms,
                "speaker_id": speaker_label
            })

    output_json = file_subdir / f"results_{input_wav_path.stem}.json"
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
    print(f"ASR for {input_wav_path} done in {time.time() - start:.2f}s. Output: {output_json}")


def extract_audio_from_video(input_video: Path, temp_audio_dir: Path, config: dict) -> Path | None:
    """Extract the audio track from a video file (.mov or .mp4) into a temporary .wav file."""
    temp_audio_dir.mkdir(parents=True, exist_ok=True)
    output_wav = temp_audio_dir / f"{input_video.stem}.wav"

    sample_rate = int(config.get("audio_sample_rate", 16000))
    channels = int(config.get("audio_channels", 1))

    try:
        (
            ffmpeg
            .input(str(input_video))
            .output(
                str(output_wav),
                ac=channels,
                ar=sample_rate,
                **{"vn": None},  # drop video 
            )
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"[language_pipeline] Extracted audio: {input_video} -> {output_wav}")
        return output_wav
    except ffmpeg.Error as e:
        print(f"[language_pipeline] Failed to extract audio from {input_video}: {e}")
        return None


def process_file(input_wav, output_dir, config, stage):
    if stage == "diarization":
        run_diarization(input_wav, output_dir, config)
    elif stage == "asr":
        run_asr(input_wav, output_dir, config)
    elif stage == "full":
        run_diarization(input_wav, output_dir, config)
        run_asr(input_wav, output_dir, config)
    else:
        print(f"Unknown stage: {stage}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_language.yaml", help="Path to config file")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing .wav, .mov, or .mp4 files (recursively searched). Ignored when --use_temp_audio is set.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--stage", type=str, choices=["diarization", "asr", "full"], default="full", help="Pipeline stage to run")
    parser.add_argument(
        "--use_temp_audio",
        action="store_true",
        help="Process from <output_dir>/_temp_audio instead of scanning input_dir. Useful for resuming after extraction.")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip work when outputs already exist (e.g., skip diarization if RTTM exists; skip ASR if results JSON exists; in 'full', run only missing steps).",
    )
    args = parser.parse_args()

    config = load_config(args.config) or {}

    device_pref = config.get("device", "auto")
    if str(device_pref).lower() in ("auto", ""):
        try:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            resolved_device = "cpu"
    else:
        resolved_device = str(device_pref)

    resolved_device = _normalise_device_name(resolved_device)
    config["device"] = resolved_device
    print(f"[language_pipeline] Selected device: {resolved_device}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    temp_audio_dir = output_dir / "_temp_audio"
    audio_files = []

    if args.use_temp_audio:
        # Resume mode: use already extracted audio from temp directory only
        audio_files = list(temp_audio_dir.rglob("*.wav"))
        if not audio_files:
            print(f"[language_pipeline] No temp audio found in {temp_audio_dir}. Did you run extraction previously?")
    else:
        # Recursively discover video files in all subdirectories
        video_files = list(input_dir.rglob("*.mov")) + list(input_dir.rglob("*.mp4"))
        for video_file in video_files:
            # If skip_existing is on and the extracted temp wav already exists, reuse it
            candidate_temp_wav = temp_audio_dir / f"{video_file.stem}.wav"
            if args.skip_existing and candidate_temp_wav.exists():
                audio_files.append(candidate_temp_wav)
                continue
            extracted = extract_audio_from_video(video_file, temp_audio_dir, config)
            if extracted is not None:
                audio_files.append(extracted)

        # Recursively discover wav files in all subdirectories
        wav_files = list(input_dir.rglob("*.wav"))
        audio_files.extend(wav_files)

    if not audio_files:
        print(f"No .wav, .mov, or .mp4 files found in {input_dir}")
        return

    print(f"[language_pipeline] Found {len(audio_files)} audio files (.wav/.mov/.mp4) in {input_dir} (recursively).")

    for audio_path in audio_files:
        audio_path = Path(audio_path)
        stem = audio_path.stem
        file_subdir = output_dir / stem
        rttm_path = file_subdir / f"{stem}.rttm"
        results_json = file_subdir / f"results_{stem}.json"

        if args.skip_existing:
            if args.stage == "diarization" and rttm_path.exists():
                print(f"[language_pipeline] Skip diarization for {audio_path} (RTTM exists: {rttm_path})")
                continue
            if args.stage == "asr" and results_json.exists():
                print(f"[language_pipeline] Skip ASR for {audio_path} (results exist: {results_json})")
                continue
            if args.stage == "full":
                if rttm_path.exists() and results_json.exists():
                    print(f"[language_pipeline] Skip full pipeline for {audio_path} (both outputs exist)")
                    continue
                if rttm_path.exists() and not results_json.exists():
                    print(f"[language_pipeline] Resume ASR only for {audio_path} (RTTM exists)")
                    run_asr(str(audio_path), str(output_dir), config)
                    continue

        print(f"[language_pipeline] Processing {audio_path} ({args.stage})")
        process_file(str(audio_path), str(output_dir), config, args.stage)

if __name__ == "__main__":
    main()



