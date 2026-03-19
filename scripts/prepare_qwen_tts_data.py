#!/usr/bin/env python3
"""
Prepare data for Qwen-TTS fine-tuning:
1. Split audio into segments using VAD
2. Transcribe segments using Whisper
3. Generate JSONL format for Qwen-TTS
"""

import os
import json
import argparse
from pathlib import Path
import torch
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.silence import split_on_silence
import soundfile as sf


def load_audio(audio_path):
    """Load audio file"""
    audio = AudioSegment.from_wav(audio_path)
    return audio


def split_audio_vad(audio_path, output_dir, min_silence_len=500, silence_thresh=-40, keep_silence=300):
    """
    Split audio using Voice Activity Detection (silence detection)

    Args:
        audio_path: input audio file path
        output_dir: output directory for segments
        min_silence_len: minimum silence length in ms
        silence_thresh: silence threshold in dBFS
        keep_silence: keep silence padding in ms
    """
    print(f"Processing {audio_path}...")
    audio = AudioSegment.from_wav(audio_path)

    # Split on silence
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(audio_path).stem
    segment_paths = []

    for i, chunk in enumerate(chunks):
        # Filter out very short segments (< 1 second)
        if len(chunk) < 1000:
            continue

        # Filter out very long segments (> 30 seconds)
        if len(chunk) > 30000:
            continue

        output_path = output_dir / f"{base_name}_seg{i:04d}.wav"
        chunk.export(output_path, format="wav")
        segment_paths.append(str(output_path.resolve()))

    print(f"Generated {len(segment_paths)} segments from {audio_path}")
    return segment_paths


def transcribe_audio(audio_path, model):
    """Transcribe audio using faster-whisper"""
    segments, info = model.transcribe(audio_path, language="zh")
    text = " ".join([seg.text for seg in segments])
    return text.strip()


def create_jsonl(segments, ref_audio, output_jsonl, whisper_model_name="large", append=False):
    """
    Create JSONL file for Qwen-TTS training

    Args:
        segments: list of audio segment paths
        ref_audio: reference audio path (for speaker identity)
        output_jsonl: output JSONL file path
        whisper_model_name: Whisper model size
        append: if True, append to existing file instead of overwriting
    """
    print(f"Loading faster-whisper model: {whisper_model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(whisper_model_name, device=device, compute_type=compute_type)

    # Load existing data if appending
    existing_audios = set()
    if append and os.path.exists(output_jsonl):
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                existing_audios.add(item['audio'])
        print(f"Found {len(existing_audios)} existing entries, will skip duplicates")

    data = []
    for i, audio_path in enumerate(segments):
        # Skip if already exists
        if audio_path in existing_audios:
            print(f"Skipping existing: {audio_path}")
            continue

        print(f"Transcribing {i+1}/{len(segments)}: {audio_path}")
        text = transcribe_audio(audio_path, model)

        if not text:
            print(f"  Skipping empty transcription")
            continue

        data.append({
            "audio": audio_path,
            "text": text,
            "ref_audio": ref_audio
        })
        print(f"  Text: {text}")

    # Write JSONL (append or overwrite)
    mode = 'a' if append else 'w'
    with open(output_jsonl, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nGenerated {len(data)} new training samples in {output_jsonl}")
    return data


def select_reference_audio(segments, output_path, target_duration=10):
    """
    Select or create a reference audio from segments
    Pick a segment close to target_duration seconds
    """
    best_segment = None
    best_diff = float('inf')

    for seg_path in segments:
        audio = AudioSegment.from_wav(seg_path)
        duration = len(audio) / 1000.0  # convert to seconds
        diff = abs(duration - target_duration)

        if diff < best_diff and 5 <= duration <= 15:
            best_diff = diff
            best_segment = seg_path

    if best_segment:
        # Copy to reference location
        audio = AudioSegment.from_wav(best_segment)
        audio.export(output_path, format="wav")
        print(f"Selected reference audio: {best_segment} ({len(audio)/1000:.1f}s)")
        return str(Path(output_path).resolve())

    # Fallback: use first segment
    if segments:
        audio = AudioSegment.from_wav(segments[0])
        audio.export(output_path, format="wav")
        print(f"Using first segment as reference: {segments[0]}")
        return str(Path(output_path).resolve())

    return None


def main():
    parser = argparse.ArgumentParser(description="Prepare data for Qwen-TTS fine-tuning")
    parser.add_argument("--raw_audio_dir", type=str, default="data/raw", help="Directory containing raw audio files")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for processed data")
    parser.add_argument("--output_jsonl", type=str, default="data/train_raw.jsonl", help="Output JSONL file")
    parser.add_argument("--whisper_model", type=str, default="large", help="Whisper model size")
    parser.add_argument("--min_silence_len", type=int, default=500, help="Minimum silence length in ms")
    parser.add_argument("--silence_thresh", type=int, default=-40, help="Silence threshold in dBFS")
    parser.add_argument("--append", action="store_true", help="Append to existing JSONL instead of overwriting")

    args = parser.parse_args()

    raw_audio_dir = Path(args.raw_audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all wav files
    all_segments = []
    for audio_file in sorted(raw_audio_dir.glob("*.wav")):
        segments = split_audio_vad(
            str(audio_file),
            output_dir / "segments",
            min_silence_len=args.min_silence_len,
            silence_thresh=args.silence_thresh
        )
        all_segments.extend(segments)

    if not all_segments:
        print("No segments generated!")
        return

    # Select reference audio (only if not appending or no ref exists)
    ref_audio_path = output_dir / "ref_audio.wav"
    if not args.append or not ref_audio_path.exists():
        ref_audio_abs = select_reference_audio(all_segments, ref_audio_path)
    else:
        ref_audio_abs = str(ref_audio_path.resolve())

    # Create JSONL with transcriptions
    create_jsonl(all_segments, ref_audio_abs, args.output_jsonl, args.whisper_model, append=args.append)

    print("\n=== Data preparation complete ===")
    print(f"Total new segments: {len(all_segments)}")
    print(f"Reference audio: {ref_audio_path}")
    print(f"Training JSONL: {args.output_jsonl}")


if __name__ == "__main__":
    main()
