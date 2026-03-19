#!/usr/bin/env python3
"""
Test inference with fine-tuned Qwen-TTS model
"""

import torch
import soundfile as sf
import argparse


def main():
    parser = argparse.ArgumentParser(description="Test Qwen-TTS inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="output/checkpoint-epoch-5",
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--speaker_name",
        type=str,
        default="qinche",
        help="Speaker name used during training",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="今天天气真不错，适合出去走走。",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output", type=str, default="test_output.wav", help="Output audio file"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    from qwen_tts import Qwen3TTSModel

    tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    print(f"Generating speech for: {args.text}")
    wavs, sr = tts.generate_custom_voice(
        text=args.text,
        speaker=args.speaker_name,
    )

    sf.write(args.output, wavs[0], sr)
    print(f"Audio saved to {args.output}")


if __name__ == "__main__":
    main()
