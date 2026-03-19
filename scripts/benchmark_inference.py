#!/usr/bin/env python3
"""
Qwen-TTS 推理加速测试脚本
对比不同加速方法的推理速度
"""

import argparse
import time
import torch
import numpy as np
from pathlib import Path


def benchmark_baseline(model_path, text, speaker, num_runs=5):
    """基线推理（无加速）"""
    from qwen_tts import Qwen3TTSModel

    print("\n=== Baseline (BF16) ===")
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Warmup
    _ = model.generate_custom_voice(text=text, speaker=speaker)

    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        wavs, sr = model.generate_custom_voice(text=text, speaker=speaker)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")

    avg_time = np.mean(times)
    print(f"  Average: {avg_time:.3f}s")

    del model
    torch.cuda.empty_cache()
    return avg_time, wavs, sr


def benchmark_flash_attention(model_path, text, speaker, num_runs=5):
    """Flash Attention 2 加速"""
    from qwen_tts import Qwen3TTSModel

    print("\n=== Flash Attention 2 ===")
    try:
        model = Qwen3TTSModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    except Exception as e:
        print(f"  Flash Attention 2 not available: {e}")
        return None, None, None

    # Warmup
    _ = model.generate_custom_voice(text=text, speaker=speaker)

    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        wavs, sr = model.generate_custom_voice(text=text, speaker=speaker)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")

    avg_time = np.mean(times)
    print(f"  Average: {avg_time:.3f}s")

    del model
    torch.cuda.empty_cache()
    return avg_time, wavs, sr


def benchmark_sdpa(model_path, text, speaker, num_runs=5):
    """SDPA (Scaled Dot Product Attention) 加速"""
    from qwen_tts import Qwen3TTSModel

    print("\n=== SDPA (PyTorch Native) ===")
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )

    # Warmup
    _ = model.generate_custom_voice(text=text, speaker=speaker)

    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        wavs, sr = model.generate_custom_voice(text=text, speaker=speaker)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")

    avg_time = np.mean(times)
    print(f"  Average: {avg_time:.3f}s")

    del model
    torch.cuda.empty_cache()
    return avg_time, wavs, sr


def benchmark_torch_compile(model_path, text, speaker, num_runs=5):
    """torch.compile 加速"""
    from qwen_tts import Qwen3TTSModel

    print("\n=== torch.compile ===")
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Compile the model
    try:
        model.model = torch.compile(model.model, mode="reduce-overhead")
    except Exception as e:
        print(f"  torch.compile failed: {e}")
        del model
        torch.cuda.empty_cache()
        return None, None, None

    # Warmup (compile happens here)
    print("  Compiling (first run)...")
    _ = model.generate_custom_voice(text=text, speaker=speaker)

    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        wavs, sr = model.generate_custom_voice(text=text, speaker=speaker)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")

    avg_time = np.mean(times)
    print(f"  Average: {avg_time:.3f}s")

    del model
    torch.cuda.empty_cache()
    return avg_time, wavs, sr


def benchmark_int8_quantization(model_path, text, speaker, num_runs=5):
    """INT8 量化加速"""
    from qwen_tts import Qwen3TTSModel

    print("\n=== INT8 Quantization (dynamic) ===")
    try:
        # 先正常加载模型
        model = Qwen3TTSModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # 使用 PyTorch 动态量化
        torch.quantization.quantize_dynamic(
            model.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
        )
    except Exception as e:
        print(f"  INT8 quantization failed: {e}")
        return None, None, None

    # Warmup
    _ = model.generate_custom_voice(text=text, speaker=speaker)

    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        wavs, sr = model.generate_custom_voice(text=text, speaker=speaker)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")

    avg_time = np.mean(times)
    print(f"  Average: {avg_time:.3f}s")

    del model
    torch.cuda.empty_cache()
    return avg_time, wavs, sr


def benchmark_int4_quantization(model_path, text, speaker, num_runs=5):
    """INT4 量化加速 - 使用 GPTQ 或跳过"""
    print("\n=== INT4 Quantization ===")
    print("  Skipped: Qwen-TTS custom architecture not compatible with bitsandbytes")
    print("  Consider using GPTQ or AWQ for INT4 quantization")
    return None, None, None


def benchmark_vllm(model_path, text, speaker, num_runs=5):
    """vLLM 加速（如果支持）"""
    print("\n=== vLLM ===")
    try:
        from vllm import LLM, SamplingParams

        print("  vLLM available, but Qwen-TTS custom architecture may not be supported")
        print("  Skipping vLLM benchmark")
        return None, None, None
    except ImportError:
        print("  vLLM not installed")
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen-TTS inference")
    parser.add_argument("--model_path", type=str, default="output/checkpoint-epoch-5")
    parser.add_argument("--speaker", type=str, default="qinche")
    parser.add_argument(
        "--text", type=str, default="大家好，我叫秦彻，今天天气真不错。"
    )
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--output", type=str, default="benchmark_results.txt")
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated methods: baseline,sdpa,flash,compile,int8,int4,vllm,all",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen-TTS Inference Benchmark")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Text: {args.text}")
    print(f"Speaker: {args.speaker}")
    print(f"Runs per method: {args.num_runs}")

    # GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    methods = (
        args.methods.split(",")
        if args.methods != "all"
        else ["baseline", "sdpa", "flash", "compile", "int8", "int4", "vllm"]
    )

    results = {}

    if "baseline" in methods:
        t, _, _ = benchmark_baseline(
            args.model_path, args.text, args.speaker, args.num_runs
        )
        if t:
            results["Baseline (BF16)"] = t

    if "sdpa" in methods:
        t, _, _ = benchmark_sdpa(
            args.model_path, args.text, args.speaker, args.num_runs
        )
        if t:
            results["SDPA"] = t

    if "flash" in methods:
        t, _, _ = benchmark_flash_attention(
            args.model_path, args.text, args.speaker, args.num_runs
        )
        if t:
            results["Flash Attention 2"] = t

    if "compile" in methods:
        t, _, _ = benchmark_torch_compile(
            args.model_path, args.text, args.speaker, args.num_runs
        )
        if t:
            results["torch.compile"] = t

    if "int8" in methods:
        t, _, _ = benchmark_int8_quantization(
            args.model_path, args.text, args.speaker, args.num_runs
        )
        if t:
            results["INT8 Quantization"] = t

    if "int4" in methods:
        t, _, _ = benchmark_int4_quantization(
            args.model_path, args.text, args.speaker, args.num_runs
        )
        if t:
            results["INT4 Quantization"] = t

    if "vllm" in methods:
        t, _, _ = benchmark_vllm(
            args.model_path, args.text, args.speaker, args.num_runs
        )
        if t:
            results["vLLM"] = t

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if results:
        baseline_time = results.get("Baseline (BF16)", list(results.values())[0])
        print(f"{'Method':<25} {'Time (s)':<12} {'Speedup':<10}")
        print("-" * 47)
        for method, t in sorted(results.items(), key=lambda x: x[1]):
            speedup = baseline_time / t if t > 0 else 0
            print(f"{method:<25} {t:<12.3f} {speedup:<10.2f}x")

        # Save results
        with open(args.output, "w") as f:
            f.write("Qwen-TTS Inference Benchmark Results\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Text: {args.text}\n")
            f.write(f"Runs: {args.num_runs}\n\n")
            for method, t in sorted(results.items(), key=lambda x: x[1]):
                speedup = baseline_time / t if t > 0 else 0
                f.write(f"{method}: {t:.3f}s ({speedup:.2f}x)\n")
        print(f"\nResults saved to {args.output}")
    else:
        print("No benchmark results collected")


if __name__ == "__main__":
    main()
