#!/usr/bin/env python3
"""
Qwen-TTS 音频质量评估脚本
评估指标：
1. Speaker Similarity - 说话人相似度（余弦相似度）
2. PESQ - 语音感知质量
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


def load_audio(audio_path, target_sr=16000):
    """加载音频并重采样到目标采样率"""
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    # 转为单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).numpy(), target_sr


class SpeakerSimilarityEvaluator:
    """使用 SpeechBrain ECAPA-TDNN 计算说话人相似度"""

    def __init__(self, device="cuda"):
        from speechbrain.inference.speaker import EncoderClassifier

        self.device = device
        print("Loading speaker encoder (ECAPA-TDNN)...")
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )

    def extract_embedding(self, audio_path):
        """提取说话人嵌入向量"""
        waveform, sr = load_audio(audio_path, target_sr=16000)
        waveform = torch.tensor(waveform).unsqueeze(0).to(self.device)
        embedding = self.encoder.encode_batch(waveform)
        return embedding.squeeze().cpu().numpy()

    def compute_similarity(self, audio1_path, audio2_path):
        """计算两个音频的说话人相似度（余弦相似度）"""
        emb1 = self.extract_embedding(audio1_path)
        emb2 = self.extract_embedding(audio2_path)

        # 余弦相似度
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)


class PESQEvaluator:
    """PESQ 语音质量评估"""

    def __init__(self, mode="wb"):
        """
        Args:
            mode: 'wb' (宽带, 16kHz) 或 'nb' (窄带, 8kHz)
        """
        from pesq import pesq

        self.pesq = pesq
        self.mode = mode
        self.sr = 16000 if mode == "wb" else 8000

    def compute_pesq(self, ref_audio_path, deg_audio_path):
        """
        计算 PESQ 分数

        Args:
            ref_audio_path: 参考音频（原始/真实音频）
            deg_audio_path: 待评估音频（生成的音频）

        Returns:
            PESQ 分数 (-0.5 ~ 4.5)
        """
        ref, _ = load_audio(ref_audio_path, target_sr=self.sr)
        deg, _ = load_audio(deg_audio_path, target_sr=self.sr)

        # 对齐长度
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]

        try:
            score = self.pesq(self.sr, ref, deg, self.mode)
            return float(score)
        except Exception as e:
            print(f"  PESQ error: {e}")
            return None


def evaluate_single(
    generated_audio,
    reference_audio,
    speaker_evaluator,
    pesq_evaluator,
):
    """评估单个生成音频"""
    results = {}

    # 说话人相似度
    if speaker_evaluator and reference_audio:
        similarity = speaker_evaluator.compute_similarity(generated_audio, reference_audio)
        results["speaker_similarity"] = similarity

    # PESQ (需要参考音频)
    if pesq_evaluator and reference_audio:
        pesq_score = pesq_evaluator.compute_pesq(reference_audio, generated_audio)
        if pesq_score is not None:
            results["pesq"] = pesq_score

    return results


def evaluate_batch(
    generated_dir,
    reference_dir,
    speaker_evaluator,
    pesq_evaluator,
    output_file=None,
):
    """批量评估目录中的音频"""
    generated_dir = Path(generated_dir)
    reference_dir = Path(reference_dir) if reference_dir else None

    generated_files = sorted(generated_dir.glob("*.wav"))
    print(f"Found {len(generated_files)} generated audio files")

    all_results = []
    similarities = []
    pesq_scores = []

    for gen_file in tqdm(generated_files, desc="Evaluating"):
        # 查找对应的参考音频
        ref_file = None
        if reference_dir:
            ref_file = reference_dir / gen_file.name
            if not ref_file.exists():
                # 尝试其他命名方式
                ref_file = None

        result = {
            "generated": str(gen_file),
            "reference": str(ref_file) if ref_file else None,
        }

        if speaker_evaluator and ref_file:
            sim = speaker_evaluator.compute_similarity(str(gen_file), str(ref_file))
            result["speaker_similarity"] = sim
            similarities.append(sim)

        if pesq_evaluator and ref_file:
            pesq_score = pesq_evaluator.compute_pesq(str(ref_file), str(gen_file))
            if pesq_score is not None:
                result["pesq"] = pesq_score
                pesq_scores.append(pesq_score)

        all_results.append(result)

    # 汇总统计
    summary = {}
    if similarities:
        summary["speaker_similarity"] = {
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
        }
    if pesq_scores:
        summary["pesq"] = {
            "mean": float(np.mean(pesq_scores)),
            "std": float(np.std(pesq_scores)),
            "min": float(np.min(pesq_scores)),
            "max": float(np.max(pesq_scores)),
        }

    # 保存结果
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "details": all_results}, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")

    return summary, all_results


def generate_and_evaluate(
    model_path,
    speaker_name,
    test_texts,
    reference_audio,
    output_dir,
    speaker_evaluator,
    pesq_evaluator,
    paired_data=None,
):
    """生成音频并评估

    Args:
        paired_data: list of {"text": ..., "audio": ...} 用于 PESQ 评估
    """
    from qwen_tts import Qwen3TTSModel

    print(f"Loading model from {model_path}...")
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    similarities = []
    pesq_scores = []

    # 如果有配对数据，用于 PESQ 评估
    if paired_data:
        print(f"\nEvaluating with {len(paired_data)} paired samples (for PESQ)...")
        for i, item in enumerate(tqdm(paired_data, desc="Generating & Evaluating (paired)")):
            text = item["text"]
            ref_audio = item["audio"]

            # 生成音频
            wavs, sr = model.generate_custom_voice(text=text, speaker=speaker_name)

            # 保存生成的音频
            output_path = output_dir / f"paired_{i:04d}.wav"
            wav_tensor = torch.tensor(wavs)
            if wav_tensor.dim() == 1:
                wav_tensor = wav_tensor.unsqueeze(0)
            elif wav_tensor.dim() > 2:
                wav_tensor = wav_tensor.squeeze()
                if wav_tensor.dim() == 1:
                    wav_tensor = wav_tensor.unsqueeze(0)
            torchaudio.save(str(output_path), wav_tensor, sr)

            result = {
                "text": text,
                "generated": str(output_path),
                "reference": ref_audio,
            }

            # 说话人相似度
            if speaker_evaluator:
                sim = speaker_evaluator.compute_similarity(str(output_path), ref_audio)
                result["speaker_similarity"] = sim
                similarities.append(sim)

            # PESQ（相同内容对比）
            if pesq_evaluator:
                pesq_score = pesq_evaluator.compute_pesq(ref_audio, str(output_path))
                if pesq_score is not None:
                    result["pesq"] = pesq_score
                    pesq_scores.append(pesq_score)
                    print(f"  [{i}] PESQ: {pesq_score:.4f}, Similarity: {result.get('speaker_similarity', 'N/A')}")

            results.append(result)

    # 普通文本生成（只评估相似度）
    if test_texts:
        print(f"\nGenerating {len(test_texts)} samples (similarity only)...")
        for i, text in enumerate(tqdm(test_texts, desc="Generating & Evaluating")):
            # 生成音频
            wavs, sr = model.generate_custom_voice(text=text, speaker=speaker_name)

            # 保存生成的音频
            output_path = output_dir / f"gen_{i:04d}.wav"
            wav_tensor = torch.tensor(wavs)
            if wav_tensor.dim() == 1:
                wav_tensor = wav_tensor.unsqueeze(0)
            elif wav_tensor.dim() > 2:
                wav_tensor = wav_tensor.squeeze()
                if wav_tensor.dim() == 1:
                    wav_tensor = wav_tensor.unsqueeze(0)
            torchaudio.save(str(output_path), wav_tensor, sr)

            result = {
                "text": text,
                "generated": str(output_path),
                "reference": reference_audio,
            }

            # 评估说话人相似度
            if speaker_evaluator and reference_audio:
                sim = speaker_evaluator.compute_similarity(str(output_path), reference_audio)
                result["speaker_similarity"] = sim
                similarities.append(sim)
                print(f"  [{i}] Similarity: {sim:.4f}")

            results.append(result)

    # 汇总
    summary = {}
    if similarities:
        summary["speaker_similarity"] = {
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
        }
    if pesq_scores:
        summary["pesq"] = {
            "mean": float(np.mean(pesq_scores)),
            "std": float(np.std(pesq_scores)),
            "min": float(np.min(pesq_scores)),
            "max": float(np.max(pesq_scores)),
        }

    return summary, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen-TTS audio quality")
    subparsers = parser.add_subparsers(dest="command", help="Evaluation mode")

    # 单文件评估
    single_parser = subparsers.add_parser("single", help="Evaluate single audio pair")
    single_parser.add_argument("--generated", type=str, required=True, help="Generated audio path")
    single_parser.add_argument("--reference", type=str, required=True, help="Reference audio path")
    single_parser.add_argument("--device", type=str, default="cuda")

    # 批量评估
    batch_parser = subparsers.add_parser("batch", help="Evaluate batch of audios")
    batch_parser.add_argument("--generated_dir", type=str, required=True, help="Directory of generated audios")
    batch_parser.add_argument("--reference_dir", type=str, required=True, help="Directory of reference audios")
    batch_parser.add_argument("--output", type=str, default="eval_results.json", help="Output JSON file")
    batch_parser.add_argument("--device", type=str, default="cuda")

    # 生成并评估
    gen_parser = subparsers.add_parser("generate", help="Generate and evaluate")
    gen_parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    gen_parser.add_argument("--speaker", type=str, default="qinche", help="Speaker name")
    gen_parser.add_argument("--reference", type=str, help="Reference audio for similarity")
    gen_parser.add_argument("--texts", type=str, nargs="+", help="Texts to generate")
    gen_parser.add_argument("--texts_file", type=str, help="File with texts (one per line)")
    gen_parser.add_argument("--paired_jsonl", type=str, help="JSONL file with paired data for PESQ (train_raw.jsonl)")
    gen_parser.add_argument("--num_paired", type=int, default=10, help="Number of paired samples to evaluate")
    gen_parser.add_argument("--output_dir", type=str, default="eval_output", help="Output directory")
    gen_parser.add_argument("--output", type=str, default="eval_results.json", help="Output JSON file")
    gen_parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    device = args.device if torch.cuda.is_available() else "cpu"

    # 初始化评估器
    print("Initializing evaluators...")
    speaker_evaluator = SpeakerSimilarityEvaluator(device=device)

    try:
        pesq_evaluator = PESQEvaluator(mode="wb")
    except ImportError:
        print("Warning: pesq not installed, skipping PESQ evaluation")
        pesq_evaluator = None

    if args.command == "single":
        print(f"\nEvaluating:")
        print(f"  Generated: {args.generated}")
        print(f"  Reference: {args.reference}")

        results = evaluate_single(
            args.generated,
            args.reference,
            speaker_evaluator,
            pesq_evaluator,
        )

        print("\nResults:")
        if "speaker_similarity" in results:
            print(f"  Speaker Similarity: {results['speaker_similarity']:.4f}")
        if "pesq" in results:
            print(f"  PESQ: {results['pesq']:.4f}")

    elif args.command == "batch":
        summary, _ = evaluate_batch(
            args.generated_dir,
            args.reference_dir,
            speaker_evaluator,
            pesq_evaluator,
            args.output,
        )

        print("\nSummary:")
        if "speaker_similarity" in summary:
            s = summary["speaker_similarity"]
            print(f"  Speaker Similarity: {s['mean']:.4f} ± {s['std']:.4f} (min: {s['min']:.4f}, max: {s['max']:.4f})")
        if "pesq" in summary:
            p = summary["pesq"]
            print(f"  PESQ: {p['mean']:.4f} ± {p['std']:.4f} (min: {p['min']:.4f}, max: {p['max']:.4f})")

    elif args.command == "generate":
        # 加载配对数据（用于 PESQ）
        paired_data = None
        if args.paired_jsonl:
            with open(args.paired_jsonl, "r", encoding="utf-8") as f:
                all_paired = [json.loads(line.strip()) for line in f if line.strip()]
            # 随机选择一部分
            import random
            random.seed(42)
            paired_data = random.sample(all_paired, min(args.num_paired, len(all_paired)))
            print(f"Loaded {len(paired_data)} paired samples for PESQ evaluation")

        # 获取测试文本
        test_texts = None
        if args.texts_file:
            with open(args.texts_file, "r", encoding="utf-8") as f:
                test_texts = [line.strip() for line in f if line.strip()]
        elif args.texts:
            test_texts = args.texts
        elif not paired_data:
            # 默认测试文本（仅当没有配对数据时）
            test_texts = [
                "大家好，我叫秦彻。",
                "今天天气真不错，适合出去走走。",
                "你好，很高兴认识你。",
                "这是一个测试语音合成的句子。",
                "我们一起去吃饭吧。",
            ]

        summary, results = generate_and_evaluate(
            args.model_path,
            args.speaker,
            test_texts,
            args.reference,
            args.output_dir,
            speaker_evaluator,
            pesq_evaluator,
            paired_data=paired_data,
        )

        # 保存结果
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "details": results}, f, indent=2, ensure_ascii=False)

        print("\nSummary:")
        if "speaker_similarity" in summary:
            s = summary["speaker_similarity"]
            print(f"  Speaker Similarity: {s['mean']:.4f} ± {s['std']:.4f}")
        if "pesq" in summary:
            p = summary["pesq"]
            print(f"  PESQ: {p['mean']:.4f} ± {p['std']:.4f}")
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
