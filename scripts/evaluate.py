#!/usr/bin/env python3
"""
评估模块 - TTS模型评估指标计算
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import librosa
import yaml
from loguru import logger
from tqdm import tqdm
import json


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_mel_cepstral_distortion(
    ref_audio: np.ndarray,
    syn_audio: np.ndarray,
    sr: int = 24000,
    n_mfcc: int = 13
) -> float:
    """
    计算梅尔倒谱失真(MCD)

    Args:
        ref_audio: 参考音频
        syn_audio: 合成音频
        sr: 采样率
        n_mfcc: MFCC系数数量

    Returns:
        MCD值(越低越好)
    """
    # 提取MFCC
    ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=sr, n_mfcc=n_mfcc)
    syn_mfcc = librosa.feature.mfcc(y=syn_audio, sr=sr, n_mfcc=n_mfcc)

    # 对齐长度
    min_len = min(ref_mfcc.shape[1], syn_mfcc.shape[1])
    ref_mfcc = ref_mfcc[:, :min_len]
    syn_mfcc = syn_mfcc[:, :min_len]

    # 计算MCD
    diff = ref_mfcc - syn_mfcc
    mcd = np.mean(np.sqrt(np.sum(diff ** 2, axis=0)))

    # 转换为dB
    mcd_db = (10.0 / np.log(10.0)) * mcd

    return mcd_db


def compute_speaker_similarity(
    ref_audio: np.ndarray,
    syn_audio: np.ndarray,
    sr: int = 24000
) -> float:
    """
    计算说话人相似度(使用简单的声学特征)

    Args:
        ref_audio: 参考音频
        syn_audio: 合成音频
        sr: 采样率

    Returns:
        相似度分数(0-1，越高越好)
    """
    # 提取声学特征
    def extract_features(audio: np.ndarray) -> np.ndarray:
        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # 基频
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=50, fmax=500, sr=sr
        )
        f0_mean = np.nanmean(f0) if not np.all(np.isnan(f0)) else 0
        f0_std = np.nanstd(f0) if not np.all(np.isnan(f0)) else 0

        # 能量
        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)

        # 组合特征
        features = np.concatenate([
            mfcc_mean, mfcc_std,
            [f0_mean, f0_std, rms_mean, rms_std]
        ])

        return features

    ref_features = extract_features(ref_audio)
    syn_features = extract_features(syn_audio)

    # 计算余弦相似度
    similarity = np.dot(ref_features, syn_features) / (
        np.linalg.norm(ref_features) * np.linalg.norm(syn_features) + 1e-8
    )

    # 归一化到0-1
    similarity = (similarity + 1) / 2

    return float(similarity)


def compute_pesq(
    ref_audio: np.ndarray,
    syn_audio: np.ndarray,
    sr: int = 16000
) -> float:
    """
    计算PESQ分数(语音质量感知评估)

    Args:
        ref_audio: 参考音频
        syn_audio: 合成音频
        sr: 采样率(PESQ需要16kHz)

    Returns:
        PESQ分数(-0.5到4.5，越高越好)
    """
    try:
        from pesq import pesq

        # PESQ需要16kHz采样率
        if sr != 16000:
            ref_audio = librosa.resample(ref_audio, orig_sr=sr, target_sr=16000)
            syn_audio = librosa.resample(syn_audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        # 对齐长度
        min_len = min(len(ref_audio), len(syn_audio))
        ref_audio = ref_audio[:min_len]
        syn_audio = syn_audio[:min_len]

        # 计算PESQ
        score = pesq(sr, ref_audio, syn_audio, "wb")  # 宽带模式

        return float(score)

    except ImportError:
        logger.warning("pesq未安装，跳过PESQ计算")
        return 0.0
    except Exception as e:
        logger.warning(f"PESQ计算失败: {e}")
        return 0.0


def compute_stoi(
    ref_audio: np.ndarray,
    syn_audio: np.ndarray,
    sr: int = 16000
) -> float:
    """
    计算STOI分数(短时客观可懂度)

    Args:
        ref_audio: 参考音频
        syn_audio: 合成音频
        sr: 采样率

    Returns:
        STOI分数(0-1，越高越好)
    """
    try:
        from pystoi import stoi

        # STOI需要特定采样率
        if sr != 16000:
            ref_audio = librosa.resample(ref_audio, orig_sr=sr, target_sr=16000)
            syn_audio = librosa.resample(syn_audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        # 对齐长度
        min_len = min(len(ref_audio), len(syn_audio))
        ref_audio = ref_audio[:min_len]
        syn_audio = syn_audio[:min_len]

        # 计算STOI
        score = stoi(ref_audio, syn_audio, sr, extended=False)

        return float(score)

    except ImportError:
        logger.warning("pystoi未安装，跳过STOI计算")
        return 0.0
    except Exception as e:
        logger.warning(f"STOI计算失败: {e}")
        return 0.0


class TTSEvaluator:
    """TTS评估器"""

    def __init__(self, config: dict):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config
        self.eval_config = config.get("evaluation", {})
        self.sample_rate = config["audio"]["sample_rate"]

    def evaluate_single(
        self,
        ref_audio_path: str,
        syn_audio_path: str
    ) -> Dict[str, float]:
        """
        评估单个样本

        Args:
            ref_audio_path: 参考音频路径
            syn_audio_path: 合成音频路径

        Returns:
            评估指标字典
        """
        # 加载音频
        ref_audio, _ = librosa.load(ref_audio_path, sr=self.sample_rate)
        syn_audio, _ = librosa.load(syn_audio_path, sr=self.sample_rate)

        results = {}

        # MCD
        results["mcd"] = compute_mel_cepstral_distortion(
            ref_audio, syn_audio, self.sample_rate
        )

        # 说话人相似度
        results["speaker_similarity"] = compute_speaker_similarity(
            ref_audio, syn_audio, self.sample_rate
        )

        # PESQ
        results["pesq"] = compute_pesq(ref_audio, syn_audio, self.sample_rate)

        # STOI
        results["stoi"] = compute_stoi(ref_audio, syn_audio, self.sample_rate)

        return results

    def evaluate_batch(
        self,
        ref_audio_paths: List[str],
        syn_audio_paths: List[str]
    ) -> Dict[str, float]:
        """
        批量评估

        Args:
            ref_audio_paths: 参考音频路径列表
            syn_audio_paths: 合成音频路径列表

        Returns:
            平均评估指标字典
        """
        all_results = []

        for ref_path, syn_path in tqdm(
            zip(ref_audio_paths, syn_audio_paths),
            total=len(ref_audio_paths),
            desc="评估中"
        ):
            try:
                results = self.evaluate_single(ref_path, syn_path)
                all_results.append(results)
            except Exception as e:
                logger.warning(f"评估失败 {ref_path}: {e}")
                continue

        if not all_results:
            return {}

        # 计算平均值
        avg_results = {}
        for key in all_results[0].keys():
            values = [r[key] for r in all_results if key in r]
            avg_results[f"avg_{key}"] = np.mean(values)
            avg_results[f"std_{key}"] = np.std(values)

        avg_results["num_samples"] = len(all_results)

        return avg_results

    def evaluate_model(
        self,
        model_inference_fn,
        test_texts: List[str],
        reference_audios: List[str],
        output_dir: str
    ) -> Dict:
        """
        评估模型

        Args:
            model_inference_fn: 模型推理函数
            test_texts: 测试文本列表
            reference_audios: 参考音频列表
            output_dir: 输出目录

        Returns:
            评估结果
        """
        os.makedirs(output_dir, exist_ok=True)

        syn_audio_paths = []

        # 生成合成音频
        logger.info("生成合成音频...")
        for i, text in enumerate(tqdm(test_texts)):
            output_path = os.path.join(output_dir, f"syn_{i:04d}.wav")
            model_inference_fn(text, output_path)
            syn_audio_paths.append(output_path)

        # 评估
        logger.info("计算评估指标...")
        results = self.evaluate_batch(reference_audios, syn_audio_paths)

        # 保存结果
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"评估结果已保存到: {results_path}")

        return results


def print_evaluation_report(results: Dict):
    """
    打印评估报告

    Args:
        results: 评估结果字典
    """
    print("\n" + "=" * 50)
    print("TTS模型评估报告")
    print("=" * 50)

    print(f"\n样本数量: {results.get('num_samples', 'N/A')}")

    print("\n音质指标:")
    print(f"  MCD (越低越好): {results.get('avg_mcd', 'N/A'):.2f} ± {results.get('std_mcd', 0):.2f} dB")
    print(f"  PESQ (越高越好): {results.get('avg_pesq', 'N/A'):.2f} ± {results.get('std_pesq', 0):.2f}")
    print(f"  STOI (越高越好): {results.get('avg_stoi', 'N/A'):.3f} ± {results.get('std_stoi', 0):.3f}")

    print("\n音色相似度:")
    print(f"  说话人相似度: {results.get('avg_speaker_similarity', 'N/A'):.3f} ± {results.get('std_speaker_similarity', 0):.3f}")

    print("\n" + "=" * 50)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="TTS评估")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--ref-dir",
        type=str,
        required=True,
        help="参考音频目录"
    )
    parser.add_argument(
        "--syn-dir",
        type=str,
        required=True,
        help="合成音频目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="输出结果文件"
    )

    args = parser.parse_args()

    # 切换到项目根目录
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    config = load_config(args.config)

    # 获取音频文件列表
    ref_files = sorted(Path(args.ref_dir).glob("*.wav"))
    syn_files = sorted(Path(args.syn_dir).glob("*.wav"))

    if len(ref_files) != len(syn_files):
        logger.warning(f"参考音频({len(ref_files)})和合成音频({len(syn_files)})数量不匹配")

    # 评估
    evaluator = TTSEvaluator(config)
    results = evaluator.evaluate_batch(
        [str(f) for f in ref_files],
        [str(f) for f in syn_files]
    )

    # 保存结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印报告
    print_evaluation_report(results)


if __name__ == "__main__":
    main()
