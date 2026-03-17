#!/usr/bin/env python3
"""
音频预处理模块 - 音频分段、降噪、格式转换
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from scipy import signal
import yaml
from loguru import logger
from tqdm import tqdm


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_audio(file_path: str, target_sr: int = 24000) -> Tuple[np.ndarray, int]:
    """
    加载音频文件

    Args:
        file_path: 音频文件路径
        target_sr: 目标采样率

    Returns:
        (音频数据, 采样率)
    """
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    音频响度归一化

    Args:
        audio: 音频数据
        target_db: 目标响度(dB)

    Returns:
        归一化后的音频
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        current_db = 20 * np.log10(rms)
        gain = 10 ** ((target_db - current_db) / 20)
        audio = audio * gain
        # 防止削波
        max_val = np.max(np.abs(audio))
        if max_val > 0.99:
            audio = audio * 0.99 / max_val
    return audio


def remove_silence(
    audio: np.ndarray,
    sr: int,
    silence_threshold: float = -40,
    min_silence_duration: float = 0.5
) -> np.ndarray:
    """
    移除音频首尾的静音部分

    Args:
        audio: 音频数据
        sr: 采样率
        silence_threshold: 静音阈值(dB)
        min_silence_duration: 最小静音时长(秒)

    Returns:
        去除静音后的音频
    """
    # 转换为pydub格式
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )

    # 检测非静音部分
    nonsilent_ranges = detect_nonsilent(
        audio_segment,
        min_silence_len=int(min_silence_duration * 1000),
        silence_thresh=silence_threshold
    )

    if not nonsilent_ranges:
        return audio

    # 获取首尾非静音位置
    start_ms = nonsilent_ranges[0][0]
    end_ms = nonsilent_ranges[-1][1]

    # 转换为采样点
    start_sample = int(start_ms * sr / 1000)
    end_sample = int(end_ms * sr / 1000)

    return audio[start_sample:end_sample]


def segment_audio_by_silence(
    audio: np.ndarray,
    sr: int,
    min_duration: float = 2.0,
    max_duration: float = 15.0,
    silence_threshold: float = -40,
    min_silence_duration: float = 0.5
) -> List[Tuple[np.ndarray, float, float]]:
    """
    基于静音检测分割音频

    Args:
        audio: 音频数据
        sr: 采样率
        min_duration: 最小片段时长(秒)
        max_duration: 最大片段时长(秒)
        silence_threshold: 静音阈值(dB)
        min_silence_duration: 最小静音时长(秒)

    Returns:
        [(音频片段, 开始时间, 结束时间), ...]
    """
    # 转换为pydub格式
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )

    # 检测非静音部分
    nonsilent_ranges = detect_nonsilent(
        audio_segment,
        min_silence_len=int(min_silence_duration * 1000),
        silence_thresh=silence_threshold
    )

    if not nonsilent_ranges:
        return []

    segments = []
    current_start = nonsilent_ranges[0][0]
    current_end = nonsilent_ranges[0][1]

    for start_ms, end_ms in nonsilent_ranges[1:]:
        # 如果当前片段加上新片段不超过最大时长，合并
        if (end_ms - current_start) / 1000 <= max_duration:
            current_end = end_ms
        else:
            # 保存当前片段(如果满足最小时长)
            duration = (current_end - current_start) / 1000
            if duration >= min_duration:
                start_sample = int(current_start * sr / 1000)
                end_sample = int(current_end * sr / 1000)
                segment_audio = audio[start_sample:end_sample]
                segments.append((
                    segment_audio,
                    current_start / 1000,
                    current_end / 1000
                ))
            # 开始新片段
            current_start = start_ms
            current_end = end_ms

    # 处理最后一个片段
    duration = (current_end - current_start) / 1000
    if duration >= min_duration:
        start_sample = int(current_start * sr / 1000)
        end_sample = int(current_end * sr / 1000)
        segment_audio = audio[start_sample:end_sample]
        segments.append((
            segment_audio,
            current_start / 1000,
            current_end / 1000
        ))

    # 处理超长片段
    final_segments = []
    for segment_audio, start_time, end_time in segments:
        duration = end_time - start_time
        if duration > max_duration:
            # 按固定长度切分
            num_splits = int(np.ceil(duration / max_duration))
            split_duration = duration / num_splits
            split_samples = int(split_duration * sr)

            for i in range(num_splits):
                split_start = i * split_samples
                split_end = min((i + 1) * split_samples, len(segment_audio))
                split_audio = segment_audio[split_start:split_end]

                if len(split_audio) / sr >= min_duration:
                    final_segments.append((
                        split_audio,
                        start_time + i * split_duration,
                        start_time + (i + 1) * split_duration
                    ))
        else:
            final_segments.append((segment_audio, start_time, end_time))

    return final_segments


def apply_noise_reduction(
    audio: np.ndarray,
    sr: int,
    noise_reduce_strength: float = 0.5
) -> np.ndarray:
    """
    简单的降噪处理(使用频谱门限)

    Args:
        audio: 音频数据
        sr: 采样率
        noise_reduce_strength: 降噪强度(0-1)

    Returns:
        降噪后的音频
    """
    # 计算短时傅里叶变换
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # ��计噪声(使用前几帧)
    noise_frames = min(10, magnitude.shape[1])
    noise_estimate = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

    # 频谱减法
    threshold = noise_estimate * (1 + noise_reduce_strength * 2)
    magnitude_cleaned = np.maximum(magnitude - threshold, 0)

    # 重建音频
    stft_cleaned = magnitude_cleaned * np.exp(1j * phase)
    audio_cleaned = librosa.istft(stft_cleaned, hop_length=hop_length)

    # 确保长度一致
    if len(audio_cleaned) < len(audio):
        audio_cleaned = np.pad(audio_cleaned, (0, len(audio) - len(audio_cleaned)))
    elif len(audio_cleaned) > len(audio):
        audio_cleaned = audio_cleaned[:len(audio)]

    return audio_cleaned


def process_audio_file(
    input_path: str,
    output_dir: str,
    config: dict,
    apply_denoise: bool = True
) -> List[str]:
    """
    处理单个音频文件

    Args:
        input_path: 输入音频路径
        output_dir: 输出目录
        config: 配置字典
        apply_denoise: 是否应用降噪

    Returns:
        输出的音频片段路径列表
    """
    audio_config = config["audio"]
    sr = audio_config["sample_rate"]
    segment_config = audio_config["segment"]

    # 加载音频
    logger.info(f"加载音频: {input_path}")
    audio, _ = load_audio(input_path, sr)

    # 降噪
    if apply_denoise:
        logger.info("应用降噪...")
        audio = apply_noise_reduction(audio, sr)

    # 归一化
    audio = normalize_audio(audio)

    # 分段
    logger.info("分割音频...")
    segments = segment_audio_by_silence(
        audio, sr,
        min_duration=segment_config["min_duration"],
        max_duration=segment_config["max_duration"],
        silence_threshold=segment_config["silence_threshold"],
        min_silence_duration=segment_config["min_silence_duration"]
    )

    logger.info(f"共分割出 {len(segments)} 个片段")

    # 保存片段
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(input_path).stem
    output_paths = []

    for i, (segment_audio, start_time, end_time) in enumerate(tqdm(segments, desc="保存片段")):
        # 对每个片段进行归一化和去除首尾静音
        segment_audio = normalize_audio(segment_audio)
        segment_audio = remove_silence(
            segment_audio, sr,
            silence_threshold=segment_config["silence_threshold"],
            min_silence_duration=0.1
        )

        # 检查处理后的长度
        duration = len(segment_audio) / sr
        if duration < segment_config["min_duration"]:
            continue

        output_path = os.path.join(
            output_dir,
            f"{base_name}_{i:04d}_{start_time:.2f}_{end_time:.2f}.wav"
        )
        sf.write(output_path, segment_audio, sr)
        output_paths.append(output_path)

    logger.info(f"保存了 {len(output_paths)} 个有效片段")
    return output_paths


def process_all_audio(config: dict, apply_denoise: bool = True) -> List[str]:
    """
    处理所有原始音频文件

    Args:
        config: 配置字典
        apply_denoise: 是否应用降噪

    Returns:
        所有输出片段的路径列表
    """
    project_root = config["paths"]["project_root"]
    raw_data_dir = os.path.join(project_root, config["paths"]["raw_data"])
    segments_dir = os.path.join(project_root, config["paths"]["audio_segments"])

    all_output_paths = []

    # 查找所有wav文件
    wav_files = list(Path(raw_data_dir).glob("*.wav"))
    logger.info(f"找到 {len(wav_files)} 个原始音频文件")

    for wav_file in wav_files:
        logger.info(f"\n处理: {wav_file.name}")
        output_paths = process_audio_file(
            str(wav_file),
            segments_dir,
            config,
            apply_denoise
        )
        all_output_paths.extend(output_paths)

    logger.info(f"\n总共生成 {len(all_output_paths)} 个音频片段")
    return all_output_paths


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="音频预处理")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="单独处理指定音频文件"
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="不应用降噪"
    )

    args = parser.parse_args()

    # 切换到项目根目录
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    config = load_config(args.config)

    if args.input:
        # 处理单个文件
        project_root = config["paths"]["project_root"]
        segments_dir = os.path.join(project_root, config["paths"]["audio_segments"])
        output_paths = process_audio_file(
            args.input,
            segments_dir,
            config,
            apply_denoise=not args.no_denoise
        )
        logger.info(f"处理完成，生成 {len(output_paths)} 个片段")
    else:
        # 处理所有文件
        all_paths = process_all_audio(config, apply_denoise=not args.no_denoise)
        logger.info(f"全部处理完成，共 {len(all_paths)} 个片段")


if __name__ == "__main__":
    main()
