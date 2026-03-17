#!/usr/bin/env python3
"""
数据集准备模块 - 创建TTS微调数据集
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import yaml
from loguru import logger
from tqdm import tqdm


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TTSDataset(Dataset):
    """TTS微调数据集"""

    def __init__(
        self,
        manifest_path: str,
        audio_base_dir: str,
        sample_rate: int = 24000,
        max_audio_length: float = 15.0,
        speaker_name: str = "秦彻"
    ):
        """
        初始化数据集

        Args:
            manifest_path: manifest文件路径
            audio_base_dir: 音频文件基础目录
            sample_rate: 采样率
            max_audio_length: 最大音频长度(秒)
            speaker_name: 说话人名称
        """
        self.audio_base_dir = audio_base_dir
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.speaker_name = speaker_name

        # 加载manifest
        self.data = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                # 过滤过长的音频
                if item.get("duration", 0) <= max_audio_length:
                    self.data.append(item)

        logger.info(f"加载了 {len(self.data)} 条数据")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        audio_path = os.path.join(self.audio_base_dir, item["audio_filepath"])
        text = item["text"]

        # 加载音频
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        return {
            "audio": audio,
            "audio_path": audio_path,
            "text": text,
            "speaker": self.speaker_name,
            "duration": len(audio) / self.sample_rate
        }


class QwenTTSDataset(Dataset):
    """
    Qwen-TTS 格式的数据集
    适配 Qwen2-Audio 或类似模型的输入格式
    """

    def __init__(
        self,
        manifest_path: str,
        audio_base_dir: str,
        processor,
        sample_rate: int = 24000,
        max_audio_length: float = 15.0,
        speaker_name: str = "秦彻"
    ):
        """
        初始化数据集

        Args:
            manifest_path: manifest文件路径
            audio_base_dir: 音频文件基础目录
            processor: 模型的processor
            sample_rate: 采样率
            max_audio_length: 最大音频长度(秒)
            speaker_name: 说话人名称
        """
        self.audio_base_dir = audio_base_dir
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.speaker_name = speaker_name

        # 加载manifest
        self.data = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get("duration", 0) <= max_audio_length:
                    self.data.append(item)

        logger.info(f"加载了 {len(self.data)} 条数据")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        audio_path = os.path.join(self.audio_base_dir, item["audio_filepath"])
        text = item["text"]

        # 加载音频
        audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # 构建对话格式
        # TTS任务: 输入文本，输出音频
        conversation = [
            {
                "role": "user",
                "content": f"请用{self.speaker_name}的声音朗读以下文本：{text}"
            },
            {
                "role": "assistant",
                "content": f"<audio>{audio_path}</audio>"
            }
        ]

        return {
            "audio": audio,
            "audio_path": audio_path,
            "text": text,
            "speaker": self.speaker_name,
            "conversation": conversation
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    数据批处理函数

    Args:
        batch: 批次数据列表

    Returns:
        批处理后的数据字典
    """
    # 找到最长的音频
    max_len = max(len(item["audio"]) for item in batch)

    # 填充音频
    audios = []
    audio_lengths = []
    for item in batch:
        audio = item["audio"]
        audio_lengths.append(len(audio))
        # 填充到最大长度
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))
        audios.append(audio)

    return {
        "audio": torch.tensor(np.array(audios), dtype=torch.float32),
        "audio_lengths": torch.tensor(audio_lengths, dtype=torch.long),
        "texts": [item["text"] for item in batch],
        "speakers": [item["speaker"] for item in batch],
        "audio_paths": [item["audio_path"] for item in batch]
    }


def split_dataset(
    manifest_path: str,
    output_dir: str,
    train_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[str, str]:
    """
    划分训练集和验证集

    Args:
        manifest_path: 原始manifest路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        seed: 随机种子

    Returns:
        (训练集路径, 验证集路径)
    """
    random.seed(seed)

    # 读取所有数据
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]

    # 打乱数据
    random.shuffle(data)

    # 划分
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # 保存
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"训练集: {len(train_data)} 条")
    logger.info(f"验证集: {len(val_data)} 条")

    return train_path, val_path


def prepare_dataset(config: dict) -> Tuple[str, str]:
    """
    准备完整的数据集

    Args:
        config: 配置字典

    Returns:
        (训练集路径, 验证集路径)
    """
    project_root = config["paths"]["project_root"]
    processed_dir = os.path.join(project_root, config["paths"]["processed_data"])

    manifest_path = os.path.join(processed_dir, "train_manifest.jsonl")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest文件不存在: {manifest_path}")

    # 划分数据集
    train_path, val_path = split_dataset(
        manifest_path,
        processed_dir,
        train_ratio=0.9
    )

    return train_path, val_path


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="准备数据集")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径"
    )

    args = parser.parse_args()

    # 切换到项目根目录
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    config = load_config(args.config)

    train_path, val_path = prepare_dataset(config)
    logger.info(f"训练集: {train_path}")
    logger.info(f"验证集: {val_path}")


if __name__ == "__main__":
    main()
