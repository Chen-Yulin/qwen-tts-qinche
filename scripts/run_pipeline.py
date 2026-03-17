#!/usr/bin/env python3
"""
完整的数据处理管线 - 从下载到准备训练数据
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
from loguru import logger

# 添加scripts目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from download_bilibili import download_all_videos
from audio_preprocessing import process_all_audio
from transcribe import transcribe_audio_segments, save_transcriptions, create_training_manifest
from prepare_dataset import split_dataset


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(
    config: dict,
    skip_download: bool = False,
    skip_preprocess: bool = False,
    skip_transcribe: bool = False,
    cookies_file: str = None,
    apply_denoise: bool = True
):
    """
    运行完整的数据处理管线

    Args:
        config: 配置字典
        skip_download: 跳过下载步骤
        skip_preprocess: 跳过预处理步骤
        skip_transcribe: 跳过转录步骤
        cookies_file: B站cookies文件
        apply_denoise: 是否应用降噪
    """
    project_root = config["paths"]["project_root"]

    # Step 1: 下载B站视频音频
    if not skip_download:
        logger.info("=" * 50)
        logger.info("Step 1: 下载B站视频音频")
        logger.info("=" * 50)

        try:
            downloaded_files = download_all_videos(config, cookies_file)
            logger.info(f"下载完成，共 {len(downloaded_files)} 个文件")
        except Exception as e:
            logger.error(f"下载失败: {e}")
            logger.info("请手动下载音频文件到 data/raw 目录")
    else:
        logger.info("跳过下载步骤")

    # Step 2: 音频预处理(分段、降噪)
    if not skip_preprocess:
        logger.info("=" * 50)
        logger.info("Step 2: 音频预处理")
        logger.info("=" * 50)

        try:
            segment_paths = process_all_audio(config, apply_denoise)
            logger.info(f"预处理完成，共 {len(segment_paths)} 个音频片段")
        except Exception as e:
            logger.error(f"预处理失败: {e}")
            raise
    else:
        logger.info("跳过预处理步骤")

    # Step 3: 语音转录
    if not skip_transcribe:
        logger.info("=" * 50)
        logger.info("Step 3: 语音转录")
        logger.info("=" * 50)

        try:
            transcriptions = transcribe_audio_segments(config)

            # 保存转录结果
            processed_dir = os.path.join(project_root, config["paths"]["processed_data"])
            trans_path = os.path.join(processed_dir, "transcriptions.jsonl")
            save_transcriptions(transcriptions, trans_path, "jsonl")

            # 创建训练manifest
            manifest_path = os.path.join(processed_dir, "train_manifest.jsonl")
            audio_base_dir = os.path.join(project_root, config["paths"]["audio_segments"])
            create_training_manifest(transcriptions, manifest_path, audio_base_dir)

            logger.info(f"转录完成，共 {len(transcriptions)} 条")
        except Exception as e:
            logger.error(f"转录失败: {e}")
            raise
    else:
        logger.info("跳过转录步骤")

    # Step 4: 划分数据集
    logger.info("=" * 50)
    logger.info("Step 4: 划分数据集")
    logger.info("=" * 50)

    try:
        processed_dir = os.path.join(project_root, config["paths"]["processed_data"])
        manifest_path = os.path.join(processed_dir, "train_manifest.jsonl")

        if os.path.exists(manifest_path):
            train_path, val_path = split_dataset(manifest_path, processed_dir)
            logger.info(f"数据集划分完成")
            logger.info(f"  训练集: {train_path}")
            logger.info(f"  验证集: {val_path}")
        else:
            logger.warning(f"Manifest文件不存在: {manifest_path}")
    except Exception as e:
        logger.error(f"数据集划分失败: {e}")
        raise

    logger.info("=" * 50)
    logger.info("数据处理管线完成!")
    logger.info("=" * 50)

    # 打印数据统计
    print_data_statistics(config)


def print_data_statistics(config: dict):
    """打印数据统计信息"""
    project_root = config["paths"]["project_root"]

    print("\n数据统计:")
    print("-" * 30)

    # 原始音频
    raw_dir = os.path.join(project_root, config["paths"]["raw_data"])
    if os.path.exists(raw_dir):
        raw_files = list(Path(raw_dir).glob("*.wav"))
        print(f"原始音频文件: {len(raw_files)}")

    # 音频片段
    segments_dir = os.path.join(project_root, config["paths"]["audio_segments"])
    if os.path.exists(segments_dir):
        segment_files = list(Path(segments_dir).glob("*.wav"))
        print(f"音频片段: {len(segment_files)}")

        # 计算总时长
        total_duration = 0
        for f in segment_files:
            import librosa
            try:
                duration = librosa.get_duration(path=str(f))
                total_duration += duration
            except:
                pass
        print(f"总时长: {total_duration / 60:.1f} 分钟")

    # 训练数据
    processed_dir = os.path.join(project_root, config["paths"]["processed_data"])
    train_path = os.path.join(processed_dir, "train.jsonl")
    val_path = os.path.join(processed_dir, "val.jsonl")

    if os.path.exists(train_path):
        with open(train_path, "r") as f:
            train_count = sum(1 for _ in f)
        print(f"训练样本: {train_count}")

    if os.path.exists(val_path):
        with open(val_path, "r") as f:
            val_count = sum(1 for _ in f)
        print(f"验证样本: {val_count}")

    print("-" * 30)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据处理管线")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过下载步骤"
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="跳过预处理步骤"
    )
    parser.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="跳过转录步骤"
    )
    parser.add_argument(
        "--cookies",
        type=str,
        default=None,
        help="B站cookies文件路径"
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

    run_pipeline(
        config,
        skip_download=args.skip_download,
        skip_preprocess=args.skip_preprocess,
        skip_transcribe=args.skip_transcribe,
        cookies_file=args.cookies,
        apply_denoise=not args.no_denoise
    )


if __name__ == "__main__":
    main()
