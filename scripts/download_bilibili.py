#!/usr/bin/env python3
"""
B站视频下载器 - 下载秦彻相关视频的音频
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import yaml
from loguru import logger


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_bvid(url: str) -> str:
    """从B站URL中提取BV号"""
    import re
    match = re.search(r'BV[\w]+', url)
    if match:
        return match.group()
    raise ValueError(f"无法从URL中提取BV号: {url}")


def download_bilibili_audio(
    url: str,
    output_dir: str,
    output_name: str,
    cookies_file: Optional[str] = None
) -> str:
    """
    使用yt-dlp下载B站视频的音频

    Args:
        url: B站视频URL
        output_dir: 输出目录
        output_name: 输出文件名(不含扩展名)
        cookies_file: cookies文件路径(可选，用于下载高清音频)

    Returns:
        下载的音频文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{output_name}.%(ext)s")

    cmd = [
        "yt-dlp",
        "-x",  # 仅提取音频
        "--audio-format", "wav",  # 转换为wav格式
        "--audio-quality", "0",  # 最高质量
        "-o", output_path,
        "--no-playlist",  # 不下载播放列表
        "--verbose",
    ]

    if cookies_file and os.path.exists(cookies_file):
        cmd.extend(["--cookies", cookies_file])

    cmd.append(url)

    logger.info(f"开始下载: {url}")
    logger.info(f"命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"下载完成: {output_name}")

        # 查找下载的文件
        for ext in ["wav", "m4a", "mp3", "opus", "webm"]:
            file_path = os.path.join(output_dir, f"{output_name}.{ext}")
            if os.path.exists(file_path):
                return file_path

        # 如果没找到，列出目录内容
        files = os.listdir(output_dir)
        for f in files:
            if output_name in f:
                return os.path.join(output_dir, f)

        raise FileNotFoundError(f"下载完成但未找到输出文件: {output_name}")

    except subprocess.CalledProcessError as e:
        logger.error(f"下载失败: {e.stderr}")
        raise


def convert_to_wav(input_path: str, output_path: str, sample_rate: int = 24000) -> str:
    """
    使用ffmpeg将音频转换为指定采样率的WAV格式

    Args:
        input_path: 输入音频路径
        output_path: 输出WAV路径
        sample_rate: 目标采样率

    Returns:
        输出文件路径
    """
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-ar", str(sample_rate),
        "-ac", "1",  # 单声道
        "-y",  # 覆盖已存在的文件
        output_path
    ]

    logger.info(f"转换音频: {input_path} -> {output_path}")

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        logger.info(f"转换完成: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"转换失败: {e.stderr}")
        raise


def download_all_videos(config: dict, cookies_file: Optional[str] = None) -> List[str]:
    """
    下载配置中所有的B站视频音频

    Args:
        config: 配置字典
        cookies_file: cookies文件路径

    Returns:
        下载的音频文件路径列表
    """
    project_root = config["paths"]["project_root"]
    raw_data_dir = os.path.join(project_root, config["paths"]["raw_data"])
    sample_rate = config["audio"]["sample_rate"]

    downloaded_files = []

    for video_info in config["data_sources"]["bilibili_videos"]:
        url = video_info["url"]
        name = video_info["name"]

        try:
            # 下载音频
            downloaded_path = download_bilibili_audio(
                url=url,
                output_dir=raw_data_dir,
                output_name=name,
                cookies_file=cookies_file
            )

            # 如果不是wav格式，转换为wav
            if not downloaded_path.endswith(".wav"):
                wav_path = os.path.join(raw_data_dir, f"{name}.wav")
                convert_to_wav(downloaded_path, wav_path, sample_rate)
                # 删除原始文件
                os.remove(downloaded_path)
                downloaded_path = wav_path

            downloaded_files.append(downloaded_path)
            logger.info(f"成功处理: {name}")

        except Exception as e:
            logger.error(f"处理视频失败 {name}: {e}")
            continue

    return downloaded_files


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="下载B站视频音频")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--cookies",
        type=str,
        default=None,
        help="B站cookies文件路径(可选)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="单独下载指定URL(可选)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="custom_video",
        help="自定义视频名称"
    )

    args = parser.parse_args()

    # 切换到项目根目录
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    config = load_config(args.config)

    if args.url:
        # 下载单个视频
        project_root = config["paths"]["project_root"]
        raw_data_dir = os.path.join(project_root, config["paths"]["raw_data"])

        downloaded_path = download_bilibili_audio(
            url=args.url,
            output_dir=raw_data_dir,
            output_name=args.name,
            cookies_file=args.cookies
        )
        logger.info(f"下载完成: {downloaded_path}")
    else:
        # 下载所有配置的视频
        downloaded_files = download_all_videos(config, args.cookies)
        logger.info(f"共下载 {len(downloaded_files)} 个文件")
        for f in downloaded_files:
            logger.info(f"  - {f}")


if __name__ == "__main__":
    main()
