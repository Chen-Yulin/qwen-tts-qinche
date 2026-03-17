#!/usr/bin/env python3
"""
语音转录模块 - 使用Whisper生成音频转录文本
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml
from loguru import logger
from tqdm import tqdm


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class WhisperTranscriber:
    """Whisper转录器"""

    def __init__(self, config: dict):
        """
        初始化转录器

        Args:
            config: 配置字典
        """
        self.config = config
        trans_config = config["transcription"]

        self.model_name = trans_config["model"]
        self.language = trans_config["language"]
        self.device = trans_config["device"]
        self.compute_type = trans_config["compute_type"]
        self.vad_filter = trans_config["vad_filter"]
        self.vad_parameters = trans_config.get("vad_parameters", {})

        self.model = None

    def load_model(self):
        """加载Whisper模型"""
        if self.model is not None:
            return

        try:
            from faster_whisper import WhisperModel
            logger.info(f"加载Whisper模型: {self.model_name}")
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info("模型加载完成")
        except ImportError:
            logger.warning("faster-whisper未安装，尝试使用funasr")
            self._use_funasr = True
            self._load_funasr()

    def _load_funasr(self):
        """加载FunASR模型作为备选"""
        from funasr import AutoModel
        logger.info("加载FunASR模型...")
        self.model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            device=self.device
        )
        logger.info("FunASR模型加载完成")

    def transcribe(self, audio_path: str) -> Dict:
        """
        转录单个音频文件

        Args:
            audio_path: 音频文件路径

        Returns:
            转录结果字典
        """
        self.load_model()

        if hasattr(self, '_use_funasr') and self._use_funasr:
            return self._transcribe_funasr(audio_path)
        else:
            return self._transcribe_whisper(audio_path)

    def _transcribe_whisper(self, audio_path: str) -> Dict:
        """使用faster-whisper转录"""
        segments, info = self.model.transcribe(
            audio_path,
            language=self.language,
            vad_filter=self.vad_filter,
            vad_parameters=self.vad_parameters
        )

        # 收集所有片段
        text_segments = []
        full_text = ""

        for segment in segments:
            text_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            full_text += segment.text

        return {
            "audio_path": audio_path,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "text": full_text.strip(),
            "segments": text_segments
        }

    def _transcribe_funasr(self, audio_path: str) -> Dict:
        """使用FunASR转录"""
        result = self.model.generate(input=audio_path)

        if result and len(result) > 0:
            text = result[0].get("text", "")
        else:
            text = ""

        return {
            "audio_path": audio_path,
            "language": "zh",
            "text": text.strip(),
            "segments": []
        }


def transcribe_audio_segments(
    config: dict,
    audio_dir: Optional[str] = None
) -> List[Dict]:
    """
    转录所有音频片段

    Args:
        config: 配置字典
        audio_dir: 音频目录(可选，默认使用配置中的路径)

    Returns:
        转录结果列表
    """
    project_root = config["paths"]["project_root"]

    if audio_dir is None:
        audio_dir = os.path.join(project_root, config["paths"]["audio_segments"])

    # 查找所有wav文件
    wav_files = sorted(Path(audio_dir).glob("*.wav"))
    logger.info(f"找到 {len(wav_files)} 个音频文件待转录")

    if not wav_files:
        logger.warning("未找到音频文件")
        return []

    # 初始化转录器
    transcriber = WhisperTranscriber(config)

    results = []
    for wav_file in tqdm(wav_files, desc="转录音频"):
        try:
            result = transcriber.transcribe(str(wav_file))
            results.append(result)
        except Exception as e:
            logger.error(f"转录失败 {wav_file.name}: {e}")
            continue

    return results


def save_transcriptions(
    results: List[Dict],
    output_path: str,
    format: str = "jsonl"
):
    """
    保存转录结果

    Args:
        results: 转录结果列表
        output_path: 输出路径
        format: 输出格式 (jsonl, json, txt)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    elif format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    elif format == "txt":
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                audio_name = Path(result["audio_path"]).stem
                f.write(f"{audio_name}\t{result['text']}\n")

    logger.info(f"转录结果已保存到: {output_path}")


def create_training_manifest(
    transcriptions: List[Dict],
    output_path: str,
    audio_base_dir: str
):
    """
    创建训练用的manifest文件

    Args:
        transcriptions: 转录结果列表
        output_path: 输出路径
        audio_base_dir: 音频文件基础目录
    """
    manifest_data = []

    for trans in transcriptions:
        audio_path = trans["audio_path"]
        text = trans["text"]

        if not text or len(text.strip()) < 2:
            continue

        # 获取相对路径
        rel_path = os.path.relpath(audio_path, audio_base_dir)

        manifest_data.append({
            "audio_filepath": rel_path,
            "text": text,
            "duration": trans.get("duration", 0)
        })

    # 保存manifest
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in manifest_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"训练manifest已保存: {output_path}")
    logger.info(f"共 {len(manifest_data)} 条有效数据")

    return manifest_data


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="音频转录")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="音频目录(可选)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["jsonl", "json", "txt"],
        default="jsonl",
        help="输出格式"
    )

    args = parser.parse_args()

    # 切换到项目根目录
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    config = load_config(args.config)
    project_root = config["paths"]["project_root"]

    # 转录
    results = transcribe_audio_segments(config, args.audio_dir)

    if not results:
        logger.warning("没有转录结果")
        return

    # 保存转录结果
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            project_root,
            config["paths"]["processed_data"],
            f"transcriptions.{args.format}"
        )

    save_transcriptions(results, output_path, args.format)

    # 创建训练manifest
    manifest_path = os.path.join(
        project_root,
        config["paths"]["processed_data"],
        "train_manifest.jsonl"
    )
    audio_base_dir = os.path.join(project_root, config["paths"]["audio_segments"])
    create_training_manifest(results, manifest_path, audio_base_dir)


if __name__ == "__main__":
    main()
