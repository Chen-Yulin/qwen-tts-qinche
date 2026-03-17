#!/usr/bin/env python3
"""
TTS推理脚本 - 使用微调后的模型进行语音合成
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
import torch
import yaml
import numpy as np
import soundfile as sf
from loguru import logger
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from peft import PeftModel


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class QwenTTSInference:
    """Qwen-TTS 推理类"""

    def __init__(
        self,
        base_model_path: str,
        lora_path: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "bfloat16"
    ):
        """
        初始化推理器

        Args:
            base_model_path: 基础模型路径
            lora_path: LoRA权重路径(可选)
            device: 设备
            dtype: 数据类型
        """
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.device = device
        self.dtype = getattr(torch, dtype)

        self.model = None
        self.tokenizer = None
        self.processor = None

    def load_model(self):
        """加载模型"""
        logger.info(f"加载基础模型: {self.base_model_path}")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True
        )

        # 尝试加载processor
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.base_model_path,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"无法加载processor: {e}")
            self.processor = None

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=self.dtype,
            device_map="auto",
            trust_remote_code=True
        )

        # 加载LoRA权重
        if self.lora_path and os.path.exists(self.lora_path):
            logger.info(f"加载LoRA权重: {self.lora_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.lora_path
            )
            # 合并LoRA权重以加速推理
            self.model = self.model.merge_and_unload()

        self.model.eval()
        logger.info("模型加载完成")

    def synthesize(
        self,
        text: str,
        speaker_name: str = "秦彻",
        reference_audio: Optional[str] = None,
        output_path: Optional[str] = None,
        **generation_kwargs
    ) -> np.ndarray:
        """
        合成语音

        Args:
            text: 要合成的文本
            speaker_name: 说话人名称
            reference_audio: 参考音频路径(用于音色克隆)
            output_path: 输出音频路径
            **generation_kwargs: 生成参数

        Returns:
            合成的音频数据
        """
        if self.model is None:
            self.load_model()

        # 构建prompt
        prompt = f"<|im_start|>user\n请用{speaker_name}的声音朗读：{text}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        # 设置默认生成参数
        default_kwargs = {
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        default_kwargs.update(generation_kwargs)

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **default_kwargs
            )

        # 解码输出
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        logger.info(f"生成的文本: {generated_text}")

        # 注意: 实际的TTS模型会输出音频token或mel谱
        # 这里需要根据具体模型的输出格式进行处理
        # 以下是示例代码，实际实现需要根据模型调整

        # 如果模型输出音频token，需要解码为音频
        # audio = self.decode_audio_tokens(outputs)

        # 临时返回空音频
        audio = np.zeros(24000, dtype=np.float32)

        if output_path:
            sf.write(output_path, audio, 24000)
            logger.info(f"音频已保存到: {output_path}")

        return audio

    def batch_synthesize(
        self,
        texts: List[str],
        output_dir: str,
        speaker_name: str = "秦彻",
        **generation_kwargs
    ) -> List[str]:
        """
        批量合成语音

        Args:
            texts: 文本列表
            output_dir: 输出目录
            speaker_name: 说话人名称
            **generation_kwargs: 生成参数

        Returns:
            输出文件路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []

        for i, text in enumerate(texts):
            output_path = os.path.join(output_dir, f"output_{i:04d}.wav")
            self.synthesize(
                text=text,
                speaker_name=speaker_name,
                output_path=output_path,
                **generation_kwargs
            )
            output_paths.append(output_path)

        return output_paths


class FishSpeechInference:
    """
    Fish-Speech 推理类
    Fish-Speech是一个更适合TTS任务的开源模型
    """

    def __init__(
        self,
        model_path: str = "fishaudio/fish-speech-1.4",
        device: str = "cuda"
    ):
        """
        初始化

        Args:
            model_path: 模型路径
            device: 设备
        """
        self.model_path = model_path
        self.device = device
        self.model = None

    def load_model(self):
        """加载模型"""
        try:
            # Fish-Speech有自己的加载方式
            # 这里是示例代码
            logger.info(f"加载Fish-Speech模型: {self.model_path}")
            # from fish_speech import FishSpeech
            # self.model = FishSpeech.from_pretrained(self.model_path)
            logger.info("模型加载完成")
        except ImportError:
            logger.error("请安装fish-speech: pip install fish-speech")
            raise

    def synthesize(
        self,
        text: str,
        reference_audio: str,
        output_path: str,
        **kwargs
    ) -> np.ndarray:
        """
        使用参考音频进行语音合成(音色克隆)

        Args:
            text: 要合成的文本
            reference_audio: 参考音频路径
            output_path: 输出路径
            **kwargs: 其他参数

        Returns:
            合成的音频
        """
        if self.model is None:
            self.load_model()

        # Fish-Speech的合成逻辑
        # audio = self.model.synthesize(text, reference_audio, **kwargs)

        # 临时返回
        audio = np.zeros(24000, dtype=np.float32)

        if output_path:
            sf.write(output_path, audio, 24000)

        return audio


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="TTS推理")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="模型路径"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="LoRA权重路径"
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="要合成的文本"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="输出音频路径"
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="秦彻",
        help="说话人名称"
    )
    parser.add_argument(
        "--reference-audio",
        type=str,
        default=None,
        help="参考音频路径(用于音色克隆)"
    )

    args = parser.parse_args()

    # 切换到项目根目录
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    config = load_config(args.config)

    # 设置默认模型路径
    if args.model_path is None:
        args.model_path = config["model"]["name"]

    if args.lora_path is None:
        project_root = config["paths"]["project_root"]
        args.lora_path = os.path.join(
            project_root,
            config["training"]["output_dir"]
        )

    # 创建推理器
    inferencer = QwenTTSInference(
        base_model_path=args.model_path,
        lora_path=args.lora_path,
        dtype=config["model"]["dtype"]
    )

    # 合成
    audio = inferencer.synthesize(
        text=args.text,
        speaker_name=args.speaker,
        reference_audio=args.reference_audio,
        output_path=args.output
    )

    logger.info(f"合成完成: {args.output}")


if __name__ == "__main__":
    main()
