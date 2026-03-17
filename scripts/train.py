#!/usr/bin/env python3
"""
Qwen-TTS 微调训练脚本
使用 LoRA 进行高效微调
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional
import json
import torch
import yaml
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset, load_dataset
import librosa
import numpy as np
from tqdm import tqdm


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class QwenTTSTrainer:
    """Qwen-TTS 微调训练器"""

    def __init__(self, config: dict):
        """
        初始化训练器

        Args:
            config: 配置字典
        """
        self.config = config
        self.model_config = config["model"]
        self.lora_config = config["lora"]
        self.training_config = config["training"]

        self.model = None
        self.processor = None
        self.tokenizer = None

    def load_model(self):
        """加载预训练模型"""
        model_name = self.model_config["name"]
        dtype = getattr(torch, self.model_config["dtype"])

        logger.info(f"加载模型: {model_name}")

        # 加载tokenizer和processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"无法加载processor: {e}")
            self.processor = None

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.model_config["device_map"],
            trust_remote_code=True
        )

        logger.info("模型加载完成")

    def setup_lora(self):
        """配置LoRA"""
        logger.info("配置LoRA...")

        lora_config = LoraConfig(
            r=self.lora_config["r"],
            lora_alpha=self.lora_config["lora_alpha"],
            lora_dropout=self.lora_config["lora_dropout"],
            target_modules=self.lora_config["target_modules"],
            bias=self.lora_config["bias"],
            task_type=TaskType.CAUSAL_LM
        )

        # 准备模型进行训练
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)

        # 打印可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"可训练参数: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def prepare_training_data(
        self,
        train_manifest: str,
        val_manifest: str,
        audio_base_dir: str
    ) -> tuple:
        """
        准备训练数据

        Args:
            train_manifest: 训练集manifest路径
            val_manifest: 验证集manifest路径
            audio_base_dir: 音频基础目录

        Returns:
            (训练数据集, 验证数据集)
        """
        def load_manifest(manifest_path: str) -> list:
            data = []
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    data.append(item)
            return data

        train_data = load_manifest(train_manifest)
        val_data = load_manifest(val_manifest)

        # 转换为对话格式
        def convert_to_conversation(item: dict) -> dict:
            text = item["text"]
            audio_path = os.path.join(audio_base_dir, item["audio_filepath"])

            # TTS任务的对话格式
            conversation = f"<|im_start|>user\n请用秦彻的声音朗读：{text}<|im_end|>\n<|im_start|>assistant\n"

            return {
                "text": text,
                "audio_path": audio_path,
                "conversation": conversation
            }

        train_dataset = Dataset.from_list([convert_to_conversation(item) for item in train_data])
        val_dataset = Dataset.from_list([convert_to_conversation(item) for item in val_data])

        return train_dataset, val_dataset

    def tokenize_function(self, examples: Dict) -> Dict:
        """
        Tokenize函数

        Args:
            examples: 输入样本

        Returns:
            tokenized结果
        """
        # 对对话进行tokenize
        tokenized = self.tokenizer(
            examples["conversation"],
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_tensors="pt"
        )

        tokenized["labels"] = tokenized["input_ids"].clone()

        return tokenized

    def train(
        self,
        train_manifest: str,
        val_manifest: str,
        audio_base_dir: str,
        output_dir: Optional[str] = None
    ):
        """
        执行训练

        Args:
            train_manifest: 训练集manifest路径
            val_manifest: 验证集manifest路径
            audio_base_dir: 音频基础目录
            output_dir: 输出目录
        """
        if output_dir is None:
            output_dir = self.training_config["output_dir"]

        # 加载模型
        self.load_model()

        # 配置LoRA
        self.setup_lora()

        # 准备数据
        logger.info("准备训练数据...")
        train_dataset, val_dataset = self.prepare_training_data(
            train_manifest, val_manifest, audio_base_dir
        )

        # Tokenize
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )

        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.training_config["num_train_epochs"],
            per_device_train_batch_size=self.training_config["per_device_train_batch_size"],
            per_device_eval_batch_size=self.training_config["per_device_eval_batch_size"],
            gradient_accumulation_steps=self.training_config["gradient_accumulation_steps"],
            learning_rate=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"],
            warmup_ratio=self.training_config["warmup_ratio"],
            lr_scheduler_type=self.training_config["lr_scheduler_type"],
            logging_steps=self.training_config["logging_steps"],
            save_steps=self.training_config["save_steps"],
            eval_steps=self.training_config["eval_steps"],
            eval_strategy="steps",
            save_total_limit=self.training_config["save_total_limit"],
            fp16=self.training_config["fp16"],
            bf16=self.training_config["bf16"],
            gradient_checkpointing=self.training_config["gradient_checkpointing"],
            dataloader_num_workers=self.training_config["dataloader_num_workers"],
            remove_unused_columns=self.training_config["remove_unused_columns"],
            report_to=self.training_config["report_to"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        # 数据整理器
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True
        )

        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        # 开始训练
        logger.info("开始训练...")
        trainer.train()

        # 保存模型
        logger.info(f"保存模型到: {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        logger.info("训练完成!")

    def save_lora_weights(self, output_dir: str):
        """
        仅保存LoRA权重

        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        logger.info(f"LoRA权重已保存到: {output_dir}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Qwen-TTS微调训练")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        default=None,
        help="训练集manifest路径"
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default=None,
        help="验证集manifest路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从checkpoint恢复训练"
    )

    args = parser.parse_args()

    # 切换到项目根目录
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    config = load_config(args.config)
    project_root = config["paths"]["project_root"]

    # 设置默认路径
    if args.train_manifest is None:
        args.train_manifest = os.path.join(
            project_root,
            config["paths"]["processed_data"],
            "train.jsonl"
        )

    if args.val_manifest is None:
        args.val_manifest = os.path.join(
            project_root,
            config["paths"]["processed_data"],
            "val.jsonl"
        )

    audio_base_dir = os.path.join(project_root, config["paths"]["audio_segments"])

    # 创建训练器并训练
    trainer = QwenTTSTrainer(config)
    trainer.train(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        audio_base_dir=audio_base_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
