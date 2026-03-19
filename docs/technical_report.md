# Qwen3-TTS 音色克隆技术报告

## 项目概述

本项目基于阿里巴巴 Qwen3-TTS 模型，通过监督微调（SFT）实现特定说话人（秦彻）的音色克隆。目标是让 TTS 模型能够生成具有目标说话人音色特征的语音。

## 技术方案

### 1. 模型架构

采用 Qwen3-TTS-12Hz-1.7B-Base 作为基础模型，主要组件包括：

| 组件 | 说明 |
|------|------|
| Audio Tokenizer | 12Hz 帧率，16 码本，2048 词汇量 |
| Speaker Encoder | 从参考音频提取说话人嵌入向量 |
| Talker | 基于 Transformer 的语言模型，预测音频 token |
| Code Predictor | 预测 1-15 层残差码本 |

### 2. 数据处理流程

```
原始视频 → 音频提取 → VAD切分 → Whisper转录 → Audio Codes提取 → 训练数据
```

#### 2.1 音频预处理
- 从视频提取音频：16kHz 采样率，单声道
- 使用 VAD（Voice Activity Detection）按静音切分
- 静音阈值：-40 dBFS，最小静音长度：500ms

#### 2.2 文本转录
- 使用 faster-whisper (large 模型) 进行语音识别
- 自动生成文本标注

#### 2.3 Audio Codes 提取
- 使用 Qwen3-TTS-Tokenizer-12Hz 将音频编码为离散 token
- 每帧 16 个整数（对应 16 个码本）
- 12Hz 帧率意味着每秒 12 个 token 帧

### 3. 微调策略

#### 3.1 训练配置
| 参数 | 值 |
|------|-----|
| 学习率 | 1e-5 |
| Batch Size | 4 |
| Gradient Accumulation | 4 |
| 混合精度 | BF16 |
| Epochs | 30 |
| 优化器 | AdamW (weight_decay=0.01) |

#### 3.2 损失函数
```
Loss = Talker_Loss + 0.3 × Sub_Talker_Loss
```
- Talker_Loss: 主码本（codec_0）的交叉熵损失
- Sub_Talker_Loss: 残差码本（codec_1~15）的预测损失

#### 3.3 说话人嵌入
- 从参考音频通过 Speaker Encoder 提取说话人嵌入
- 训练时将嵌入向量写入 codec_embedding 的特定位置（index=3000）
- 推理时通过 speaker_name 索引该嵌入

## 数据统计

| 指标 | 数值 |
|------|------|
| 原始音频文件 | 7 个 |
| 切分后片段数 | 485 个 |
| 有效训练样本 | 481 条 |
| 总音频时长 | 约 14 分钟 |
| 数据大小 | 31 MB |

## 实验结果

### 训练过程
- 完成 30 个 epoch 的训练
- 保存了 epoch-9, epoch-19, epoch-29 的 checkpoint

### 当前问题
生成的语音音调偏高于目标说话人，可能原因：
1. 训练数据量不足（14分钟，建议 30 分钟以上）
2. 需要更多 epoch 或调整学习率
3. 参考音频选择可优化

### 改进方向
1. 增加训练数据量
2. 添加 Validation Loss 监控过拟合
3. 尝试不同的学习率（2e-5, 5e-5）
4. 筛选更具代表性的参考音频

## 项目结构

```
qwen-tts-qc/
├── src/                          # 核心训练代码
│   ├── sft_12hz.py              # 微调主脚本
│   ├── dataset.py               # 数据集定义
│   └── prepare_data.py          # Audio codes 提取
├── scripts/
│   ├── prepare_qwen_tts_data.py # 数据预处理（VAD+转录）
│   ├── run_finetuning.sh        # 训练启动脚本
│   └── test_inference.py        # 推理测试
├── data/
│   ├── raw/                     # 原始音频
│   ├── processed/segments/      # 切分后的片段
│   ├── train_raw.jsonl          # 原始训练数据
│   └── train_with_codes.jsonl   # 含 audio_codes 的训练数据
├── models/                       # 预训练模型
│   ├── Qwen3-TTS-Tokenizer-12Hz
│   └── Qwen3-TTS-12Hz-1.7B-Base
└── output/                       # 微调后的模型
    └── checkpoint-epoch-29/
```

## 使用方法

### 训练
```bash
bash scripts/run_finetuning.sh
```

### 推理
```python
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained("output/checkpoint-epoch-29")
wavs, sr = tts.generate_custom_voice(
    text="大家好，我叫秦彻",
    speaker="qinche"
)
```

## 依赖环境

- Python 3.10+
- PyTorch 2.0+
- qwen-tts
- faster-whisper
- accelerate
- pydub
- librosa

## 参考资料

- [Qwen3-TTS 官方仓库](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS 技术报告](https://arxiv.org/abs/xxx)
