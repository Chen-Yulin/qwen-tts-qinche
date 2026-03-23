# Qwen3-TTS 秦彻音色微调

## 项目概述

本项目基于Qwen3-TTS模型，通过监督微调（SFT）实现特定说话人（秦彻）的音色克隆。目标是让 TTS 模型能够生成具有目标说话人音色特征的语音。
仓库：https://github.com/Chen-Yulin/qwen-tts-qinche

## 技术方案

### 1. 模型架构

采用 Qwen3-TTS-12Hz-1.7B-Base 作为基础模型，主要组件包括：

| 组件 | 说明 |
|------|------|
| Audio Tokenizer | 12Hz 帧率，16 码本，2048 词汇量 |
| Speaker Encoder | 从参考音频提取speaker嵌入向量 |
| Talker | 基于 Transformer 的语言模型，预测音频 token |
| Code Predictor | 预测 1-15 层残差码本 |

### 2. 数据处理流程

```
原始视频 → 音频提取 → VAD切分 → Whisper转录为text → Audio Codes提取 → 训练数据(data/train_with_codes.jsonl
```

#### 2.1 音频预处理
- 从视频提取音频：16kHz 采样率，单声道
    - 包含7个bilibili视频
- 使用 VAD（Voice Activity Detection）按静音切分
- 静音阈值：-40 dBFS，最小静音长度：500ms
- 人工筛选无效音频片段

#### 2.2 文本转录
- 使用 faster-whisper (large 模型) 进行语音识别
- 自动生成文本标注

#### 2.3 Audio Codes 提取
- 使用 Qwen3-TTS-Tokenizer-12Hz 将音频编码为离散 token
- 每帧 16 个整数（对应 16 个码本）
- 12Hz 帧率意味着每秒 12 个 token 帧

### 3. 微调配置

| 参数 | 值 |
|------|-----|
| 学习率 | 5e-6 |
| Batch Size | 4 |
| Epochs | 6 |


## 数据统计

| 指标 | 数值 |
|------|------|
| 原始音频文件 | 7 个 |
| 有效样本 | 481 条(433train/48val) |
| 总音频时长 | 约 14 分钟 |

## 实验结果
经多次实验，当学习率为5e-6时，大约在**epoch=6时开始过拟合**，所以取epoch为5的checkpoint。

### 评估结果

使用 20 条配对数据进行评估（checkpoint-epoch-5）：

| 指标 | 均值 | 标准差 | 最小值 | 最大值 |
|------|------|--------|--------|--------|
| Speaker Similarity | 0.430 | 0.138 | 0.119 | 0.605 |
| PESQ | 1.063 | 0.044 | 1.026 | 1.205 |

说明：
- Speaker Similarity: 说话人相似度（余弦相似度），0.7+ 为较好，当前结果偏低
- PESQ: 语音感知质量，范围 -0.5~4.5，3.0+ 为较好，当前结果偏低

### 推理加速测试

测试环境：NVIDIA L20X, CUDA 12.1, PyTorch 2.4.0

| 方法 | 推理时间 | 加速比 |
|------|----------|--------|
| SDPA (PyTorch Native) | 3.675s | 1.30x |
| torch.compile | 3.687s | 1.30x |
| Flash Attention 2 | 4.788s | 1.00x (baseline) |



### 当前待解决问题
1. 训练数据量不足（14分钟）
2. 参考音频选择可优化
3. transcript均为直接转录，还没有人工validate
4. 和其他开源tts模型的benchmark

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
python scripts/test_inference.py
```

## 参考资料

- [Qwen3-TTS 官方仓库](https://github.com/QwenLM/Qwen3-TTS)
