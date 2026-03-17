# Qwen-TTS 秦彻音色微调项目

基于 Qwen-TTS 模型，使用B站秦彻语音数据进行微调，实现音色复刻。

## 项目结构

```
qwen-tts-qc/
├── configs/
│   └── config.yaml          # 配置文件
├── data/
│   ├── raw/                  # 原始下载的音频
│   ├── processed/            # 处理后的数据
│   └── audio_segments/       # 分割后的音频片段
├── scripts/
│   ├── download_bilibili.py  # B站视频下载
│   ├── audio_preprocessing.py # 音频预处理
│   ├── transcribe.py         # 语音转录
│   ├── prepare_dataset.py    # 数据集准备
│   ├── train.py              # 模型训练
│   ├── inference.py          # 推理测试
│   ├── accelerate_inference.py # 推理加速
│   ├── evaluate.py           # 模型评估
│   └── run_pipeline.py       # 完整管线
├── models/                   # 模型存储
├── outputs/                  # 训练输出
├── logs/                     # 日志
└── requirements.txt          # 依赖
```

## 快速开始

### 1. 安装依赖

```bash
cd qwen-tts-qc
pip install -r requirements.txt

# 安装ffmpeg (用于音频处理)
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg
```

### 2. 运行完整数据处理管线

```bash
# 运行完整管线(下载 -> 预处理 -> 转录 -> 划分数据集)
python scripts/run_pipeline.py

# 如果已有音频文件，跳过下载
python scripts/run_pipeline.py --skip-download

# 使用cookies下载高清音频(可选)
python scripts/run_pipeline.py --cookies cookies.txt
```

### 3. 分步执行

```bash
# Step 1: 下载B站视频音频
python scripts/download_bilibili.py

# Step 2: 音频预处理(分段、降噪)
python scripts/audio_preprocessing.py

# Step 3: 语音转录
python scripts/transcribe.py

# Step 4: 准备数据集
python scripts/prepare_dataset.py
```

### 4. 模型训练

```bash
# 使用LoRA微调
python scripts/train.py

# 指定配置
python scripts/train.py --config configs/config.yaml

# 从checkpoint恢复
python scripts/train.py --resume outputs/qinche_finetune/checkpoint-xxx
```

### 5. 推理测试

```bash
# 基础推理
python scripts/inference.py --text "你好，我是秦彻" --output output.wav

# 使用微调后的模型
python scripts/inference.py \
    --text "你好，我是秦彻" \
    --lora-path outputs/qinche_finetune \
    --output output.wav
```

### 6. 模型评估

```bash
# 评估合成质量
python scripts/evaluate.py \
    --ref-dir data/audio_segments \
    --syn-dir outputs/synthesized \
    --output evaluation_results.json
```

### 7. 推理加速

```bash
# 导出ONNX模型
python scripts/accelerate_inference.py \
    --action export-onnx \
    --model-path outputs/qinche_finetune \
    --output models/model.onnx

# 转换为TensorRT
python scripts/accelerate_inference.py \
    --action convert-trt \
    --model-path models/model.onnx \
    --output models/model.trt

# 性能基准测试
python scripts/accelerate_inference.py \
    --action benchmark \
    --model-path outputs/qinche_finetune
```

## 配置说明

主要配置项在 `configs/config.yaml`:

- `data_sources`: B站视频URL列表
- `audio`: 音频处理参数(采样率、分段配置等)
- `transcription`: Whisper转录配置
- `model`: 基础模型配置
- `lora`: LoRA微调参数
- `training`: 训练超参数
- `acceleration`: 推理加速配置

## 评估指标

- **MCD (Mel Cepstral Distortion)**: 梅尔倒谱失真，越低越好
- **Speaker Similarity**: 说话人相似度，越高越好
- **PESQ**: 语音质量感知评估，-0.5到4.5，越高越好
- **STOI**: 短时客观可懂度，0到1，越高越好

## 推理加速方案

1. **ONNX Runtime**: 跨平台优化，支持CPU/GPU
2. **TensorRT**: NVIDIA GPU专用优化，FP16加速
3. **vLLM**: 大模型推理优化，支持连续批处理

## 注意事项

1. B站视频下载可能需要cookies才能获取高清音频
2. Whisper转录需要GPU，建议使用large-v3模型
3. 训练建议使用至少24GB显存的GPU
4. 音频数据质量直接影响微调效果

## 数据来源

- https://www.bilibili.com/video/BV1NjzAB2EPY (11分钟)
- https://www.bilibili.com/video/BV168UQBeEKp (35分钟)

## License

仅供学习研究使用，请遵守相关版权规定。
