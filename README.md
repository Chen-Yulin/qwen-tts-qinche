# Qwen3-TTS 秦彻音色微调项目

基于 Qwen3-TTS 模型，使用秦彻语音数据进行 SFT 微调，实现音色克隆。

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
│   ├── raw_videos/              # 原始视频（可选）
│   ├── processed/segments/      # 切分后的片段
│   ├── train_raw.jsonl          # 原始训练数据
│   └── train_with_codes.jsonl   # 含 audio_codes 的训练数据
├── models/                       # 预训练模型
│   ├── Qwen3-TTS-Tokenizer-12Hz
│   └── Qwen3-TTS-12Hz-1.7B-Base
├── output/                       # 微调后的模型
├── docs/
│   └── technical_report.md      # 技术报告
└── requirements.txt
```

## 快速开始

### 1. 依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

#### 下载预处理数据（推荐）

```bash
huggingface-cli download CallMeChen/qwen-tts-qinche-data --local-dir data --repo-type dataset
```

#### 自行处理数据

##### 2.1 从视频提取音频（可选）

```bash
mkdir -p data/raw
for f in data/raw_videos/*.mp4; do
  name=$(basename "$f" .mp4)
  ffmpeg -i "$f" -vn -acodec pcm_s16le -ar 16000 -ac 1 "data/raw/${name}.wav" -y
done
```

##### 2.2 处理音频（VAD切分 + Whisper转录）

```bash
python scripts/prepare_qwen_tts_data.py \
  --raw_audio_dir data/raw \
  --output_dir data/processed \
  --output_jsonl data/train_raw.jsonl \
  --whisper_model large
```

追加新数据：
```bash
python scripts/prepare_qwen_tts_data.py \
  --raw_audio_dir data/raw \
  --output_dir data/processed \
  --output_jsonl data/train_raw.jsonl \
  --append
```

### 3. 训练

```bash
bash scripts/run_finetuning.sh
```

训练流程：
1. 提取 audio_codes
2. 进行 SFT 微调
3. 保存 checkpoint 到 `output/`

### 4. 推理测试

```bash
python scripts/test_inference.py \
  --model_path output/checkpoint-epoch-5 \
  --speaker_name qinche \
  --text "大家好，我叫秦彻" \
  --output test_output.wav
```

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| BATCH_SIZE | 4 | 批次大小 |
| LR | 1e-5 | 学习率 |
| EPOCHS | 6 | 训练轮数 |
| SPEAKER_NAME | qinche | 说话人名称 |

修改 `scripts/run_finetuning.sh` 中的配置即可调整。

## 数据格式

### train_raw.jsonl
```json
{"audio": "path/to/segment.wav", "text": "转录文本", "ref_audio": "path/to/ref.wav"}
```

### train_with_codes.jsonl
```json
{"audio": "...", "text": "...", "ref_audio": "...", "audio_codes": [[...], [...], ...]}
```

## 技术文档

详细技术方案见 [docs/technical_report.md](docs/technical_report.md)
