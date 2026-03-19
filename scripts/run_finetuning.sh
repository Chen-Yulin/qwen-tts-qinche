#!/usr/bin/env bash
set -e

# Configuration
DEVICE="cuda:0"
BASE_DIR="/workspace/chenyl29@xiaopeng.com/qwen-tts-qc"
TOKENIZER_MODEL_PATH="${BASE_DIR}/models/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="${BASE_DIR}/models/Qwen3-TTS-12Hz-1.7B-Base"

RAW_JSONL="${BASE_DIR}/data/train_raw.jsonl"
TRAIN_JSONL="${BASE_DIR}/data/train_with_codes.jsonl"
OUTPUT_DIR="${BASE_DIR}/output"

BATCH_SIZE=4
LR=5e-6
EPOCHS=6
SPEAKER_NAME="qinche"

SRC_DIR="${BASE_DIR}/src"

echo "=== Step 1: Prepare data (extract audio_codes) ==="
cd ${SRC_DIR}
python prepare_data.py \
  --device ${DEVICE} \
  --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
  --input_jsonl ${RAW_JSONL} \
  --output_jsonl ${TRAIN_JSONL}

echo ""
echo "=== Step 2: Fine-tune model ==="
python sft_12hz.py \
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --speaker_name ${SPEAKER_NAME}

echo ""
echo "=== Fine-tuning complete ==="
echo "Model saved to: ${OUTPUT_DIR}"
echo "Speaker name: ${SPEAKER_NAME}"
echo "=== test inference ==="
cd ${BASE_DIR}
python scripts/test_inference.py --model_path output/checkpoint-epoch-6
