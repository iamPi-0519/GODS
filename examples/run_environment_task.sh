#!/bin/bash

TASK_ID="1"
MODEL="Qwen/Qwen2.5-3B-Instruct"
DATASET="https://huggingface.co/datasets/TuringEnterprises/Turing-Open-Reasoning/resolve/main/Computational_STEM_QA_Dataset.json?download=true"
DATASET_TYPE='{
  "environment_name": "alfworld"
}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=12

# For uploading the outputs
HUGGINGFACE_TOKEN="Your Huggingface Token"
WANDB_TOKEN=""
HUGGINGFACE_USERNAME="Your Huggingface Username"
EXPECTED_REPO_NAME="environment_test"
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"
DOCKER_BUILDKIT=1

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
DEBUG_DIR="$(pwd)/alfworld_debug"
mkdir -p "$CHECKPOINTS_DIR"
chmod 777 "$CHECKPOINTS_DIR"
mkdir -p "$OUTPUTS_DIR"
chmod 777 "$OUTPUTS_DIR"
mkdir -p "$DEBUG_DIR"
chmod 777 "$DEBUG_DIR"

docker network create -d bridge train-environment-network

docker run -d \
  --name environment-server-0 \
  --network train-environment-network \
  -p 10000:8000 \
  affinefoundation/agentgym:alfworld

docker run -d \
  --name environment-server-1 \
  --network train-environment-network \
  -p 10001:8000 \
  affinefoundation/agentgym:alfworld

# Build the downloader image
docker build -t trainer-downloader -f dockerfiles/trainer-downloader.dockerfile .

# Build the trainer image
docker build -t standalone-text-trainer -f dockerfiles/standalone-text-trainer.dockerfile .

#Download model and dataset
echo "Downloading model and dataset..."
docker run --rm \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --name downloader-image \
  trainer-downloader \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --file-format "$FILE_FORMAT" \
  --task-type "EnvTask"


docker run --rm --gpus all \
  --network train-environment-network \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=64g \
  --cpus=8 \
  -e WANDB_API_KEY="$WANDB_TOKEN" \
  -e WANDB_TOKEN="$WANDB_TOKEN" \
  -e HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
  -e HUGGINGFACE_USERNAME="$HUGGINGFACE_USERNAME" \
  -e WANDB_MODE="online" \
  -e ENVIRONMENT_SERVER_URLS="http://environment-server-0:8000,http://environment-server-1:8000" \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --volume "$DEBUG_DIR:/workspace/axolotl/alfworld_debug:rw" \
  --name grpo-text-trainer-example \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "EnvTask" \
  --file-format "$FILE_FORMAT" \
  --hours-to-complete "$HOURS_TO_COMPLETE" \
  --expected-repo-name "$EXPECTED_REPO_NAME" \