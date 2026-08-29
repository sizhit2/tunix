#!/usr/bin/env bash
# Runs the #2023-based GSM8K demo (branch sizhi-metrics-pr2023) on real GSM8K
# with the shaped reward, and checks whether the reward variance produces a
# nonzero advantage. Reads WANDB_API_KEY from the environment or ~/.env.
set -euo pipefail
[[ -f "$HOME/.env" ]] && source "$HOME/.env"
: "${WANDB_API_KEY:?set WANDB_API_KEY or put it in ~/.env}"

# The TPU driver does not release HBM the instant a container is removed, and
# starting while the previous run still holds it makes the trainer die at
# startup with "Attempting to allocate 593.50M. There are 310.72M free."
docker rm -f pr2023-verify >/dev/null 2>&1 || true

# #2023 moved the checkpoint root to ${ARTIFACT_ROOT}/checkpoints; the repo-root
# "checkpoints" dir is the obsolete pre-#2023 location. A stale checkpoint here
# is restored on startup and fails on shape mismatch when the model changes
# ("Requested shape: (1024, 3072) is not compatible with the stored shape:
# (2048, 6144)"), and collides with StepAlreadyExistsError on a same-model
# rerun. The sibling data/ dir holds the downloaded GSM8K and is kept.
sudo rm -rf "$HOME/tunix-pr1983/artifacts/qwen3_dist_gsm8k/checkpoints" /mnt/disk/gsm8k_checkpoints
mkdir -p /mnt/disk/gsm8k_checkpoints

echo -n "waiting for TPU HBM to be free"
for _ in $(seq 1 30); do
  if docker run --rm --privileged --net=host \
       us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/sizhi/tunix_base_image:pr1983-wandb-tpusync-323e1f7 \
       python3 -c "import jax,sys; sys.exit(0 if jax.local_devices()[0].memory_stats()['bytes_in_use'] < (1<<30) else 1)" \
       >/dev/null 2>&1; then
    echo " - free"; break
  fi
  echo -n "."; sleep 10
done

RUN_NAME="pr2023-${SAMPLER:-vanilla}-sync-$(date +%H%M%S)"
echo "run name: $RUN_NAME"
docker rm -f pr2023-verify >/dev/null 2>&1 || true
docker run -d --name pr2023-verify --privileged --net=host --shm-size=16g \
  -v "$HOME/tunix-pr1983:/workspace" -w /workspace \
  -v "${MODEL_HOST_DIR:-/mnt/disk/hf_models/qwen3-0.6b}":/models \
  -v /mnt/disk/gsm8k_checkpoints:/checkpoints \
  -e PYTHONPATH=/workspace \
  -e SAMPLER="${SAMPLER:-vanilla}" \
  -e SYNC_WEIGHTS="${SYNC_WEIGHTS:-1}" \
  -e WEIGHT_SYNC_MODE="${WEIGHT_SYNC_MODE:-raiden}" \
  -e WEIGHT_SYNC_BACKEND="${WEIGHT_SYNC_BACKEND:-raiden}" \
  -e TRACE_ROLLOUTS=1 -e LOG_LEVEL=INFO \
  -e BATCH_SIZE=4 -e NUM_GENERATIONS="${NUM_GENERATIONS:-4}" \
  -e MAX_STEPS="${MAX_STEPS:-3}" -e TRAIN_MICRO_BATCH_SIZE=1 \
  -e MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}" \
  -e MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}" \
  -e REWARD_MODE=exact \
  -e BETA="${BETA:-0}" \
  -e VLLM_INIT_WITH_RANDOM_WEIGHTS=false \
  -e MODEL_NAME="${MODEL_NAME:-Qwen3-0.6B}" \
  -e CHECKPOINT_ROOT_DIRECTORY=/checkpoints \
  -e CHECKPOINT_MAX_TO_KEEP="${CHECKPOINT_MAX_TO_KEEP:-1}" \
  -e MODEL_DIR=/models -e TOKENIZER_PATH=/models -e LOG_ROOT=/workspace/runlogs \
  -e WANDB_PROJECT=trellis-gsm8k -e WANDB_RUN_NAME="$RUN_NAME" \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/sizhi/tunix_base_image:pr1983-wandb-tpusync-323e1f7 \
  bash -c 'apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq procps iproute2 >/dev/null 2>&1; \
           bash /workspace/tunix/experimental/examples/math_gsm8k_dist/launcher.sh'
echo "started; follow with: docker logs -f pr2023-verify"
