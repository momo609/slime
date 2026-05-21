#!/bin/bash

# Qwen3.5-35B-A3B VL RL training on geo3k dataset

# pip install -U transformers

# IMPORTANT: This branch is specially modified for slime's current Megatron
# version and Qwen3.5 from the main Megatron Bridge. Other models are not verified!
# To restore the original Megatron Bridge, run:
#   pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
# TODO: Remove this once Megatron & Megatron Bridge are upgraded upstream.
# pip install git+https://github.com/coding-famer/Megatron-Bridge-slime.git@qwen35 --no-build-isolation

# Configuration
TRAIN_BACKEND="megatron"
MODEL_NAME="Qwen3_5-9B"
DATASET_NAME=${SLIME_SCRIPT_DATASET_NAME:-"chenhegu/geo3k_imgurl"}
DATASET_LOCAL_NAME=$(basename "$DATASET_NAME")

MODEL_NAME_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

# External Ray flag
if [ -z "$SLIME_SCRIPT_EXTERNAL_RAY" ] || [ "$SLIME_SCRIPT_EXTERNAL_RAY" = "0" ]; then
   USE_EXTERNAL_RAY=0
else
   USE_EXTERNAL_RAY=1
fi

# Cleanup
pkill -9 sglang
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   ray stop --force
   pkill -9 ray
fi
pkill -9 slime
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   pkill -9 ray
fi
pkill -9 slime
pkill -9 redis

set -ex
ulimit -n 65535

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY

# cann
source /home/c00944022/851b080/ascend-toolkit/set_env.sh
source /home/c00944022/851b080/nnal/atb/set_env.sh

export PYTHONBUFFERED=16
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)  # or any free port
export PYTHONPATH="/home/c00944022/slime-proj/Megatron-Bridge-slime/src:/home/c00944022/slime-proj/Megatron-LM/:/home/c00944022/slime-proj/sglang/python:$PYTHONPATH"
export STREAMS_PER_DEVICE=32
export HCCL_BUFFSIZE=1000
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
# export ASCEND_LAUNCH_BLOCKING=1
export DISABLE_L2_CACHE=1

# Common args
CKPT_ARGS=(
   --hf-checkpoint /home/data/${MODEL_NAME}
   --load /home/data/${MODEL_NAME}
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data /home/c00944022/slime-proj/dataset/geo3k_imgurl/train.parquet
   --input-key problem
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.8
   --global-batch-size 256
)

# required for vlm datasets
MULTIMODAL_KEYS='{"image": "images"}'

# EVAL_ARGS=(
#    --eval-interval 20
#    --eval-prompt-data ${DATASET_LOCAL_NAME} /home/c00944022/slime-proj/dataset/geo3k_imgurl/test.parquet
#    --n-samples-per-eval-prompt 1
#    --eval-max-response-len 4096
# )

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.7
   # --sglang-ep-size 4
   --sglang-cuda-graph-bs 4 8 16 24 32 40 48 56 64 72
   --sglang-disable-cuda-graph

   # MTP speculative decoding
   # --sglang-speculative-algorithm EAGLE
   # --sglang-speculative-num-steps 2
   # --sglang-speculative-eagle-topk 1
   # --sglang-speculative-num-draft-tokens 3

   # --sglang-max-running-requests 512

   --sglang-attention-backend ascend
   --sglang-device npu
   --sglang-disable-radix-cache
   --sglang-enable-multimodal
   --sglang-mm-attention-backend ascend_attn
   --sglang-dtype bfloat16
   --sglang-chunked-prefill-size 32768
   --sglang-max-prefill-tokens 4000
   --sglang-max-total-tokens 327680
   # --sglang-node-rank 0
   # --sglang-nnodes 1
   --num-gpus-per-node 16
)

# Wandb args (only if WANDB_API_KEY is set)
if [ -n "$WANDB_API_KEY" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project slime-geo3k-vlm
      --wandb-group ${MODEL_NAME_LOWER}-${TRAIN_BACKEND}
      --wandb-key ${WANDB_API_KEY}
      --disable-wandb-random-suffix
   )
else
   WANDB_ARGS=()
fi

MISC_ARGS=(
   # --colocate
   --use-flash-attn
   # --debug-rollout-only
   # --debug-train-only
   # --load-debug-rollout-data /home/c00944022/slime-proj/debug/data_{rollout_id}.pt
   --no-check-for-nan-in-loss-and-grad
)

# Backend-specific args
# megatron backend
BACKEND_ARGS=(
   --train-backend megatron
   # Qwen3.5-9B has num_query_groups = 4
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash

   # Packing is not supported for GDN currently
   --qkv-format bshd
   --micro-batch-size 1
)

SLIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
source "${SLIME_DIR}/scripts/models/qwen3.5-9B.sh"

# Start Ray if not using external Ray
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head --node-ip-address ${MASTER_ADDR} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

# Build runtime env
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES\": \"1\",
    \"ASCEND_TOOLKIT_HOME\": \"/home/c00944022/851b080/cann-8.5.1/\",
    \"ASCEND_OPP_PATH\": \"/home/c00944022/851b080/cann-8.5.1/opp/\",
    \"ASCEND_AICPU_PATH\": \"/home/c00944022/851b080/cann-8.5.1/\",
    \"ASCEND_HOME_PATH\": \"/home/c00944022/851b080/cann-8.5.1/\",
    \"HYDRA_FULL_ERROR\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 16 \
   --rollout-num-gpus 16 \
   --colocate \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${BACKEND_ARGS[@]} \
   ${MISC_ARGS[@]}
