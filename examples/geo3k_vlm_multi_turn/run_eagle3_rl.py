"""
Example: Using Eagle3 Online Training in Slime RL

This example demonstrates how to enable Eagle3 drafter online training
in Slime's RL training pipeline.
"""

import os

import slime.utils.misc as U
from slime.utils.external_utils.command_utils import execute_train_npu

MODEL_NAME = os.environ.get("SLIME_SCRIPT_MODEL_NAME", "Qwen3-VL-8B-Instruct")
TRAIN_BACKEND = os.environ.get("SLIME_SCRIPT_TRAIN_BACKEND", "megatron").lower()

DATASET_NAME = "VeraIsHere/geo3k_imgurl_processed"
DATA_ROOT = "/home/vllm/w00664509/geo3k_imgurl_processed/"
TRAIN_DATA_PATH = os.path.join(DATA_ROOT, "train.parquet")
print("#############main222############")

def get_megatron_model_type(model_name: str) -> str:
    model_type = model_name.replace("-Instruct", "").replace("-Thinking", "")
    model_type = model_type.replace("Qwen3-VL-", "qwen3-")
    return model_type.replace("-2B", "-1.7B")

def execute():
    """Execute RL training with Eagle3 online training enabled."""
    
    print("#############execute 29############")
    # Checkpoint arguments
    ckpt_args = f"--hf-checkpoint /home/vllm/w00664509/{MODEL_NAME} "
    print("#############execute 31############")
    
    # Wandb arguments (optional)
    wandb_args = (
        (
            "--use-wandb "
            "--wandb-project slime-dev "
            "--wandb-group geo3k_vlm_multi_turn "
            f"--wandb-key '{wandb_api_key}' "
        )
        if (wandb_api_key := os.environ.get("WANDB_API_KEY"))
        else ""
    )
    
    # Rollout arguments
    rollout_args = (
        f"--prompt-data {TRAIN_DATA_PATH} "
        "--input-key problem "
        "--label-key answer "
        '--multimodal-keys \'{"image": "images"}\' '
        "--rm-type math "
        "--apply-chat-template "
        "--custom-generate-function-path examples.geo3k_vlm_multi_turn.rollout.generate "
        "--custom-config-path examples/geo3k_vlm_multi_turn/geo3k_vlm_multi_turn_config.yaml "
        "--rollout-shuffle "
        "--num-rollout 3000 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
    )
    
    # GRPO algorithm arguments
    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--use-kl-loss "
    )
    
    # Optimizer arguments
    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
        "--use-fused-rmsnorm "
        "--use-fused-swiglu "
    )
    
    # SGLang arguments with Eagle3 speculative decoding
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
        f"--sglang-cuda-graph-bs {' '.join(map(str, [4, 8] + list(range(16, 257, 8))))} "
        "--sglang-mm-attention-backend ascend_attn "
        "--sglang-device npu "
        "--sglang-disable-radix-cache "
        "--sglang-chunked-prefill-size 32768 "
        "--sglang-max-prefill-tokens 4000 "
        "--sglang-max-total-tokens 327680 "
        
        "--sglang-enable-return-hidden-states "
        # Eagle3 speculative decoding settings
        "--sglang-speculative-algorithm EAGLE3 "
        "--sglang-speculative-num-steps 3 "
        "--sglang-speculative-eagle-topk 1 "
        "--sglang-speculative-num-draft-tokens 1 "
        "--sglang-speculative-draft-model-path /home/vllm/w00664509/Qwen3-VL-8B-Instruct-eagle3 "
    )
    
    # Eagle3 online training arguments
    eagle3_args = (
        # Enable Eagle3 online training
        "--enable-eagle3-training "
        "--eagle3-collect-hidden-states "
        
        # Training configuration
        "--eagle3-training-interval-steps 10 "
        "--eagle3-batch-size-per-gpu 2 "
        "--eagle3-max-seq-len 8192 "
        "--eagle3-max-epochs 10 "
        "--eagle3-checkpoint-path /home/vllm/w00664509/qwen3-eagle "
        
        # Model architecture
        "--eagle3-num-layers 1 "
        
        # Optimizer settings
        "--eagle3-lr 1e-6 "
        "--eagle3-lr-warmup-steps 1000 "
        "--eagle3-weight-decay 1e-2 "
        "--eagle3-warmup-style constant "
        
        # Offload settings (optional, for memory-constrained setups)
        "--eagle3-offload-param "
        "--eagle3-offload-optimizer "
        
        # Weight update settings
        "--update-weights-bucket-megabytes 512 "
    )
    
    # Megatron backend arguments
    megatron_args = (
        "--train-backend megatron "
        f"--load /home/vllm/w00664509/{MODEL_NAME} "
        f"--ref-load /home/vllm/w00664509/{MODEL_NAME} "
        # f"--save /home/data/{MODEL_NAME}_eagle3_rl "
        # f"--save-hf /path/to/{MODEL_NAME}_eagle3_rl_hf/rollout_{rollout_id} "
        # "--save-interval 10000 "
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
        "--balance-data "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--megatron-to-hf-mode bridge "
    )
    
    # Miscellaneous arguments
    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node 4 "
        f"--rollout-num-gpus 4 "
        "--no-gradient-accumulation-fusion "
        "--use-flash-attn "
    )
    
    
    print("#############execute 181############")
    backend_args = megatron_args
    megatron_model_type = get_megatron_model_type(MODEL_NAME)
    os.environ["MODEL_ARGS_ROTARY_BASE"] = "5000000"
    print("#############execute 185############")
    
    # Combine all arguments
    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{sglang_args} "
        f"{eagle3_args} "
        f"{megatron_args} "
        f"{misc_args} "
        f"{wandb_args} "
    )
    
    print("#############execute 200############")
    # Execute training
    execute_train_npu(
        train_args=train_args,
        train_script="train_async.py",
        megatron_model_type=megatron_model_type,
        extra_env_vars=({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]} if os.environ.get("WANDB_API_KEY") else {}),
    )

if __name__ == "__main__":
    print("#############main############")
    execute()
