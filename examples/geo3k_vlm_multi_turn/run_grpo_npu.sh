export SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-8B-Instruct
export SLIME_SCRIPT_TRAIN_BACKEND=megatron
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Replace placeholder paths with actual installation paths
export PYTHONPATH="/home/vllm/w00664509/slime-re/Megatron-Bridge/src:/home/vllm/w00664509/slime-re/Megatron-LM/:/home/vllm/w00664509/sglang-main/sglang/python:$PYTHONPATH"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export HYDRA_FULL_ERROR=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)  # or any free port
# export RAY_memory_monitor_refresh_ms=0
# export SGLANG_VIT_ENABLE_CUDA_GRAPH=True
export DISABLE_L2_CACHE=1
source /home/vllm/cann8.5.2/ascend-toolkit/set_env.sh
source /home/vllm/cann8.5.2/nnal/atb/set_env.sh

python /home/vllm/w00664509/slime-re/slime/examples/geo3k_vlm_multi_turn/run_eagle3_rl.py
# python /home/vllm/miles_project/miles/examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py