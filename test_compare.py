import safetensors
import torch
import torch
from safetensors.torch import load_file
import os
# 加载两个文件
checkpoint_path ="/home/vllm/w00664509/Qwen3-VL-8B-Instruct-eagle3/"
safetensors_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors')]
if safetensors_files:
    state_dict = {}
    for f in safetensors_files:
        state_dict.update(load_file(os.path.join(checkpoint_path, f)))
print(state_dict.keys())

print("#############################################")
checkpoint_path = "/home/vllm/w00664509/qwen3-eagle/eagle3_step_100/"
safetensors_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors')]
if safetensors_files:
    state_dict2 = {}
    for f in safetensors_files:
        state_dict2.update(load_file(os.path.join(checkpoint_path, f)))
print(state_dict2.keys())
# 比较键集合
keys1 = set(state_dict.keys())
keys2 = set(state_dict2.keys())
print("Large only:", keys1 - keys2)
print("Small only:", keys2 - keys1)
print("Common keys:", keys1 & keys2)

# 比较公共键的张量形状
for k in keys1 & keys2:
    shape1 = state_dict[k].shape
    shape2 = state_dict2[k].shape
    if shape1 != shape2:
        print(f"Shape mismatch on {k}: {shape1} vs {shape2}")
