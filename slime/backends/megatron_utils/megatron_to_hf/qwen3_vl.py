import re
import torch
from megatron.core import parallel_state as mpu


DEBUG_MODE = False
# if True, will compare the updated parameters from megatron with huggingface checkpoint
# only for debugging purpose before first rollout, best to set it to False during actual training
# currently, only support Qwen3-VL-4B and Qwen3-VL-8B


def compare_with_hf(args, mcore_name_param):

    import os
    import pandas as pd
    from safetensors import safe_open

    if args.num_layers == 36 and args.hidden_size == 2560:
        # 4B
        hf_dir = "/mnt/shared-storage-user/ailab-sys/shipengcheng/CKPT/Qwen/Qwen3-VL-4B-Instruct/"
    else:
        # 8B
        hf_dir = "/mnt/shared-storage-user/ailab-sys/shipengcheng/CKPT/Qwen/Qwen3-VL-8B-Instruct/"

    weight_mapping = pd.read_json(os.path.join(hf_dir, "model.safetensors.index.json"))["weight_map"]

    for key, megatron_param in mcore_name_param:

        weight_path = os.path.join(hf_dir, weight_mapping[key])

        with safe_open(weight_path, framework="pt", device="cpu") as f:
            # 获取所有键名
            hf_param = f.get_tensor(key)

            if megatron_param.shape == hf_param.shape:
                if abs(megatron_param.cpu() - hf_param).sum() > 0:
                    print(
                        "[Debug] uncompatible",
                        key,
                        megatron_param.shape,
                        hf_param.shape,
                        abs(megatron_param.cpu() - hf_param).sum(),
                    )
            else:
                print("[Debug] uncompatible shape", key, megatron_param.shape, hf_param.shape)


def convert_qwen3vl_to_hf(args, name, param):
    """
    Convert Megatron-style Qwen3-VL parameter names to HF-style names.
    Supports both language model and vision model parameters.

    Args:
        args: megatron model args (num_attention_heads, kv_channels, etc.)
        name: str, Megatron parameter name
        param: torch.Tensor, parameter value

    Returns:
        List of tuples [(hf_name, hf_param), ...]
    """

    hf_name_param = None

    # ----------------------------
    # 1. language model & vision model parameters
    # ----------------------------

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except (ZeroDivisionError, TypeError):
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups
    language_num_layers = args.num_layers
    
    pp_size = args.pipeline_model_parallel_size
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    if language_num_layers % pp_size != 0:
        raise ValueError(
            f"Language model layers ({language_num_layers}) must be divisible by "
            f"pipeline parallel size ({pp_size}) for distributed training."
        )

    num_layers_per_rank = language_num_layers // pp_size
    offsets = pp_rank * num_layers_per_rank



    # ----------------------------
    # 2. LM Embeddings & output
    # ----------------------------
    if name == "module.module.language_model.embedding.word_embeddings.weight":
        hf_name_param = [("model.language_model.embed_tokens.weight", param)]
    elif name == "module.module.language_model.decoder.final_layernorm.weight":
        hf_name_param = [("model.language_model.norm.weight", param)]
    elif name == "module.module.language_model.output_layer.weight":
        if not args.untie_embeddings_and_output_weights:
            return [("model.language_model.embed_tokens.weight", param)]
        else:
            return [("lm_head.weight", param)]

    else:

        decoder_layers_pattern = r"module\.module\.language_model.decoder\.layers\.(\d+)\.(.+)"
        vision_pattern = r"module\.module\.vision_model\.(.+)"

        match = re.match(decoder_layers_pattern, name)
        if match:
            # ----------------------------
            # 3. Attention and MLP layers in language model
            # ----------------------------

            layer_idx, rest = match.groups()
            layer_idx = str(int(layer_idx) + offsets)
            # Self-attention projection
            if rest == "self_attention.linear_proj.weight":
                hf_name_param = [(f"model.language_model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
            elif rest == "self_attention.linear_qkv.weight":
                param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
                q_param, k_param, v_param = torch.split(
                    param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1
                )
                q_param = q_param.reshape(-1, args.hidden_size)
                k_param = k_param.reshape(-1, args.hidden_size)
                v_param = v_param.reshape(-1, args.hidden_size)
                hf_name_param = [
                    (f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                    (f"model.language_model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                    (f"model.language_model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
                ]

            # MLP layers
            elif rest == "mlp.linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)

                hf_name_param = [
                    (f"model.language_model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                    (f"model.language_model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
                ]

            elif rest == "mlp.linear_fc2.weight":
                hf_name_param = [(f"model.language_model.layers.{layer_idx}.mlp.down_proj.weight", param)]

            # LayerNorms
            elif rest == "self_attention.linear_qkv.layer_norm_weight":
                hf_name_param = [(f"model.language_model.layers.{layer_idx}.input_layernorm.weight", param)]
            elif rest == "mlp.linear_fc1.layer_norm_weight":
                hf_name_param = [(f"model.language_model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
            elif rest == "self_attention.q_layernorm.weight":
                hf_name_param = [(f"model.language_model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
            elif rest == "self_attention.k_layernorm.weight":
                hf_name_param = [(f"model.language_model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

        else:
            match_v = re.match(vision_pattern, name)
            if match_v:
                # ----------------------------
                # 4. Vision model parameters
                # ----------------------------
                deepstack_merger_pattern = r"deepstack_merger_list\.(\d+)\.(.+)"
                decoder_layer_pattern = r"blocks\.(\d+)\.(.+)"

                rest = match_v.groups()[0]

                match_layer = re.match(deepstack_merger_pattern, rest)
                if match_layer:
                    layer_idx, layer_rest = match_layer.groups()
                    layer_idx = str(int(layer_idx) + offsets)
                    hf_name_param = [(f"model.visual.deepstack_merger_list.{layer_idx}.{layer_rest}", param)]

                # Decoder layers
                else:
                    match_layer = re.match(decoder_layer_pattern, rest)
                    if match_layer:
                        layer_idx, layer_rest = match_layer.groups()
                        layer_idx = str(int(layer_idx) + offsets)
                        hf_name_param = [(f"model.visual.blocks.{layer_idx}.{layer_rest}", param)]
                    else:
                        hf_name_param = [(f"model.visual.{rest}", param)]

    if hf_name_param is None:
        raise ValueError(f"Unknown parameter name: {name}")
    else:
        if DEBUG_MODE:
            compare_with_hf(args, hf_name_param)

        return hf_name_param
