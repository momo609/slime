from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any,Optional

import torch
from examples.geo3k_vlm_multi_turn.base_env import BaseInteractionEnv

# When executed as a module: python -m examples.vlm_multi_turn.rollout
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

DEFAULT_ENV_MODULE = "examples.vlm_multi_turn.env_geo3k"

# Dummy messages used for calculating trim length in chat template encoding
DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]
import logging
import os
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("SLIME_LOGGING_LEVEL", "INFO"))

def _load_env_module(env_path: str | None):
    """Load the interaction environment module from a module path or a file path."""
    target = env_path or DEFAULT_ENV_MODULE
    module_path = Path(target)
    if module_path.suffix == ".py" and module_path.exists():
        spec = importlib.util.spec_from_file_location(f"rollout_env_{module_path.stem}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import environment module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(target)


def _build_env(env_module, sample: Sample, args: Any):
    """Instantiate the interaction environment using the provided module."""
    build_fn = env_module.build_env
    if not callable(build_fn):
        raise ValueError("Environment module must expose a callable `build_env(sample, args)`.")
    try:
        return build_fn(sample=sample, args=args)
    except TypeError:
        # Fallback to positional signature
        return build_fn(sample, args)


def _encode_observation_for_generation(
    tokenizer,
    processor,
    message: dict,
    metadata: dict | None,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
):
    """
    Encode a single observation turn that may include images/videos in the content list.
    Trim out the system/tool preamble added by the chat template so only the observation tokens remain.
    """
    tools = metadata.get("tools") if metadata else None
    apply_kwargs = apply_chat_template_kwargs or {}

    trim_length = 0

    if apply_chat_template:
        dummy_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            **apply_kwargs,
        )
        formatted_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES + [message],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **apply_kwargs,
        )
        trim_length = len(tokenizer.encode(dummy_prompt, add_special_tokens=False))
    else:
        formatted_prompt = [message]

    multimodal_inputs = None
    multimodal_train_inputs = None
    if processor:
        # Convert content-embedded images/videos into multimodal inputs for the processor.
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info([message])
        multimodal_inputs = {"images": images, "videos": videos}
        processor_output = processor(text=formatted_prompt, **multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    if trim_length:
        prompt_ids = prompt_ids[trim_length:]

    image_data = []
    if multimodal_inputs and multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in multimodal_inputs["images"]]
    return prompt_ids, image_data, multimodal_inputs, multimodal_train_inputs


def _merge_multimodal_train_inputs(chunks: list[dict | None]) -> dict | None:
    """
    Merge per-turn multimodal_train_inputs with a single concat per key.

    Note: Only torch.Tensor values are merged; non-tensor fields are ignored by design.
    """
    if not chunks:
        return None

    values_by_key = {}
    for chunk in chunks:
        if not chunk:
            continue
        for key, val in chunk.items():
            if val is None:
                continue
            values_by_key.setdefault(key, []).append(val)

    merged = {}
    for key, values in values_by_key.items():
        if all(isinstance(v, torch.Tensor) for v in values):
            merged[key] = torch.cat(values, dim=0)

    return merged


def _initialize_resources(args: Any, sample: Sample):
    env_module = _load_env_module(args.rollout_interaction_env_path)
    max_turns = args.max_turns
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path in the custom config file.")
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sample.metadata = sample.metadata or {}
    env = _build_env(env_module, sample, args)
    config = {"max_turns": max_turns}
    return env, env_module, config, state, url


def _prepare_initial_inputs(sample: Sample, processor, tokenizer):
    if processor:
        processor_output = processor(text=sample.prompt, **(sample.multimodal_inputs or {}))
        prompt_ids = processor_output["input_ids"][0]
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)

    image_data = []
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in sample.multimodal_inputs["images"]]
    return prompt_ids, image_data, sample.multimodal_train_inputs


def _prepare_start_state(sample: Sample, state, args: Any, sampling_params: dict):
    prompt_ids, image_data, init_mm_train = _prepare_initial_inputs(sample, state.processor, state.tokenizer)
    current_image_data = image_data
    multimodal_train_inputs_buffer: list[dict | None] = []
    if init_mm_train:
        multimodal_train_inputs_buffer.append(init_mm_train)

    if not sample.tokens:
        sample.tokens = list(prompt_ids)
    response_tokens: list[int] = sample.tokens[len(prompt_ids) :] if len(sample.tokens) >= len(prompt_ids) else []
    sample.loss_mask = sample.loss_mask or []
    sample.rollout_log_probs = sample.rollout_log_probs or []
    sample.response_length = len(response_tokens)

    budget = None
    if args.rollout_max_context_len is not None:
        budget = args.rollout_max_context_len - len(sample.tokens)
    elif sampling_params.get("max_new_tokens") is not None:
        budget = sampling_params["max_new_tokens"] - len(sample.tokens)
    return current_image_data, response_tokens, budget, multimodal_train_inputs_buffer

def _scatter_topk_logprobs_with_tail(logprobs: torch.Tensor, indices: torch.Tensor, vocab_size: int) -> torch.Tensor:
    dense_logprob_view = torch.full(
        (logprobs.size(0), vocab_size),
        float("-inf"),
        dtype=logprobs.dtype,
        device=logprobs.device,
    )
    if logprobs.numel() == 0:
        return dense_logprob_view

    valid = torch.isfinite(logprobs) & (indices >= 0) & (indices < vocab_size)
    if not valid.any():
        return dense_logprob_view

    valid_count = valid.sum(dim=-1)
    has_valid = valid_count > 0
    # print("###########logprobs",logprobs)
    # print("###########valid",valid)
    topk_mass = torch.where(valid, logprobs.float().exp(), torch.zeros_like(logprobs, dtype=torch.float32)).sum(dim=-1)
    print("###########topk_mass",torch.where(valid, logprobs.float().exp(), torch.zeros_like(logprobs, dtype=torch.float32)).shape)
    remaining_mass = (1.0 - topk_mass).clamp(min=torch.finfo(torch.float32).tiny)
    print("###########remaining_mass",remaining_mass.shape)
    remaining_count = (vocab_size - valid_count).clamp(min=1).to(torch.float32)
    tail_logprob = (remaining_mass.log() - remaining_count.log()).to(logprobs.dtype)
    print("###########tail_logprob",tail_logprob.shape)
    dense_logprob_view = torch.where(
        has_valid.unsqueeze(-1),
        tail_logprob.unsqueeze(-1).expand(-1, vocab_size),
        dense_logprob_view,
    )

    row_indices = torch.arange(logprobs.size(0), device=logprobs.device).unsqueeze(1).expand_as(indices)
    dense_logprob_view[row_indices[valid], indices[valid]] = logprobs[valid]
    print("###########dense_logprob_view",dense_logprob_view.shape)
    return dense_logprob_view

import math
def _top_logprobs_to_tensor(top_logprobs: list, topk: int) -> Optional[torch.Tensor]:
    if topk <= 0:
        return None

    rows = []
    for step_top_logprobs in top_logprobs:
        if isinstance(step_top_logprobs, dict):
            entries = list(step_top_logprobs.values())
        else:
            entries = list(step_top_logprobs or [])

        row = []
        for entry in entries[:topk]:
            if isinstance(entry, dict):
                logprob = entry.get("logprob", entry.get("log_probs", entry.get("log_prob")))
                token_id = entry.get("token_id", entry.get("idx", entry.get("id")))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                logprob, token_id = entry[0], entry[1]
            else:
                continue

            try:
                row.append([float(logprob), float(int(token_id))])
            except (TypeError, ValueError):
                continue

        if not row:
            row = [[-math.inf, -1.0] for _ in range(topk)]
        while len(row) < topk:
            row.append([-math.inf, -1.0])
        rows.append(row)

    if not rows:
        return None
    return torch.tensor(rows, dtype=torch.float32)

def _response_aligned_top_logprobs_to_tensor(
    prompt_len: int,
    output_top_logprobs: list,
    topk: int,
) -> torch.Tensor:
    if not output_top_logprobs:
        return None

    # Drafter loss is response-only. Some SGLang GPU speculative paths may
    # return partial input_top_logprobs, so keep prompt target rows as masked
    # placeholders and align finite rows only from generated response tokens.
    prompt_target_padding = [[] for _ in range(max(int(prompt_len) - 1, 0))]
    return _top_logprobs_to_tensor(prompt_target_padding + output_top_logprobs, topk)

def reconstruct_dense_logprob_view(target_topk_logprobs, topk, vocab_size):
    if topk <= 0:
        raise ValueError(f"topk must be positive when reconstructing dense logprob view, got {topk}")
    if target_topk_logprobs.dim() != 3 or target_topk_logprobs.size(-1) < 2:
        raise ValueError(
            "target_topk_logprobs must have shape [seq, topk, 2+] when reconstructing a dense logprob view, "
            f"but got shape={tuple(target_topk_logprobs.shape)}"
        )
    if target_topk_logprobs.numel() == 0:
        return torch.full(
            (
                target_topk_logprobs.shape[0],
                vocab_size,
            ),
            float("-inf"),
            dtype=target_topk_logprobs.dtype,
            device=target_topk_logprobs.device,
        )
    logprobs = target_topk_logprobs[..., 0]
    indices = target_topk_logprobs[..., 1].to(torch.long)
    return _scatter_topk_logprobs_with_tail(logprobs, indices, vocab_size)

async def _run_inference_step(url: str, tokens: list[int], sampling_params: dict, image_data, tokenizer, rollout_id):
    if rollout_id % 10 == 0:
        payload = {
            "input_ids": tokens,
            "sampling_params": sampling_params,
            "return_logprob": True,
            "return_hidden_states": True,
            "top_logprobs_num": 128,
            "logprob_start_len": 0,
        }
        if image_data:
            payload["image_data"] = image_data

        output = await post(url, payload)
        response_text = output["text"]
        if "output_token_logprobs" in output["meta_info"]:
            new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            new_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            new_tokens, new_log_probs = [], []
        finish_type = output["meta_info"]["finish_reason"]["type"]
        hs_data = output["meta_info"]["hidden_states"]
        hidden_states_list = []
        for i in range(len(hs_data)):
            # Convert hidden states to tensors
            h_state = torch.tensor(hs_data[i], dtype=torch.bfloat16)
            # Skip empty tensors
            if h_state.numel() == 0:
                continue
            # Ensure proper dimensions [1, hidden_dim] or [seq_len, hidden_dim]
            if h_state.dim() == 1:
                h_state = h_state.unsqueeze(0)
            elif h_state.dim() == 3:
                h_state = h_state.squeeze(0)
            hidden_states_list.append(h_state)
            # Concatenate non-empty tensors
        if hidden_states_list:
            hidden_states = torch.cat(hidden_states_list, dim=0)
            print(f"########hidden_state_shape: {hidden_states.shape}")
            engine_hidden_states = hidden_states
        output_top = output.get("meta_info", {}).get("output_top_logprobs", [])
        # print("############output_top",output_top)
        target_logprobs = _response_aligned_top_logprobs_to_tensor(
            prompt_len=len(tokens),
            output_top_logprobs=output_top,
            topk=128,
        )
        if target_logprobs is None:
            logger.warning("Failed to convert output top_logprobs to tensor; skip target_logprobs collection")
        target_scores = reconstruct_dense_logprob_view(
                            target_logprobs.squeeze(0),
                            topk=128,
                            vocab_size=32000,
                        ).unsqueeze(0)
    else:
        payload = {
            "input_ids": tokens,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }
        if image_data:
            payload["image_data"] = image_data

        output = await post(url, payload)
        response_text = output["text"]
        if "output_token_logprobs" in output["meta_info"]:
            new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            new_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
        else:
            new_tokens, new_log_probs = [], []
        finish_type = output["meta_info"]["finish_reason"]["type"]
        engine_hidden_states = None
        target_scores = None
    return response_text, new_tokens, new_log_probs, finish_type, engine_hidden_states, target_scores


def _process_env_step(env: BaseInteractionEnv, response_text: str, tokenizer, processor, args, sample_metadata):
    observation, done, _ = env.step(response_text)
    if done:
        return None, None, None, None, True

    next_user_message = env.format_observation(observation)
    obs_prompt_ids, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs = (
        _encode_observation_for_generation(
            tokenizer,
            processor,
            next_user_message,
            sample_metadata,
            args.apply_chat_template,
            args.apply_chat_template_kwargs,
        )
    )

    bos_id = tokenizer.bos_token_id
    if bos_id is not None and obs_prompt_ids and obs_prompt_ids[0] == bos_id:
        obs_prompt_ids = obs_prompt_ids[1:]

    return obs_prompt_ids, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs, False


def _append_to_sample(
    sample: Sample,
    response_tokens: list[int],
    tokens_to_add: list[int],
    logprobs: list[float],
    loss_mask_val: int,
) -> None:
    sample.tokens.extend(tokens_to_add)
    response_tokens.extend(tokens_to_add)
    sample.loss_mask.extend([loss_mask_val] * len(tokens_to_add))
    sample.rollout_log_probs.extend(logprobs)
    sample.response_length = len(response_tokens)


def _update_multimodal_state(
    sample: Sample,
    current_image_data,
    obs_image_data,
    obs_multimodal_inputs,
    obs_multimodal_train_inputs,
    multimodal_train_inputs_buffer: list[dict | None],
):
    if obs_image_data:
        current_image_data = (current_image_data or []) + obs_image_data

    if obs_multimodal_inputs:
        if not sample.multimodal_inputs:
            sample.multimodal_inputs = obs_multimodal_inputs
        elif isinstance(sample.multimodal_inputs, dict) and isinstance(obs_multimodal_inputs, dict):
            for key, val in obs_multimodal_inputs.items():
                if val is None:
                    continue
                if (
                    key in sample.multimodal_inputs
                    and isinstance(sample.multimodal_inputs[key], list)
                    and isinstance(val, list)
                ):
                    sample.multimodal_inputs[key].extend(val)
        else:
            sample.multimodal_inputs = obs_multimodal_inputs

    if obs_multimodal_train_inputs:
        multimodal_train_inputs_buffer.append(obs_multimodal_train_inputs)

    return current_image_data


def _should_stop_on_finish(sample: Sample, finish_type: str) -> bool:
    match finish_type:
        case "length":
            sample.status = Sample.Status.TRUNCATED
            return True
        case "abort":
            sample.status = Sample.Status.ABORTED
            return True
    return False


def _update_budget(budget, consumed: int):
    if budget is None:
        return None
    return budget - consumed


def _finalize_sample(sample: Sample, tokenizer, response_tokens, multimodal_train_inputs_buffer):
    sample.multimodal_train_inputs = _merge_multimodal_train_inputs(multimodal_train_inputs_buffer)
    sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    if sample.status is None:
        sample.status = Sample.Status.COMPLETED
    return sample


async def generate(args: Any, sample: Sample, sampling_params, rollout_id: int) -> Sample:
    """Custom multi-turn rollout that interacts with a pluggable environment."""
    assert not args.partial_rollout, "Partial rollout is not supported for interaction rollouts."

    env, env_module, config, state, url = _initialize_resources(args, sample)
    sampling_params = sampling_params.copy()
    current_image_data, response_tokens, budget, multimodal_train_inputs_buffer = _prepare_start_state(
        sample, state, args, sampling_params
    )
    try:
        env.reset()
        if budget is not None and budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            return sample

        cur_sampling_params = sampling_params
        for turn_idx in range(config["max_turns"]):
            if budget is not None:
                cur_sampling_params["max_new_tokens"] = budget

            response_text, new_response_tokens, new_response_log_probs, finish_type, engine_hidden_states, target_scores = await _run_inference_step(
                url, sample.tokens, cur_sampling_params, current_image_data, state.tokenizer, rollout_id=rollout_id
            )
            _append_to_sample(sample, response_tokens, new_response_tokens, new_response_log_probs, loss_mask_val=1)
            sample.hidden_states = engine_hidden_states
            sample.target_scores = target_scores
            # print("###############sample.hidden_states########")
            budget = _update_budget(budget, len(new_response_tokens))

            if _should_stop_on_finish(sample, finish_type):
                break
            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break

            obs_prompt_ids, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs, done = (
                _process_env_step(env, response_text, state.tokenizer, state.processor, args, sample.metadata)
            )
            if done:
                sample.status = Sample.Status.COMPLETED
                break

            obs_log_probs = [0.0] * len(obs_prompt_ids)
            _append_to_sample(sample, response_tokens, obs_prompt_ids, obs_log_probs, loss_mask_val=0)
            budget = _update_budget(budget, len(obs_prompt_ids))

            current_image_data = _update_multimodal_state(
                sample,
                current_image_data,
                obs_image_data,
                obs_multimodal_inputs,
                obs_multimodal_train_inputs,
                multimodal_train_inputs_buffer,
            )

            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break
            if turn_idx + 1 >= config["max_turns"]:
                sample.status = Sample.Status.COMPLETED
                break

        return _finalize_sample(sample, state.tokenizer, response_tokens, multimodal_train_inputs_buffer)
    finally:
        try:
            env.close()
        except Exception:
            pass
