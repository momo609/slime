import importlib
import logging
import os
import re
import textwrap
from functools import wraps
from typing import Any

import torch

try:
    import sglang.srt.entrypoints.engine
    from sglang.srt.utils import MultiprocessingSerializer as _MultiprocessingSerializer
except Exception:
    pass

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("SLIME_LOGGING_LEVEL", "WARN"))

_DISABLE_PATCH_ENV = "SLIME_DISABLE_SGLANG_PATCH"
_DRAFTER_RETURN_LAST_HIDDEN_ENV = "SLIME_SGLANG_DRAFTER_RETURN_LAST_HIDDEN"
_DRAFTER_HIDDEN_WINDOW_PARAM = "_slime_drafter_hidden_state_window"
_HIDDEN_STATE_FRONT_TOKENS_PARAM = "_slime_hidden_state_front_tokens_per_sample"
_HIDDEN_STATE_MAX_ROWS_PARAM = "_slime_hidden_state_max_rows"
_HIDDEN_STATE_PROMPT_LEN_PARAM = "_slime_prompt_len"
_HIDDEN_STATE_METADATA_MARKER = "__slime_hidden_state_metadata__"
_DRAFTER_RETURN_LAST_HIDDEN_PARAM = "_slime_drafter_return_last_hidden"
_DRAFTER_LAST_HIDDEN_STATES_ATTR = "_slime_drafter_last_hidden_states"

_SCHEDULER_PROCESS_PATCH_ATTR = "_slime_patched_scheduler_process"
_SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED = False
_SGLANG_DRAFTER_LAST_HIDDEN_OUTPUT_PATCHED = False
_SGLANG_EAGLE_VERIFY_HIDDEN_STATES_PATCHED = False
_SGLANG_SCHEDULER_PROCESS_PATCHED = False

_ORIGINAL_SGLANG_RUN_SCHEDULER_PROCESS = None
_ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS = None


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    return normalized in {"1", "true", "on", "yes", "y"}


def _sglang_drafter_return_last_hidden_enabled() -> bool:
    return _env_flag_enabled(_DRAFTER_RETURN_LAST_HIDDEN_ENV, default=False)


def _sglang_verl_patches_disabled() -> bool:
    return _env_flag_enabled(_DISABLE_PATCH_ENV, default=False)


def _is_torch_tensor(value: Any) -> bool:
    is_tensor = getattr(torch, "is_tensor", None)
    return bool(callable(is_tensor) and is_tensor(value))


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _custom_flag_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "on", "yes", "y"}
    return bool(value)


def _sglang_req_custom_params(req) -> dict[str, Any]:
    sampling_params = getattr(req, "sampling_params", None)
    custom_params = getattr(sampling_params, "custom_params", None)
    return custom_params if isinstance(custom_params, dict) else {}


def _sglang_req_requests_last_hidden_for_drafter(req) -> bool:
    return _custom_flag_enabled(
        _sglang_req_custom_params(req).get(_DRAFTER_RETURN_LAST_HIDDEN_PARAM, False)
    )


def _sglang_forward_batch_requests_last_hidden_for_drafter(forward_batch) -> bool:
    if _sglang_drafter_return_last_hidden_enabled():
        return True
    for req in getattr(forward_batch, "reqs", []) or []:
        if (
            getattr(req, "return_hidden_states", False)
            and _sglang_req_requests_last_hidden_for_drafter(req)
        ):
            return True
    return False


def _sglang_req_prompt_len(req) -> int:
    custom_prompt_len = _int_or_none(_sglang_req_custom_params(req).get(_HIDDEN_STATE_PROMPT_LEN_PARAM))
    if custom_prompt_len is not None:
        return max(custom_prompt_len, 0)
    try:
        return len(getattr(req, "origin_input_ids", []) or [])
    except TypeError:
        return 0


def _sglang_req_hidden_prefix_cache_rows(req) -> int:
    value = _int_or_none(getattr(req, "_slime_hidden_prefix_cache_rows", None))
    if value is not None:
        return max(value, 0)
    return 0


def _sglang_hidden_window_config(req) -> dict[str, int] | None:
    custom_params = _sglang_req_custom_params(req)
    if not _custom_flag_enabled(custom_params.get(_DRAFTER_HIDDEN_WINDOW_PARAM, False)):
        return None

    front_tokens = _positive_int_or_none(custom_params.get(_HIDDEN_STATE_FRONT_TOKENS_PARAM))
    if front_tokens is None:
        front_tokens = _positive_int_or_none(custom_params.get(_HIDDEN_STATE_MAX_ROWS_PARAM))
    if front_tokens is None:
        return None

    prompt_len = _sglang_req_prompt_len(req)
    prefix_cache_rows = _sglang_req_hidden_prefix_cache_rows(req)
    window_start = max(prefix_cache_rows, max(prompt_len - 1, 0))
    return {
        "prompt_len": prompt_len,
        "prefix_cache_rows": prefix_cache_rows,
        "window_start": window_start,
        "window_end": window_start + front_tokens,
    }


def _sglang_hidden_chunk_rows(chunk) -> int:
    if _is_torch_tensor(chunk):
        if chunk.dim() <= 1:
            return 1
        return int(chunk.shape[0])
    try:
        return len(chunk)
    except TypeError:
        return 1


def _slice_sglang_hidden_chunk(chunk, start: int, end: int):
    if start <= 0 and end >= _sglang_hidden_chunk_rows(chunk):
        return chunk
    return chunk[start:end]


def _to_cpu_sglang_hidden_chunk(chunk):
    if _is_torch_tensor(chunk):
        return chunk.detach().to("cpu", copy=True)
    return chunk


def _sglang_concat_last_hidden_for_drafter(req, logits_output, hidden_chunk, last_hidden_chunk):
    if not _sglang_req_requests_last_hidden_for_drafter(req):
        return hidden_chunk
    if last_hidden_chunk is None:
        return hidden_chunk
    if not (_is_torch_tensor(hidden_chunk) and _is_torch_tensor(last_hidden_chunk)):
        return hidden_chunk
    if tuple(hidden_chunk.shape[:-1]) != tuple(last_hidden_chunk.shape[:-1]):
        return hidden_chunk
    if last_hidden_chunk.device != hidden_chunk.device:
        last_hidden_chunk = last_hidden_chunk.to(hidden_chunk.device)
    return torch.cat((hidden_chunk, last_hidden_chunk), dim=-1)


def _slice_sglang_drafter_last_hidden_output(logits_output, index):
    last_hidden_states = getattr(logits_output, _DRAFTER_LAST_HIDDEN_STATES_ATTR, None)
    if last_hidden_states is None:
        return None
    return last_hidden_states[index]


def _append_sglang_hidden_chunk_payload(req, chunk, metadata: dict[str, int] | None = None) -> int:
    appended = _to_cpu_sglang_hidden_chunk(chunk)
    appended_rows = _sglang_hidden_chunk_rows(appended)
    if metadata is None:
        req.hidden_states.append(appended)
    else:
        req.hidden_states.append(
            {
                _HIDDEN_STATE_METADATA_MARKER: True,
                "hidden_states": appended,
                **metadata,
            }
        )
    return appended_rows


def _finish_sglang_hidden_state_capture(req, batch=None) -> None:
    req.return_hidden_states = False
    if batch is not None:
        setattr(batch, "return_hidden_states", _sglang_batch_requests_hidden_states(batch))


def _append_sglang_hidden_state_chunk_with_budget(
    req,
    chunk,
    *,
    position_start: int | None = None,
    prefix_cache_rows: int | None = None,
    batch=None,
) -> None:
    if not getattr(req, "return_hidden_states", False):
        return

    if prefix_cache_rows is not None:
        setattr(req, "_slime_hidden_prefix_cache_rows", max(int(prefix_cache_rows), 0))

    chunk_rows = _sglang_hidden_chunk_rows(chunk)
    if chunk_rows <= 0:
        return

    if position_start is None:
        position_start = _int_or_none(getattr(req, "_slime_hidden_next_position", None))
        if position_start is None:
            position_start = _sglang_req_hidden_prefix_cache_rows(req)
    position_start = max(int(position_start), 0)
    position_end = position_start + chunk_rows
    setattr(req, "_slime_hidden_next_position", position_end)

    window_config = _sglang_hidden_window_config(req)
    if window_config is not None:
        if getattr(req, "_slime_hidden_state_window_done", False):
            return
        window_start = window_config["window_start"]
        window_end = window_config["window_end"]
        clipped_start = max(position_start, window_start)
        clipped_end = min(position_end, window_end)
        if clipped_start >= clipped_end:
            if position_end >= window_end:
                setattr(req, "_slime_hidden_state_window_done", True)
                _finish_sglang_hidden_state_capture(req, batch)
            return
        local_start = clipped_start - position_start
        local_end = clipped_end - position_start
        clipped_chunk = _slice_sglang_hidden_chunk(chunk, local_start, local_end)
        _append_sglang_hidden_chunk_payload(req, clipped_chunk, {
            "position_start": clipped_start, "position_end": clipped_end,
            "prefix_cache_rows": window_config["prefix_cache_rows"],
            "window_start": window_start, "window_end": window_end,
        })
        if clipped_end >= window_end:
            setattr(req, "_slime_hidden_state_window_done", True)
            _finish_sglang_hidden_state_capture(req, batch)
        return

    if getattr(req, "_slime_hidden_state_budget_done", False):
        return

    max_rows = getattr(req, "_slime_hidden_state_max_rows", None)
    if max_rows is not None:
        try:
            max_rows = int(max_rows)
        except (TypeError, ValueError):
            max_rows = None

    collected_rows = int(getattr(req, "_slime_hidden_state_rows", 0) or 0)
    if max_rows is not None and max_rows > 0:
        remaining_rows = max_rows - collected_rows
        if remaining_rows <= 0:
            setattr(req, "_slime_hidden_state_budget_done", True)
            _finish_sglang_hidden_state_capture(req, batch)
            return
        if _is_torch_tensor(chunk) and chunk.dim() > 0 and int(chunk.shape[0]) > remaining_rows:
            chunk = chunk[:remaining_rows]

    appended_rows = _append_sglang_hidden_chunk_payload(req, chunk)
    collected_rows += appended_rows
    setattr(req, "_slime_hidden_state_rows", collected_rows)
    if max_rows is not None and max_rows > 0 and collected_rows >= max_rows:
        setattr(req, "_slime_hidden_state_budget_done", True)
        _finish_sglang_hidden_state_capture(req, batch)


def _append_sglang_prefill_hidden_states(req, logits_output, hidden_state_offset: int, extend_input_len: int, batch=None) -> int:
    hidden_states = getattr(logits_output, "hidden_states", None)
    if hidden_states is None:
        return hidden_state_offset

    try:
        rows = max(int(extend_input_len), 0)
    except (TypeError, ValueError):
        rows = len(getattr(req, "origin_input_ids", []) or [])

    prompt_len = len(getattr(req, "origin_input_ids", []) or [])
    prefix_cache_rows = max(prompt_len - rows, 0)
    end = hidden_state_offset + rows
    chunk = hidden_states[hidden_state_offset:end]
    chunk = _sglang_concat_last_hidden_for_drafter(
        req, logits_output, chunk,
        _slice_sglang_drafter_last_hidden_output(logits_output, slice(hidden_state_offset, end)),
    )
    _append_sglang_hidden_state_chunk_with_budget(
        req, chunk, position_start=prefix_cache_rows, prefix_cache_rows=prefix_cache_rows, batch=batch,
    )
    return end


def _append_sglang_decode_hidden_states(req, logits_output, result, req_index: int, hidden_state_offset: int, batch=None) -> int:
    hidden_states = getattr(logits_output, "hidden_states", None)
    if hidden_states is None:
        return hidden_state_offset

    accept_lengths = getattr(result, "accept_length_per_req_cpu", None)
    if accept_lengths is not None and req_index < len(accept_lengths) and _is_torch_tensor(hidden_states):
        rows = max(int(accept_lengths[req_index]) + 1, 1)
        position_start = max(
            len(getattr(req, "origin_input_ids", []) or []) + len(getattr(req, "output_ids", []) or []) - rows, 0,
        )
        if hidden_states.dim() == 3 and req_index < int(hidden_states.shape[0]):
            if int(hidden_states.shape[1]) >= rows:
                chunk = hidden_states[req_index, :rows]
                chunk = _sglang_concat_last_hidden_for_drafter(
                    req, logits_output, chunk,
                    _slice_sglang_drafter_last_hidden_output(logits_output, (req_index, slice(0, rows))),
                )
                _append_sglang_hidden_state_chunk_with_budget(
                    req, chunk, position_start=position_start, batch=batch,
                )
                return hidden_state_offset + rows

        expected_rows = sum(max(int(accept_len) + 1, 1) for accept_len in accept_lengths)
        end = hidden_state_offset + rows
        if hidden_states.dim() >= 2 and int(hidden_states.shape[0]) >= expected_rows and end <= int(hidden_states.shape[0]):
            chunk = hidden_states[hidden_state_offset:end]
            chunk = _sglang_concat_last_hidden_for_drafter(
                req, logits_output, chunk,
                _slice_sglang_drafter_last_hidden_output(logits_output, slice(hidden_state_offset, end)),
            )
            _append_sglang_hidden_state_chunk_with_budget(
                req, chunk, position_start=position_start, batch=batch,
            )
            return end

    if not getattr(req, "return_hidden_states", False):
        return hidden_state_offset
    position_start = max(
        len(getattr(req, "origin_input_ids", []) or []) + len(getattr(req, "output_ids", []) or []) - 2, 0,
    )
    if _is_torch_tensor(hidden_states):
        chunk = hidden_states[req_index]
        chunk = _sglang_concat_last_hidden_for_drafter(
            req, logits_output, chunk,
            _slice_sglang_drafter_last_hidden_output(logits_output, req_index),
        )
        _append_sglang_hidden_state_chunk_with_budget(
            req, chunk, position_start=position_start, batch=batch,
        )
    else:
        _append_sglang_hidden_state_chunk_with_budget(
            req, hidden_states[req_index], position_start=position_start, batch=batch,
        )
    return hidden_state_offset


def _sglang_batch_requests_hidden_states(batch) -> bool:
    return any(bool(getattr(req, "return_hidden_states", False)) for req in getattr(batch, "reqs", []) or [])


def _ensure_sglang_eagle_verify_full_hidden_mode(batch, spec_info) -> None:
    if not _sglang_batch_requests_hidden_states(batch):
        return
    try:
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
    except Exception as exc:
        logger.debug("Cannot import SGLang CaptureHiddenMode for EAGLE hidden-state patch: %s", exc)
        return
    spec_info.capture_hidden_mode = CaptureHiddenMode.FULL


# ============ LogitsProcessor Patch ============

def _make_sglang_drafter_last_hidden_forward_patch(original_method):
    @wraps(original_method)
    def patched_logits_processor_forward(
        self, input_ids, hidden_states, lm_head, logits_metadata,
        aux_hidden_states=None, hidden_states_before_norm=None,
    ):
        return_last_hidden = False
        if _sglang_drafter_return_last_hidden_enabled():
            return_last_hidden = True
        else:
            return_last_hidden = _sglang_forward_batch_requests_last_hidden_for_drafter(logits_metadata)

        output = original_method(
            self, input_ids, hidden_states, lm_head, logits_metadata,
            aux_hidden_states, hidden_states_before_norm,
        )
        if return_last_hidden and getattr(output, "hidden_states", None) is not None and hidden_states is not None:
            setattr(output, _DRAFTER_LAST_HIDDEN_STATES_ATTR, hidden_states)
        return output

    patched_logits_processor_forward._slime_patched_drafter_last_hidden_output = True
    return patched_logits_processor_forward


def patch_sglang_drafter_last_hidden_output() -> None:
    global _SGLANG_DRAFTER_LAST_HIDDEN_OUTPUT_PATCHED
    if _SGLANG_DRAFTER_LAST_HIDDEN_OUTPUT_PATCHED:
        return

    try:
        logits_module = importlib.import_module("sglang.srt.layers.logits_processor")
        logits_processor_cls = getattr(logits_module, "LogitsProcessor")
    except Exception as exc:
        logger.debug("Skip SGLang drafter last-hidden output patch: %s", exc)
        return

    original_forward = getattr(logits_processor_cls, "forward", None)
    if original_forward is None or getattr(original_forward, "_slime_patched_drafter_last_hidden_output", False):
        if original_forward is not None:
            _SGLANG_DRAFTER_LAST_HIDDEN_OUTPUT_PATCHED = True
        return

    setattr(logits_processor_cls, "forward", _make_sglang_drafter_last_hidden_forward_patch(original_forward))
    _SGLANG_DRAFTER_LAST_HIDDEN_OUTPUT_PATCHED = True
    logger.info("SGLang drafter last-hidden output patch active")


# ============ Hidden States Tensor Output Patch ============

_SGLANG_HIDDEN_STATES_LIST_OUTPUT_PATTERN = re.compile(r"\.cpu\(\)\s*\.clone\(\)\s*\.tolist\(\)")

_SGLANG_DECODE_REQUEST_LOOP_PATTERN = re.compile(
    r"(?ms)^(?P<indent>[ \t]+)for i, \(req, next_token_id\) in enumerate\(\s*"
    r"zip\(batch\.reqs,\s*next_token_ids\)\s*\):\r?\n",
)

_SGLANG_DECODE_HIDDEN_STATES_APPEND_PATTERN = re.compile(
    r"(?ms)^(?P<indent>[ \t]+)if\s*(?:\(\s*)?req\.return_hidden_states\s+"
    r"and\s+logits_output\.hidden_states\s+is\s+not\s+None\s*(?:\))?\s*:\r?\n"
    r"(?P=indent)[ \t]+req\.hidden_states\.append\(\r?\n"
    r".*?"
    r"^(?P=indent)[ \t]+\)\r?\n",
)

_SGLANG_PREFILL_HIDDEN_STATES_APPEND_PATTERN = re.compile(
    r"(?ms)^(?P<indent>[ \t]+)if\s*(?:\(\s*)?req\.return_hidden_states\s+"
    r"and\s+logits_output\.hidden_states\s+is\s+not\s+None\s*(?:\))?\s*:\r?\n"
    r"(?P=indent)[ \t]+req\.hidden_states\.append\(\r?\n"
    r".*?"
    r"(?:\.detach\(\)\.to\(\"cpu\",\s*copy=True\)|\.cpu\(\)\s*\.clone\(\)\s*\.tolist\(\))\r?\n"
    r"(?P=indent)[ \t]+\)\r?\n",
)


def _replace_sglang_hidden_states_list_output(source: str) -> str:
    return _SGLANG_HIDDEN_STATES_LIST_OUTPUT_PATTERN.sub('.detach().to("cpu", copy=True)', source)


def _render_sglang_decode_hidden_states_append(match: re.Match) -> str:
    indent = match.group("indent")
    return (
        f"{indent}hidden_state_offset = _append_sglang_decode_hidden_states(\n"
        f"{indent}    req, logits_output, result, i, hidden_state_offset, batch,\n"
        f"{indent})\n"
    )


def _render_sglang_prefill_hidden_states_append(match: re.Match) -> str:
    indent = match.group("indent")
    return (
        f"{indent}hidden_state_offset = _append_sglang_prefill_hidden_states(\n"
        f"{indent}    req, logits_output, hidden_state_offset,\n"
        f"{indent}    (extend_input_len_per_req[i] if extend_input_len_per_req is not None else len(req.origin_input_ids)),\n"
        f"{indent}    batch,\n"
        f"{indent})\n"
    )


def _insert_sglang_decode_hidden_state_offset(source: str) -> str | None:
    if re.search(r"(?m)^[ \t]+hidden_state_offset = 0\s*$", source):
        return source
    patched_source, loop_count = _SGLANG_DECODE_REQUEST_LOOP_PATTERN.subn(
        lambda match: f"{match.group('indent')}hidden_state_offset = 0\n\n{match.group(0)}",
        source, count=1,
    )
    if loop_count <= 0:
        return None
    return patched_source


def _patch_sglang_decode_hidden_states_source(source: str) -> str | None:
    patched_source, count = _SGLANG_DECODE_HIDDEN_STATES_APPEND_PATTERN.subn(
        _render_sglang_decode_hidden_states_append, source, count=1,
    )
    if count <= 0:
        return None
    return _insert_sglang_decode_hidden_state_offset(patched_source)


def _patch_sglang_prefill_hidden_states_source(source: str) -> str | None:
    patched_source, count = _SGLANG_PREFILL_HIDDEN_STATES_APPEND_PATTERN.subn(
        _render_sglang_prefill_hidden_states_append, source, count=1,
    )
    if count <= 0:
        return None
    return patched_source


def _make_sglang_hidden_states_tensor_output_patch(original_method):
    try:
        source = inspect = __import__("inspect")
        source = inspect.getsource(original_method)
    except Exception:
        return None

    source = textwrap.dedent(source)
    patched_source, conversion_count = _SGLANG_HIDDEN_STATES_LIST_OUTPUT_PATTERN.subn(
        '.detach().to("cpu", copy=True)', source,
    )

    if original_method.__name__ == "process_batch_result_prefill":
        patched_prefill = _patch_sglang_prefill_hidden_states_source(patched_source)
        if patched_prefill is None:
            return None
        patched_source = patched_prefill
    elif original_method.__name__ == "process_batch_result_decode":
        patched_decode = _patch_sglang_decode_hidden_states_source(patched_source)
        if patched_decode is None:
            return None
        patched_source = patched_decode
    elif conversion_count <= 0:
        return None

    if patched_source == source:
        return None

    globals_dict = original_method.__globals__
    globals_dict["_append_sglang_prefill_hidden_states"] = _append_sglang_prefill_hidden_states
    globals_dict["_append_sglang_decode_hidden_states"] = _append_sglang_decode_hidden_states
    namespace = {}

    import inspect as _inspect
    exec(
        "from __future__ import annotations\n" + patched_source,
        globals_dict,
        namespace,
    )
    patched_method = namespace[original_method.__name__]
    patched_method = wraps(original_method)(patched_method)
    patched_method._slime_patched_hidden_states_tensor_output = True
    return patched_method


def patch_sglang_hidden_states_tensor_output() -> None:
    global _SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED
    patch_sglang_drafter_last_hidden_output()

    if _SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED:
        return

    try:
        module = importlib.import_module("sglang.srt.managers.scheduler_output_processor_mixin")
        processor_cls = getattr(module, "SchedulerOutputProcessorMixin")
    except Exception as exc:
        logger.debug("Skip SGLang hidden-state tensor output patch: %s", exc)
        return

    for method_name in ("process_batch_result_prefill", "process_batch_result_decode"):
        original_method = getattr(processor_cls, method_name, None)
        if original_method is None or getattr(original_method, "_slime_patched_hidden_states_tensor_output", False):
            continue
        patched_method = _make_sglang_hidden_states_tensor_output_patch(original_method)
        if patched_method is None:
            logger.debug("Skip SGLang hidden-state tensor output patch for %s", method_name)
            continue
        setattr(processor_cls, method_name, patched_method)

    _SGLANG_HIDDEN_STATES_TENSOR_OUTPUT_PATCHED = True
    logger.info("SGLang hidden-state tensor output patch active")


# ============ EAGLE Verify Full Hidden States Patch ============

def _sglang_hidden_state_rows(hidden_states) -> int:
    if not torch.is_tensor(hidden_states):
        try:
            return len(hidden_states)
        except TypeError:
            return 0
    if hidden_states.dim() == 0:
        return 1
    if hidden_states.dim() >= 3:
        return int(hidden_states.shape[0]) * int(hidden_states.shape[1])
    return int(hidden_states.shape[0])


def _sglang_eagle_verify_expected_hidden_rows(batch, spec_info) -> int:
    batch_size = len(getattr(batch, "reqs", []) or [])
    draft_token_num = int(getattr(spec_info, "draft_token_num", 0) or 0)
    return batch_size * draft_token_num


def _sglang_eagle_verify_hidden_states_incomplete(batch, spec_info, logits_output) -> bool:
    if not _sglang_batch_requests_hidden_states(batch):
        return False
    expected_rows = _sglang_eagle_verify_expected_hidden_rows(batch, spec_info)
    if expected_rows <= 0:
        return False
    hidden_states = getattr(logits_output, "hidden_states", None)
    return hidden_states is None or _sglang_hidden_state_rows(hidden_states) < expected_rows


def _rerun_sglang_eagle_verify_without_graph(worker, model_worker_batch):
    target_worker = getattr(worker, "target_worker", None)
    model_runner = getattr(target_worker, "model_runner", None)
    graph_runner = getattr(model_runner, "graph_runner", None)
    try:
        if model_runner is not None:
            model_runner.graph_runner = None
        return target_worker.forward_batch_generation(model_worker_batch, is_verify=True)
    finally:
        if model_runner is not None:
            model_runner.graph_runner = graph_runner


def _validate_sglang_eagle_verify_hidden_states(batch, spec_info, logits_output) -> None:
    if not _sglang_eagle_verify_hidden_states_incomplete(batch, spec_info, logits_output):
        return
    hidden_states = getattr(logits_output, "hidden_states", None)
    shape = tuple(hidden_states.shape) if torch.is_tensor(hidden_states) else None
    expected_rows = _sglang_eagle_verify_expected_hidden_rows(batch, spec_info)
    actual_rows = _sglang_hidden_state_rows(hidden_states)
    raise RuntimeError(
        f"SGLang EAGLE verify did not return full hidden states for drafter training: "
        f"actual_rows={actual_rows}, expected_rows={expected_rows}, shape={shape}."
    )


def _make_sglang_eagle_verify_full_hidden_patch(original_method):
    import inspect as _inspect
    try:
        source = _inspect.getsource(original_method)
    except Exception:
        return None

    source = textwrap.dedent(source)
    patched_source = source

    old_prepare = "        spec_info.prepare_for_verify(batch, self.page_size)\n"
    new_prepare = (
        "        spec_info.prepare_for_verify(batch, self.page_size)\n"
        "        _ensure_sglang_eagle_verify_full_hidden_mode(batch, spec_info)\n"
    )
    if old_prepare in patched_source:
        patched_source = patched_source.replace(old_prepare, new_prepare, 1)
    else:
        return None

    globals_dict = original_method.__globals__
    globals_dict["_ensure_sglang_eagle_verify_full_hidden_mode"] = _ensure_sglang_eagle_verify_full_hidden_mode
    namespace = {}
    exec(
        "from __future__ import annotations\n" + patched_source,
        globals_dict,
        namespace,
    )
    patched_method = namespace[original_method.__name__]
    patched_method = wraps(original_method)(patched_method)
    patched_method._slime_patched_eagle_verify_full_hidden_states = True
    return patched_method


def patch_sglang_eagle_verify_hidden_states_full() -> None:
    global _SGLANG_EAGLE_VERIFY_HIDDEN_STATES_PATCHED
    if _SGLANG_EAGLE_VERIFY_HIDDEN_STATES_PATCHED:
        return

    targets = (
        ("sglang.srt.speculative.eagle_worker", "EAGLEWorker"),
        ("sglang.srt.speculative.multi_layer_eagle_worker", "MultiLayerEagleWorker"),
    )
    patched_targets = []
    for module_name, class_name in targets:
        try:
            module = importlib.import_module(module_name)
            worker_cls = getattr(module, class_name)
            original_method = getattr(worker_cls, "verify", None)
        except Exception as exc:
            logger.debug("Skip SGLang EAGLE full hidden-state patch for %s.%s: %s", module_name, class_name, exc)
            continue
        if original_method is None or getattr(original_method, "_slime_patched_eagle_verify_full_hidden_states", False):
            if original_method is not None:
                patched_targets.append(f"{module_name}.{class_name}.verify")
            continue
        patched_method = _make_sglang_eagle_verify_full_hidden_patch(original_method)
        if patched_method is not None:
            setattr(worker_cls, "verify", patched_method)
            patched_targets.append(f"{module_name}.{class_name}.verify")
        else:
            logger.debug("Skip SGLang EAGLE full hidden-state patch for %s.%s", module_name, class_name)

    if patched_targets:
        _SGLANG_EAGLE_VERIFY_HIDDEN_STATES_PATCHED = True
        logger.info("SGLang EAGLE verify full hidden states patch active for %s", ", ".join(patched_targets))


# ============ Scheduler Process Entrypoint Patch ============

def _apply_sglang_child_process_patches() -> None:
    if _sglang_verl_patches_disabled():
        logger.info("Skip all slime SGLang patches because %s=1.", _DISABLE_PATCH_ENV)
        return
    logger.info("Applying slime SGLang patches in scheduler subprocess.")
    patch_sglang_hidden_states_tensor_output()
    patch_sglang_eagle_verify_hidden_states_full()


def _run_scheduler_process_with_slime_patches(*args, **kwargs):
    global _ORIGINAL_SGLANG_RUN_SCHEDULER_PROCESS
    _apply_sglang_child_process_patches()
    if _ORIGINAL_SGLANG_RUN_SCHEDULER_PROCESS is None:
        _ORIGINAL_SGLANG_RUN_SCHEDULER_PROCESS = sglang.srt.entrypoints.engine.run_scheduler_process
    return _ORIGINAL_SGLANG_RUN_SCHEDULER_PROCESS(*args, **kwargs)


_run_scheduler_process_with_slime_patches._slime_patched = True
setattr(_run_scheduler_process_with_slime_patches, _SCHEDULER_PROCESS_PATCH_ATTR, True)


def _run_direct_scheduler_process_with_slime_patches(*args, **kwargs):
    global _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS
    _apply_sglang_child_process_patches()
    if _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS is None:
        scheduler_module = importlib.import_module("sglang.srt.managers.scheduler")
        _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS = scheduler_module.run_scheduler_process
    return _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS(*args, **kwargs)


_run_direct_scheduler_process_with_slime_patches._slime_patched = True
setattr(_run_direct_scheduler_process_with_slime_patches, _SCHEDULER_PROCESS_PATCH_ATTR, True)


def patch_slime_sglang_scheduler_entrypoints() -> None:
    global _SGLANG_SCHEDULER_PROCESS_PATCHED
    if _SGLANG_SCHEDULER_PROCESS_PATCHED:
        return

    modules = [sglang.srt.entrypoints.engine]
    try:
        modules.append(importlib.import_module("sglang.srt.managers.scheduler"))
    except Exception as exc:
        logger.debug("Skip direct scheduler entrypoint patch: %s", exc)

    for module in modules:
        original_run_scheduler = getattr(module, "run_scheduler_process", None)
        if original_run_scheduler is None or getattr(
            original_run_scheduler, _SCHEDULER_PROCESS_PATCH_ATTR, False,
        ):
            continue
        if module is sglang.srt.entrypoints.engine:
            module.run_scheduler_process = _run_scheduler_process_with_slime_patches
        else:
            global _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS
            _ORIGINAL_SGLANG_DIRECT_RUN_SCHEDULER_PROCESS = original_run_scheduler
            module.run_scheduler_process = _run_direct_scheduler_process_with_slime_patches

    _SGLANG_SCHEDULER_PROCESS_PATCHED = True
    logger.info("Slime SGLang scheduler entrypoints patched")


def install_slime_sglang_patches() -> None:
    if _sglang_verl_patches_disabled():
        logger.info("Skip installing slime SGLang patches because %s=1.", _DISABLE_PATCH_ENV)
        return

    patch_slime_sglang_scheduler_entrypoints()
    logger.info("Slime SGLang patches installed")