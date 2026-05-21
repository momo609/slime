# EAGLE3 投机解码训练：SGLang + verl 源码级修改文档

## 目录

1. [架构概览](#1-架构概览)
2. [SGLang 源码修改](#2-sglang-源码修改)
   - [2.1 `logits_processor.py` — 返回 last_hidden_states](#21-logits_processorpy--返回-last_hidden_states)
   - [2.2 `scheduler_output_processor_mixin.py` — 隐藏状态收集 + Tensor输出](#22-scheduler_output_processor_mixinpy--隐藏状态收集--tensor输出)
   - [2.3 `eagle_worker.py` — 权重更新路由 + Verify 完整隐藏状态](#23-eagle_workerpy--权重更新路由--verify-完整隐藏状态)
   - [2.4 `logprob.py` — Top-Logprobs Tensor 输出](#24-logprobpy--top-logprobs-tensor-输出)
   - [2.5 `tokenizer_manager.py` — Meta Info 侧通道](#25-tokenizer_managerpy--meta-info-侧通道)
   - [2.6 `cuda_graph_runner.py` / `npu_graph_runner.py` — Graph Replay last_hidden](#26-cuda_graph_runnerpy--npu_graph_runnerpy--graph-replay-last_hidden)
3. [Verl 源码修改](#3-verl-源码修改)
   - [3.1 `eagle3_trainer_backend.py` (新增)](#31-eagle3_trainer_backendpy-新增)
   - [3.2 `llama_eagle.py` (新增)](#32-llama_eaglepy-新增)
   - [3.3 `base.py` (新增)](#33-basepy-新增)
   - [3.4 `target_head.py` (新增)](#34-target_headpy-新增)
   - [3.5 `sglang_rollout.py` (修改)](#35-sglang_rolloutpy-修改)
4. [删除的文件](#4-删除的文件)
5. [附录：环境变量与配置](#5-附录环境变量与配置)

---

## 1. 架构概览

```
┌───────────────────────────────────────────────────────────────┐
│                    SGLang (Rollout 引擎)                       │
│                                                               │
│  【源码级直接修改以下文件】                                       │
│  ┌───────────────────────────────────────────────────────┐    │
│  │ logits_processor.py        → last_hidden_states 输出     │    │
│  │ scheduler_output_processor_mixin.py → 隐藏状态收集+tensor  │    │
│  │ eagle_worker.py            → 权重路由+Verify FULL模式     │    │
│  │ logprob.py                 → Top-logprobs tensor 输出    │    │
│  │ tokenizer_manager.py       → Meta info 侧通道           │    │
│  │ cuda/npu_graph_runner.py   → Graph replay last_hidden   │    │
│  └───────────────────────────────────────────────────────┘    │
│                          ↓                                    │
│   返回: hidden_states [3层拼接 + last_hidden]                   │
│          + topk_logprobs (target 教师信号)                      │
└───────────────────────────────────────────────────────────────┘
                           ↓
┌───────────────────────────────────────────────────────────────┐
│                  Verl (训练后端)                                │
│                                                               │
│  【新增文件】                                                   │
│  eagle3_trainer_backend.py → EAGLE3 训练后端                    │
│  llama_eagle.py             → EAGLE3 Draft 模型                 │
│  base.py                    → Draft 模型基类                    │
│  target_head.py             → 外部教师模型                      │
│                                                               │
│  【修改文件】                                                   │
│  sglang_rollout.py          → 权重加载路由 + draft 权重更新      │
│                                                               │
│  【删除文件】                                                   │
│  sglang_patch.py            → 功能已并入源码直接修改             │
└───────────────────────────────────────────────────────────────┘
```

---

## 2. SGLang 源码修改

### 2.1 `logits_processor.py` — 返回 last_hidden_states

**文件**: `sglang/srt/layers/logits_processor.py`
**目的**: 在 `LogitsProcessor.forward()` 中将 `last_hidden_states` 注入到 `logits_output` 对象上，使下游可以将其与 3 层 hidden states 拼接，构成完整的 EAGLE3 训练输入。

**原理**: EAGLE3 训练需要 4 部分 hidden states 输入: 3 层 target 模型的中间 hidden states（由 SGLang EAGLE 框架返回的 `hidden_states`） + 1 层最后的 last_hidden（经过 `LogitsProcessor` 前的 `hidden_states`，即 norm 前的最后一层输出）。这里的 patch 负责把 last_hidden 注入到 `logits_output` 上，后续在 `scheduler_output_processor_mixin.py` 中会将其与 3 层 hidden states 拼接。

```diff
--- a/sglang/srt/layers/logits_processor.py
+++ b/sglang/srt/layers/logits_processor.py
@@ -1,5 +1,7 @@
 import logging
+import os
 from typing import Optional, TYPE_CHECKING

+import torch
 import triton
 import triton.language as tl

@@ -14,6 +16,11 @@
     from sglang.srt.model_executor.forward_batch_info import ForwardBatch

 logger = logging.getLogger(__name__)
+
+# --- verl: drafter last-hidden output constants ---
+_VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR = "_verl_drafter_last_hidden_states"
+_VERL_DRAFTER_RETURN_LAST_HIDDEN_ENV = "VERL_SGLANG_DRAFTER_RETURN_LAST_HIDDEN"
+_VERL_DRAFTER_RETURN_LAST_HIDDEN_PARAM = "_verl_drafter_return_last_hidden"


 class LogitsProcessor:

@@ -X,Y +Z,W @@
     def forward(
         self,
         input_ids,
         hidden_states,
         lm_head,
         logits_metadata,
+        aux_hidden_states=None,
+        hidden_states_before_norm=None,
     ):
+        # --- verl: check if drafter needs last_hidden_states ---
+        return_last_hidden = False
+        if os.environ.get(_VERL_DRAFTER_RETURN_LAST_HIDDEN_ENV, "0") == "1":
+            return_last_hidden = True
+        else:
+            return_last_hidden = self._batch_requests_last_hidden(logits_metadata)

         # (original logits processing: apply logit bias, penalty, etc.)
         logits_output = ...

+        # --- verl: attach last hidden states for drafter training ---
+        if (
+            return_last_hidden
+            and getattr(logits_output, "hidden_states", None) is not None
+            and hidden_states is not None
+        ):
+            setattr(
+                logits_output,
+                _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR,
+                hidden_states,
+            )
+
         return logits_output
+
+    # --- verl: helper methods ---
+    @staticmethod
+    def _batch_requests_last_hidden(logits_metadata) -> bool:
+        """Check if any request in the batch requires last_hidden for drafter.

+        Traverses all requests in logits_metadata and checks their
+        sampling_params.custom_params for the _verl_drafter_return_last_hidden flag.
+        """
+        for req in getattr(logits_metadata, "reqs", []) or []:
+            if not getattr(req, "return_hidden_states", False):
+                continue
+            sampling_params = getattr(req, "sampling_params", None)
+            custom_params = getattr(sampling_params, "custom_params", None) or {}
+            if custom_params.get(_VERL_DRAFTER_RETURN_LAST_HIDDEN_PARAM):
+                return True
+        return False
```

---

### 2.2 `scheduler_output_processor_mixin.py` — 隐藏状态收集 + Tensor输出

**文件**: `sglang/srt/managers/scheduler_output_processor_mixin.py`
**目的**: 这是改动最大的文件。在 SGLang 的输出处理流程中注入隐藏状态收集逻辑，同时将 `.tolist()` 换为 tensor 直接返回。

#### 2.2.1 新增辅助函数（文件顶部）

```diff
--- a/sglang/srt/managers/scheduler_output_processor_mixin.py
+++ b/sglang/srt/managers/scheduler_output_processor_mixin.py
@@ -1,5 +1,7 @@
+import torch
 from typing import TYPE_CHECKING

 if TYPE_CHECKING:
     from sglang.srt.managers.schedule_batch import ScheduleBatch


+# =============================================================================
+# verl: EAGLE3 hidden state collection helpers
+# =============================================================================
+
+_DRAFTER_LAST_HIDDEN_STATES_ATTR = "_verl_drafter_last_hidden_states"
+
+
+def _is_tensor(x) -> bool:
+    is_tensor_fn = getattr(torch, "is_tensor", None)
+    return bool(callable(is_tensor_fn) and is_tensor_fn(x))
+
+
+def _append_prefill_hidden_states(
+    req, logits_output, hidden_state_offset: int, extend_input_len: int, batch=None,
+) -> int:
+    """Append hidden states for a single request during prefill.
+
+    Extracts hidden_states[hidden_state_offset : hidden_state_offset + rows]
+    and concatenates with last_hidden_states if available.
+
+    Returns the updated hidden_state_offset.
+    """
+    hidden_states = getattr(logits_output, "hidden_states", None)
+    if hidden_states is None:
+        return hidden_state_offset
+
+    try:
+        rows = max(int(extend_input_len), 0)
+    except (TypeError, ValueError):
+        rows = len(getattr(req, "origin_input_ids", []) or [])
+
+    prompt_len = len(getattr(req, "origin_input_ids", []) or [])
+    prefix_cache_rows = max(prompt_len - rows, 0)
+
+    if rows <= 0 or hidden_state_offset + rows > int(hidden_states.shape[0]):
+        return hidden_state_offset
+
+    end = hidden_state_offset + rows
+    chunk = hidden_states[hidden_state_offset:end]
+
+    # Windows mode with prefix cache awareness
+    custom_params = getattr(getattr(req, "sampling_params", None), "custom_params", None) or {}
+    window_enabled = custom_params.get("_verl_drafter_hidden_window") in ("1", "true", "True")
+    max_rows = custom_params.get("_verl_hidden_state_max_rows")
+    front_tokens = custom_params.get("_verl_hidden_state_front_tokens_per_sample")
+
+    if window_enabled:
+        front = int(front_tokens or max_rows or rows)
+        if prefix_cache_rows > 0:
+            chunk = chunk[prefix_cache_rows:prefix_cache_rows + front]
+        else:
+            chunk = chunk[:front]
+    elif max_rows:
+        chunk = chunk[:int(max_rows)]
+
+    # Attach last_hidden if available
+    last_hidden = getattr(logits_output, _DRAFTER_LAST_HIDDEN_STATES_ATTR, None)
+    if last_hidden is not None and _is_tensor(chunk) and _is_tensor(last_hidden):
+        last_hidden_chunk = last_hidden[hidden_state_offset:end]
+        if window_enabled:
+            if prefix_cache_rows > 0:
+                last_hidden_chunk = last_hidden_chunk[prefix_cache_rows:prefix_cache_rows + front]
+            else:
+                last_hidden_chunk = last_hidden_chunk[:front]
+        elif max_rows:
+            last_hidden_chunk = last_hidden_chunk[:int(max_rows)]
+        if last_hidden_chunk.shape[:-1] == chunk.shape[:-1]:
+            chunk = torch.cat((chunk, last_hidden_chunk), dim=-1)
+
+    req.hidden_states.append(chunk.detach().to("cpu", copy=True))
+    return hidden_state_offset + rows


+def _append_decode_hidden_states(
+    req, logits_output, result, req_index: int, hidden_state_offset: int, batch=None,
+) -> int:
+    """Append hidden states for a single request during decode.
+
+    Uses accept_lengths from EAGLE speculative decoding to determine
+    how many hidden state rows to collect.
+
+    Returns the updated hidden_state_offset.
+    """
+    hidden_states = getattr(logits_output, "hidden_states", None)
+    if hidden_states is None:
+        return hidden_state_offset
+
+    accept_lengths = getattr(result, "accept_length_per_req_cpu", None)
+    if accept_lengths is not None and req_index < len(accept_lengths) and _is_tensor(hidden_states):
+        rows = max(int(accept_lengths[req_index]) + 1, 1)
+        if hidden_states.dim() == 3 and req_index < int(hidden_states.shape[0]):
+            if int(hidden_states.shape[1]) >= rows:
+                chunk = hidden_states[req_index, :rows]
+            else:
+                chunk = hidden_states[req_index]
+        else:
+            chunk = hidden_states[hidden_state_offset:hidden_state_offset + rows]
+    else:
+        chunk = hidden_states[hidden_state_offset:hidden_state_offset + 1]
+        rows = 1

+    if _is_tensor(chunk):
+        # Attach last_hidden if available
+        last_hidden = getattr(logits_output, _DRAFTER_LAST_HIDDEN_STATES_ATTR, None)
+        if last_hidden is not None and _is_tensor(last_hidden):
+            if hidden_states.dim() == 3:
+                if req_index < int(last_hidden.shape[0]) and last_hidden.shape[1] >= rows:
+                    last_hidden_chunk = last_hidden[req_index, :rows]
+                else:
+                    last_hidden_chunk = last_hidden[req_index]
+            else:
+                last_hidden_chunk = last_hidden[hidden_state_offset:hidden_state_offset + rows]
+            if last_hidden_chunk.shape[:-1] == chunk.shape[:-1]:
+                chunk = torch.cat((chunk, last_hidden_chunk), dim=-1)
+
+        req.hidden_states.append(chunk.detach().to("cpu", copy=True))
+
+    return hidden_state_offset + rows
```

#### 2.2.2 `process_batch_result_prefill` 修改

```diff
@@ class SchedulerOutputProcessorMixin:
     def process_batch_result_prefill(self, batch, result):
         ...
-        for i, req in enumerate(batch.reqs):
+        hidden_state_offset = 0
+        for i, req in enumerate(batch.reqs):
             ...
-            # (原始代码)
-            if req.return_hidden_states and logits_output.hidden_states is not None:
-                req.hidden_states.append(
-                    logits_output.hidden_states[i].cpu().clone().tolist()
-                )
+            # --- verl: hidden state collection (replaces .cpu().clone().tolist()) ---
+            hidden_state_offset = _append_prefill_hidden_states(
+                req,
+                logits_output,
+                hidden_state_offset,
+                (
+                    extend_input_len_per_req[i]
+                    if extend_input_len_per_req is not None
+                    else len(req.origin_input_ids)
+                ),
+                batch,
+            )
```

#### 2.2.3 `process_batch_result_decode` 修改

```diff
@@ class SchedulerOutputProcessorMixin:
     def process_batch_result_decode(self, batch, result):
         ...
+        hidden_state_offset = 0
         for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
             ...
-            # (原始代码)
-            if req.return_hidden_states and logits_output.hidden_states is not None:
-                req.hidden_states.append(
-                    logits_output.hidden_states[i].cpu().clone().tolist()
-                )
+            # --- verl: hidden state collection for decode (handles EAGLE accept) ---
+            hidden_state_offset = _append_decode_hidden_states(
+                req,
+                logits_output,
+                result,
+                i,
+                hidden_state_offset,
+                batch,
+            )
```

---

### 2.3 `eagle_worker.py` — 权重更新路由 + Verify 完整隐藏状态

**文件**: `sglang/srt/speculative/eagle_worker.py`
**目的**:
1. 训练时区分 target-only / draft-only 权重更新
2. EAGLE verify 时强制 `CaptureHiddenMode.FULL`（多 token 完整隐藏状态）
3. CUDA/NPU Graph Replay 缺少隐藏状态时自动 fallback 重跑
4. 验证后按 `accepted_indices` 过滤 last_hidden_states

**架构说明**: 这部分的 patch 由以下层次组成：
- **文件级常量 & 工具函数** — 放在模块最外层
- **Patch 1 (helper)**: `_ensure_sglang_eagle_verify_full_hidden_mode` — 设置 `CaptureHiddenMode.FULL`
- **Patch 2 (helpers)**: `_sglang_hidden_state_rows` / `_sglang_eagle_verify_expected_hidden_rows` / `_sglang_eagle_verify_hidden_states_incomplete` / `_rerun_sglang_eagle_verify_without_graph` / `_validate_sglang_eagle_verify_hidden_states` — 完整性检查 + Graph fallback + 验证
- **Patch 3 (helper)**: `_filter_sglang_drafter_last_hidden_output` — 按 accepted_indices 裁剪 last_hidden

```diff
--- a/sglang/srt/speculative/eagle_worker.py
+++ b/sglang/srt/speculative/eagle_worker.py
@@ -1,5 +1,6 @@
 import logging
+import os
 from typing import List, Optional, TYPE_CHECKING

 import torch
 from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

+from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

 logger = logging.getLogger(__name__)
+
+# =============================================================================
+# verl: constants and helper functions for EAGLE hidden-state patches
+# =============================================================================
+
+_VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR = "_verl_drafter_last_hidden_states"
+
+
+def _sglang_batch_requests_hidden_states(batch) -> bool:
+    """Check if any request in the batch needs hidden states."""
+    return any(
+        bool(getattr(req, "return_hidden_states", False))
+        for req in getattr(batch, "reqs", []) or []
+    )
```

#### Patch 1 helper — 强制 CaptureHiddenMode.FULL

```diff


+# =============================================================================
+# verl Patch 1: force SGLang to return FULL per-token hidden states
+# =============================================================================
+
+def _ensure_sglang_eagle_verify_full_hidden_mode(batch, spec_info) -> None:
+    """Force CaptureHiddenMode.FULL when any request needs hidden states.

+    By default SGLang EAGLE uses CaptureHiddenMode.LAST which only returns
+    the last token's hidden state. EAGLE3 training requires all token hidden
+    states for every accepted position, so we must switch to FULL mode.
+    """
+    if not _sglang_batch_requests_hidden_states(batch):
+        return
+    try:
+        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
+    except Exception as exc:
+        logger.debug(
+            "Cannot import SGLang CaptureHiddenMode for EAGLE hidden-state patch: %s",
+            exc,
+        )
+        return
+    spec_info.capture_hidden_mode = CaptureHiddenMode.FULL
```

#### Patch 2 helpers — 完整性检查 + Graph fallback + 验证

```diff


+# =============================================================================
+# verl Patch 2: hidden-state completeness check, graph fallback & validation
+# =============================================================================
+
+def _sglang_hidden_state_rows(hidden_states) -> int:
+    """Compute number of rows in hidden states, handling all shape layouts.

+    SGLang may return hidden states as:
+      - 2D [total_tokens, hidden_dim] (prefill / batched decode)
+      - 3D [batch, max_tokens, hidden_dim] (verify with padding)
+    """
+    if not torch.is_tensor(hidden_states):
+        try:
+            return len(hidden_states)
+        except TypeError:
+            return 0
+    if hidden_states.dim() == 0:
+        return 1
+    if hidden_states.dim() >= 3:
+        return int(hidden_states.shape[0]) * int(hidden_states.shape[1])
+    return int(hidden_states.shape[0])
+
+
+def _sglang_eagle_verify_expected_hidden_rows(batch, spec_info) -> int:
+    """Calculate expected row count for verify-phase hidden states.

+    Expects: batch_size x draft_token_num rows.
+    draft_token_num = number of speculative tokens + 1 (bonus token).
+    """
+    batch_size = len(getattr(batch, "reqs", []) or [])
+    draft_token_num = int(getattr(spec_info, "draft_token_num", 0) or 0)
+    return batch_size * draft_token_num
+
+
+def _sglang_eagle_verify_hidden_states_incomplete(
+    batch, spec_info, logits_output,
+) -> bool:
+    """Check whether verify-phase hidden states are incomplete.

+    Returns True if hidden_states is missing or has fewer rows than expected.
+    This typically happens when CUDA/NPU Graph Replay captures only the last
+    token's hidden state (CaptureHiddenMode.LAST) instead of all tokens.
+    """
+    if not _sglang_batch_requests_hidden_states(batch):
+        return False
+    expected_rows = _sglang_eagle_verify_expected_hidden_rows(batch, spec_info)
+    if expected_rows <= 0:
+        return False
+    hidden_states = getattr(logits_output, "hidden_states", None)
+    return (
+        hidden_states is None
+        or _sglang_hidden_state_rows(hidden_states) < expected_rows
+    )
+
+
+def _rerun_sglang_eagle_verify_without_graph(worker, model_worker_batch):
+    """Re-run forward_batch_generation without CUDA/NPU Graph.

+    Temporarily sets model_runner.graph_runner = None to force eager-mode
+    execution, which bypasses the compiled graph and produces full hidden
+    states. Restores the graph_runner after the forward pass completes.
+    """
+    target_worker = getattr(worker, "target_worker", None)
+    model_runner = getattr(target_worker, "model_runner", None)
+    graph_runner = getattr(model_runner, "graph_runner", None)
+    try:
+        if model_runner is not None:
+            model_runner.graph_runner = None
+        return target_worker.forward_batch_generation(
+            model_worker_batch, is_verify=True,
+        )
+    finally:
+        if model_runner is not None:
+            model_runner.graph_runner = graph_runner
+
+
+def _validate_sglang_eagle_verify_hidden_states(
+    batch, spec_info, logits_output,
+) -> None:
+    """Raise RuntimeError if hidden states are still incomplete after fallback.

+    This is a hard guardrail: if even eager-mode execution cannot produce
+    full hidden states, training would use partial/incorrect hidden alignment
+    and produce garbage gradients.
+    """
+    if not _sglang_eagle_verify_hidden_states_incomplete(
+        batch, spec_info, logits_output,
+    ):
+        return
+    hidden_states = getattr(logits_output, "hidden_states", None)
+    shape = (
+        tuple(hidden_states.shape) if torch.is_tensor(hidden_states) else None
+    )
+    expected_rows = _sglang_eagle_verify_expected_hidden_rows(batch, spec_info)
+    actual_rows = _sglang_hidden_state_rows(hidden_states)
+    raise RuntimeError(
+        "SGLang EAGLE verify did not return full hidden states "
+        "for drafter training: "
+        f"actual_rows={actual_rows}, expected_rows={expected_rows}, "
+        f"shape={shape}. "
+        "This would train on partial/incorrect hidden alignment."
+    )
```

#### Patch 3 helper — 按 accepted_indices 过滤 last_hidden

```diff


+# =============================================================================
+# verl Patch 3: filter last_hidden_states by EAGLE accepted indices
+# =============================================================================
+
+def _filter_sglang_drafter_last_hidden_output(logits_output, index) -> None:
+    """Filter last_hidden_states by accepted indices after EAGLE verify.

+    During EAGLE verify, the model processes draft_token_num speculative
+    tokens but only accept_length+1 are actually accepted. The hidden states
+    are filtered by accepted_indices to keep only the accepted positions.
+    last_hidden_states must be filtered the same way to stay aligned.

+    This function is idempotent: if hidden_rows already equals index_len,
+    the filtering is skipped (may have been applied by a source-level patch).
+    """
+    last_hidden_states = getattr(
+        logits_output, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, None,
+    )
+    if last_hidden_states is None:
+        return

+    try:
+        index_len = (
+            int(index.numel()) if torch.is_tensor(index) else len(index)
+        )
+    except Exception:
+        index_len = None

+    if (
+        torch.is_tensor(last_hidden_states)
+        and index_len is not None
+        and last_hidden_states.dim() > 0
+    ):
+        hidden_rows = int(last_hidden_states.shape[0])
+        if hidden_rows == index_len:
+            return
+        if hidden_rows < index_len:
+            logger.warning(
+                "Skip filtering SGLang drafter last-hidden output: "
+                "hidden_rows=%s < index_len=%s",
+                hidden_rows,
+                index_len,
+            )
+            return

+    try:
+        filtered = last_hidden_states[index]
+    except Exception as exc:
+        logger.warning(
+            "Failed to filter SGLang drafter last-hidden output "
+            "by accepted indices: %s",
+            exc,
+        )
+        return
+    setattr(
+        logits_output,
+        _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR,
+        filtered,
+    )
```

#### EAGLEWorker 类 — 权重路由 + verify 方法 (3 个 Patch 调用点)

```diff


+# =============================================================================
+# verl: EAGLEWorker class modifications
+# =============================================================================

 class EAGLEWorker:
+    # --- verl: weight update routing flags ---
+    disable_draft_model: bool = False
+    disable_target_model: bool = False

     def update_weights_from_tensor(
         self,
         named_tensors: List[tuple[str, torch.Tensor]],
         load_format: Optional[str] = None,
     ):
-        # original: directly update all weights
-        for name, tensor in named_tensors:
-            self._load_weight(name, tensor)
+        # --- verl: target/draft routing ---
+        for name, tensor in named_tensors:
+            if self.disable_draft_model and self._is_eagle_draft_param(name):
+                continue
+            if self.disable_target_model and not self._is_eagle_draft_param(name):
+                continue
+            self._load_weight(name, tensor)
+
+    @staticmethod
+    def _is_eagle_draft_param(name: str) -> bool:
+        """Detect whether a parameter belongs to the EAGLE draft model."""
+        draft_prefixes = (
+            "drafter.", "draft_model.", "eagle_drafter.",
+            "fc.", "midlayer.", "lm_head.", "norm.",
+        )
+        return name.startswith(draft_prefixes)
```

```diff
@@ verify method @@

     def verify(self, batch, spec_info):
         spec_info.prepare_for_verify(batch, self.page_size)

+        # ================================================================
+        # verl Patch 1: force FULL hidden states capture
+        # ================================================================
+        _ensure_sglang_eagle_verify_full_hidden_mode(batch, spec_info)

         # (original verify setup: build model_worker_batch, etc.)
         ...

         # Forward
         batch_result = self.target_worker.forward_batch_generation(
             model_worker_batch, is_verify=True
         )
         logits_output, can_run_cuda_graph = (
             batch_result.logits_output,
             batch_result.can_run_cuda_graph,
         )

+        # ================================================================
+        # verl Patch 2: completeness check -> graph fallback -> validate
+        # ================================================================
+        if _sglang_eagle_verify_hidden_states_incomplete(
+            batch, spec_info, logits_output,
+        ):
+            logger.warning(
+                "SGLang EAGLE verify returned incomplete hidden states; "
+                "rerunning without graph for full hidden output."
+            )
+            batch_result = _rerun_sglang_eagle_verify_without_graph(
+                self, model_worker_batch,
+            )
+            logits_output, can_run_cuda_graph = (
+                batch_result.logits_output,
+                batch_result.can_run_cuda_graph,
+            )
+        _validate_sglang_eagle_verify_hidden_states(
+            batch, spec_info, logits_output,
+        )

         # (original verify logic: compute res, verify, etc.)
         ...

         # Filter hidden states by accepted indices
         logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]

+        # ================================================================
+        # verl Patch 3: filter last_hidden by same accepted indices
+        # ================================================================
+        _filter_sglang_drafter_last_hidden_output(
+            logits_output, res.accepted_indices,
+        )

         return logits_output, next_token_ids, res.can_run_cuda_graph
```

#### MultiLayerEagleWorker — 通过继承获得相同修改

```diff

+# =============================================================================
+# verl: MultiLayerEagleWorker — same modifications via inheritance
+# =============================================================================

 class MultiLayerEagleWorker(EAGLEWorker):
     """
     Same weight update routing and verify modifications as EAGLEWorker.
-    All three patches (CaptureHiddenMode.FULL, Graph fallback, last_hidden filter)
-    are applied identically via class inheritance.
+    All three patches (CaptureHiddenMode.FULL, Graph fallback,
+    last_hidden filter) are inherited from EAGLEWorker.verify().
+    No additional modification needed.
     """
     pass
```

---

### 2.4 `logprob.py` — Top-Logprobs Tensor 输出

**文件**: `sglang/srt/layers/utils/logprob.py`
**目的**: 将 SGLang 返回的 top-k logprobs 结果以 tensor 形式通过 verl 专属侧通道传递，避免 Python list 序列化损耗。

```diff
--- a/sglang/srt/layers/utils/logprob.py
+++ b/sglang/srt/layers/utils/logprob.py
@@ -1,5 +1,6 @@
+import os
 from enum import Enum
+import torch
 from typing import List, Optional


+_VERL_TOP_LOGPROBS_VALUES_DTYPE_ENV = "VERL_SGLANG_TOP_LOGPROBS_VALUES_DTYPE"

 def get_top_logprobs_raw(
     logprobs: torch.Tensor,
     top_logprobs_nums: List[int],
+    stage: 'LogprobStage' = LogprobStage.DECODE,
+    extend_logprob_pruned_lens_cpu: Optional[torch.Tensor] = None,
 ) -> List:
+    # --- verl: return top-logprobs as a dense tensor side-channel ---
+    top_logprobs_val = []
+    top_logprobs_idx = []
+
+    # (original computation: compute topk, gather values and indices)
     ...
+
+    # --- verl: pack into a single tensor for efficient transfer ---
+    max_topk = max(top_logprobs_nums) if top_logprobs_nums else 0
+    if max_topk > 0:
+        packed_val = torch.full(
+            (len(top_logprobs_nums), max_topk),
+            float("-inf"),
+            device=logprobs.device,
+        )
+        packed_idx = torch.full(
+            (len(top_logprobs_nums), max_topk),
+            -1,
+            dtype=torch.long,
+            device=logprobs.device,
+        )
+        for i, topk in enumerate(top_logprobs_nums):
+            if topk > 0:
+                packed_val[i, :topk] = top_logprobs_val[i][:topk]
+                packed_idx[i, :topk] = top_logprobs_idx[i][:topk]
+
+        # Convert dtype if configured
+        dtype_str = os.environ.get(_VERL_TOP_LOGPROBS_VALUES_DTYPE_ENV, "")
+        if dtype_str:
+            packed_val = packed_val.to(getattr(torch, dtype_str, torch.float32))
+
+        return packed_val, packed_idx
+
+    return [], []
```

---

### 2.5 `tokenizer_manager.py` — Meta Info 侧通道

**文件**: `sglang/srt/managers/tokenizer_manager.py`
**目的**: 在 `add_logprob_to_meta_info` 中将 top-logprobs tensor 通过 `meta_info` 传递给 verl，而不是通过 Python list 序列化。

```diff
--- a/sglang/srt/managers/tokenizer_manager.py
+++ b/sglang/srt/managers/tokenizer_manager.py
@@ -1,5 +1,6 @@
+import torch

+_VERL_OUTPUT_TOP_LOGPROBS_TENSOR_KEY = "_verl_output_top_logprobs_tensor"

 class TokenizerManager:
     def add_logprob_to_meta_info(self, state, meta_info, ...):
         ...
+        # --- verl: pack top-logprobs as a single tensor ---
+        topk = int(state.top_logprobs_num or 0)
+        if topk > 0 and hasattr(state, "output_top_logprobs_val"):
+            packed_tensor = self._pack_top_logprobs_tensor(
+                state.output_top_logprobs_val,
+                state.output_top_logprobs_idx,
+                topk,
+            )
+            if packed_tensor is not None:
+                meta_info[_VERL_OUTPUT_TOP_LOGPROBS_TENSOR_KEY] = packed_tensor
+                # Keep public fields empty—data is in the tensor side-channel
+                meta_info["input_top_logprobs"] = []
+                meta_info["output_top_logprobs"] = []
+
         # (original meta_info assembly continues)
+
+    def _pack_top_logprobs_tensor(self, val_list, idx_list, topk: int):
+        """Pack top-logprobs into a dense [seq, topk, 2] tensor."""
+        rows = []
+        for vals, idxs in zip(val_list, idx_list):
+            row = []
+            for v, i in zip(vals[:topk], idxs[:topk]):
+                row.append([float(v), int(i)])
+            while len(row) < topk:
+                row.append([float("-inf"), -1])
+            rows.append(row)
+        if not rows:
+            return None
+        return torch.tensor(rows, dtype=torch.float32)
```

---

### 2.6 `cuda_graph_runner.py` / `npu_graph_runner.py` — Graph Replay last_hidden

**文件**:
- `sglang/srt/model_executor/cuda_graph_runner.py`
- `sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py`

**目的**: CUDA/NPU Graph Replay 会绕过 `LogitsProcessor.forward`（因为整个计算图被编译后直接 replay），导致 `last_hidden_states` 未注入到 `logits_output`。需要从 replay 用的 output buffer 中单独拷贝 `last_hidden_states` 到结果对象上。

**说明**: Graph replay 时，`LogitsProcessor.forward` 的输出被缓存在 `self.output_buffers` 字典中。由于 section 2.1 的 patch 将 `last_hidden_states` 注入到了 `LogitsProcessor.forward` 的输出上，所以 replay 后的 output buffer 中已经包含了 `last_hidden_states`，只需要拷贝出来并截取前 `raw_num_token` 行即可。

```diff
--- a/sglang/srt/model_executor/cuda_graph_runner.py
+++ b/sglang/srt/model_executor/cuda_graph_runner.py
@@ -1,5 +1,6 @@
+import torch

+_VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR = "_verl_drafter_last_hidden_states"

 class CudaGraphRunner:
     def replay(self, *args, **kwargs):
-        # (original graph replay: run compiled graph, return result)
-        return result
+        # (original graph replay: run compiled graph)
+        result = ...  # original replay output
+
+        # --- verl: copy last_hidden from output buffer to result ---
+        output = None
+        keys = []
+        if getattr(self, "enable_pdmux", False):
+            get_current_stream_idx = (
+                self._original_replay.__globals__.get("get_current_stream_idx")
+            )
+            if callable(get_current_stream_idx):
+                keys.append(f"{get_current_stream_idx()}_{self.bs}")
+        keys.append(self.bs)
+        for key in keys:
+            try:
+                output = self.output_buffers[key]
+                break
+            except Exception:
+                continue
+
+        if output is not None:
+            last_hidden = getattr(output, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, None)
+            if last_hidden is not None:
+                setattr(
+                    result,
+                    _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR,
+                    last_hidden[:self.raw_num_token],
+                )
+
+        return result
```

```diff
--- a/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py
+++ b/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py
@@
--- (identical modification as CudaGraphRunner above) ---
++
++ _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR = "_verl_drafter_last_hidden_states"
++
++ class NPUGraphRunner:
++     def replay(self, *args, **kwargs):
++         result = ...  # original replay output
++
++         # --- verl: copy last_hidden from output buffer to result ---
++         output = None
++         keys = [self.bs]
++         for key in keys:
++             try:
++                 output = self.output_buffers[key]
++                 break
++             except Exception:
++                 continue
++
++         if output is not None:
++             last_hidden = getattr(output, _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR, None)
++             if last_hidden is not None:
++                 setattr(
++                     result,
++                     _VERL_DRAFTER_LAST_HIDDEN_STATES_ATTR,
++                     last_hidden[:self.raw_num_token],
++                 )
++
++         return result
```

---

## 3. Verl 源码修改

### 3.1 `eagle3_trainer_backend.py` (新增)

**文件**: `verl/verl/workers/drafter/eagle3_trainer_backend.py`
**性质**: **新增 970 行**

```diff
new file mode 100644
--- /dev/null
+++ b/verl/verl/workers/drafter/eagle3_trainer_backend.py
@@ -0,0 +1,970 @@
+"""EAGLE3 投机解码草稿模型训练后端
+
+从 EagleTrainerBackend 继承，专为 EAGLE3 (3层隐藏状态 + last_hidden) 架构设计。
+核心差异:
+  - 输入: 3+1 层 target 隐藏状态 (vs. EAGLE v1/v2 的单层)
+  - 损失: soft cross-entropy (vs. EAGLE v1/v2 的 SmoothL1Loss)
+  - 教师: TargetHead 或 topk logprobs (两种模式)
+  - 多步: Teacher Forcing TTT 循环 (gamma=0.8 衰减)
+"""
+
+import logging
+import os
+from copy import deepcopy
+
+import torch
+from torch.nn import functional as F
+from transformers import AutoConfig
+
+from .model.auto import AutoDraftModelConfig, AutoEagle3DraftModel
+from .eagle_trainer_backend import EagleTrainerBackend
+from .model.target.target_head import TargetHead
+from verl.utils.fsdp_utils import get_device_id
+from verl.utils.device import get_device_name
+
+
+logger = logging.getLogger(__name__)
+device_name = get_device_name()
+
+
+# =============================================================================
+# 核心工具函数
+# =============================================================================
+
+def _scatter_topk_logprobs_with_tail(
+    logprobs: torch.Tensor,    # [seq, topk]
+    indices: torch.Tensor,     # [seq, topk]
+    vocab_size: int,
+) -> torch.Tensor:             # [seq, vocab_size]
+    """从稀疏 top-k logprobs 重建完整 dense logprob 向量。
+
+    算法:
+    1. 创建 [seq, vocab_size] 的 -inf 张量
+    2. 计算 topk 概率总质量: topk_mass = Σ exp(logprobs_valid)
+    3. 剩余质量均匀分配: tail_logprob = log(1 - topk_mass) - log(remaining_count)
+    4. 将 topk logprobs 填入对应 index 位置
+
+    用途: verl rollout 返回稀疏 topk logprobs，训练时需要完整词表分布作为 soft target。
+    """
+    dense = torch.full(
+        (logprobs.size(0), vocab_size),
+        float("-inf"),
+        dtype=logprobs.dtype,
+        device=logprobs.device,
+    )
+    if logprobs.numel() == 0:
+        return dense
+
+    valid = torch.isfinite(logprobs) & (indices >= 0) & (indices < vocab_size)
+    if not valid.any():
+        return dense
+
+    valid_count = valid.sum(dim=-1)
+    has_valid = valid_count > 0
+    topk_mass = torch.where(
+        valid, logprobs.float().exp(),
+        torch.zeros_like(logprobs, dtype=torch.float32),
+    ).sum(dim=-1)
+    remaining_mass = (1.0 - topk_mass).clamp(min=torch.finfo(torch.float32).tiny)
+    remaining_count = (vocab_size - valid_count).clamp(min=1).to(torch.float32)
+    tail_logprob = (remaining_mass.log() - remaining_count.log()).to(logprobs.dtype)
+
+    dense[:] = torch.where(
+        has_valid.unsqueeze(-1),
+        tail_logprob.unsqueeze(-1).expand(-1, vocab_size),
+        dense,
+    )
+    row_idx = torch.arange(logprobs.size(0), device=logprobs.device).unsqueeze(1).expand_as(indices)
+    dense[row_idx[valid], indices[valid]] = logprobs[valid]
+    return dense


+def _masked_soft_cross_entropy(
+    logits: torch.Tensor,
+    target_p: torch.Tensor,
+    position_mask: torch.Tensor,
+) -> tuple[torch.Tensor, torch.Tensor]:
+    """Soft cross-entropy: -(target_p * log_softmax(logits)).sum(-1).
+
+    过滤 -inf logits 和全零 target rows。
+    返回 (per_token_ploss, valid_position)。
+    """
+    logits = logits.float()
+    target_p = target_p.float()
+    finite_logits = torch.isfinite(logits).all(dim=-1)
+    finite_target = torch.isfinite(target_p).all(dim=-1) & (target_p.sum(dim=-1) > 0)
+    valid_position = (position_mask > 0) & finite_logits & finite_target
+
+    safe_logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
+    safe_target = torch.where(torch.isfinite(target_p), target_p, torch.zeros_like(safe_target))
+    safe_target = torch.where(valid_position.unsqueeze(-1), safe_target, torch.zeros_like(safe_target))
+    log_probs = F.log_softmax(safe_logits, dim=-1)
+
+    per_token_ploss = -(safe_target * log_probs).sum(dim=-1)
+    per_token_ploss = torch.where(valid_position, per_token_ploss, torch.zeros_like(per_token_ploss))
+    return per_token_ploss, valid_position


+# (继续: _sparse_restricted_topk_cross_entropy, _log_topk_draft_vocab_coverage,
+#  _build_topk_draft_vocab_coverage_mask, reconstruct_dense_logprob_view 等工具函数)
+# (详见完整文件 e:\ascendproject\canndev4\eagle3\verl\verl\workers\drafter\eagle3_trainer_backend.py)


+# =============================================================================
+# Eagle3TrainerBackend 类
+# =============================================================================

+class Eagle3TrainerBackend(EagleTrainerBackend):
+    """EAGLE3 draft model training backend.
+
+    关键方法:
+    - build_model(): 构建 LlamaForCausalLMEagle3 模型
+    - preprocess_individual_items(): 处理 hidden states + target_logprobs
+    - compute_loss(): 多步 Teacher Forcing TTT 损失
+    """
+
+    def __init__(self, config, target_model_config):
+        super().__init__(config, target_model_config)
+        self.target_model = None  # 外部 Teacher 模型 (TargetHead)
+        self.vocab_size = None
+
+    def build_model(self):
+        """构建 EAGLE3 draft 模型。
+
+        架构:
+          embed_tokens (freeze) → fc(3*target_hidden → hidden)
+          → midlayer(1× DecoderLayer) → norm → lm_head → draft_vocab_size
+
+        教师信号有两种模式:
+          1. use_logits=False: TargetHead(last_hidden) → target_scores
+          2. use_logits=True:  rollout 返回的 topk logprobs → dense 重建
+        """
+        spec_model_path = self.config.rollout.drafter.model_path
+        config_path = os.path.join(spec_model_path, "config.json")
+        target_hf_config = self._get_target_hf_config()
+
+        if os.path.exists(config_path):
+            drafter_config = AutoDraftModelConfig.from_file(config_path)
+        else:
+            drafter_config = deepcopy(target_hf_config)
+            drafter_config.num_hidden_layers = 1
+            drafter_config.torch_dtype = torch.bfloat16
+            drafter_config.tie_word_embeddings = False
+            drafter_config.architectures = ["LlamaForCausalLMEagle3"]
+
+        if not hasattr(drafter_config, "draft_vocab_size"):
+            drafter_config.draft_vocab_size = drafter_config.vocab_size
+        if not hasattr(drafter_config, "target_hidden_size"):
+            drafter_config.target_hidden_size = target_hf_config.hidden_size
+
+        self.vocab_size = drafter_config.vocab_size
+        drafter_module = AutoEagle3DraftModel.from_config(drafter_config)
+
+        # 加载 checkpoint (如果有)
+        if spec_model_path and os.path.exists(spec_model_path):
+            drafter_module = AutoEagle3DraftModel.from_pretrained(spec_model_path)
+
+        # 复用 target 模型的 Embedding (冻结)
+        target_model_path = self.config.model.path
+        drafter_module.load_embedding(target_model_path)
+        drafter_module.freeze_embedding()
+
+        # 验证 t2d/d2t 词表映射
+        self._validate_vocab_mapping(drafter_module)
+
+        # 构建外部教师模型 (可选)
+        use_logits = self.config.rollout.drafter.training.get("use_logits", False)
+        if not use_logits:
+            target_device = torch.device(f"{device_name}:{get_device_id()}")
+            self.target_model = (
+                self._build_target_model(target_model_path).to(target_device).eval()
+            )
+            for param in self.target_model.parameters():
+                param.requires_grad_(False)
+
+        return drafter_module, drafter_config
+
+    def preprocess_individual_items(self, items, device, model_config):
+        """数据预处理。
+
+        hidden_states 格式:
+          [3层 target hidden | 1层 last_hidden] = 4 * hidden_size (use_logits=False)
+          [3层 target hidden] = 3 * hidden_size (use_logits=True)
+
+        拆分:
+          h_states = full_h[:, :3*hidden_size]      → draft 模型输入
+          last_h = full_h[:, 3*hidden_size:]          → TargetHead 输入
+
+        target_logprobs: 截取 [start : end-1] (末位无 next-token target)
+        """
+        use_logits = bool(self.config.rollout.drafter.training.get("use_logits", False))
+        h_dim = getattr(model_config, "target_hidden_size", model_config.hidden_size)

+        res = {'ids': [], 'h_states': [], 'masks': [], 'position_ids': [],
+               'last_h_states': [], 'target_logprobs': []}

+        for item in items:
+            ids = item["input_ids"].to(device, non_blocking=True)
+            raw_h = item["hidden_states"]
+            full_h = (
+                torch.cat(raw_h, dim=-1).to(device, dtype=torch.bfloat16)
+                if isinstance(raw_h, (list, tuple))
+                else raw_h.to(device, dtype=torch.bfloat16)
+            )
+
+            h_states = full_h[:, :3 * h_dim]
+            if not use_logits:
+                last_h = full_h[:, 3 * h_dim:4 * h_dim]

+            # 生成 loss_mask (response 部分 = 1)
+            full_len = ids.size(0)
+            if "loss_mask" not in item:
+                loss_mask = self._make_loss_mask(item, full_len, device)
+            else:
+                loss_mask = item["loss_mask"].to(device, dtype=torch.float32)

+            res['ids'].append(ids)
+            res['h_states'].append(h_states)
+            res['masks'].append(loss_mask)
+            if not use_logits:
+                res['last_h_states'].append(last_h)

+            # target_logprobs: 比 hidden_states 短1 (末位无 target)
+            if item.get("target_logprobs") is not None:
+                tg = item["target_logprobs"].to(device, dtype=torch.float32)
+                res['target_logprobs'].append(tg[:full_len - 1])
+            else:
+                res['target_logprobs'].append(None)

+        return res

+    def compute_loss(self, model, batch, _current_pad_size):
+        """EAGLE3 多步 Teacher Forcing TTT 损失。
+
+        流程:
+          1. 前向: model(hidden_states) → {logits[0..N], position_masks[0..N]}
+          2. 教师信号:
+             use_logits=True:  reconstruct_dense_logprob_view()
+             use_logits=False: TargetHead(last_hidden) → _compute_target_p()
+          3. 循环 TTT 步:
+             for idx in range(ttt_length):
+                 logits = outputs[idx]
+                 target = teacher_signal[idx:]  (右移对齐)
+                 loss += 0.8^idx × soft_cross_entropy(logits, target)
+          4. 统计: top1/top5 准确率
+
+        Gamma 衰减: loss_total = Σ 0.8^i × loss_i
+        Ulysses SP: 通过 gather_outputs_and_unpad 支持
+        """
+        ttt_length = int(self.config.rollout.drafter.training.get("ttt_length", 1))
+        use_logits = self.config.rollout.drafter.training.use_logits
+        gamma = 0.8

+        # 前向传播 → TTT 多步 logits
+        outputs = model(
+            input_ids=batch["input_ids"],
+            hidden_states=batch["hidden_states"],
+            attention_mask=batch["attention_mask"],
+            loss_mask=batch["loss_mask"],
+            position_ids=batch["position_ids"],
+            ttt_length=ttt_length,
+        )

+        # 教师信号 → target_p
+        if use_logits:
+            target_scores = reconstruct_dense_logprob_view(
+                batch["target_logprobs"],
+                topk=self.config.rollout.drafter.training.logits_topk,
+                vocab_size=self.vocab_size,
+            )
+        else:
+            with torch.no_grad():
+                target_scores = self.target_model(batch["last_hidden_states"])

+        # 扩展到 ttt_length 步对齐
+        target_p, position_mask = self._compute_target_p(
+            target_scores, model.t2d, batch["loss_mask"]
+        )

+        total_loss = torch.tensor(0.0, device=batch["input_ids"].device)
+        for idx in range(len(outputs["logits"])):
+            p_loss, _ = _masked_soft_cross_entropy(
+                outputs["logits"][idx],
+                target_p[:, idx:],
+                outputs["position_masks"][idx] * position_mask[:, idx:],
+            )
+            total_loss += (gamma ** idx) * p_loss.sum()

+        return {
+            "total_local_vloss": torch.tensor(0.0),
+            "total_local_ploss": total_loss,
+            "local_num_tokens": position_mask.sum(),
+        }

+    def _compute_target_p(self, target_scores, t2d, loss_mask):
+        """Filter target scores through t2d vocab mask → softmax → target_p."""
+        t2d = t2d.to(device=target_scores.device, dtype=torch.bool)
+        subset = target_scores[..., t2d].float()
+        finite_mask = torch.isfinite(subset).any(dim=-1)
+        position_mask = finite_mask.float() * loss_mask.float()
+
+        finite_scores = torch.isfinite(subset)
+        floor = torch.finfo(subset.dtype).min
+        subset = torch.where(finite_scores, subset, torch.full_like(subset, floor))
+        subset = torch.where(finite_mask.unsqueeze(-1), subset, torch.zeros_like(subset))
+        target_p = F.softmax(subset, dim=-1)
+        target_p = torch.where(
+            finite_scores & finite_mask.unsqueeze(-1),
+            target_p,
+            torch.zeros_like(target_p),
+        )
+        return target_p.detach(), position_mask

+    def _build_target_model(self, target_model_path: str):
+        """Build TargetHead from pretrained lm_head weights."""
+        return TargetHead.from_pretrained(model_path=target_model_path)
```

### 3.2 `llama_eagle.py` (新增)

**文件**: `verl/verl/workers/drafter/model/eagle/llama_eagle.py`
**性质**: **新增 1669 行**

```diff
new file mode 100644
--- /dev/null
+++ b/verl/verl/workers/drafter/model/eagle/llama_eagle.py
@@ -0,0 +1,1669 @@
+"""EAGLE3 Draft Model — Llama architecture.
+
+核心类: LlamaForCausalLMEagle3
+
+架构:
+  ┌───────────────────────────────────────────┐
+  │ embed_tokens: vocab_size → hidden_size     │
+  │ midlayer: 1× LlamaDecoderLayer             │
+  │   - Q/K/V projection: hidden_size×2→...    │
+  │     (因为输入 = cat(embeds, hidden_states)) │
+  │ fc: target_hidden_size×3 → hidden_size     │
+  │ norm: RMSNorm                              │
+  │ lm_head: hidden → draft_vocab_size         │
+  │                                           │
+  │ t2d: [vocab_size] bool (target→draft mask) │
+  │ d2t: [draft_vocab_size] int64 (draft→target)│
+  └───────────────────────────────────────────┘
+
+forward: Teacher Forcing 多步预测
+  for idx in range(ttt_length):
+    inputs_embeds = embed(current_input_ids)
+    hidden = backbone(inputs_embeds, current_hidden_states)
+    logits = lm_head(norm(hidden))
+    save(logits)
+    if not is_last:
+      current_input_ids = _shift_right(current_input_ids)  # Teacher Forcing
+      current_loss_mask = _shift_right(current_loss_mask)
+    position_ids += 1
+"""
+
+import logging
+import math
+from typing import Optional
+
+import torch
+from torch import nn
+from torch.nn import functional as F
+from transformers.activations import ACT2FN
+from transformers.cache_utils import Cache
+
+from .base import Eagle3DraftModel
+
+
+logger = logging.getLogger(__name__)


+class LlamaAttention(nn.Module):
+    """EAGLE3 Attention with multi-branch KV cache merging.
+
+    Supports tree-structured KV cache (from EAGLE speculative decoding).
+    Q/K/V projection input dim = 2 × hidden_size because
+    input = cat(input_embeds, target_hidden_states).
+    """
+    # (full implementation: 240+ lines, see source file)


+class LlamaDecoderLayer(nn.Module):
+    """EAGLE3 single decoder layer.
+
+    Forward receives two tensors:
+      input_embeds: embedded current input tokens
+      hidden_states: projected target model hidden states
+
+    Concatenates them along last dim → passes through attention + MLP.
+    """
+    def forward(self, input_embeds, hidden_states, cache_hidden=None,
+                attention_mask=None, position_ids=None, past_key_values=None,
+                use_cache=False):
+        concat_input = torch.cat((input_embeds, hidden_states), dim=-1)
+        # (attention + MLP forward)


+class LlamaForCausalLMEagle3(Eagle3DraftModel):
+    """EAGLE3 Draft Model for Llama architecture.
+
+    Key methods:
+      forward(): Teacher-Forcing multi-step TTT
+      project_hidden_states(): fc(hidden_states)
+      embed_input_ids(): embed_tokens(input_ids)
+      compute_logits(): lm_head(norm(hidden))
+      backbone(): midlayer(inputs_embeds, hidden_states, ...)
+    """

+    def __init__(self, config):
+        super().__init__(config)
+        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
+        self.midlayer = LlamaDecoderLayer(config, layer_idx=0)
+
+        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
+        self.fc = nn.Linear(target_hidden_size * 3, config.hidden_size, bias=False)
+        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
+        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)
+
+        # Vocab mapping buffers
+        self.register_buffer("t2d", ...)  # [vocab_size] bool
+        self.register_buffer("d2t", ...)  # [draft_vocab_size] int64

+    def forward(
+        self,
+        input_ids: torch.Tensor,
+        hidden_states: torch.Tensor,
+        attention_mask: Optional[torch.Tensor] = None,
+        loss_mask: Optional[torch.Tensor] = None,
+        position_ids: Optional[torch.LongTensor] = None,
+        ttt_length: int = 1,
+    ) -> dict:
+        """
+        Returns: {
+            "logits": list of [B, seq, draft_vocab] tensors,
+            "position_masks": list of [B, seq] masks,
+        }
+        """
+        current_hidden_states = self.project_hidden_states(hidden_states)
+        current_input_ids = input_ids
+        current_position_ids = position_ids
+        current_loss_mask = loss_mask

+        all_logits = []
+        all_position_masks = []

+        for idx in range(ttt_length):
+            inputs_embeds = self.embed_input_ids(current_input_ids)
+            hidden = self.backbone(
+                input_embeds=inputs_embeds,
+                hidden_states=current_hidden_states,
+                attention_mask=attention_mask,
+                position_ids=current_position_ids,
+            )
+            logits = self.compute_logits(hidden)
+            all_logits.append(logits)
+            all_position_masks.append(current_loss_mask)

+            if idx < ttt_length - 1:
+                # Teacher Forcing: shift inputs right by 1
+                current_input_ids = self._shift_right(current_input_ids)
+                current_loss_mask = self._shift_right(current_loss_mask)

+            current_position_ids = current_position_ids + 1

+        return {
+            "logits": all_logits,
+            "position_masks": all_position_masks,
+        }

+    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
+        return self.fc(hidden_states)

+    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
+        return self.embed_tokens(input_ids)

+    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
+        return self.lm_head(self.norm(hidden_states))

+    def backbone(self, input_embeds, hidden_states, **kwargs):
+        return self.midlayer(input_embeds, hidden_states, **kwargs)

+    @staticmethod
+    def _shift_right(x: torch.Tensor) -> torch.Tensor:
+        """Shift tensor right by 1, padding with zeros on the left."""
+        padded = F.pad(x, (1, 0), value=0)
+        return padded[..., :-1]
```

### 3.3 `base.py` (新增)

**文件**: `verl/verl/workers/drafter/model/eagle/base.py`
**性质**: **新增 167 行**

```diff
new file mode 100644
--- /dev/null
+++ b/verl/verl/workers/drafter/model/eagle/base.py
@@ -0,0 +1,167 @@
+"""Draft model base classes for EAGLE / EAGLE3."""
+
+from abc import abstractmethod
+import torch
+from torch import nn
+from transformers import PreTrainedModel


+class DraftModel(PreTrainedModel):
+    """Base class for all draft models (EAGLE v1/v2, EAGLE3)."""
+
+    @abstractmethod
+    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
+        """Embed input token IDs."""
+
+    @abstractmethod
+    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
+        """Compute logits from hidden states."""
+
+    @abstractmethod
+    def backbone(self, input_embeds, hidden_states, **kwargs):
+        """Main model backbone."""
+
+    @abstractmethod
+    def freeze_embedding(self):
+        """Freeze the embedding layer."""
+
+    @abstractmethod
+    def load_embedding(self, target_model_path: str):
+        """Load embedding weights from target model."""


+class Eagle3DraftModel(DraftModel):
+    """EAGLE3-specific base class with vocab mapping support."""

+    @property
+    def drafter_model_type(self) -> str:
+        return "LlamaForCausalLMEagle3"

+    @abstractmethod
+    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
+        """Project 3-layer concatenated target hidden states to draft space."""

+    def load_vocab_mapping(self, t2d: torch.Tensor, d2t: torch.Tensor):
+        """Load vocabulary mapping buffers."""
+        self.register_buffer("t2d", t2d)
+        self.register_buffer("d2t", d2t)


+class EagleDraftModel(DraftModel):
+    """EAGLE v1/v2 base class with lm_head loading support."""

+    @property
+    def drafter_model_type(self) -> str:
+        return "LlamaForCausalLMEagle"

+    def load_lm_head(self, target_model_path: str):
+        """Load lm_head weights from target model."""
+
+    def freeze_lm_head(self):
+        """Freeze lm_head parameters."""
```

### 3.4 `target_head.py` (新增)

**文件**: `verl/verl/workers/drafter/model/target/target_head.py`
**性质**: **新增 99 行**

```diff
new file mode 100644
--- /dev/null
+++ b/verl/verl/workers/drafter/model/target/target_head.py
@@ -0,0 +1,99 @@
+"""TargetHead: external teacher model for EAGLE3 training.
+
+Extracts the single linear layer (lm_head) from a pretrained target model
+to convert last_hidden_states back to logits for computing teacher signal.
+"""
+
+import json
+import os
+
+import torch
+from torch import nn
+from transformers import AutoConfig
+
+from verl.utils.fsdp_utils import get_device_id
+from verl.utils.device import get_device_name


+class TargetHead(nn.Module):
+    """Single frozen linear layer extracted from target model lm_head.
+
+    Architecture: fc(hidden_states) → vocab_size logits
+    """
+
+    def __init__(self, hidden_size: int, vocab_size: int):
+        super().__init__()
+        self.fc = nn.Linear(hidden_size, vocab_size, bias=False)

+    @classmethod
+    def from_pretrained(cls, model_path: str) -> "TargetHead":
+        """Create TargetHead from a pretrained model path.
+
+        Loads the lm_head.weight from the model's weight files
+        and freezes it for teacher signal computation.
+        """
+        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
+        hidden_size = config.hidden_size
+        vocab_size = config.vocab_size

+        head = cls(hidden_size, vocab_size)
+        device = f"{torch.device('cpu')}"
+        head.load_weights(model_path, device)
+        head.freeze_weights()
+        return head.to(dtype=torch.bfloat16).eval()

+    def load_weights(self, model_path: str, device):
+        """Load lm_head.weight from safetensors or pytorch bin."""
+        index_path = os.path.join(model_path, "model.safetensors.index.json")
+        if os.path.exists(index_path):
+            with open(index_path) as f:
+                index = json.load(f)
+            weight_file = index["weight_map"].get("lm_head.weight")
+            if weight_file:
+                from safetensors.torch import load_file
+                state = load_file(os.path.join(model_path, weight_file))
+                self.fc.weight.data.copy_(state["lm_head.weight"].to(device))
+                return
+
+        # Fallback: pytorch_model.bin
+        bin_path = os.path.join(model_path, "pytorch_model.bin")
+        if os.path.exists(bin_path):
+            state = torch.load(bin_path, map_location=device, weights_only=True)
+            if "lm_head.weight" in state:
+                self.fc.weight.data.copy_(state["lm_head.weight"].to(device))

+    def freeze_weights(self):
+        """Freeze all parameters."""
+        for p in self.parameters():
+            p.requires_grad_(False)

+    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
+        return self.fc(hidden_states)
```

### 3.5 `sglang_rollout.py` (修改)

**文件**: `verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py`
**性质**: **修改已有文件**，增加 weight loader 路由 + `update_draft_weights`

```diff
--- a/verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py
+++ b/verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py
@@ -1,4 +1,5 @@
 import logging
+import sglang.srt.entrypoints.engine
 from typing import List, Optional


+# =============================================================================
+# verl: EAGLE weight loader constants
+# =============================================================================
+
+VERL_SGLANG_TARGET_WEIGHT_LOADER = "verl_sglang_target_weight_loader"
+VERL_SGLANG_DRAFT_WEIGHT_LOADER = "verl_sglang_draft_weight_loader"
+
+
+def _is_sglang_eagle_draft_model(weight, weight_name, model_config) -> bool:
+    """Detect EAGLE draft model parameters.
+
+    Heuristics:
+      - model_config.model_type contains "eagle"
+      - model_config.architectures includes "LlamaForCausalLMEagle"
+      - model_config has draft_vocab_size attribute
+    """
+    if "eagle" not in str(getattr(model_config, "model_type", "")).lower():
+        return False
+    architectures = getattr(model_config, "architectures", None) or []
+    if any("LlamaForCausalLMEagle" in arch for arch in architectures):
+        return True
+    if hasattr(model_config, "draft_vocab_size"):
+        return True
+    return False


+def verl_sglang_target_weight_loader(
+    model_config, model_path, quant_config, load_format, ...
+):
+    """Load only target model weights. Skip EAGLE draft model."""
+    ...
+    if _is_sglang_eagle_draft_model(weight, weight_name, model_config):
+        return  # Skip draft model weights
+    default_loader(model_config, model_path, ...)


+def verl_sglang_draft_weight_loader(
+    model_config, model_path, quant_config, load_format, ...
+):
+    """Load only EAGLE draft model weights. Skip target model."""
+    ...
+    if not _is_sglang_eagle_draft_model(weight, weight_name, model_config):
+        return  # Skip target model weights
+    default_loader(model_config, model_path, ...)


+# =============================================================================
+# verl: _sgl_update_weights_with_route
+# =============================================================================
+
+def _sgl_update_weights_with_route(
+    named_tensors, model_config, model_path, *,
+    disable_draft_model=False, disable_target_model=False,
+):
+    """Weight update with target/draft routing.
+
+    disable_draft_model=True  → Update target model only
+    disable_target_model=True → Update draft model only
+    """
+    ...


@@ class ServerAdapter:
+    # --- verl: EAGLE training mode weight update ---
     def update_weights(self, named_tensors):
-        # (original: update all model weights)
+        if self.config.drafter.enable:
+            load_format = VERL_SGLANG_TARGET_WEIGHT_LOADER
+            disable_draft_model = True
+            # Pause generation & flush cache during weight update
+            self.pause_generation()
+            self.flush_cache()
+        else:
+            load_format = self.load_format
+        ...

+    # --- verl: EAGLE draft-specific weight update ---
+    def update_draft_weights(self, named_tensors):
+        """Update only EAGLE draft model weights.
+
+        Mutually exclusive with update_weights():
+          disable_draft_model=False, disable_target_model=True
+
+        Supports:
+          - Bucket-based batch updates (draft_update_weights_bucket_megabytes)
+          - pause_generation(mode="retract")
+          - Weight statistics logging
+        """
+        ...
```

---

## 4. 删除的文件

```diff
--- a/verl/verl/workers/drafter/sglang_patch.py
+++ /dev/null
@@ -1,3179 +0,0 @@
-"""EAGLE3 SGLang runtime patches.
-
-This file's functionality has been incorporated into direct source
-modifications of SGLang. See the git patches in section 2 of this document.
-"""
```

---

## 5. 附录：环境变量与配置

| 环境变量 | 用途 | 默认值 |
|---------|------|-------|
| `VERL_SGLANG_DRAFTER_RETURN_LAST_HIDDEN` | 全局启用 last_hidden 返回 | `0` |
| `VERL_SGLANG_TOP_LOGPROBS_VALUES_DTYPE` | top-logprobs tensor 精度 | `float32` |

**训练配置** (`config.yaml`):

```yaml
rollout:
  drafter:
    model_path: /path/to/eagle3/model
    training:
      use_logits: true          # true=topk logprobs, false=TargetHead
      logits_loss_mode: dense_tail  # dense_tail | sparse_restricted
      logits_topk: 16
      ttt_length: 1             # Teacher Forcing 多步预测步数
```