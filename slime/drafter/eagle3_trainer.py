"""
Eagle3 Drafter Background Trainer for Slime

This module implements online training for Eagle3 draft model to optimize
speculative decoding performance during RL training.
"""

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("SLIME_LOGGING_LEVEL", "INFO"))


@dataclass
class Eagle3TrainConfig:
    """Configuration for Eagle3 drafter training."""
    enable: bool = False
    spec_model_path: str = ""
    training_interval_steps: int = 10
    batch_size_per_gpu: int = 2
    max_seq_len: int = 8192
    max_epochs: int = 10
    checkpoint_path: Optional[str] = None
    min_workers_for_training: int = 1
    collect_hidden_states_from_sgl: bool = False
    buffer_max_samples: int = 2000
    data_buffer_max_size: int = 10000
    vloss_weight: float = 1.0
    ploss_weight: float = 0.1
    enable_step_barrier: bool = False
    ulysses_sequence_parallel_size: int = 1
    
    # Optimizer config
    lr: float = 1e-6
    lr_warmup_steps: int = 1000
    weight_decay: float = 1e-2
    warmup_style: str = "constant"
    
    # Offload config
    is_offload_param: bool = False
    is_offload_optimizer: bool = False


class DataBuffer:
    """Buffer for storing training data across RL steps."""
    
    def __init__(self, max_size: int = 10000, store_hidden_states: bool = False):
        self.max_size = max_size
        self.store_hidden_states = store_hidden_states
        self.data = []
        self.step_boundaries = [0]  # Track step boundaries
        self.current_step = 0
    
    def add_batch(self, batch: dict, hidden_states: list = None):
        """Add a batch of data to the buffer."""
        # logger.exception(f"add_batch 67 {batch}")
        # logger.exception(f"add_batch 68  {hidden_states}")
        logger.exception(f"add_batch 67")
        batch_size = batch.get("tokens", []).__len__() if isinstance(batch.get("tokens"), list) else batch.get("tokens").size(0)
        logger.exception(f"add_batch 71")
        for i in range(batch_size):
            item = {
                "input_ids": batch["tokens"][i] if isinstance(batch["tokens"], list) else batch["tokens"][i].cpu(),
                "step": self.current_step,
            }
            
            if "prompts" in batch and "responses" in batch:
                item["prompts"] = batch["prompts"][i] if isinstance(batch["prompts"], list) else batch["prompts"][i].cpu()
                item["responses"] = batch["responses"][i] if isinstance(batch["responses"], list) else batch["responses"][i].cpu()
            
            if self.store_hidden_states and hidden_states and i < len(hidden_states):
                item["hidden_states"] = hidden_states[i].cpu() if hasattr(hidden_states[i], 'cpu') else hidden_states[i]
            
            self.data.append(item)
        
        # Trim if exceeds max size
        if len(self.data) > self.max_size:
            excess = len(self.data) - self.max_size
            self.data = self.data[excess:]
            # Update step boundaries
            self.step_boundaries = [0]
            step_count = 0
            for item in self.data:
                if item["step"] != step_count:
                    step_count = item["step"]
                if len(self.step_boundaries) <= step_count:
                    self.step_boundaries.append(len(self.data))
    
    def increment_step(self):
        """Mark the end of current RL step."""
        self.current_step += 1
        self.step_boundaries.append(len(self.data))
    
    def get_data_from_last_n_steps(self, n: int) -> list:
        """Get data from the last n RL steps."""
        logger.exception(f"get_data_from_last_n_steps  104 {len(self.step_boundaries)}")
        if len(self.step_boundaries) < 2:
            return self.data
        logger.exception(f"get_data_from_last_n_steps  107")
        
        start_idx = max(0, self.step_boundaries[-(n + 1)])
        logger.exception(f"get_data_from_last_n_steps  110")
        return self.data[start_idx:]
    
    def get_current_step(self) -> int:
        return self.current_step
    
    def __len__(self) -> int:
        return len(self.data)
    
    def clear(self):
        self.data = []
        self.step_boundaries = [0]
        self.current_step = 0


class Eagle3BackgroundTrainer:
    """Background trainer for Eagle3 drafter model with online training capability."""
    
    def __init__(
        self,
        drafter_model,
        drafter_optimizer,
        drafter_lr_scheduler,
        train_config: Eagle3TrainConfig,
        device_mesh: DeviceMesh,
        model_config=None,
    ):
        self.model = drafter_model
        self.optimizer = drafter_optimizer
        self.lr_scheduler = drafter_lr_scheduler
        self.config = train_config
        self.device_mesh = device_mesh
        self.model_config = model_config
        
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        self._training_initialized = False
        self._training_active = False
        self.training_steps = 0
        
        # Data buffers
        self.collected_data = deque(maxlen=int(train_config.buffer_max_samples))
        self.data_buffer = DataBuffer(
            max_size=int(train_config.data_buffer_max_size),
            store_hidden_states=train_config.collect_hidden_states_from_sgl
        )
        self.batch_size = int(train_config.batch_size_per_gpu)

        # Checkpoint management
        self.checkpoint_dir = train_config.checkpoint_path
        self._last_ckpt_step = -1
        self._pending_checkpoint_future = None
        self._frozen_param_names = {"model.embed_tokens.weight", "lm_head.weight"}
        
        # Ulysses SP support
        self.use_ulysses_sp = train_config.ulysses_sequence_parallel_size > 1
        
        if self.rank == 0:
            logger.info(f"Eagle3BackgroundTrainer initialized with enable={train_config.enable}")
    
    def _get_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Get state dict excluding frozen layers."""
        full_state_dict = self.model.state_dict()
        trainable_state_dict = {}
        
        for name, param in full_state_dict.items():
            # Skip frozen parameters (embed_tokens, lm_head)
            if any(frozen_name in name for frozen_name in self._frozen_param_names):
                continue
            trainable_state_dict[name] = param
        
        return trainable_state_dict
    
    def _save_checkpoint_async(self, step: int, is_final: bool = False):
        """Asynchronously save checkpoint."""
        if not self.checkpoint_dir:
            return None
        
        try:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"eagle3_step_{step}")
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Get trainable state dict
            model_state_dict = self._get_trainable_state_dict()
            optimizer_state_dict = self.optimizer.state_dict() if self.optimizer else {}
            
            state_dict = {
                "model": model_state_dict,
                "optimizer": optimizer_state_dict,
                "step": step,
            }
            
            # Save checkpoint
            torch.save(state_dict, os.path.join(checkpoint_path, "trainer_state.pt"))
            
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Checkpoint save failed on rank {self.rank}: {e}")
            return None
    
    async def activate_training_model(self, device_mesh: DeviceMesh, training_ranks: list[int]) -> bool:
        """Activate training mode for the drafter model."""
        start_ts = time.time()
        try:
            logger.info(f"[EagleTrainer rank {self.rank}] Activating training model, training_ranks={training_ranks}")
            
            # Load model to GPU if offloaded
            first_param = next(self.model.parameters(), None)
            param_device = first_param.device.type if first_param is not None else None
            
            if self.config.is_offload_param or param_device != "cuda":
                self.model.to("cuda")
                logger.debug("Loaded drafter model to GPU for training")
            
            if self.optimizer is not None:
                for param_group in self.optimizer.param_groups:
                    for param in param_group['params']:
                        if param.device.type != "cuda":
                            param.data = param.data.to("cuda")
                logger.debug("Loaded drafter optimizer to GPU for training")
            
            self.training_device_mesh = device_mesh
            self._training_initialized = True
            self._training_active = True
            
            logger.info(f"[EagleTrainer rank {self.rank}] Training activated in {time.time() - start_ts:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"[EagleTrainer rank {self.rank}] Activation failed: {e}")
            return False
    
    def collect_online_data(self, batch: dict, hidden_states: list = None, target_logprobs: list = None):
        """Collect online data from rollout for Eagle3 training."""
        logger.exception(
            f"collect_online_data tokens 254",
        )
        input_ids = batch.get("tokens")
        logger.exception(
            f"collect_online_data tokens 258",
        )
        if input_ids is None:
            return
        logger.exception(
            f"collect_online_data tokens 263",
        )
        self.data_buffer.add_batch(batch, hidden_states)

        batch_size = len(input_ids) if isinstance(input_ids, list) else input_ids.size(0)
        logger.exception(
            f"collect_online_data tokens 271 {batch_size}",
        )
        for i in range(batch_size):
            seq = input_ids[i] if isinstance(input_ids, list) else input_ids[i]

            loss_mask = torch.zeros_like(seq, dtype=torch.float32)
            if "prompts" in batch and "responses" in batch:
                prompts = batch["prompts"][i] if isinstance(batch["prompts"], list) else batch["prompts"][i]
                responses = batch["responses"][i] if isinstance(batch["responses"], list) else batch["responses"][i]
                prompt_len = len(prompts) if isinstance(prompts, torch.Tensor) else prompts.size(0)
                response_len = len(responses) if isinstance(responses, torch.Tensor) else responses.size(0)
                pad_token_id = getattr(self.model_config, "pad_token_id", 0)
                for j in range(response_len):
                    if responses[j] != pad_token_id:
                        loss_mask[prompt_len + j] = 1.0

            h_state = None
            if hidden_states and i < len(hidden_states):
                h_state = hidden_states[i]
                if hasattr(h_state, 'cpu'):
                    h_state = h_state.cpu()

            t_logprobs = None
            if target_logprobs and i < len(target_logprobs):
                t_logprobs = target_logprobs[i]
                if hasattr(t_logprobs, 'cpu'):
                    t_logprobs = t_logprobs.cpu()

            if h_state is not None:
                item = {
                    "input_ids": seq.cpu() if hasattr(seq, 'cpu') else seq,
                    "loss_mask": loss_mask.cpu() if hasattr(loss_mask, 'cpu') else loss_mask,
                    "hidden_states": h_state,
                }
                if t_logprobs is not None:
                    item["target_logprobs"] = t_logprobs
                self.collected_data.append(item)
            
    def _prepare_training_batch(self, use_buffer_data: bool = True, buffer_steps: int = 2):
        """Prepare a batch for training."""
        effective_batch_size = min(self.batch_size, 4)
        
        logger.exception(f"_prepare_training_batch  291 {len(self.data_buffer)} use_buffer_data {use_buffer_data}")
        logger.exception(f"_prepare_training_batch  29122 {len(self.collected_data)} effective_batch_size {effective_batch_size}")
        logger.exception(f"_prepare_training_batch  312 {buffer_steps}")
        # Determine data source
        if use_buffer_data and len(self.data_buffer) > 0:
            available_data = self.data_buffer.get_data_from_last_n_steps(buffer_steps)
            logger.exception(f"_prepare_training_batch  295")
            if len(available_data) < effective_batch_size:
                if len(available_data) >= min(2, effective_batch_size // 2):
                    logger.exception(f"_prepare_training_batch  298")
                    items = available_data
                else:
                    return None
            else:
                import random
                items = random.sample(available_data, min(len(available_data), effective_batch_size))
        else:
            if len(self.collected_data) < effective_batch_size:
                if len(self.collected_data) >= min(2, effective_batch_size // 2):
                    items = list(self.collected_data)
                else:
                    return None
            else:
                items = list(self.collected_data)[:effective_batch_size]
        
        logger.exception(f"_prepare_training_batch  314")
        # Filter items with hidden states
        items = [item for item in items if "hidden_states" in item]
        if len(items) == 0:
            return None
        
        logger.exception(f"_prepare_training_batch  320")
        pad_id = int(getattr(self.model_config, "pad_token_id", 0) or 0)
        dev = next(self.model.parameters()).device
        
        # Prepare batch tensors
        input_ids_list = []
        loss_mask_list = []
        hidden_states_list = []
        target_logprobs_list = []
        
        for item in items:
            full_len = item["input_ids"].numel() if hasattr(item["input_ids"], 'numel') else len(item["input_ids"])
            
            # Compute loss mask if not present
            if "loss_mask" not in item:
                item_loss_mask = torch.zeros_like(item["input_ids"], dtype=torch.float32)
                if "responses" in item:
                    response_start = full_len - item["responses"].size(0)
                    pad_id = int(getattr(self.model_config, "pad_token_id", 0) or 0)
                    response_mask = (item["responses"] != pad_id).float()
                    item_loss_mask[response_start:] = response_mask
                else:
                    item_loss_mask[:] = 1.0
            else:
                item_loss_mask = item["loss_mask"]
            
            # Limit sequence length
            max_len = min(full_len, 512)
            
            # Select window around response tokens
            nonzero = torch.nonzero(item_loss_mask).flatten().cpu()
            if nonzero.numel() > 0:
                resp_start_idx = int(nonzero[0].item())
                resp_end_idx = int(nonzero[-1].item()) + 1
                window_span = max_len
                start = max(0, min(resp_start_idx, full_len - window_span))
                if resp_end_idx - start > window_span:
                    start = resp_end_idx - window_span
                end = min(full_len, start + window_span)
            else:
                start = max(0, full_len - max_len)
                end = full_len
            
            # Extract window
            seq_input_ids = item["input_ids"][start:end].to(dev, non_blocking=True)
            seq_loss_mask = item_loss_mask[start:end].to(dev, non_blocking=True)
            # print("#############item hidden_states", type(item["hidden_states"]))
            h_states = item["hidden_states"].to(dev, torch.bfloat16, non_blocking=True)
            h_seq_len = h_states.size(0)
            window_len = end - start
            
            if h_seq_len < window_len:
                pad_len = window_len - h_seq_len
                padding = torch.zeros(pad_len, h_states.size(-1), dtype=h_states.dtype, device=dev)
                seq_hidden_states = torch.cat([h_states, padding], dim=0)
            elif h_seq_len > window_len:
                if start < h_seq_len:
                    actual_end = min(h_seq_len, start + window_len)
                    seq_hidden_states = h_states[start:actual_end]
                    if seq_hidden_states.size(0) < window_len:
                        pad_len = window_len - seq_hidden_states.size(0)
                        padding = torch.zeros(pad_len, h_states.size(-1), dtype=h_states.dtype, device=dev)
                        seq_hidden_states = torch.cat([seq_hidden_states, padding], dim=0)
                else:
                    seq_hidden_states = h_states[-window_len:]
            else:
                seq_hidden_states = h_states
            
            input_ids_list.append(seq_input_ids)
            loss_mask_list.append(seq_loss_mask)
            hidden_states_list.append(seq_hidden_states)

            if "target_logprobs" in item:
                tg = item["target_logprobs"]
                tg = tg.to(dev, torch.float32, non_blocking=True)
                tg_seq_len = tg.size(0)
                if tg_seq_len < window_len:
                    pad_len = window_len - tg_seq_len
                    padding = torch.full((pad_len, tg.size(-1)), float("-inf"), dtype=tg.dtype, device=dev)
                    seq_target_logprobs = torch.cat([tg, padding], dim=0)
                elif tg_seq_len > window_len:
                    seq_target_logprobs = tg[start:start + window_len]
                else:
                    seq_target_logprobs = tg
                target_logprobs_list.append(seq_target_logprobs)
        
        if len(input_ids_list) == 0:
            return None
        
        # Concatenate sequences
        input_ids_concat = torch.cat(input_ids_list, dim=0).unsqueeze(0)
        loss_mask_concat = torch.cat(loss_mask_list, dim=0).unsqueeze(0)
        hidden_states_concat = torch.cat(hidden_states_list, dim=0).unsqueeze(0)
        
        total_seq_len = input_ids_concat.size(1)
        attn_mask = torch.ones((1, total_seq_len), dtype=torch.long, device=dev)
        
        batch_dict = {
            "input_ids": input_ids_concat,
            "attention_mask": attn_mask,
            "hidden_states": hidden_states_concat,
            "loss_mask": loss_mask_concat,
        }
        if target_logprobs_list:
            target_logprobs_concat = torch.cat(target_logprobs_list, dim=0).unsqueeze(0)
            batch_dict["target_logprobs"] = target_logprobs_concat
        return batch_dict
    
    async def training_step(self, step: int) -> bool:
        """Execute a single training step."""
        try:
            with torch.enable_grad():
                logger.exception(f"Training step {step} enter")
                return await self._training_step_impl(step)
        except Exception as e:
            logger.exception(f"Training step {step} failed: {e}")
            return False
    
    async def _training_step_impl(self, step: int) -> bool:
        """Implementation of a single training step."""
        logger.exception(f"_training_step_impl step {step} enter")
        if not self.model:
            logger.exception("No model available for training")
            return False
        
        logger.exception(f"_training_step_impl step {step} enter 430")
        if not self.config.collect_hidden_states_from_sgl:
            logger.exception(f"Skipping training step {step} (collect_hidden_states_from_sgl=False)")
            return False
        
        logger.exception(f"_training_step_impl step {step} enter 435")
        batch = self._prepare_training_batch()
        if batch is None:
            logger.exception(f"Not enough data at step {step}")
            return False
        logger.exception(f"_training_step_impl step {step} enter 440")
        
        # Optional barrier
        if self.config.enable_step_barrier and self.device_mesh is not None and self.device_mesh.size() > 1:
            try:
                dist.barrier(self.device_mesh.get_group())
            except Exception as e:
                logger.exception(f"Training barrier failed at step {step}: {e}")
        
        logger.exception(f"_training_step_impl step {step} enter 449")
        self.model.train()
        self.optimizer.zero_grad()
        logger.exception(f"_training_step_impl step {step} enter 452")
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss_kwargs = dict(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                hidden_states=batch["hidden_states"],
                loss_mask=batch["loss_mask"],
            )
            if "target_logprobs" in batch:
                loss_kwargs["target_logprobs"] = batch["target_logprobs"]
            loss = self.model(**loss_kwargs)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        
        self.training_steps += 1
        if self.training_steps % 10 == 0:
            logger.info(
                f"Step {self.training_steps}: loss={float(loss.item()):.4f}"
            )
        
        # Save checkpoint periodically
        if self.checkpoint_dir and (step // 100) > self._last_ckpt_step:
            self._save_checkpoint_async(step, is_final=False)
            self._last_ckpt_step = step // 100
        
        return True
    
    def get_model_state_dict(self) -> Optional[dict[str, torch.Tensor]]:
        """Get trainable model state dict."""
        if not self.model:
            return None
        trainable_state = self._get_trainable_state_dict()
        return {k: v.detach().cpu() for k, v in trainable_state.items() if v.requires_grad}
    
    def increment_rl_step(self):
        """Increment the RL step counter in data buffer."""
        self.data_buffer.increment_step()
        logger.debug(f"DataBuffer RL step incremented to {self.data_buffer.get_current_step()}")
    
    async def cleanup_training(self):
        """Cleanup training resources."""
        self._training_active = False
        
        # Save final checkpoint
        if self.checkpoint_dir and self.model is not None:
            self._save_checkpoint_async(self.training_steps, is_final=True)
        
        # Offload model and optimizer
        if self.model is not None:
            self.model.cpu()
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    param.data = param.data.cpu()
        
        self.collected_data.clear()
        self.data_buffer.clear()
        self.training_steps = 0
        self._training_initialized = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @property
    def is_training_initialized(self) -> bool:
        return self._training_initialized
    
    @property
    def is_training_active(self) -> bool:
        return self._training_active
    
    def set_training_active(self, active: bool):
        self._training_active = active
