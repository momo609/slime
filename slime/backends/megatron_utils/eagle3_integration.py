"""
Eagle3 Integration for Megatron Training Actor

This module integrates Eagle3 drafter training and weight updates into
the Megatron training workflow in Slime.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from megatron.core import mpu
from torch.distributed.device_mesh import DeviceMesh

from slime.drafter import Eagle3BackgroundTrainer, Eagle3TrainConfig
from slime.drafter.eagle3_weight_updater import Eagle3WeightUpdater

logger = logging.getLogger(__name__)


@dataclass
class Eagle3ModelConfig:
    """Configuration for Eagle3 drafter model."""
    spec_model_path: str = ""
    num_hidden_layers: int = 1
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    vocab_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    rms_norm_eps: Optional[float] = None
    rope_theta: Optional[float] = None
    pad_token_id : int = 0



def create_eagle3_drafter_model(
    base_model_config,
    drafter_config: Eagle3ModelConfig,
):
    """Create Eagle3 drafter model from base model config.
    
    This function creates a single-layer drafter model that shares
    embeddings with the base model but has its own attention and FFN layers.
    
    Args:
        base_model_config: Configuration of the base model
        drafter_config: Configuration for the drafter model
        
    Returns:
        Eagle3 drafter model instance
    """
    # Import here to avoid circular dependencies
    from slime.drafter.models.qwen2_eagle3 import Qwen2ForCausalLMEagle3
    from slime.drafter.models.llama_eagle import LlamaForCausalLMEagle3
    from transformers import LlamaConfig, Qwen2Config
    
    # Create config from base model
    from transformers import AutoConfig
    
    if hasattr(base_model_config, 'to_dict'):
        config_dict = base_model_config.to_dict()
    else:
        config_dict = base_model_config
    
    # Override for drafter model
    config_dict['num_hidden_layers'] = drafter_config.num_hidden_layers
    config_dict['torch_dtype'] = 'bfloat16'
    config_dict['tie_word_embeddings'] = False
    
    # Override with explicit drafter config if provided
    if drafter_config.hidden_size is not None:
        config_dict['hidden_size'] = drafter_config.hidden_size
    if drafter_config.intermediate_size is not None:
        config_dict['intermediate_size'] = drafter_config.intermediate_size
    if drafter_config.num_attention_heads is not None:
        config_dict['num_attention_heads'] = drafter_config.num_attention_heads
    if drafter_config.num_key_value_heads is not None:
        config_dict['num_key_value_heads'] = drafter_config.num_key_value_heads
    if drafter_config.vocab_size is not None:
        config_dict['vocab_size'] = drafter_config.vocab_size
    if drafter_config.max_position_embeddings is not None:
        config_dict['max_position_embeddings'] = drafter_config.max_position_embeddings
    if drafter_config.rms_norm_eps is not None:
        config_dict['rms_norm_eps'] = drafter_config.rms_norm_eps
    if drafter_config.rope_theta is not None:
        config_dict['rope_theta'] = drafter_config.rope_theta
    if drafter_config.rope_theta is not None:
        config_dict['pad_token_id'] = drafter_config.pad_token_id
    
    # Determine model type and create appropriate config and model
    model_type = config_dict.get('model_type', '').lower()
    config_class = None
    
    
    config_class = LlamaConfig
    
    
    hf_config = config_class.from_dict(config_dict)
    
    # Create Eagle3 model based on config type
    if isinstance(hf_config, LlamaConfig):
        model = LlamaForCausalLMEagle3(hf_config)
        logger.info(f"Created LlamaForCausalLMEagle3 drafter model")
        print("###############Created LlamaForCausalLMEagle3 model", model)
    elif isinstance(hf_config, Qwen2Config):
        model = Qwen2ForCausalLMEagle3(hf_config)
        logger.info(f"Created Qwen2ForCausalLMEagle3 drafter model")
    else:
        # Default to Qwen2 for backward compatibility
        model = Qwen2ForCausalLMEagle3(hf_config)
        logger.warning(f"Unknown config type {type(hf_config)}, defaulting to Qwen2ForCausalLMEagle3")
    
    return model



def load_eagle3_checkpoint(
    model,
    checkpoint_path: str,
):
    """Load Eagle3 drafter model from checkpoint.
    
    Args:
        model: Eagle3 drafter model
        checkpoint_path: Path to checkpoint directory or file
    """
    logger.info(f"Loading Eagle3 drafter from checkpoint: {checkpoint_path}")
    
    if os.path.isfile(checkpoint_path):
        # Single file checkpoint
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    elif os.path.isdir(checkpoint_path):
        # Directory checkpoint
        checkpoint_file = os.path.join(checkpoint_path, "trainer_state.pt")
        if os.path.exists(checkpoint_file):
            state_dict = torch.load(checkpoint_file, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        else:
            # Try loading from HuggingFace format
            from safetensors.torch import load_file
            
            safetensors_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors')]
            if safetensors_files:
                state_dict = {}
                for f in safetensors_files:
                    state_dict.update(load_file(os.path.join(checkpoint_path, f)))
            else:
                # Try .bin files
                bin_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.bin')]
                if bin_files:
                    state_dict = {}
                    for f in bin_files:
                        checkpoint = torch.load(os.path.join(checkpoint_path, f), map_location="cpu")
                        state_dict.update(checkpoint)
                else:
                    raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
    
    # Rename keys if necessary
    renamed_checkpoint = {}
    for key, value in state_dict.items():
        renamed_checkpoint[key] = value
        print("#############renamed_checkpoint######", key, "####", value.shape)
        # if not key.startswith("model.") and not key.startswith("lm_head"):
        #     renamed_checkpoint[f"model.{key}"] = value
        #     print("#############renamed_checkpoint model######", f"model.{key}")
        # else:
        #     renamed_checkpoint[key] = value
        #     print("#############renamed_checkpoint######", key)
    print("###########load_eagle3_checkpoint model########", model)
    # Load state dict with strict=False to allow partial loading
    missing_keys, unexpected_keys = model.load_state_dict(renamed_checkpoint, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys when loading Eagle3 checkpoint: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading Eagle3 checkpoint: {unexpected_keys}")
    
    # Load vocabulary mapping (t2d/d2t) if available in checkpoint
    if hasattr(model, "load_vocab_mapping"):
        model.load_vocab_mapping(renamed_checkpoint)
    
    logger.info("Eagle3 drafter checkpoint loaded successfully")


def setup_eagle3_optimizer(
    model,
    train_config: Eagle3TrainConfig,
):
    """Setup optimizer and learning rate scheduler for Eagle3 training.
    
    Args:
        model: Eagle3 drafter model
        train_config: Training configuration
        
    Returns:
        Tuple of (optimizer, lr_scheduler)
    """
    # Filter trainable parameters (exclude frozen layers like embed_tokens and lm_head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if len(trainable_params) == 0:
        logger.warning("No trainable parameters found in Eagle3 model")
        return None, None
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_config.lr,
        betas=(0.9, 0.95),
        weight_decay=train_config.weight_decay,
    )
    
    # Create learning rate scheduler
    total_steps = train_config.max_epochs * 1000  # Approximate
    num_warmup_steps = train_config.lr_warmup_steps
    
    if train_config.warmup_style == "constant":
        from torch.optim.lr_scheduler import ConstantLR
        
        lr_scheduler = ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=num_warmup_steps,
        )
    elif train_config.warmup_style == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, ConstantLR
        
        warmup_scheduler = ConstantLR(optimizer, factor=0.1, total_iters=num_warmup_steps)
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - num_warmup_steps,
            eta_min=train_config.lr * 0.1,
        )
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[num_warmup_steps],
        )
    else:
        from torch.optim.lr_scheduler import ConstantLR
        
        lr_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=num_warmup_steps)
    
    logger.info(f"Eagle3 optimizer setup: {len(trainable_params)} trainable parameters")
    
    return optimizer, lr_scheduler


class Eagle3TrainingManager:
    """Manages Eagle3 drafter training integration with Megatron training."""
    
    def __init__(self):
        self.drafter_model = None
        self.drafter_optimizer = None
        self.drafter_lr_scheduler = None
        self.eagle3_trainer = None
        self.weight_updater = None
        self.train_config = None
        self.device_mesh = None
        self.rollout_manager = None  # Add rollout_manager reference
        self.is_initialized = False
        
    def initialize(
        self,
        base_model,
        base_model_config,
        train_config: Eagle3TrainConfig,
        drafter_model_config: Eagle3ModelConfig,
        rollout_manager,
    ):
        """Initialize Eagle3 drafter training.
        
        Args:
            base_model: Base model (for sharing embeddings)
            base_model_config: Base model configuration
            train_config: Eagle3 training configuration
            drafter_model_config: Eagle3 model configuration
            rollout_manager: Rollout manager instance
        """
        if not train_config.enable:
            logger.info("Eagle3 training is disabled")
            return
        
        logger.info("Initializing Eagle3 drafter training...")
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Create device mesh for drafter training
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Handle PyTorch version compatibility
        try:
            # PyTorch 2.1+ API
            from torch.distributed.device_mesh import DeviceMesh
            self.device_mesh = DeviceMesh("cuda", mesh_shape=(world_size,))
        except TypeError:
            # PyTorch 2.8 API
            from torch.distributed.device_mesh import DeviceMesh
            # PyTorch 2.8 uses different parameters
            self.device_mesh = DeviceMesh(
                device_type="cuda",
                mesh=[[i] for i in range(world_size)]
            )
        
        # Create drafter model
        self.drafter_model = create_eagle3_drafter_model(
            base_model_config, drafter_model_config
        )
        
        # # Load checkpoint if provided
        if train_config.spec_model_path and os.path.exists(train_config.spec_model_path):
            load_eagle3_checkpoint(self.drafter_model, train_config.spec_model_path)
        else:
            logger.info("Initializing Eagle3 drafter from scratch")
        
        # Share embeddings with base model
        if hasattr(base_model, 'lm_head'):
            self.drafter_model.lm_head = base_model.lm_head
            for param in self.drafter_model.lm_head.parameters():
                param.requires_grad = False
            logger.info("Shared lm_head with base model")
        
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'embed_tokens'):
            self.drafter_model.model.embed_tokens = base_model.model.embed_tokens
            for param in self.drafter_model.model.embed_tokens.parameters():
                param.requires_grad = False
            logger.info("Shared embed_tokens with base model")
        
        # Move model to GPU
        self.drafter_model.cuda()
        
        # Setup optimizer
        self.drafter_optimizer, self.drafter_lr_scheduler = setup_eagle3_optimizer(
            self.drafter_model, train_config
        )
        
        # Create background trainer
        self.eagle3_trainer = Eagle3BackgroundTrainer(
            drafter_model=self.drafter_model,
            drafter_optimizer=self.drafter_optimizer,
            drafter_lr_scheduler=self.drafter_lr_scheduler,
            train_config=train_config,
            device_mesh=self.device_mesh,
            model_config=base_model_config,
        )
        
        # # Create weight updater (will be initialized with rollout engine later)
        # self.weight_updater = None
        # weight_updater = Eagle3WeightUpdater(
        #     engine=rollout_engine,
        #     device_mesh=device_mesh,
        #     update_weights_bucket_megabytes=update_weights_bucket_megabytes,
        # )
        
        # Store rollout_manager reference
        self.rollout_manager = rollout_manager
        
        self.train_config = train_config
        self.is_initialized = True
        
        if rank == 0:
            logger.info("Eagle3 drafter training initialized successfully")
    
    def set_weight_updater(self, weight_updater: Eagle3WeightUpdater):
        """Set the weight updater for drafter model."""
        self.weight_updater = weight_updater
    
    def collect_rollout_data(self, rollout_data, hidden_states=None):
        """Collect data from rollout for Eagle3 training.
        
        Args:
            rollout_data: Rollout data dictionary
            hidden_states: Optional hidden states from rollout
        """
        if not self.is_initialized or not self.eagle3_trainer:
            return
        # logger.exception(
        #     f"collect_rollout_data rollout_data: {rollout_data}",
        # )
        self.eagle3_trainer.collect_online_data(rollout_data, hidden_states)
    
    async def train_drafter(self, rollout_id: int):
        """Train Eagle3 drafter model.

        Args:
            rollout_id: Current rollout iteration ID
        """
        logger.info(f"[Eagle3 Manager] train_drafter called with rollout_id={rollout_id}")
        logger.info(f"[Eagle3 Manager] is_initialized={self.is_initialized}, eagle3_trainer={self.eagle3_trainer}")

        if not self.is_initialized or not self.eagle3_trainer:
            logger.warning(f"[Eagle3 Manager] Not initialized or no trainer, skipping training")
            return

        # Check if we should train based on interval
        logger.info(f"[Eagle3 Manager] Checking training interval: rollout_id={rollout_id}, interval={self.train_config.training_interval_steps}")
        if rollout_id % self.train_config.training_interval_steps != 0:
            logger.info(f"[Eagle3 Manager] Rollout {rollout_id} not at training interval, skipping")
            return

        logger.info(f"[Eagle3 Manager] Training Eagle3 drafter at rollout {rollout_id}")

        # Activate training mode
        training_ranks = list(range(self.device_mesh.size()))
        logger.info(f"[Eagle3 Manager] Activating training model on ranks {training_ranks}")
        await self.eagle3_trainer.activate_training_model(
            self.device_mesh, training_ranks
        )
        for step in range(50):
            # Perform training step
            success = await self.eagle3_trainer.training_step(step)
        
            if success:
                logger.info(f"Eagle3 training step {step} completed")
            else:
                logger.warning(f"Eagle3 training step {step} failed")
        
        # Increment RL step counter
        self.eagle3_trainer.increment_rl_step()
    
    async def update_drafter_weights(self):
        """Update Eagle3 drafter weights in rollout engine.
        
        This should be called after training to sync weights to SGLang.
        """
        if not self.is_initialized or not self.weight_updater:
            return
        
        logger.info("Updating Eagle3 drafter weights in rollout engine")
        
        # Get current model state dict
        drafter_params = self.eagle3_trainer.get_model_state_dict()
        
        if drafter_params is None:
            logger.warning("No drafter parameters to update")
            return
        
        # Update weights via weight updater
        await self.weight_updater.update_weights(drafter_params)
        
        logger.info("Eagle3 drafter weights updated successfully")
    
    def cleanup(self):
        """Cleanup Eagle3 training resources."""
        if not self.is_initialized:
            return
        
        logger.info("Cleaning up Eagle3 training resources...")
        
        if self.eagle3_trainer:
            # Run async cleanup
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.eagle3_trainer.cleanup_training())
            finally:
                loop.close()
        
        self.drafter_model = None
        self.drafter_optimizer = None
        self.drafter_lr_scheduler = None
        self.eagle3_trainer = None
        self.weight_updater = None
        self.is_initialized = False
        
        logger.info("Eagle3 training cleanup completed")


# Global singleton instance
_eagle3_manager = None


def get_eagle3_manager() -> Eagle3TrainingManager:
    """Get the global Eagle3 training manager instance."""
    global _eagle3_manager
    if _eagle3_manager is None:
        _eagle3_manager = Eagle3TrainingManager()
    return _eagle3_manager
