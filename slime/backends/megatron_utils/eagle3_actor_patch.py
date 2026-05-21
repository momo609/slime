"""
Eagle3 Integration Patch for MegatronTrainRayActor

This module patches the MegatronTrainRayActor to integrate Eagle3 drafter
training and weight updates into the training loop.
"""

import asyncio
import logging
from typing import Optional

import ray
import torch.distributed as dist

from slime.backends.megatron_utils.eagle3_integration import (
    Eagle3ModelConfig,
    Eagle3TrainingManager,
    get_eagle3_manager,
)
from slime.drafter import Eagle3TrainConfig
from slime.drafter.eagle3_weight_updater import Eagle3WeightUpdater

logger = logging.getLogger(__name__)


def enable_eagle3_training(args):
    """Check if Eagle3 training should be enabled based on args."""
    return (
        hasattr(args, 'sglang_speculative_algorithm') 
        and args.sglang_speculative_algorithm == 'EAGLE3'
        and hasattr(args, 'enable_eagle3_training')
        and args.enable_eagle3_training
    )


def init_eagle3_in_actor(actor_instance):
    """Initialize Eagle3 training in the actor.
    
    This should be called in the init() method of MegatronTrainRayActor
    after the base model is initialized.
    
    Args:
        actor_instance: MegatronTrainRayActor instance
    """
    if not enable_eagle3_training(actor_instance.args):
        logger.info("Eagle3 training is not enabled")
        return
    
    args = actor_instance.args
    
    # Create Eagle3 training configuration
    train_config = Eagle3TrainConfig(
        enable=True,
        spec_model_path=getattr(args, 'sglang_speculative_draft_model_path', ''),
        training_interval_steps=getattr(args, 'eagle3_training_interval_steps', 10),
        batch_size_per_gpu=getattr(args, 'eagle3_batch_size_per_gpu', 2),
        max_seq_len=getattr(args, 'eagle3_max_seq_len', 8192),
        max_epochs=getattr(args, 'eagle3_max_epochs', 10),
        checkpoint_path=getattr(args, 'eagle3_checkpoint_path', None),
        min_workers_for_training=getattr(args, 'eagle3_min_workers_for_training', 1),
        collect_hidden_states_from_sgl=getattr(args, 'eagle3_collect_hidden_states', True),
        lr=getattr(args, 'eagle3_lr', 1e-6),
        lr_warmup_steps=getattr(args, 'eagle3_lr_warmup_steps', 1000),
        weight_decay=getattr(args, 'eagle3_weight_decay', 1e-2),
        warmup_style=getattr(args, 'eagle3_warmup_style', 'constant'),
        is_offload_param=getattr(args, 'eagle3_offload_param', False),
        is_offload_optimizer=getattr(args, 'eagle3_offload_optimizer', False),
    )
    
    # Create Eagle3 model configuration
    drafter_model_config = Eagle3ModelConfig(
        spec_model_path=train_config.spec_model_path,
        num_hidden_layers=getattr(args, 'eagle3_num_layers', 1),
        hidden_size=getattr(args, 'eagle3_hidden_size', actor_instance.hf_config.text_config.hidden_size),
        intermediate_size=getattr(args, 'eagle3_intermediate_size', None),
        num_attention_heads=getattr(args, 'eagle3_num_attention_heads', None),
        num_key_value_heads=getattr(args, 'eagle3_num_key_value_heads', None),
        vocab_size=151936,
        max_position_embeddings=actor_instance.hf_config.text_config.max_position_embeddings,
        rms_norm_eps=actor_instance.hf_config.text_config.rms_norm_eps,
        rope_theta=getattr(actor_instance.hf_config.text_config, 'rope_theta', None),
        pad_token_id=getattr(args, 'pad_token_id', 0),  
    )
    
    # Initialize Eagle3 manager
    eagle3_manager = get_eagle3_manager()
    eagle3_manager.initialize(
        base_model=actor_instance.model,
        base_model_config=actor_instance.hf_config.text_config,
        train_config=train_config,
        drafter_model_config=drafter_model_config,
        rollout_manager=None,  # rollout_manager will be set later in set_rollout_manager
    )
    
    # Store reference in actor
    actor_instance.eagle3_manager = eagle3_manager
    
    logger.info("Eagle3 training initialized in actor")


def patch_update_weights_method_for_eagle3(actor_class):
    """Patch the update_weights method to include Eagle3 weight updates.
    
    Args:
        actor_class: MegatronTrainRayActor class to patch
    """
    original_update_weights = actor_class.update_weights
    
    def update_weights_with_eagle3(self):
        """Update weights method with Eagle3 integration."""
        # Call original update_weights
        original_update_weights(self)
        print("##############original_update_weights#######")
        
        # Update Eagle3 drafter weights if enabled
        if hasattr(self, 'eagle3_manager') and self.eagle3_manager.is_initialized:
            eagle3_manager = self.eagle3_manager
            
            # Run async update
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(eagle3_manager.update_drafter_weights())
            finally:
                loop.close()
    
    actor_class.update_weights = update_weights_with_eagle3


def patch_save_model_method_for_eagle3(actor_class):
    """Patch the save_model method to include Eagle3 checkpoint saving.
    
    Args:
        actor_class: MegatronTrainRayActor class to patch
    """
    original_save_model = actor_class.save_model
    
    def save_model_with_eagle3(self, rollout_id, force_sync=False):
        """Save model method with Eagle3 integration."""
        # Call original save_model
        original_save_model(self, rollout_id, force_sync)
        print("##############save_model_with_eagle3#######")
        
        # Save Eagle3 checkpoint if enabled
        if hasattr(self, 'eagle3_manager') and self.eagle3_manager.is_initialized:
            eagle3_manager = self.eagle3_manager
            
            # Save Eagle3 checkpoint
            if eagle3_manager.eagle3_trainer:
                eagle3_manager.eagle3_trainer._save_checkpoint_async(
                    rollout_id, is_final=force_sync
                )
    
    actor_class.save_model = save_model_with_eagle3


def apply_eagle3_patches(actor_class):
    """Apply all Eagle3 patches to the actor class.
    
    Args:
        actor_class: MegatronTrainRayActor class to patch
    """
    logger.info("Applying Eagle3 patches to MegatronTrainRayActor")
    
    patch_update_weights_method_for_eagle3(actor_class)
    patch_save_model_method_for_eagle3(actor_class)
    
    logger.info("Eagle3 patches applied successfully")


def setup_eagle3_weight_updater(actor_instance, rollout_engines):
    """Setup Eagle3 weight updater with rollout engines.
    
    This should be called after rollout engines are initialized.
    
    Args:
        actor_instance: MegatronTrainRayActor instance
        rollout_engines: List of rollout engine actors
    """
    if not hasattr(actor_instance, 'eagle3_manager') or not actor_instance.eagle3_manager.is_initialized:
        return
    
    eagle3_manager = actor_instance.eagle3_manager
    
    # Create weight updater for the first rollout engine
    if rollout_engines and rollout_engines[0]:
        rollout_engine = rollout_engines[0]
        
        # Get device mesh
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Handle PyTorch version compatibility
        try:
            from torch.distributed.device_mesh import DeviceMesh
            # PyTorch 2.1+ API
            device_mesh = DeviceMesh("cuda", mesh_shape=(world_size,))
        except TypeError:
            from torch.distributed.device_mesh import DeviceMesh
            # PyTorch 2.8 API
            device_mesh = DeviceMesh(
                device_type="cuda",
                mesh=[[i] for i in range(world_size)]
            )
        
        # Create weight updater
        update_weights_bucket_megabytes = getattr(
            actor_instance.args, 
            'update_weights_bucket_megabytes', 
            512
        )
        
        weight_updater = Eagle3WeightUpdater(
            engine=rollout_engine,
            device_mesh=device_mesh,
            update_weights_bucket_megabytes=update_weights_bucket_megabytes,
        )
        
        # Set weight updater in manager
        eagle3_manager.set_weight_updater(weight_updater)
        
        logger.info("Eagle3 weight updater initialized")


def apply_eagle3_patches_if_enabled(actor_instance):
    """Apply Eagle3 patches if enabled in arguments.
    
    This should be called in the init() method of MegatronTrainRayActor
    after the base model is initialized.
    
    Args:
        actor_instance: MegatronTrainRayActor instance
    """
    if not enable_eagle3_training(actor_instance.args):
        logger.info("Eagle3 training is not enabled, skipping patches")
        return
    
    logger.info("Applying Eagle3 patches to MegatronTrainRayActor")
    
    # Get the class of the instance
    actor_class = actor_instance.__class__
    
    # Check if patches are already applied
    if hasattr(actor_class, '_eagle3_patched'):
        logger.info("Eagle3 patches already applied, skipping")
        return
    
    # Apply patches
    # patch_train_method_for_eagle3(actor_class)
    patch_update_weights_method_for_eagle3(actor_class)
    patch_save_model_method_for_eagle3(actor_class)
    
    
    print("###########actor_class",actor_class.train)
    # Mark as patched
    actor_class._eagle3_patched = True
    
    logger.info("Eagle3 patches applied successfully")
    
    # Initialize Eagle3 in actor
    init_eagle3_in_actor(actor_instance)
    
    # Patch set_rollout_manager to setup weight updater
    original_set_rollout_manager = actor_class.set_rollout_manager
    
    # def set_rollout_manager_with_eagle3(self, rollout_manager):
    #     """Set rollout manager with Eagle3 weight updater setup."""
    #     # Call original method
    #     original_set_rollout_manager(self, rollout_manager)
        
    #     # Store rollout_manager reference for Eagle3
    #     if hasattr(self, 'eagle3_manager') and self.eagle3_manager.is_initialized:
    #         self.eagle3_manager.rollout_manager = rollout_manager
        
    #     # Setup Eagle3 weight updater if enabled
    #     if hasattr(self, 'eagle3_manager') and self.eagle3_manager.is_initialized:
    #         logger.info("Setting up Eagle3 weight updater")
    #         # Get rollout engines from rollout manager
    #         rollout_engines = getattr(rollout_manager, 'rollout_engines', [])
    #         setup_eagle3_weight_updater(self, rollout_engines)
    
    # actor_class.set_rollout_manager = set_rollout_manager_with_eagle3
