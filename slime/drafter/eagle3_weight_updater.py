"""
Eagle3 Weight Updater for Slime

This module handles updating Eagle3 drafter weights in SGLang inference engine.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def get_named_tensor_buckets(named_tensors: List[tuple], bucket_bytes: int) -> List[List[tuple]]:
    """Split named tensors into buckets of specified size."""
    buckets = []
    current_bucket = []
    current_size = 0
    
    for name, tensor in named_tensors:
        tensor_size = tensor.numel() * tensor.element_size()
        
        if current_size + tensor_size > bucket_bytes and current_bucket:
            buckets.append(current_bucket)
            current_bucket = []
            current_size = 0
        
        current_bucket.append((name, tensor))
        current_size += tensor_size
    
    if current_bucket:
        buckets.append(current_bucket)
    
    return buckets


async def sgl_update_weights(
    engine,
    params_batch: List[tuple],
    is_draft_model: bool = False,
):
    """Update weights in SGLang engine."""
    # This is a placeholder - actual implementation depends on SGLang API
    # In fastrl, this calls sglang's weight update endpoint
    
    # For now, we'll implement a simple version that updates the model directly
    # In production, this should use SGLang's distributed weight update API
    
    for name, tensor in params_batch:
        if is_draft_model:
            # Update drafter model weights
            if hasattr(engine, 'draft_model'):
                # Navigate to the correct parameter
                parts = name.split('.')
                param = engine.draft_model
                for part in parts:
                    param = getattr(param, part)
                param.data.copy_(tensor.to(param.device))
        else:
            # Update target model weights
            if hasattr(engine, 'model'):
                parts = name.split('.')
                param = engine.model
                for part in parts:
                    param = getattr(param, part)
                param.data.copy_(tensor.to(param.device))


async def update_drafter_weights(
    engine,
    drafter_params: Dict[str, torch.Tensor],
    update_weights_bucket_megabytes: int = 512,
):
    """Update Eagle3 drafter weights in SGLang engine.
    
    Args:
        engine: SGLang engine instance
        drafter_params: Dictionary of parameter name to tensor
        device_mesh: Device mesh for distributed operations
        update_weights_bucket_megabytes: Bucket size for weight updates (in MB)
    """
    if drafter_params is None or len(drafter_params) == 0:
        return
    
    logger.info(f"Updating Eagle3 drafter weights with {len(drafter_params)} parameters")
    
    # Convert to list of named tensors
    named_tensors = [(k, v) for k, v in drafter_params.items()]
    
    # Split into buckets
    update_weights_bucket_bytes = int(update_weights_bucket_megabytes) << 20
    buckets = get_named_tensor_buckets(named_tensors, update_weights_bucket_bytes)
    
    logger.info(f"Split {len(named_tensors)} parameters into {len(buckets)} buckets")
    
    # Update weights bucket by bucket
    for i, params_batch in enumerate(buckets):
        logger.debug(f"Updating bucket {i + 1}/{len(buckets)} with {len(params_batch)} parameters")
        
        try:
            await sgl_update_weights(
                engine=engine,
                params_batch=params_batch,
                is_draft_model=True,
            )
        except Exception as e:
            logger.error(f"Failed to update bucket {i + 1}: {e}")
            raise
    
    logger.info("Eagle3 drafter weights updated successfully")


def convert_weight_keys(
    state_dict: Dict[str, torch.Tensor],
    model,
) -> Dict[str, torch.Tensor]:
    """Convert weight keys to match the model config.
    
    This handles any necessary renaming of parameter keys between
    the drafter model and the inference engine's expected format.
    """
    converted = {}
    
    for key, value in state_dict.items():
        # Handle common key transformations
        new_key = key
        
        # For Eagle3 models, ensure keys match the expected format
        if not new_key.startswith("model.") and not new_key.startswith("lm_head"):
            new_key = f"model.{new_key}"
        
        converted[new_key] = value
    
    return converted


class Eagle3WeightUpdater:
    """Manages Eagle3 drafter weight updates to SGLang engine."""
    
    def __init__(
        self,
        engine,
        update_weights_bucket_megabytes: int = 512,
    ):
        self.engine = engine
        self.update_weights_bucket_megabytes = update_weights_bucket_megabytes
        
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        if self.rank == 0:
            logger.info("Eagle3WeightUpdater initialized")
    
    async def update_weights(self, drafter_params: Dict[str, torch.Tensor]):
        """Update drafter weights in the inference engine.
        
        Args:
            drafter_params: Dictionary of parameter name to tensor
        """
        if drafter_params is None or len(drafter_params) == 0:
            logger.warning("No drafter parameters to update")
            return
        
        # Convert parameter keys if needed
        drafter_params = convert_weight_keys(drafter_params, self.engine)
        
        # Update weights
        await update_drafter_weights(
            engine=self.engine,
            drafter_params=drafter_params,
            update_weights_bucket_megabytes=self.update_weights_bucket_megabytes,
        )
        
        if self.rank == 0:
            logger.info("Eagle3 drafter weights update completed")
