"""
Eagle3 Configuration for Slime

This module provides configuration utilities for Eagle3 drafter training.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Eagle3Config:
    """Configuration for Eagle3 speculative decoding and training."""
    
    # Speculative decoding settings
    sglang_speculative_algorithm: str = "EAGLE3"
    sglang_speculative_num_steps: int = 3
    sglang_speculative_eagle_topk: int = 1
    sglang_speculative_num_draft_tokens: int = 1
    sglang_speculative_draft_model_path: str = ""
    
    # Eagle3 training settings
    enable_eagle3_training: bool = False
    eagle3_training_interval_steps: int = 10
    eagle3_batch_size_per_gpu: int = 2
    eagle3_max_seq_len: int = 8192
    eagle3_max_epochs: int = 10
    eagle3_checkpoint_path: Optional[str] = None
    eagle3_min_workers_for_training: int = 1
    eagle3_collect_hidden_states: bool = True
    
    # Eagle3 model architecture
    eagle3_num_layers: int = 1
    eagle3_hidden_size: Optional[int] = None
    eagle3_intermediate_size: Optional[int] = None
    eagle3_num_attention_heads: Optional[int] = None
    eagle3_num_key_value_heads: Optional[int] = None
    
    # Eagle3 optimizer settings
    eagle3_lr: float = 1e-6
    eagle3_lr_warmup_steps: int = 1000
    eagle3_weight_decay: float = 1e-2
    eagle3_warmup_style: str = "constant"
    
    # Eagle3 offload settings
    eagle3_offload_param: bool = False
    eagle3_offload_optimizer: bool = False
    
    # Weight update settings
    update_weights_bucket_megabytes: int = 512
    
    @classmethod
    def from_args(cls, args) -> "Eagle3Config":
        """Create Eagle3Config from argparse arguments."""
        config = cls()
        
        # Speculative decoding settings
        config.sglang_speculative_algorithm = getattr(
            args, 'sglang_speculative_algorithm', 'EAGLE3'
        )
        config.sglang_speculative_num_steps = getattr(
            args, 'sglang_speculative_num_steps', 3
        )
        config.sglang_speculative_eagle_topk = getattr(
            args, 'sglang_speculative_eagle_topk', 1
        )
        config.sglang_speculative_num_draft_tokens = getattr(
            args, 'sglang_speculative_num_draft_tokens', 1
        )
        config.sglang_speculative_draft_model_path = getattr(
            args, 'sglang_speculative_draft_model_path', ''
        )
        
        # Training settings
        config.enable_eagle3_training = getattr(
            args, 'enable_eagle3_training', False
        )
        config.eagle3_training_interval_steps = getattr(
            args, 'eagle3_training_interval_steps', 10
        )
        config.eagle3_batch_size_per_gpu = getattr(
            args, 'eagle3_batch_size_per_gpu', 2
        )
        config.eagle3_max_seq_len = getattr(
            args, 'eagle3_max_seq_len', 8192
        )
        config.eagle3_max_epochs = getattr(
            args, 'eagle3_max_epochs', 10
        )
        config.eagle3_checkpoint_path = getattr(
            args, 'eagle3_checkpoint_path', None
        )
        config.eagle3_min_workers_for_training = getattr(
            args, 'eagle3_min_workers_for_training', 1
        )
        config.eagle3_collect_hidden_states = getattr(
            args, 'eagle3_collect_hidden_states', True
        )
        
        # Model architecture
        config.eagle3_num_layers = getattr(
            args, 'eagle3_num_layers', 1
        )
        config.eagle3_hidden_size = getattr(
            args, 'eagle3_hidden_size', None
        )
        config.eagle3_intermediate_size = getattr(
            args, 'eagle3_intermediate_size', None
        )
        config.eagle3_num_attention_heads = getattr(
            args, 'eagle3_num_attention_heads', None
        )
        config.eagle3_num_key_value_heads = getattr(
            args, 'eagle3_num_key_value_heads', None
        )
        
        # Optimizer settings
        config.eagle3_lr = getattr(
            args, 'eagle3_lr', 1e-6
        )
        config.eagle3_lr_warmup_steps = getattr(
            args, 'eagle3_lr_warmup_steps', 1000
        )
        config.eagle3_weight_decay = getattr(
            args, 'eagle3_weight_decay', 1e-2
        )
        config.eagle3_warmup_style = getattr(
            args, 'eagle3_warmup_style', 'constant'
        )
        
        # Offload settings
        config.eagle3_offload_param = getattr(
            args, 'eagle3_offload_param', False
        )
        config.eagle3_offload_optimizer = getattr(
            args, 'eagle3_offload_optimizer', False
        )
        
        # Weight update settings
        config.update_weights_bucket_megabytes = getattr(
            args, 'update_weights_bucket_megabytes', 512
        )
        
        return config
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "sglang_speculative_algorithm": self.sglang_speculative_algorithm,
            "sglang_speculative_num_steps": self.sglang_speculative_num_steps,
            "sglang_speculative_eagle_topk": self.sglang_speculative_eagle_topk,
            "sglang_speculative_num_draft_tokens": self.sglang_speculative_num_draft_tokens,
            "sglang_speculative_draft_model_path": self.sglang_speculative_draft_model_path,
            "enable_eagle3_training": self.enable_eagle3_training,
            "eagle3_training_interval_steps": self.eagle3_training_interval_steps,
            "eagle3_batch_size_per_gpu": self.eagle3_batch_size_per_gpu,
            "eagle3_max_seq_len": self.eagle3_max_seq_len,
            "eagle3_max_epochs": self.eagle3_max_epochs,
            "eagle3_checkpoint_path": self.eagle3_checkpoint_path,
            "eagle3_min_workers_for_training": self.eagle3_min_workers_for_training,
            "eagle3_collect_hidden_states": self.eagle3_collect_hidden_states,
            "eagle3_num_layers": self.eagle3_num_layers,
            "eagle3_hidden_size": self.eagle3_hidden_size,
            "eagle3_intermediate_size": self.eagle3_intermediate_size,
            "eagle3_num_attention_heads": self.eagle3_num_attention_heads,
            "eagle3_num_key_value_heads": self.eagle3_num_key_value_heads,
            "eagle3_lr": self.eagle3_lr,
            "eagle3_lr_warmup_steps": self.eagle3_lr_warmup_steps,
            "eagle3_weight_decay": self.eagle3_weight_decay,
            "eagle3_warmup_style": self.eagle3_warmup_style,
            "eagle3_offload_param": self.eagle3_offload_param,
            "eagle3_offload_optimizer": self.eagle3_offload_optimizer,
            "update_weights_bucket_megabytes": self.update_weights_bucket_megabytes,
        }


def add_eagle3_arguments(parser):
    """Add Eagle3 arguments to argument parser.
    
    Args:
        parser: argparse.ArgumentParser instance
    """
    # Speculative decoding arguments
    parser.add_argument(
        '--sglang-speculative-algorithm',
        type=str,
        default='EAGLE3',
        choices=['EAGLE', 'EAGLE3', 'MEDUSA', 'NONE'],
        help='Speculative decoding algorithm'
    )
    parser.add_argument(
        '--sglang-speculative-num-steps',
        type=int,
        default=3,
        help='Number of speculative steps'
    )
    parser.add_argument(
        '--sglang-speculative-eagle-topk',
        type=int,
        default=1,
        help='Top-k for Eagle sampling'
    )
    parser.add_argument(
        '--sglang-speculative-num-draft-tokens',
        type=int,
        default=1,
        help='Number of draft tokens'
    )
    parser.add_argument(
        '--sglang-speculative-draft-model-path',
        type=str,
        default='',
        help='Path to Eagle3 drafter model checkpoint'
    )
    
    # Eagle3 training arguments
    parser.add_argument(
        '--enable-eagle3-training',
        action='store_true',
        help='Enable Eagle3 drafter training'
    )
    parser.add_argument(
        '--eagle3-training-interval-steps',
        type=int,
        default=10,
        help='Training interval in rollout steps'
    )
    parser.add_argument(
        '--eagle3-batch-size-per-gpu',
        type=int,
        default=2,
        help='Batch size per GPU for Eagle3 training'
    )
    parser.add_argument(
        '--eagle3-max-seq-len',
        type=int,
        default=8192,
        help='Maximum sequence length for Eagle3 training'
    )
    parser.add_argument(
        '--eagle3-max-epochs',
        type=int,
        default=10,
        help='Maximum epochs for Eagle3 training'
    )
    parser.add_argument(
        '--eagle3-checkpoint-path',
        type=str,
        default=None,
        help='Path to save Eagle3 checkpoints'
    )
    parser.add_argument(
        '--eagle3-min-workers-for-training',
        type=int,
        default=1,
        help='Minimum workers for Eagle3 training'
    )
    parser.add_argument(
        '--eagle3-collect-hidden-states',
        action='store_true',
        default=True,
        help='Collect hidden states from SGLang for training'
    )
    
    # Eagle3 model architecture
    parser.add_argument(
        '--eagle3-num-layers',
        type=int,
        default=1,
        help='Number of layers in Eagle3 drafter model'
    )
    parser.add_argument(
        '--eagle3-hidden-size',
        type=int,
        default=None,
        help='Hidden size for Eagle3 drafter model'
    )
    parser.add_argument(
        '--eagle3-intermediate-size',
        type=int,
        default=None,
        help='Intermediate size for Eagle3 drafter model'
    )
    parser.add_argument(
        '--eagle3-num-attention-heads',
        type=int,
        default=None,
        help='Number of attention heads for Eagle3 drafter model'
    )
    parser.add_argument(
        '--eagle3-num-key-value-heads',
        type=int,
        default=None,
        help='Number of key-value heads for Eagle3 drafter model'
    )
    
    # Eagle3 optimizer settings
    parser.add_argument(
        '--eagle3-lr',
        type=float,
        default=1e-6,
        help='Learning rate for Eagle3 training'
    )
    parser.add_argument(
        '--eagle3-lr-warmup-steps',
        type=int,
        default=1000,
        help='Learning rate warmup steps for Eagle3'
    )
    parser.add_argument(
        '--eagle3-weight-decay',
        type=float,
        default=1e-2,
        help='Weight decay for Eagle3 optimizer'
    )
    parser.add_argument(
        '--eagle3-warmup-style',
        type=str,
        default='constant',
        choices=['constant', 'cosine', 'linear'],
        help='Learning rate warmup style for Eagle3'
    )
    
    # Eagle3 offload settings
    parser.add_argument(
        '--eagle3-offload-param',
        action='store_true',
        help='Offload Eagle3 parameters to CPU'
    )
    parser.add_argument(
        '--eagle3-offload-optimizer',
        action='store_true',
        help='Offload Eagle3 optimizer to CPU'
    )
    
    # Weight update settings
    parser.add_argument(
        '--update-weights-bucket-megabytes',
        type=int,
        default=512,
        help='Bucket size for weight updates (in MB)'
    )
