"""
Eagle3 Model Implementations

This module provides Eagle3 drafter model implementations for different
base model architectures.
"""

from .qwen2_eagle3 import Qwen2ForCausalLMEagle3, Qwen2EagleDecoderLayer
from .llama_eagle import LlamaForCausalLMEagle3, LlamaDecoderLayer

__all__ = [
    "Qwen2ForCausalLMEagle3",
    "Qwen2EagleDecoderLayer",
    "LlamaForCausalLMEagle3",
    "LlamaDecoderLayer",
]
