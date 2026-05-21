"""
Eagle3 Drafter Module for Slime

This module provides online training and weight update capabilities for
Eagle3 draft model to optimize speculative decoding during RL training.
"""

from .eagle3_trainer import Eagle3BackgroundTrainer, Eagle3TrainConfig, DataBuffer

__all__ = [
    "Eagle3BackgroundTrainer",
    "Eagle3TrainConfig",
    "DataBuffer",
]
