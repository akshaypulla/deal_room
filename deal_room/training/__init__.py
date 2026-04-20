"""
Training module for DealRoom v3 - GRPO trainer.
"""

from .grpo_trainer import (
    GRPOTrainer,
    TrainingMetrics,
    EpisodeTrajectory,
)

__all__ = [
    "GRPOTrainer",
    "TrainingMetrics",
    "EpisodeTrajectory",
]
