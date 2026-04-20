"""
Curriculum module for DealRoom v3 - adaptive curriculum generator.
"""

from .adaptive_generator import (
    AdaptiveCurriculumGenerator,
    CurriculumConfig,
    FailureAnalysis,
    create_curriculum_generator,
)

__all__ = [
    "AdaptiveCurriculumGenerator",
    "CurriculumConfig",
    "FailureAnalysis",
    "create_curriculum_generator",
]
