"""
Lunar Image Registration Pipeline
ISRO SIH 2026 — Problem Statement 26166

Multi-modal, sun-angle and scale-invariant image correspondence
using Chandrayaan-2 optical images (OHRC, TMC-2, IIRS).
"""

from .orchestrator import RegistrationOrchestrator
from .preprocessing import LunarPreprocessor
from .feature_extraction import FeatureEngine
from .registration import TransformEstimator
from .subpixel import SubPixelRefiner
from .validation import RegistrationEvaluator

__all__ = [
    "RegistrationOrchestrator",
    "LunarPreprocessor",
    "FeatureEngine",
    "TransformEstimator",
    "SubPixelRefiner",
    "RegistrationEvaluator",
]
