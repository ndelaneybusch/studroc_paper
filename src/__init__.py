"""ROC Confidence Band Analysis Suite.

This package provides tools for generating, evaluating, and visualizing
confidence bands for ROC curves using various methods including bootstrap,
Working-Hotelling, and Kolmogorov-Smirnov approaches.
"""

from . import datagen, eval, methods, sampling, viz

__all__ = [
    "datagen",
    "eval",
    "methods",
    "sampling",
    "viz",
]
