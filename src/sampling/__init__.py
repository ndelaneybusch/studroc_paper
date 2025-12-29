"""Sampling utilities for bootstrap and experimental design."""

from .bootstrap_grid import generate_bootstrap_grid, reconstruct_ranks
from .lhs import iman_conover_transform, maximin_lhs
from .optim_k_b import (
    allocate_budget,
    compute_beta,
    construct_grid,
    estimate_D,
    estimate_D_theoretical,
)

__all__ = [
    "allocate_budget",
    "compute_beta",
    "construct_grid",
    "estimate_D",
    "estimate_D_theoretical",
    "generate_bootstrap_grid",
    "iman_conover_transform",
    "maximin_lhs",
    "reconstruct_ranks",
]
