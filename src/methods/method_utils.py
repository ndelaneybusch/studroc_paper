"""Shared utilities for PyTorch-based methods."""

import torch
from torch import Tensor


def numpy_to_torch(arr, device: torch.device | None = None) -> Tensor:
    """Convert numpy array to tensor, preserving dtype.

    Args:
        arr: Input numpy array.
        device: Target device (defaults to CUDA if available).

    Returns:
        PyTorch tensor on the specified device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.from_numpy(arr).to(device)


def torch_to_numpy(tensor: Tensor):
    """Convert tensor back to numpy array, preserving dtype.

    Args:
        tensor: Input PyTorch tensor.

    Returns:
        Numpy array with preserved dtype.
    """
    return tensor.detach().cpu().numpy()


def torch_step_interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """
    Step-function interpolation (right-continuous).

    For x between xp[j] and xp[j+1], returns fp[j].
    """
    # Find insertion indices
    indices = torch.searchsorted(xp, x, right=True) - 1
    indices = torch.clamp(indices, 0, len(fp) - 1)
    return fp[indices]


def torch_interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Linear interpolation equivalent to np.interp.

    Args:
        x: X-coordinates at which to evaluate.
        xp: X-coordinates of data points (must be increasing).
        fp: Y-coordinates of data points.

    Returns:
        Interpolated values at x positions.
    """
    # Find indices where x would be inserted
    indices = torch.searchsorted(xp, x)

    # Clamp indices for boundary handling
    indices = torch.clamp(indices, 1, len(xp) - 1)

    # Get left and right points
    x0 = xp[indices - 1]
    x1 = xp[indices]
    y0 = fp[indices - 1]
    y1 = fp[indices]

    # Linear interpolation
    t = (x - x0) / (x1 - x0 + 1e-12)
    return y0 + t * (y1 - y0)
