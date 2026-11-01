"""Class split of the M3 confidence budget.

M3 composes one exact order-statistic band per class. The two class
coverages multiply, ``(1 - alpha_F)(1 - alpha_G) = 1 - alpha``, and the split
ratio ``rho`` sets ``1 - alpha_F = (1 - alpha) ** rho`` for the negative
class. Any ``rho`` fixed before the ranks are seen preserves the theorem
(theory doc Proposition 12), including a ``rho`` that depends on the class
sizes, because the sampling design fixes them.

``"sample_size"`` names the rule

    rho(n0, n1) = n0 ** -0.5 / (n0 ** -0.5 + n1 ** -0.5),

which hands the larger share of the error allowance to the smaller class:
``rho = 0.75`` at 100/900 and ``0.25`` at 900/100. Its motivation is that a
class band's half-width scales as ``n ** -0.5``, so the rule spends the
budget where it buys the most width. It reduces to the Sidak split
``rho = 0.5`` whenever ``n0 == n1``.
"""

import math
from typing import Literal

SplitRatio = float | Literal["sample_size"]


def validate_split_ratio(split_ratio: SplitRatio) -> None:
    """Reject split ratios outside ``(0, 1)`` and unknown rule names.

    Args:
        split_ratio: A fixed ratio in ``(0, 1)`` or the name ``"sample_size"``.

    Raises:
        ValueError: If the value is neither a ratio in ``(0, 1)`` nor a known
            rule name.
    """
    if isinstance(split_ratio, str):
        if split_ratio != "sample_size":
            raise ValueError(
                f"split_ratio must be a float in (0, 1) or 'sample_size', "
                f"got {split_ratio!r}"
            )
        return
    if not 0.0 < split_ratio < 1.0:
        raise ValueError(f"split_ratio must be in (0, 1), got {split_ratio}")


def resolve_split_ratio(*, split_ratio: SplitRatio, n0: int, n1: int) -> float:
    """Return the negative-class share of the log confidence budget.

    Args:
        split_ratio: A fixed ratio in ``(0, 1)`` or ``"sample_size"``.
        n0: Negative-class sample size.
        n1: Positive-class sample size.

    Returns:
        The ratio ``rho`` in ``(0, 1)``.

    Raises:
        ValueError: If ``split_ratio`` is invalid or a class is empty.
    """
    validate_split_ratio(split_ratio)
    if n0 < 1 or n1 < 1:
        raise ValueError(f"Both classes must be present (n0={n0}, n1={n1})")
    if split_ratio == "sample_size":
        w0, w1 = 1.0 / math.sqrt(n0), 1.0 / math.sqrt(n1)
        return w0 / (w0 + w1)
    return float(split_ratio)
