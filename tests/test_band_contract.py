"""Smoke contract for the competitor band methods.

Threat model: the paper's comparison figures (envelope vs. WH/KS/ellipse/...) are
only meaningful if every competitor actually returns a well-formed band on clean
data. These tests do not judge a competitor's calibration — that is the point of
the study — they only guarantee that none of them is silently broken (crashing,
mis-shaped output, NaNs, values outside the unit square, or a crossed band), which
would invalidate any comparison drawn against it.

Each method has its own calling convention (some take raw scores, some take a
shared bootstrap TPR matrix), so a per-method adapter normalizes the call; the
assertions are shared via the ``assert_valid_band`` contract.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from studroc_paper.methods import (
    bp_smoothed_bootstrap_band,
    ellipse_envelope_band,
    fixed_width_ks_band,
    hsieh_turnbull_band,
    logit_bootstrap_band,
    pointwise_bootstrap_band,
    variance_model_band,
    wilson_band,
    wilson_rectangle_band,
    working_hotelling_band,
)

Adapter = Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]

K = 101
ALPHA = 0.05


def _wh(yt, ys, boot, grid):
    return working_hotelling_band(yt, ys, k=K, alpha=ALPHA)


def _ks(yt, ys, boot, grid):
    return fixed_width_ks_band(yt, ys, k=K, alpha=ALPHA)


def _ellipse(yt, ys, boot, grid):
    return ellipse_envelope_band(yt, ys, num_grid_points=K, alpha=ALPHA)


def _ht(yt, ys, boot, grid):
    return hsieh_turnbull_band(yt, ys, k=K, alpha=ALPHA)


def _wilson(yt, ys, boot, grid):
    return wilson_band(yt, ys, k=K, alpha=ALPHA)


def _wilson_rect(yt, ys, boot, grid):
    return wilson_rectangle_band(yt, ys, k=K, alpha=ALPHA)


def _pointwise(yt, ys, boot, grid):
    return pointwise_bootstrap_band(boot, grid, alpha=ALPHA)


def _logit(yt, ys, boot, grid):
    return logit_bootstrap_band(boot, grid, yt, ys, alpha=ALPHA)


def _varmodel(yt, ys, boot, grid):
    return variance_model_band(boot, grid, yt, ys, alpha=ALPHA)


def _bp(yt, ys, boot, grid):
    return bp_smoothed_bootstrap_band(yt, ys, grid, alpha=ALPHA, n_bootstrap=200)


_METHODS: dict[str, Adapter] = {
    "working_hotelling": _wh,
    "fixed_width_ks": _ks,
    "ellipse_envelope": _ellipse,
    "hsieh_turnbull": _ht,
    "wilson": _wilson,
    "wilson_rectangle": _wilson_rect,
    "pointwise_bootstrap": _pointwise,
    "logit_bootstrap": _logit,
    "variance_model": _varmodel,
    "bp_smoothed": _bp,
}


@pytest.mark.parametrize("method", _METHODS.values(), ids=list(_METHODS))
def test_competitor_band_satisfies_universal_contract(
    method, gaussian_scores, boot_tpr_matrix, fpr_grid, assert_valid_band
):
    y_true, y_score = gaussian_scores
    fpr, lower, upper = method(y_true, y_score, boot_tpr_matrix, fpr_grid)

    assert_valid_band(fpr, lower, upper)
    # The FPR axis must itself be a valid sorted grid spanning the unit interval.
    fpr = np.asarray(fpr)
    assert fpr[0] >= -1e-6 and fpr[-1] <= 1.0 + 1e-6
    assert np.all(np.diff(fpr) >= -1e-9)
