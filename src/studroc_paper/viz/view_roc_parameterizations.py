"""
Visualization of true ROC curves across parameter spaces.

Displays ROC curves with color gradients and subplot layouts to explore
how DGP parameters affect ROC curve shape.
"""

import warnings
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ..datagen.roc_to_dgp import map_lhs_to_dgp
from ..datagen.true_rocs import (
    DGP,
    make_beta_opposing_skew_dgp,
    make_bimodal_negative_dgp,
    make_exponential_dgp,
    make_gamma_dgp,
    make_heteroskedastic_gaussian_dgp,
    make_logitnormal_dgp,
    make_lognormal_dgp,
    make_student_t_dgp,
    make_weibull_dgp,
)
from .plot_aggregate import set_publication_style

# =============================================================================
# DGP Factory Registry
# =============================================================================

DGP_FACTORIES: dict[str, Callable[..., DGP]] = {
    "lognormal": make_lognormal_dgp,
    "logitnormal": make_logitnormal_dgp,
    "hetero_gaussian": make_heteroskedastic_gaussian_dgp,
    "exponential": make_exponential_dgp,
    "weibull": make_weibull_dgp,
    "student_t": make_student_t_dgp,
    "beta_opposing": make_beta_opposing_skew_dgp,
    "gamma": make_gamma_dgp,
    "bimodal_negative": make_bimodal_negative_dgp,
}


# =============================================================================
# Helper Functions
# =============================================================================


def _get_dgp_factory(dgp_type: str) -> Callable[..., DGP]:
    """Get the DGP factory function for a given type.

    Args:
        dgp_type: One of the registered DGP types.

    Returns:
        Factory function that creates DGP instances.

    Raises:
        ValueError: If dgp_type is not recognized.
    """
    if dgp_type not in DGP_FACTORIES:
        valid_types = ", ".join(sorted(DGP_FACTORIES.keys()))
        raise ValueError(f"Unknown DGP type: {dgp_type}. Valid types: {valid_types}")
    return DGP_FACTORIES[dgp_type]


def _validate_and_parse_params(
    params_dict: dict[str, tuple[float, float]],
    n_color: int,
    n_col: int | None,
    n_row: int | None,
) -> tuple[list[str], list[np.ndarray], tuple[int, int]]:
    """Validate params_dict and return parameter grids and subplot shape.

    Args:
        params_dict: Dictionary mapping parameter names to (min, max) bounds.
        n_color: Number of color gradient values for first parameter.
        n_col: Number of columns (for 2nd parameter). Required if len(params_dict) >= 2.
        n_row: Number of rows (for 3rd parameter). Required if len(params_dict) >= 3.

    Returns:
        Tuple of (param_names, param_grids, (nrows, ncols)).

    Raises:
        ValueError: If params_dict has invalid number of keys or required n_col/n_row missing.
    """
    n_params = len(params_dict)

    if n_params < 1 or n_params > 3:
        raise ValueError(f"params_dict must have 1-3 keys, got {n_params}")

    if n_params >= 2 and n_col is None:
        raise ValueError("n_col is required when params_dict has 2+ parameters")

    if n_params >= 3 and n_row is None:
        raise ValueError("n_row is required when params_dict has 3 parameters")

    param_names = list(params_dict.keys())
    param_grids = []

    # First parameter: color gradient
    bounds = params_dict[param_names[0]]
    param_grids.append(np.linspace(bounds[0], bounds[1], n_color))

    # Second parameter: columns
    if n_params >= 2:
        bounds = params_dict[param_names[1]]
        param_grids.append(np.linspace(bounds[0], bounds[1], n_col))

    # Third parameter: rows
    if n_params >= 3:
        bounds = params_dict[param_names[2]]
        param_grids.append(np.linspace(bounds[0], bounds[1], n_row))

    # Determine subplot shape
    nrows = n_row if n_params >= 3 else 1
    ncols = n_col if n_params >= 2 else 1

    return param_names, param_grids, (nrows, ncols)


def _extract_scalar_dgp_params(dgp_params: dict, idx: int = 0) -> dict:
    """Extract scalar parameters from dgp_params at given index.

    Handles the special case of bimodal_negative which returns nested lists.

    Args:
        dgp_params: Dictionary from map_lhs_to_dgp.
        idx: Index to extract if values are arrays.

    Returns:
        Dictionary with scalar values suitable for DGP factory.
    """
    result = {}
    for key, value in dgp_params.items():
        if isinstance(value, np.ndarray):
            result[key] = value.item() if value.ndim == 0 else value[idx]
        elif isinstance(value, list) and len(value) > 0:
            # Handle bimodal_negative's nested structure
            item = value[idx] if idx < len(value) else value[0]
            if isinstance(item, (list, tuple)):
                result[key] = list(item)
            else:
                result[key] = item
        else:
            result[key] = value
    return result


def _format_param_name(name: str) -> str:
    """Format parameter name for display.

    Args:
        name: Raw parameter name.

    Returns:
        Formatted name with Greek letters and subscripts.
    """
    replacements = {
        "sigma": "σ",
        "sigma_ratio": "σ_ratio",
        "alpha": "α",
        "beta": "β",
        "delta_mu": "Δμ",
        "delta_loc": "Δloc",
        "auc": "AUC",
        "df": "df",
        "shape": "shape",
        "mixture_weight": "w_mix",
        "mode_separation": "Δmode",
    }
    return replacements.get(name, name)


def _add_subplot_labels(
    axes: np.ndarray,
    param_names: list[str],
    param_grids: list[np.ndarray],
) -> None:
    """Add column titles and row labels to subplot grid.

    Args:
        axes: 2D array of axes.
        param_names: List of parameter names.
        param_grids: List of parameter value grids.
    """
    nrows, ncols = axes.shape

    # Column titles (2nd parameter values)
    if len(param_names) >= 2:
        col_values = param_grids[1]
        col_name = _format_param_name(param_names[1])
        for j in range(ncols):
            axes[0, j].set_title(f"{col_name} = {col_values[j]:.2g}", fontsize=10)

    # Row labels (3rd parameter values)
    if len(param_names) >= 3:
        row_values = param_grids[2]
        row_name = _format_param_name(param_names[2])
        for i in range(nrows):
            axes[i, 0].set_ylabel(f"{row_name} = {row_values[i]:.2g}\n\nTPR", fontsize=9)


def _add_colorbar(
    fig: plt.Figure,
    axes: np.ndarray,
    param_name: str,
    param_grid: np.ndarray,
    cmap: str,
) -> None:
    """Add colorbar with parameter name label.

    Args:
        fig: Matplotlib figure.
        axes: Array of axes.
        param_name: Name of the color parameter.
        param_grid: Values for the color parameter.
        cmap: Colormap name.
    """
    norm = Normalize(vmin=param_grid.min(), vmax=param_grid.max())
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Position colorbar to the right of all subplots
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    cbar.set_label(_format_param_name(param_name), fontsize=10)


# =============================================================================
# Main Plotting Function
# =============================================================================


def plot_roc_parameterizations(
    dgp_type: str,
    params_dict: dict[str, tuple[float, float]],
    n_color: int = 5,
    n_col: int | None = None,
    n_row: int | None = None,
    grid_points: int = 201,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
    title: str | None = None,
    show_colorbar: bool = True,
    show_diagonal: bool = True,
) -> Axes | np.ndarray:
    """Plot true ROC curves across a parameter space using color gradients and subplots.

    Visualizes how ROC curve shape varies with DGP parameters. The first parameter
    in params_dict controls color (using a gradient), the second controls subplot
    columns, and the third controls subplot rows.

    Args:
        dgp_type: Type of DGP. One of: 'lognormal', 'logitnormal', 'hetero_gaussian',
            'exponential', 'weibull', 'student_t', 'beta_opposing', 'gamma',
            'bimodal_negative'.
        params_dict: Dictionary mapping parameter names to (min, max) bounds.
            Must contain 'auc' and any DGP-specific shape parameters.
            Order matters: 1st key -> color, 2nd key -> columns, 3rd key -> rows.
        n_color: Number of values for the color gradient (1st parameter).
        n_col: Number of columns (values for 2nd parameter). Required if 2+ params.
        n_row: Number of rows (values for 3rd parameter). Required if 3 params.
        grid_points: Number of FPR grid points for ROC curves.
        figsize: Figure size (width, height). If None, auto-calculated.
        cmap: Matplotlib colormap name for color gradient.
        title: Figure title. If None, auto-generated from dgp_type.
        show_colorbar: Whether to show colorbar for color parameter.
        show_diagonal: Whether to show diagonal reference line.

    Returns:
        Single Axes if 1 subplot, or 2D ndarray of Axes if multiple subplots.

    Raises:
        ValueError: If dgp_type is unknown or params_dict has invalid structure.

    Examples:
        Single parameter (color only):
        >>> ax = plot_roc_parameterizations(
        ...     dgp_type="lognormal",
        ...     params_dict={"auc": (0.6, 0.95), "sigma": (0.5, 2.0)},
        ...     n_color=5,
        ... )

        Two parameters (color + columns):
        >>> axes = plot_roc_parameterizations(
        ...     dgp_type="student_t",
        ...     params_dict={"auc": (0.6, 0.95), "df": (2, 30)},
        ...     n_color=5,
        ...     n_col=4,
        ... )

        Three parameters (color + columns + rows):
        >>> axes = plot_roc_parameterizations(
        ...     dgp_type="gamma",
        ...     params_dict={"auc": (0.6, 0.95), "shape": (0.5, 5.0), "extra": (1, 10)},
        ...     n_color=5,
        ...     n_col=3,
        ...     n_row=2,
        ... )
    """
    set_publication_style()

    # Validate inputs
    factory = _get_dgp_factory(dgp_type)
    param_names, param_grids, (nrows, ncols) = _validate_and_parse_params(
        params_dict, n_color, n_col, n_row
    )

    # Calculate figure size if not provided
    if figsize is None:
        figsize = (3.5 * ncols + (1.0 if show_colorbar else 0), 3.5 * nrows)

    # Create figure and axes
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    # FPR grid for ROC curves
    fpr_grid = np.linspace(0, 1, grid_points)

    # Color gradient setup
    color_values = param_grids[0]
    colormap = plt.get_cmap(cmap)
    norm = Normalize(vmin=color_values.min(), vmax=color_values.max())

    # Generate ROC curves for each subplot
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]

            # Determine parameter values for this subplot
            for k, color_val in enumerate(color_values):
                # Build LHS params dict
                lhs_params = {param_names[0]: np.array([color_val])}

                if len(param_names) >= 2:
                    lhs_params[param_names[1]] = np.array([param_grids[1][j]])

                if len(param_names) >= 3:
                    lhs_params[param_names[2]] = np.array([param_grids[2][i]])

                # Map LHS to DGP parameters
                try:
                    dgp_params = map_lhs_to_dgp(dgp_type, lhs_params)
                except Exception as e:
                    warnings.warn(
                        f"Failed to map parameters for {dgp_type} with "
                        f"{lhs_params}: {e}"
                    )
                    continue

                # Extract scalar params
                scalar_params = _extract_scalar_dgp_params(dgp_params, idx=0)

                # Check for NaN in solved parameters
                if any(
                    np.isnan(v) if isinstance(v, (int, float, np.number)) else False
                    for v in scalar_params.values()
                ):
                    warnings.warn(
                        f"NaN in solved parameters for {dgp_type} with {lhs_params}"
                    )
                    continue

                # Create DGP and get ROC
                try:
                    dgp = factory(**scalar_params)
                    tpr = dgp.get_true_roc(fpr_grid)
                except Exception as e:
                    warnings.warn(
                        f"Failed to create DGP or compute ROC for {dgp_type} with "
                        f"{scalar_params}: {e}"
                    )
                    continue

                # Plot ROC curve
                color = colormap(norm(color_val))
                ax.plot(fpr_grid, tpr, color=color, linewidth=1.5, alpha=0.9)

            # Diagonal reference line
            if show_diagonal:
                ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)

            # Axis formatting
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")

            # Only label bottom and left edges
            if i == nrows - 1:
                ax.set_xlabel("FPR")
            if j == 0 and len(param_names) < 3:
                ax.set_ylabel("TPR")

    # Add subplot labels
    _add_subplot_labels(axes, param_names, param_grids)

    # Figure title
    if title is None:
        title = f"ROC Curves: {dgp_type}"
    fig.suptitle(title, fontsize=12, y=1.02)

    # Layout adjustment before colorbar
    fig.tight_layout()

    # Add colorbar after tight_layout to avoid conflict
    if show_colorbar:
        _add_colorbar(fig, axes, param_names[0], param_grids[0], cmap)

    # Return single axes or array
    if nrows == 1 and ncols == 1:
        return axes[0, 0]
    return axes
