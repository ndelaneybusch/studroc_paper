"""
Module for ingesting simulation result files into pandas DataFrames.

Two ingestion paths are provided:

1. Aggregated JSON files ("*_aggregated.json"), which produce two types of
   DataFrames per alpha level:
   - "standard": One row per model result per file, with flattened scalar
     metrics (coverage_rate, coverage_se, coverage_ci_lower/upper,
     violation_rate_above/below, direction_test_pvalue, mean_band_area,
     std_band_area, mean_band_width, flattened width percentiles,
     mean_max_violation, percentile_95_max_violation, and
     mean_violation_area_above/below)
   - "curve": One row per FPR region per model result per file, with
     region-specific width and violation-rate metrics

2. Trial-level feather files ("*_individual.feather") via
   load_individual_results, which concatenates per-simulation rows across
   DGPs and sample-size configurations with memory-efficient dtypes.
"""

import json
import os
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def process_single_model_result(
    model_name: str,
    model_data: Mapping[str, Any],
    metadata: Mapping[str, Any],
    source_file: str,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Process a single model's results from a JSON file.

    Extracts scalar metrics and curve data (FPR region metrics) from a model's
    results at all alpha levels. Produces two types of records per alpha level:
    standard rows with scalar metrics and curve rows with region-specific metrics.

    Args:
        model_name: Name of the model (e.g., "envelope_standard", "ks").
        model_data: The model's result data containing alpha-level results.
        metadata: File-level metadata to include in each row.
        source_file: Name of the source file for traceability.

    Returns:
        Nested dict: {alpha_key: {"standard": [row_dict], "curve": [row_dicts]}}.
        The "standard" list contains one dict with flattened scalar metrics.
        The "curve" list contains one dict per FPR region with region metrics.

    Examples:
        >>> model_data = {
        ...     "alpha_0.05": {
        ...         "nominal_alpha": 0.05,
        ...         "confidence_level": 0.95,
        ...         "mean_width": 1.23,
        ...         "width_by_fpr_region": {"0-10": 1.1, "10-20": 1.3},
        ...     }
        ... }
        >>> result = process_single_model_result(
        ...     model_name="envelope_standard",
        ...     model_data=model_data,
        ...     metadata={"n": 100, "dist": "gaussian"},
        ...     source_file="results.json",
        ... )
        >>> result.keys()
        dict_keys(['alpha_0.05'])
        >>> result["alpha_0.05"]["standard"][0]["model"]
        'envelope_standard'
        >>> len(result["alpha_0.05"]["curve"])
        2
    """
    results = {}

    for alpha_key, alpha_data in model_data.items():
        if not alpha_key.startswith("alpha_"):
            continue

        # Initialize containers for this alpha level
        results[alpha_key] = {"standard": [], "curve": []}

        # Build base row with metadata and identifiers
        base_row = {"source_file": source_file, "model": model_name, **metadata}

        # Build standard row (scalar metrics only)
        standard_row = base_row.copy()

        # Fields that are curve data (to exclude from standard)
        curve_fields = {"width_by_fpr_region", "violation_rate_by_region"}

        for key, value in alpha_data.items():
            if key in curve_fields:
                continue
            elif key == "width_percentiles":
                # Flatten width percentiles
                for pct_key, pct_val in value.items():
                    standard_row[f"width_{pct_key}"] = pct_val
            else:
                standard_row[key] = value

        results[alpha_key]["standard"].append(standard_row)

        # Build curve rows (one per FPR region)
        width_by_region = alpha_data.get("width_by_fpr_region", {})
        violation_by_region = alpha_data.get("violation_rate_by_region", {})

        # Get union of all regions
        all_regions = set(width_by_region.keys()) | set(violation_by_region.keys())

        for region in sorted(all_regions, key=_parse_region_start):
            curve_row = base_row.copy()
            curve_row["nominal_alpha"] = alpha_data.get("nominal_alpha")
            curve_row["confidence_level"] = alpha_data.get("confidence_level")
            curve_row["fpr_region"] = region
            curve_row["region_width"] = width_by_region.get(region)
            curve_row["region_violation_rate"] = violation_by_region.get(region)
            results[alpha_key]["curve"].append(curve_row)

    return results


def _parse_region_start(region: str) -> int:
    """Extract starting value from region string for sorting.

    Args:
        region: Region string in format "start-end" (e.g., "0-10", "10-20").

    Returns:
        Starting value of the region as an integer. Returns 0 if parsing fails.

    Examples:
        >>> _parse_region_start("0-10")
        0
        >>> _parse_region_start("10-20")
        10
        >>> _parse_region_start("invalid")
        0
    """
    try:
        return int(region.split("-")[0])
    except (ValueError, IndexError):
        return 0


def process_json_file(filepath: str | Path) -> dict[str, pd.DataFrame]:
    """Process a single JSON file into per-alpha DataFrames.

    Reads an aggregated simulation result JSON file and produces separate
    DataFrames for each alpha level and type (standard vs. curve).

    Args:
        filepath: Path to the JSON file containing aggregated results.

    Returns:
        Dict of DataFrames keyed by "{alpha}_standard" and "{alpha}_curve".
        Each standard DataFrame has one row per model result with scalar metrics.
        Each curve DataFrame has one row per FPR region per model result.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.

    Examples:
        >>> dfs = process_json_file("results/gaussian_n100_aggregated.json")
        >>> dfs.keys()
        dict_keys(['alpha_0.05_standard', 'alpha_0.05_curve', 'alpha_0.10_standard', ...])
        >>> dfs["alpha_0.05_standard"].columns
        Index(['source_file', 'model', 'n', 'dist', 'mean_width', ...], dtype='object')
        >>> dfs["alpha_0.05_curve"].columns
        Index(['source_file', 'model', 'n', 'dist', 'fpr_region', 'region_width', ...], dtype='object')
    """
    filepath = Path(filepath)

    with open(filepath, "r") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    source_file = filepath.name

    # Collect all rows by alpha and type
    all_results: dict[str, dict[str, list[dict]]] = {}

    for model_name, model_data in data.items():
        if model_name == "metadata":
            continue
        if not isinstance(model_data, dict):
            continue

        model_results = process_single_model_result(
            model_name=model_name,
            model_data=model_data,
            metadata=metadata,
            source_file=source_file,
        )

        # Merge into all_results
        for alpha_key, type_dict in model_results.items():
            if alpha_key not in all_results:
                all_results[alpha_key] = {"standard": [], "curve": []}
            all_results[alpha_key]["standard"].extend(type_dict["standard"])
            all_results[alpha_key]["curve"].extend(type_dict["curve"])

    # Convert to DataFrames
    dataframes = {}
    for alpha_key, type_dict in all_results.items():
        for df_type, rows in type_dict.items():
            if rows:
                key = f"{alpha_key}_{df_type}"
                dataframes[key] = pd.DataFrame(rows)

    return dataframes


def process_folder(
    folder_path: str | Path, pattern: str = "*_aggregated.json"
) -> dict[str, pd.DataFrame]:
    """Process all matching JSON files in a folder into concatenated DataFrames.

    Reads all JSON files matching the pattern in the specified folder and
    concatenates results across files for each alpha level and type.

    Args:
        folder_path: Path to folder containing JSON files.
        pattern: Glob pattern to match JSON files. Defaults to "*_aggregated.json".

    Returns:
        Dict of concatenated DataFrames keyed by "{alpha}_standard" and "{alpha}_curve".
        Each DataFrame combines results from all matching files.

    Raises:
        FileNotFoundError: If no files matching the pattern are found in folder_path.

    Examples:
        >>> dfs = process_folder("data/results")
        >>> dfs.keys()
        dict_keys(['alpha_0.05_standard', 'alpha_0.05_curve', ...])
        >>> len(dfs["alpha_0.05_standard"])
        150
        >>> dfs["alpha_0.05_standard"]["source_file"].nunique()
        5
    """
    folder_path = Path(folder_path)
    json_files = sorted(folder_path.glob(pattern))

    if not json_files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {folder_path}")

    # Collect all dataframes by key
    collected: dict[str, list[pd.DataFrame]] = {}

    for json_file in json_files:
        try:
            file_dfs = process_json_file(json_file)
            for key, df in file_dfs.items():
                if key not in collected:
                    collected[key] = []
                collected[key].append(df)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Skipping {json_file.name} due to error: {e}")
            continue

    # Concatenate all dataframes for each key
    result = {}
    for key, df_list in collected.items():
        if df_list:
            result[key] = pd.concat(df_list, ignore_index=True)

    return result


def _optimize_trial_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast a trial-level DataFrame for memory-efficient concatenation.

    String identifier columns become categoricals and float64 metric columns
    become float32. Object columns (e.g., per-row parameter arrays for mixture
    DGPs) and boolean columns are left untouched.

    Args:
        df: Trial-level DataFrame from a single "*_individual.feather" file.

    Returns:
        The same DataFrame with optimized dtypes (modified in place).
    """
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            first_valid = df[col].dropna()
            if len(first_valid) and isinstance(first_valid.iloc[0], str):
                df[col] = df[col].astype("category")
        elif df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    return df


def load_individual_results(
    folder_path: os.PathLike | str,
    *,
    pattern: str = "*_individual.feather",
    columns: Sequence[str] | None = None,
    optimize_dtypes: bool = True,
) -> pd.DataFrame:
    """Load and concatenate trial-level simulation results from feather files.

    Each feather file holds one row per (method, alpha, LHS combination,
    simulation repeat) for a single DGP and sample-size configuration. Files
    may have DGP-specific parameter columns (e.g., "dgp_df" for student_t);
    the concatenated result takes the union of columns with NaN elsewhere.

    Two derived columns are added when their inputs are present:
    - "max_violation": elementwise max of max_violation_above/below
    - "violation_area": sum of violation_area_above/below

    Args:
        folder_path: Folder containing the feather files.
        pattern: Glob pattern selecting trial-level files.
        columns: Optional subset of columns to load from each file. Derived
            columns are only added if their inputs are included.
        optimize_dtypes: Whether to downcast strings to categoricals and
            float64 to float32 to reduce memory (roughly halves footprint).

    Returns:
        Concatenated trial-level DataFrame across all matching files.

    Raises:
        FileNotFoundError: If no files matching the pattern are found.

    Examples:
        >>> df = load_individual_results("data/results/final_20260611")
        >>> df.groupby(["dgp_type", "n_total"]).size()
    """
    folder_path = Path(folder_path)
    files = sorted(folder_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {folder_path}")

    frames = []
    for fp in files:
        df = pd.read_feather(fp, columns=list(columns) if columns else None)
        if optimize_dtypes:
            df = _optimize_trial_dtypes(df)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    if {"max_violation_above", "max_violation_below"}.issubset(out.columns):
        out["max_violation"] = np.maximum(
            out["max_violation_above"], out["max_violation_below"]
        )
    if {"violation_area_above", "violation_area_below"}.issubset(out.columns):
        out["violation_area"] = (
            out["violation_area_above"] + out["violation_area_below"]
        )
    return out


def get_available_keys(dataframes: Mapping[str, pd.DataFrame]) -> dict[str, list[str]]:
    """Get available alpha levels and DataFrame types from a result dict.

    Extracts and categorizes the keys from DataFrames produced by process_folder
    or process_json_file to show what data is available.

    Args:
        dataframes: Result from process_folder or process_json_file.

    Returns:
        Dict with three keys:
        - "alphas": List of alpha level keys (e.g., ["alpha_0.05", "alpha_0.10"])
        - "types": List of DataFrame types (e.g., ["standard", "curve"])
        - "keys": List of all DataFrame keys (e.g., ["alpha_0.05_standard", ...])

    Examples:
        >>> dfs = process_folder("data/results")
        >>> info = get_available_keys(dfs)
        >>> info["alphas"]
        ['alpha_0.05', 'alpha_0.10']
        >>> info["types"]
        ['curve', 'standard']
        >>> len(info["keys"])
        4
    """
    keys = list(dataframes.keys())
    alphas = sorted(set(k.rsplit("_", 1)[0] for k in keys))
    types = sorted(set(k.rsplit("_", 1)[1] for k in keys))

    return {"alphas": alphas, "types": types, "keys": keys}


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = "."

    output_folder = None
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]

    dfs = process_folder(folder)
    info = get_available_keys(dfs)

    print(f"Loaded {len(dfs)} DataFrames:")
    for key in info["keys"]:
        df = dfs[key]
        print(f"  {key}: {len(df)} rows x {len(df.columns)} columns")

    if output_folder is not None:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        pickle_file = output_path / "aggregated_results.pkl"
        with open(pickle_file, "wb") as f:
            pickle.dump(dfs, f)
        print(f"\nSaved DataFrames to {pickle_file}")
