#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot whole-brain pre- versus post-sleep EEG spectral power.

Input arrays were generated using Step2_generate_ersp_from_epochs.py.
This script reads:

    dataset/
        ERSP_f.npy
        Figure_3_source_data/
            pre_mean/ERSP_<SubXX>_pre_mean.npy
            post_normalized/ERSP_<SubXX>_post.npy

Expected shapes:
    pre mean:  region x frequency
    post data: trial x region x frequency

For each subject and frequency band:
- Pre is the mean across the four regions and the frequency bins in that band.
- Post is first averaged across post trials and frequency bins within each region,
  then averaged across the four regions.
"""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, t


# =============================================================================
# Configuration
# =============================================================================

SHALLOW = [
    "Sub01", "Sub03", "Sub04", "Sub05", "Sub09", "Sub12", "Sub13",
    "Sub24", "Sub25", "Sub26", "Sub28", "Sub30", "Sub32", "Sub37",
    "Sub41", "Sub47", "Sub50", "Sub54", "Sub55", "Sub62", "Sub64",
]

DEEP = [
    "Sub02", "Sub06", "Sub08", "Sub15", "Sub17", "Sub18", "Sub21",
    "Sub22", "Sub29", "Sub31", "Sub36", "Sub38", "Sub39", "Sub40",
    "Sub43", "Sub45", "Sub46", "Sub48", "Sub51", "Sub53", "Sub59",
    "Sub61", "Sub63",
]

SUBJECTS = SHALLOW + DEEP

# =============================================================================
# Paths
# =============================================================================

# Directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent

# Root directory of the shared dataset
DATASET_DIR = SCRIPT_DIR.parent

# Source data for Figure 3
SOURCE_DATA_DIR = DATASET_DIR / "Data_2"

# Input files
PRE_DIR = SOURCE_DATA_DIR / "pre_mean"
POST_DIR = SOURCE_DATA_DIR / "post"
FREQUENCY_PATH = DATASET_DIR / "ERSP_f.npy"

# Output directory
OUT_ROOT = DATASET_DIR / "results" / "Figure_3"

# Automatically create output directories
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Configuration Parameters
# =============================================================================

plt.rcParams["font.size"] = 16

BOX_LINEWIDTH = 1.5
GROUP_SPACING = 3.0
INTRA_GROUP_OFFSET = 0.5
FIRST_GROUP_OFFSET = 1.0


# =============================================================================
# Input setup
# =============================================================================

if not FREQUENCY_PATH.exists():
    raise FileNotFoundError(
        f"Frequency file not found: {FREQUENCY_PATH}\n"
        "Check DATA_ROOT."
    )

ERSP_F = np.load(FREQUENCY_PATH, allow_pickle=True)

BANDS = {
    "Delta": np.where((ERSP_F > 1) & (ERSP_F < 4))[0],
    "Theta": np.where((ERSP_F > 4) & (ERSP_F < 8))[0],
    "Alpha": np.where((ERSP_F > 8) & (ERSP_F < 12))[0],
    "Beta1": np.where((ERSP_F > 12) & (ERSP_F < 16))[0],
    "Beta2": np.where((ERSP_F > 16) & (ERSP_F < 30))[0],
}

for band_name, indices in BANDS.items():
    if indices.size == 0:
        raise ValueError(
            f"No frequency bins were found for {band_name}. "
            f"Check {FREQUENCY_PATH} and the band limits."
        )


# =============================================================================
# Helpers
# =============================================================================

def load_subject_arrays(subject: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load one subject's saved pre mean and post trial-wise power arrays."""
    pre_path = PRE_DIR / f"ERSP_{subject}_pre_mean.npy"
    post_path = POST_DIR / f"ERSP_{subject}_post.npy"

    try:
        pre_array = np.load(pre_path, allow_pickle=True)
        post_array = np.load(post_path, allow_pickle=True)
    except Exception as exc:
        print(f"[SKIP] {subject}: failed to load: {exc}")
        return None, None

    if pre_array.ndim != 2:
        print(
            f"[SKIP] {subject}: expected pre shape region x frequency, "
            f"got {pre_array.shape}: {pre_path}"
        )
        return None, None

    if post_array.ndim != 3:
        print(
            f"[SKIP] {subject}: expected post shape trial x region x frequency, "
            f"got {post_array.shape}: {post_path}"
        )
        return None, None

    if pre_array.shape[0] != 4 or post_array.shape[1] != 4:
        print(
            f"[SKIP] {subject}: expected four brain regions, "
            f"got pre {pre_array.shape} and post {post_array.shape}."
        )
        return None, None

    if pre_array.shape[1] != len(ERSP_F) or post_array.shape[2] != len(ERSP_F):
        print(
            f"[SKIP] {subject}: frequency dimension does not match ERSP_f.npy. "
            f"Pre={pre_array.shape}, Post={post_array.shape}, "
            f"n_frequencies={len(ERSP_F)}."
        )
        return None, None

    return pre_array, post_array


def compute_wholebrain_values(
    subjects: list[str],
    bands: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Compute one whole-brain Pre and Post value per subject and band."""
    rows = []
    all_wholebrain_values = []

    for subject in subjects:
        pre_array, post_array = load_subject_arrays(subject)
        if pre_array is None or post_array is None:
            continue

        for band_name, band_indices in bands.items():
            # pre_array shape: region x frequency
            # One value per region, averaged across the selected frequency bins.
            pre_values_by_region = np.nanmean(
                pre_array[:, band_indices],
                axis=1,
            )

            # post_array shape: trial x region x frequency
            # One value per region, averaged across post trials and frequency bins.
            post_values_by_region = np.nanmean(
                post_array[:, :, band_indices],
                axis=(0, 2),
            )

            pre_wholebrain = float(np.nanmean(pre_values_by_region))
            post_wholebrain = float(np.nanmean(post_values_by_region))
            difference = post_wholebrain - pre_wholebrain

            rows.append(
                [
                    subject,
                    band_name,
                    "wholebrain",
                    pre_wholebrain,
                    post_wholebrain,
                    difference,
                ]
            )
            all_wholebrain_values.extend([pre_wholebrain, post_wholebrain])

    values_df = pd.DataFrame(
        rows,
        columns=["Subject", "Band", "Region", "Pre", "Post", "Diff"],
    )

    finite_values = np.asarray(all_wholebrain_values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]

    if finite_values.size == 0:
        unified_ylim = (0.0, 1.0)
    else:
        ymin = float(np.min(finite_values))
        ymax = float(np.max(finite_values))

        if math.isclose(ymin, ymax):
            padding = 1e-6 if ymin == 0 else abs(ymin) * 0.05
            unified_ylim = (ymin - padding, ymax + padding)
        else:
            value_range = ymax - ymin
            unified_ylim = (
                ymin - 0.05 * value_range,
                ymax + 0.05 * value_range,
            )

    return values_df, unified_ylim


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction without extra dependencies."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    valid_mask = np.isfinite(p_values)
    valid_p = p_values[valid_mask]

    if valid_p.size == 0:
        return adjusted

    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    n_tests = ranked_p.size

    ranked_adjusted = ranked_p * n_tests / np.arange(1, n_tests + 1)
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0.0, 1.0)

    valid_adjusted = np.empty_like(ranked_adjusted)
    valid_adjusted[order] = ranked_adjusted
    adjusted[valid_mask] = valid_adjusted
    return adjusted


def compute_pre_post_statistics(
    values_df: pd.DataFrame,
    band_order: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run paired t-tests for Pre versus Post and FDR-correct five bands."""
    rows = []

    for band_name in band_order:
        band_df = values_df[
            (values_df["Band"] == band_name)
            & (values_df["Region"] == "wholebrain")
        ][["Subject", "Pre", "Post"]].dropna()

        pre_values = band_df["Pre"].astype(float).to_numpy()
        post_values = band_df["Post"].astype(float).to_numpy()
        differences = post_values - pre_values
        n_subjects = differences.size

        if n_subjects < 2:
            t_value = np.nan
            p_value = np.nan
            mean_difference = np.nan
            ci_low = np.nan
            ci_high = np.nan
            cohens_dz = np.nan
        else:
            test_result = ttest_rel(post_values, pre_values, nan_policy="omit")
            t_value = float(test_result.statistic)
            p_value = float(test_result.pvalue)
            mean_difference = float(np.mean(differences))
            difference_sd = float(np.std(differences, ddof=1))
            difference_se = difference_sd / np.sqrt(n_subjects)
            t_critical = float(t.ppf(1.0 - alpha / 2.0, df=n_subjects - 1))
            ci_low = mean_difference - t_critical * difference_se
            ci_high = mean_difference + t_critical * difference_se
            cohens_dz = (
                mean_difference / difference_sd
                if difference_sd > 0
                else np.nan
            )

        rows.append(
            {
                "Band": band_name,
                "N": n_subjects,
                "Pre_Mean": float(np.mean(pre_values)) if n_subjects else np.nan,
                "Pre_SD": float(np.std(pre_values, ddof=1)) if n_subjects > 1 else np.nan,
                "Post_Mean": float(np.mean(post_values)) if n_subjects else np.nan,
                "Post_SD": float(np.std(post_values, ddof=1)) if n_subjects > 1 else np.nan,
                "Mean_Difference_Post_minus_Pre": mean_difference,
                "Difference_95CI_Low": ci_low,
                "Difference_95CI_High": ci_high,
                "t": t_value,
                "df": n_subjects - 1 if n_subjects >= 2 else np.nan,
                "p_uncorrected": p_value,
                "Cohens_dz": cohens_dz,
            }
        )

    statistics_df = pd.DataFrame(rows)
    statistics_df["p_FDR_BH"] = benjamini_hochberg(
        statistics_df["p_uncorrected"].to_numpy()
    )
    statistics_df["Significant_FDR_0.05"] = (
        statistics_df["p_FDR_BH"] < alpha
    )
    return statistics_df


def p_to_symbol(p_value: float) -> str:
    """Convert an FDR-adjusted p value to a figure annotation."""
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def plot_wholebrain_all_bands(
    values_df: pd.DataFrame,
    statistics_df: pd.DataFrame,
    band_order: list[str],
    ylabel: str,
    unified_ylim: tuple[float, float],
    save_prefix: str,
) -> None:
    """Draw paired Pre/Post boxplots for all five frequency bands."""
    if values_df.empty:
        raise RuntimeError(
            "No subject data were loaded. Check DATA_ROOT and the saved filenames."
        )

    fig, ax = plt.subplots(figsize=(12, 5), dpi=600)

    original_ymin, original_ymax = unified_ylim
    original_range = original_ymax - original_ymin
    if original_range <= 0:
        original_range = 1.0
    annotation_space = 0.22 * original_range
    ax.set_ylim(original_ymin, original_ymax + annotation_space)

    centers = FIRST_GROUP_OFFSET + np.arange(len(band_order)) * GROUP_SPACING
    # ax.set_xlim(-0.5, centers[-1] + 1.0)

    median_properties = {"color": "black", "linewidth": BOX_LINEWIDTH}
    pre_color = "#F0F0F0"
    post_color = "#A0A0A0"

    # Reproducible horizontal jitter for individual data points
    rng = np.random.default_rng(42)
    
    all_point_x = []
    for index, band_name in enumerate(band_order):
        center = centers[index]
        band_df = values_df[
            (values_df["Band"] == band_name)
            & (values_df["Region"] == "wholebrain")
        ][["Subject", "Pre", "Post"]].dropna()

        pre_values = band_df["Pre"].astype(float).to_numpy()
        post_values = band_df["Post"].astype(float).to_numpy()

        if pre_values.size == 0 or post_values.size == 0:
            print(f"[WARN] No complete Pre/Post values for {band_name}; boxplot omitted.")
            continue

        positions = [
            center - INTRA_GROUP_OFFSET,
            center + INTRA_GROUP_OFFSET,
        ]

        boxplot = ax.boxplot(
            [pre_values, post_values],
            positions=positions,
            widths=0.45,
            showfliers=False,
            patch_artist=True,
            medianprops=median_properties,
        )

        for patch, color in zip(boxplot["boxes"], [pre_color, post_color]):
            patch.set_facecolor(color)

        for element in ["boxes", "whiskers", "caps"]:
            for item in boxplot[element]:
                item.set_linewidth(BOX_LINEWIDTH)

        statistics_row = statistics_df[statistics_df["Band"] == band_name]
        
        # Individual data points placed beside the boxplots
        for condition_idx, (x_center, values) in enumerate(
            zip(positions, [pre_values, post_values])
        ):
        
            # Pre: left of the box; Post: right of the box
            direction = -1 if condition_idx == 0 else 1
        
            side_offset = 0.48
            jitter_width = 0.10
        
            jitter = rng.uniform(
                -jitter_width,
                jitter_width,
                size=len(values)
            )
        
            x_points = x_center + direction * side_offset + jitter
            
            all_point_x.extend(x_points)
        
            ax.scatter(
                x_points,
                values,
                s=9,
                color="#666666",
                alpha=0.55,
                edgecolors="none",
                zorder=4
            )
        
        if not statistics_row.empty:
            q_value = float(statistics_row.iloc[0]["p_FDR_BH"])
        
            # Only annotate FDR-significant comparisons
            if np.isfinite(q_value) and q_value < 0.05:
                symbol = p_to_symbol(q_value)
        
                pair_max = np.nanmax(np.concatenate([pre_values, post_values]))
                line_y = pair_max + 0.07 * original_range
                cap_height = 0.02 * original_range
        
                ax.plot(
                    [positions[0], positions[0], positions[1], positions[1]],
                    [line_y, line_y + cap_height, line_y + cap_height, line_y],
                    color="black",
                    linewidth=BOX_LINEWIDTH,
                )
        
                ax.text(
                    center,
                    line_y + cap_height + 0.01 * original_range,
                    symbol,
                    ha="center",
                    va="bottom",
                    fontsize=14,
                )

    if all_point_x:
        ax.set_xlim(min(all_point_x) - 0.15, max(all_point_x) + 0.15)
    else:
        ax.set_xlim(centers[0] - 0.8, centers[-1] + 0.8)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(BOX_LINEWIDTH)

    ax.set_ylabel(ylabel)

    pre_post_positions = []
    for center in centers:
        pre_post_positions.extend(
            [center - INTRA_GROUP_OFFSET, center + INTRA_GROUP_OFFSET]
        )

    ax.set_xticks(pre_post_positions, minor=True)
    ax.tick_params(axis="x", which="minor", labelsize=14)
    ax.set_xticklabels(["Pre", "Post"] * len(band_order), minor=True)

    ax.set_xticks(centers)
    ax.set_xticklabels(band_order)
    ax.tick_params(axis="x", which="major", pad=20, labelsize=14)

    plt.tight_layout()

    for extension in ["png", "pdf", "svg"]:
        output_path = OUT_ROOT / f"{save_prefix}.{extension}"
        fig.savefig(output_path)
        print(f"[FIG] {output_path}")

    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    band_order = ["Delta", "Theta", "Alpha", "Beta1", "Beta2"]
    values_df, unified_ylim = compute_wholebrain_values(SUBJECTS, BANDS)

    statistics_df = compute_pre_post_statistics(
        values_df=values_df,
        band_order=band_order,
    )

    statistics_csv = OUT_ROOT / "wholebrain_pre_post_paired_statistics.csv"
    statistics_df.to_csv(statistics_csv, index=False)
    print(f"[STATS] {statistics_csv}")
    print(
        statistics_df[
            ["Band", "N", "t", "df", "p_uncorrected", "p_FDR_BH", "Cohens_dz"]
        ].to_string(index=False)
    )

    # Save the subject-level values used in the figure for later checking.
    values_csv = OUT_ROOT / "wholebrain_pre_post_values.csv"
    values_df.to_csv(values_csv, index=False)
    print(f"[DATA] {values_csv}")

    plot_wholebrain_all_bands(
        values_df=values_df,
        statistics_df=statistics_df,
        band_order=band_order,
        ylabel="Mean power",
        unified_ylim=unified_ylim,
        save_prefix="Figure_3_pre_vs_post_power",
    )

    print("[DONE] Figure 3 generated successfully.")


if __name__ == "__main__":
    main()
