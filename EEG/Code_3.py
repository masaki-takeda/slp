#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster-based permutation tests and time-frequency heatmaps for
Figure 4, Figure S5, and Figure S6.

This script performs one-sample cluster-based permutation tests on
post-minus-pre EEG spectral power for the deep-NREM and shallow-NREM groups.

The analyses are performed separately for four brain regions:
frontal, parietal, temporal, and occipital.

Input data
----------
The script is located in SLP_dataset/code/.

SLP_dataset/
    code/
        Figure_4_S5_S6_permutation_test_and_heatmap.py

    ERSP_f.npy

    Figure_4_source_data/
        ERSP_SubXX.npy

    Figure_S5_source_data/
        ERSP_SubXX.npy

    Figure_S6_source_data/
        ERSP_SubXX.npy

Each subject-level array has the shape:
    trials × brain regions × frequencies

The three figures correspond to different analysis periods:
    Figure 4:  full trial
    Figure S5: encoding period
    Figure S6: test period (fixed 8-second window)

Output
------
The script generates combined time-frequency heatmaps for each figure,
showing four brain regions and two sleep-depth groups.

Significant clusters are displayed in color over grayscale
statistical maps.

The deep-NREM significant-cluster masks are also saved for
reproducing Figure 5.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

from scipy.stats import t
from mne.stats import permutation_cluster_1samp_test


# =============================================================================
# Configuration
# =============================================================================

PVAL = 0.01
N_PERMUTATIONS = 5000
RANDOM_SEED = 42

GLOBAL_VMIN = -5
GLOBAL_VMAX = 5

CLUSTER_DISPLAY_P = 0.05
MIN_CLUSTER_TIME_SPAN = 3

REGIONS = ["Frontal", "Parietal", "Temporal", "Occipital"]
GROUP_TITLE = {"deep": "deep-NREM", "shallow": "shallow-NREM"}


PERIODS = ["full_trial", "encoding", "retrieval_fixed8"]

PERIOD_MAIN_TITLE = {
    "full_trial": None,
    "encoding": "Digit sequence presentation (encoding) period",
    "retrieval_fixed8": "Test period",
}

FIGURE_BASENAME_BY_PERIOD = {
    "full_trial": "Figure_4",
    "encoding": "Figure_S5",
    "retrieval_fixed8": "Figure_S6",
}


SHALLOW_IDS = [
    "Sub01", "Sub03", "Sub04", "Sub05", "Sub09", "Sub12", "Sub13",
    "Sub24", "Sub25", "Sub26", "Sub28", "Sub30", "Sub32", "Sub37",
    "Sub41", "Sub47", "Sub50", "Sub54", "Sub55", "Sub62", "Sub64",
]

DEEP_IDS = [
    "Sub02", "Sub06", "Sub08", "Sub15", "Sub17", "Sub18", "Sub21",
    "Sub22", "Sub29", "Sub31", "Sub36", "Sub38", "Sub39", "Sub40",
    "Sub43", "Sub45", "Sub46", "Sub48", "Sub51", "Sub53", "Sub59",
    "Sub61", "Sub63",
]


# =============================================================================
# Repository-relative paths
# =============================================================================

# Script location
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent

# Figure source data directories
SOURCE_DATA_DIRS = {
    "full_trial": DATASET_DIR / "Figure_4_source_data",
    "encoding": DATASET_DIR / "Figure_S5_source_data",
    "retrieval_fixed8": DATASET_DIR / "Figure_S6_source_data",
}

# Shared frequency vector
FREQ_FILE = DATASET_DIR / "ERSP_f.npy"

# Output directory
RESULTS_DIR = DATASET_DIR / "results"

OUTPUT_ROOT = RESULTS_DIR

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

FIGURE_FOLDER_BY_PERIOD = {
    "full_trial": "Figure_4",
    "encoding": "Figure_S5",
    "retrieval_fixed8": "Figure_S6",
}


def make_output_dir(output_root: Path, period: str) -> Path:
    """Create the output directory for Figure 4, S5, or S6."""

    output_dir = output_root / FIGURE_FOLDER_BY_PERIOD[period]
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


# =============================================================================
# Data helpers
# =============================================================================

def load_frequency_vector(freq_file: Path) -> np.ndarray:
    return np.asarray(np.load(freq_file, allow_pickle=True), dtype=float)


def broad_frequency_indices(ersp_f: np.ndarray) -> list[int]:
    return [i for i, x in enumerate(ersp_f) if 1 < x < 30]


def get_available_subjects(subject_ids: list[str], data_dir: Path) -> list[str]:
    return [sub for sub in subject_ids if (data_dir / f"ERSP_{sub}.npy").exists()]


def load_group_data(subject_ids: list[str], data_dir: Path) -> np.ndarray:
    arrays = []
    for sub in subject_ids:
        file_path = data_dir / f"ERSP_{sub}.npy"
        data = np.load(file_path, allow_pickle=True)
        if data.ndim != 3:
            raise ValueError(
                f"{file_path} should be 3D: trial x region x frequency. Got {data.shape}."
            )
        if data.shape[1] != len(REGIONS):
            raise ValueError(
                f"{file_path}: region dimension should be {len(REGIONS)}. Got {data.shape}."
            )
        arrays.append(data)

    if not arrays:
        raise ValueError(f"No subject data found in {data_dir}")

    return np.stack(arrays, axis=0)


def validate_group_data(shallow_data: np.ndarray, deep_data: np.ndarray) -> None:
    print("  Shallow data shape:", shallow_data.shape)
    print("  Deep data shape:   ", deep_data.shape)

    if shallow_data.shape[1:] != deep_data.shape[1:]:
        raise ValueError(f"Shape mismatch: {shallow_data.shape} vs {deep_data.shape}")

    print("  Shallow NaN values:", int(np.isnan(shallow_data).sum()))
    print("  Deep NaN values:   ", int(np.isnan(deep_data).sum()))
    print("  Shallow inf values:", int(np.isinf(shallow_data).sum()))
    print("  Deep inf values:   ", int(np.isinf(deep_data).sum()))


def calc_threshold_1samp(pval: float, n_freqs: int, n_trials: int) -> float:
    df = n_freqs * n_trials - 1
    return t.ppf(1 - pval / 2, df=df)


# =============================================================================
# Plot helpers
# =============================================================================

def mask_significant_clusters(
    stat_obs: np.ndarray,
    clusters: list,
    cluster_pvals: np.ndarray,
) -> np.ndarray:
    mask = np.full(stat_obs.shape, np.nan)

    for cluster, p_value in zip(clusters, cluster_pvals):
        if len(cluster) < 2:
            continue
        time_span = max(cluster[1]) - min(cluster[1])
        if p_value <= CLUSTER_DISPLAY_P and time_span >= MIN_CLUSTER_TIME_SPAN:
            mask[cluster] = stat_obs[cluster]

    return mask


def format_subplot_title(region: str, group_key: str) -> str:
    return f"{region} region – {GROUP_TITLE[group_key]} group"


def style_heatmap_axis(
    ax,
    n_trials: int,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    ax.set_xlim(1, n_trials)
    ax.set_ylim(1, 30)
    ax.set_yticks([1, 10, 20, 30])
    ax.set_yticklabels(["1", "10", "20", "30"])

    x_ticks = [1] + list(range(10, n_trials + 1, 10))
    ax.set_xticks(x_ticks)
    if show_xlabel:
        ax.set_xlabel("Trials", fontsize=9)
        ax.set_xticklabels([str(x) for x in x_ticks])
        ax.tick_params(axis="x", labelbottom=True)
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)

    if show_ylabel:
        ax.set_ylabel("Frequency (Hz)", fontsize=9)
    else:
        ax.set_ylabel("")

    ax.tick_params(axis="both", labelsize=8, width=0.5, length=2, direction="out")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.10,
        1.15,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )


def plot_combined_heatmap(
    combined: dict[str, dict[str, tuple[np.ndarray, list, np.ndarray]]],
    ersp_f: np.ndarray,
    broad: list[int],
    period: str,
    figures_dir: Path,
) -> None:
    """
    Save a 2 x 2 regional panel for Figure 4 / Figure S5 / Figure S6.
    Each regional panel contains deep-NREM on top and shallow-NREM on bottom.
    """
    first_region = next(iter(combined.values()))
    n_trials = first_region["deep"][0].shape[1]
    trial_axis = np.arange(1, n_trials + 1)
    freq_axis = ersp_f[broad]
    norm = mcolors.Normalize(vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX)

    main_title = PERIOD_MAIN_TITLE.get(period)
    top = 0.96 if main_title is None else 0.89

    fig = plt.figure(figsize=(10, 8), dpi=600)
    outer_gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1, 1],
        width_ratios=[1, 1],
        hspace=0.35,
        wspace=0.225,
        right=0.84,
        top=top,
    )


    gray_mesh = None
    color_mesh = None
    panel_labels = ["A", "B", "C", "D"]

    for panel_idx, region in enumerate(REGIONS):
        inner_gs = outer_gs[panel_idx // 2, panel_idx % 2].subgridspec(2, 1, hspace=0.35)
        axes_pair = []

        for row_idx, group_key in enumerate(["deep", "shallow"]):
            ax = fig.add_subplot(inner_gs[row_idx, 0])
            axes_pair.append(ax)

            stat_obs, clusters, pvals = combined[region][group_key]
            mask = mask_significant_clusters(stat_obs, clusters, pvals)

            gray_mesh = ax.pcolormesh(
                trial_axis,
                freq_axis,
                stat_obs,
                cmap="gray",
                shading="auto",
                norm=norm,
            )
            color_mesh = ax.pcolormesh(
                trial_axis,
                freq_axis,
                mask,
                cmap="RdBu_r",
                shading="auto",
                norm=norm,
            )

            ax.set_title(format_subplot_title(region, group_key), fontsize=10, pad=6)
            style_heatmap_axis(
                ax,
                n_trials=n_trials,
                show_xlabel=(row_idx == 1),
                show_ylabel=(panel_idx % 2 == 0),
            )

        add_panel_label(axes_pair[0], panel_labels[panel_idx])

    if main_title is not None:
        fig.suptitle(main_title, fontsize=12, y=0.972)

    if gray_mesh is not None and color_mesh is not None:
        cbar_gray_ax = fig.add_axes([0.88, 0.16, 0.018, 0.68])
        cbar_gray = fig.colorbar(gray_mesh, cax=cbar_gray_ax)
        cbar_gray.set_ticks([])
        cbar_gray.ax.tick_params(width=0.5, length=0)
        cbar_gray.outline.set_linewidth(0.8)

        cbar_color_ax = fig.add_axes([0.91, 0.16, 0.018, 0.68])
        cbar_color = fig.colorbar(color_mesh, cax=cbar_color_ax)
        cbar_color.set_label(r"$\it{t}$ value", rotation=270, labelpad=13, fontsize=9)
        cbar_color.set_ticks(np.linspace(GLOBAL_VMIN, GLOBAL_VMAX, 5))
        cbar_color.ax.tick_params(labelsize=8, width=0.5, length=2)
        cbar_color.outline.set_linewidth(0.8)

    base_name = FIGURE_BASENAME_BY_PERIOD[period]
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(figures_dir / f"{base_name}.{ext}", dpi=600, bbox_inches="tight")

    plt.close(fig)


# =============================================================================
# Main analysis
# =============================================================================

def run_cluster_permutation_for_period(
    period: str,
    output_root: Path,
    freq_file: Path,
    n_permutations: int = N_PERMUTATIONS,
) -> None:

    # Select source data according to the analysis period
    data_dir = SOURCE_DATA_DIRS[period]
    period_output_dir = make_output_dir(output_root, period)
    figures_dir = period_output_dir

    print("\n" + "=" * 80)
    print(f"Period: {period}")
    print(f"Data dir: {data_dir}")
    print(f"Figures dir: {figures_dir}")

    if not data_dir.exists():
        print(f"  SKIP: data directory not found: {data_dir}")
        return

    shallow_subjects = get_available_subjects(SHALLOW_IDS, data_dir)
    deep_subjects = get_available_subjects(DEEP_IDS, data_dir)

    missing_shallow = sorted(set(SHALLOW_IDS) - set(shallow_subjects))
    missing_deep = sorted(set(DEEP_IDS) - set(deep_subjects))

    print(f"  Available shallow subjects: {len(shallow_subjects)}/{len(SHALLOW_IDS)}")
    print(f"  Available deep subjects:    {len(deep_subjects)}/{len(DEEP_IDS)}")
    if missing_shallow:
        print("  Missing shallow:", ", ".join(missing_shallow))
    if missing_deep:
        print("  Missing deep:   ", ", ".join(missing_deep))

    if len(shallow_subjects) < 2 or len(deep_subjects) < 2:
        print("  SKIP: not enough subjects for permutation test.")
        return

    ersp_f = load_frequency_vector(freq_file)
    broad = broad_frequency_indices(ersp_f)

    print(f"  Broad frequency bins: {len(broad)}")
    print(f"  Frequency range: {ersp_f[broad][0]:.3f} to {ersp_f[broad][-1]:.3f} Hz")

    shallow_data = load_group_data(shallow_subjects, data_dir)
    deep_data = load_group_data(deep_subjects, data_dir)
    validate_group_data(shallow_data, deep_data)

    n_trials = shallow_data.shape[1]
    threshold_t = calc_threshold_1samp(PVAL, len(broad), n_trials)

    print(f"  One-sample cluster-forming t threshold: {threshold_t:.6f}")
    print(f"  n_permutations: {n_permutations}")

    combined_heatmaps = {}

    for region_idx, region in enumerate(REGIONS):
        print(f"\n  Region: {region}")

        # Input shape: subject x trial x region x frequency
        # Test input shape: subject x frequency x trial
        shallow_region = shallow_data[:, :, region_idx, :][:, :, broad].transpose(0, 2, 1)
        deep_region = deep_data[:, :, region_idx, :][:, :, broad].transpose(0, 2, 1)


        shallow_region = np.nan_to_num(shallow_region, nan=0.0, posinf=0.0, neginf=0.0)
        deep_region = np.nan_to_num(deep_region, nan=0.0, posinf=0.0, neginf=0.0)

        stat_shallow, clusters_shallow, pvals_shallow, _ = permutation_cluster_1samp_test(
            shallow_region,
            n_permutations=n_permutations,
            threshold=threshold_t,
            seed=RANDOM_SEED,
            tail=0,
        )

        stat_deep, clusters_deep, pvals_deep, _ = permutation_cluster_1samp_test(
            deep_region,
            n_permutations=n_permutations,
            threshold=threshold_t,
            seed=RANDOM_SEED,
            tail=0,
        )

        combined_heatmaps[region] = {
            "shallow": (stat_shallow, clusters_shallow, pvals_shallow),
            "deep": (stat_deep, clusters_deep, pvals_deep),
        }

        # Save the displayed deep-NREM cluster masks for Figure 5.
        # Only the full-trial analysis is used to generate Figure 5.
        if period == "full_trial":
        
            deep_cluster_mask = mask_significant_clusters(
                stat_deep,
                clusters_deep,
                pvals_deep
            )
        
            mask_path = (
                period_output_dir
                / f"deep_cluster_mask_{region.lower()}.npy"
            )
        
            np.save(mask_path, deep_cluster_mask)
            print(f"  Saved cluster mask: {mask_path}")

        n_shallow_clusters = int(np.sum(np.asarray(pvals_shallow) <= CLUSTER_DISPLAY_P))
        n_deep_clusters = int(np.sum(np.asarray(pvals_deep) <= CLUSTER_DISPLAY_P))
        print(f"    p<={CLUSTER_DISPLAY_P} clusters: shallow={n_shallow_clusters}, deep={n_deep_clusters}")

    plot_combined_heatmap(
        combined=combined_heatmaps,
        ersp_f=ersp_f,
        broad=broad,
        period=period,
        figures_dir=figures_dir,
    )

    print(f"\n  Combined figure saved to: {figures_dir}")


def run_all(
    periods: Iterable[str],
    output_root: Path,
    freq_file: Path,
    n_permutations: int = N_PERMUTATIONS,
) -> None:
    
    for period in periods:
        if period not in PERIODS:
            raise ValueError(f"Unknown period: {period}. Valid periods: {PERIODS}")

        run_cluster_permutation_for_period(
            period=period,
            output_root=output_root,
            freq_file=freq_file,
            n_permutations=n_permutations,
        )


def parse_list(values: list[str] | None, valid_values: list[str]) -> list[str]:
    if values is None or len(values) == 0:
        return valid_values

    out = []
    for value in values:
        parts = [p.strip() for p in str(value).split(",") if p.strip()]
        out.extend(parts)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one-sample cluster permutation tests and save combined heatmaps."
    )
    parser.add_argument(
        "--periods",
        nargs="*",
        default=None,
        help=f"Periods to run. Valid: {PERIODS}",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=N_PERMUTATIONS,
        help="Number of permutations.",
    )
    args = parser.parse_args()

    periods = parse_list(args.periods, PERIODS)

    # Check that all requested source-data directories exist
    for period in periods:
        source_dir = SOURCE_DATA_DIRS[period]
    
        if not source_dir.is_dir():
            raise FileNotFoundError(
                f"Source data directory not found: {source_dir}"
            )
    
    # Check the shared frequency vector
    if not FREQ_FILE.is_file():
        raise FileNotFoundError(
            f"Frequency file not found: {FREQ_FILE}. "
            "Please place the original ERSP_f.npy in SLP_dataset/."
        )

    print("Source data directories:")
    
    for period in periods:
        print(f"  {period}: {SOURCE_DATA_DIRS[period]}")
    
    print("Output root:", OUTPUT_ROOT)
    print("Frequency file:", FREQ_FILE)
    print("Periods:", periods)
    print("n_permutations:", args.n_permutations)

    run_all(
        periods=periods,

        output_root=OUTPUT_ROOT,
        freq_file=FREQ_FILE,
        n_permutations=args.n_permutations,
    )

    print("\nAll requested permutation figures are complete.")


if __name__ == "__main__":
    main()
