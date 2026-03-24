"""Structured output path generation for organized file storage.

This module provides centralized utilities for generating organized output paths
across different analysis modes (no_split, year_split, period4, etc.).
"""

from __future__ import annotations

import os
from typing import Dict, Optional


def sanitize_task_name(task: str) -> str:
    """Sanitize task name for use in filenames.

    Args:
        task: Raw task name (may contain spaces, special chars)

    Returns:
        Sanitized task name safe for filenames
    """
    # Replace common separators
    name = task.replace(" ", "_")
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    name = name.replace(":", "_")
    name = name.replace("*", "_")
    name = name.replace("?", "_")
    name = name.replace('"', "_")
    name = name.replace("<", "_")
    name = name.replace(">", "_")
    name = name.replace("|", "_")
    return name


class SigmoidOutputPaths:
    """Path generator for sigmoid frontier outputs."""

    def __init__(
        self,
        base_dir: str,
        mode: str = "no_split",
        train_mode: Optional[str] = None,
        legacy: bool = False,
    ):
        """Initialize path generator.

        Args:
            base_dir: Base output directory
            mode: Split mode ('no_split', 'year_split', 'period4')
            train_mode: For period4: 'cumulative' or 'single_k'
            legacy: Use legacy flat structure if True
        """
        self.base_dir = base_dir
        self.mode = mode
        self.train_mode = train_mode or "cumulative"
        self.legacy = legacy

    def get_root(self) -> str:
        """Get root directory for this mode."""
        if self.legacy:
            return self.base_dir

        if self.mode == "no_split":
            return os.path.join(self.base_dir, "sigmoid", "no_split")
        elif self.mode == "year_split":
            return os.path.join(self.base_dir, "sigmoid", "year_split")
        elif self.mode == "period4":
            return os.path.join(
                self.base_dir, "sigmoid", "period4", self.train_mode
            )
        else:
            return self.base_dir

    def get_plot_path(self, task: str, suffix: str = "") -> str:
        """Get path for plot file.

        Args:
            task: Task name
            suffix: Optional suffix (e.g., 'split', 'period4')

        Returns:
            Path for plot file (without extension)
        """
        task_clean = sanitize_task_name(task)
        root = self.get_root()

        if self.legacy:
            # Legacy flat structure
            if suffix:
                return os.path.join(root, f"smooth_frontier_{suffix}__{task}")
            return os.path.join(root, f"smooth_frontier__{task}")

        # New structured path
        plots_dir = os.path.join(root, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        if suffix:
            return os.path.join(plots_dir, f"{task_clean}_{suffix}")
        return os.path.join(plots_dir, task_clean)

    def get_curve_path(
        self, task: str, group: Optional[str] = None, k: Optional[int] = None
    ) -> str:
        """Get path for curve CSV file.

        Args:
            task: Task name
            group: Group label (e.g., 'pre2025', '2025')
            k: Period index for period4 mode

        Returns:
            Path for curve CSV file
        """
        task_clean = sanitize_task_name(task)
        root = self.get_root()

        if self.legacy:
            # Legacy flat structure
            if k is not None:
                return os.path.join(root, f"smooth_frontier__{task}__k{k}.csv")
            elif group:
                group_clean = group.replace(" ", "_")
                return os.path.join(
                    root, f"smooth_frontier__{task}__{group_clean}.csv"
                )
            return os.path.join(root, f"smooth_frontier__{task}.csv")

        # New structured path
        curves_dir = os.path.join(root, "curves")
        os.makedirs(curves_dir, exist_ok=True)

        if k is not None:
            return os.path.join(curves_dir, f"{task_clean}_k{k}.csv")
        elif group:
            group_clean = group.replace(" ", "_").lower()
            return os.path.join(curves_dir, f"{task_clean}_{group_clean}.csv")
        return os.path.join(curves_dir, f"{task_clean}.csv")

    def get_manifest_dir(self, k: Optional[int] = None) -> str:
        """Get directory for manifest files.

        Args:
            k: Period index for period4 mode

        Returns:
            Directory path for manifests
        """
        root = self.get_root()

        if self.legacy:
            if k is not None:
                manifest_dir = os.path.join(root, f"k{k}")
            else:
                manifest_dir = root
        else:
            if k is not None:
                manifest_dir = os.path.join(root, "manifests", f"k{k}")
            else:
                manifest_dir = os.path.join(root, "manifests")

        os.makedirs(manifest_dir, exist_ok=True)
        return manifest_dir


class EvaluationOutputPaths:
    """Path generator for evaluation outputs."""

    def __init__(
        self,
        base_dir: str,
        mode: str = "year_split",
        train_mode: Optional[str] = None,
        legacy: bool = False,
    ):
        """Initialize path generator.

        Args:
            base_dir: Base output directory
            mode: Split mode ('year_split' or 'period4')
            train_mode: For period4: 'cumulative' or 'single_k'
            legacy: Use legacy flat structure if True
        """
        self.base_dir = base_dir
        self.mode = mode
        self.train_mode = train_mode or "cumulative"
        self.legacy = legacy

    def get_root(self, k: Optional[int] = None) -> str:
        """Get root directory for evaluation outputs.

        Args:
            k: Period index for period4 mode

        Returns:
            Root directory path
        """
        if self.legacy:
            if k is not None:
                return os.path.join(self.base_dir, f"k{k}")
            return self.base_dir

        if self.mode == "year_split":
            return os.path.join(self.base_dir, "evaluation", "sigmoid", "year_split")
        elif self.mode == "period4":
            root = os.path.join(
                self.base_dir, "evaluation", "sigmoid", "period4", self.train_mode
            )
            if k is not None:
                root = os.path.join(root, f"k{k}")
            return root
        else:
            return self.base_dir

    def get_bins_path(self, task: str, era: str, k: Optional[int] = None) -> str:
        """Get path for bins CSV file.

        Args:
            task: Task name
            era: Era label ('train', 'test_overlap', etc.)
            k: Period index for period4 mode

        Returns:
            Path for bins CSV file
        """
        task_clean = sanitize_task_name(task)
        root = self.get_root(k)

        if self.legacy:
            return os.path.join(root, f"{task}__bins_{era}.csv")

        bins_dir = os.path.join(root, "bins")
        os.makedirs(bins_dir, exist_ok=True)
        return os.path.join(bins_dir, f"{task_clean}_bins_{era}.csv")

    def get_summary_path(self, task: str, k: Optional[int] = None) -> str:
        """Get path for task summary CSV file.

        Args:
            task: Task name
            k: Period index for period4 mode

        Returns:
            Path for summary CSV file
        """
        task_clean = sanitize_task_name(task)
        root = self.get_root(k)

        if self.legacy:
            return os.path.join(root, f"{task}__summary.csv")

        summaries_dir = os.path.join(root, "summaries")
        os.makedirs(summaries_dir, exist_ok=True)
        return os.path.join(summaries_dir, f"{task_clean}_summary.csv")

    def get_aggregate_path(self, k: Optional[int] = None) -> str:
        """Get path for aggregate summary CSV file.

        Args:
            k: Period index for period4 mode

        Returns:
            Path for aggregate summary CSV file
        """
        root = self.get_root(k)

        if self.legacy:
            return os.path.join(root, "summary_over_tasks.csv")

        aggregate_dir = os.path.join(root, "aggregate")
        os.makedirs(aggregate_dir, exist_ok=True)
        return os.path.join(aggregate_dir, "summary_over_tasks.csv")


class EvaluationRunPaths:
    """Path generator for a single evaluation run directory.

    This helper organizes evaluation outputs under a provided root directory by
    creating `bins/`, `summaries/`, and `aggregate/` subdirectories.

    Unlike `EvaluationOutputPaths`, this class does not impose a global
    `outputs/evaluation/sigmoid/...` layout; callers fully control `root_dir`.
    """

    def __init__(self, root_dir: str, legacy: bool = False):
        """Initialize path generator.

        Args:
            root_dir: Root directory for this evaluation run (e.g., an out_dir or k*/ folder)
            legacy: Use legacy flat structure if True
        """
        self.root_dir = root_dir
        self.legacy = legacy

    def get_root(self) -> str:
        """Get root directory for this evaluation run."""
        return self.root_dir

    def get_bins_path(self, task: str, era: str) -> str:
        """Get path for bins CSV file.

        Args:
            task: Task name
            era: Era label ('train', 'test_overlap', 'test_fixed', etc.)

        Returns:
            Path for bins CSV file
        """
        root = self.get_root()
        if self.legacy:
            task_legacy = task.replace("/", "_").replace("\\", "_")
            return os.path.join(root, f"{task_legacy}__bins_{era}.csv")

        task_clean = sanitize_task_name(task)
        bins_dir = os.path.join(root, "bins")
        os.makedirs(bins_dir, exist_ok=True)
        return os.path.join(bins_dir, f"{task_clean}_bins_{era}.csv")

    def get_summary_path(self, task: str) -> str:
        """Get path for task summary CSV file."""
        root = self.get_root()
        if self.legacy:
            task_legacy = task.replace("/", "_").replace("\\", "_")
            return os.path.join(root, f"{task_legacy}__summary.csv")

        task_clean = sanitize_task_name(task)
        summaries_dir = os.path.join(root, "summaries")
        os.makedirs(summaries_dir, exist_ok=True)
        return os.path.join(summaries_dir, f"{task_clean}_summary.csv")

    def get_aggregate_path(self) -> str:
        """Get path for aggregate summary CSV file."""
        root = self.get_root()
        if self.legacy:
            return os.path.join(root, "summary_over_tasks.csv")

        aggregate_dir = os.path.join(root, "aggregate")
        os.makedirs(aggregate_dir, exist_ok=True)
        return os.path.join(aggregate_dir, "summary_over_tasks.csv")


class FrontierOutputPaths:
    """Path generator for multi-skill frontier outputs."""

    def __init__(self, base_dir: str, dataset_name: str = "default", legacy: bool = False):
        """Initialize path generator.

        Args:
            base_dir: Base output directory
            dataset_name: Name of dataset (for organizing multiple datasets)
            legacy: Use legacy flat structure if True
        """
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.legacy = legacy

    def get_root(self) -> str:
        """Get root directory for frontier outputs."""
        if self.legacy:
            return self.base_dir
        return os.path.join(self.base_dir, "frontier", self.dataset_name)

    def get_csv_dir(self) -> str:
        """Get directory for CSV outputs."""
        root = self.get_root()
        if self.legacy:
            return root
        csv_dir = os.path.join(root, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        return csv_dir

    def get_vertices_dir(self) -> str:
        """Get directory for vertex CSV files."""
        csv_dir = self.get_csv_dir()
        if self.legacy:
            return csv_dir
        vertices_dir = os.path.join(csv_dir, "vertices")
        os.makedirs(vertices_dir, exist_ok=True)
        return vertices_dir

    def get_plot_dir(self, plot_type: str = "per_task") -> str:
        """Get directory for plot outputs.

        Args:
            plot_type: Type of plots ('per_task' or 'pairwise')

        Returns:
            Directory path for plots
        """
        root = self.get_root()
        if self.legacy:
            return os.path.join(root, "figures")

        plots_dir = os.path.join(root, "plots", plot_type)
        os.makedirs(plots_dir, exist_ok=True)
        return plots_dir


def get_budget_output_paths(
    base_dir: str,
    mode: str = "year_split",
    train_mode: Optional[str] = None,
    legacy: bool = False,
) -> Dict[str, str]:
    """Get output paths for budget-constrained design.

    Args:
        base_dir: Base output directory
        mode: Split mode ('year_split' or 'period4')
        train_mode: For period4: 'cumulative' or 'single_k'
        legacy: Use legacy flat structure if True

    Returns:
        Dictionary with paths for manifests, plots, curves
        Modes:
            - "no_split": use base_dir directly
            - "year_split": nested under budget/year_split
            - "period4": nested under budget/period4/<train_mode>
    """
    if legacy:
        return {
            "root": base_dir,
            "manifests": base_dir,
            "plots": os.path.join(base_dir, "plots"),
            "curves": os.path.join(base_dir, "plots"),
        }

    if mode == "no_split":
        root = base_dir
    elif mode == "year_split":
        root = os.path.join(base_dir, "budget", "year_split")
    elif mode == "period4":
        train_suffix = train_mode or "cumulative"
        root = os.path.join(base_dir, "budget", "period4", train_suffix)
    else:
        root = os.path.join(base_dir, "budget")

    return {
        "root": root,
        "manifests": os.path.join(root, "manifests"),
        "plots": os.path.join(root, "plots"),
        "curves": os.path.join(root, "curves"),
    }
