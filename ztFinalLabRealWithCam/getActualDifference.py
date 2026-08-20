#!/usr/bin/env python3

from pathlib import Path
import json
import re
import numpy as np
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.signal import find_peaks


# ============================================================
# CONFIGURATION
# ============================================================

# Change this to your experiment folder.
EXPERIMENT_FOLDER = Path("./mergedData/")

# Output files
RESULT_FILE = EXPERIMENT_FOLDER / "result.csv"
PROGRESS_FILE = EXPERIMENT_FOLDER / ".bifurcation_progress.json"

# Robot trajectory columns
X_COLUMN = "robot_x"
Y_COLUMN = "robot_y"

# DOA columns
DOA_COLUMNS = [f"doa_{i}" for i in range(120)]

# Column added to result.csv
SOURCE_COLUMN = "source_file"

# ------------------------------------------------------------
# Selection settings
# ------------------------------------------------------------

# Number of points that must be selected from each file.
MAX_SELECTIONS = 5

# Plot settings
MARKER_SIZE = 55
SELECTED_MARKER_SIZE = 190
PICK_RADIUS_POINTS = 10

# Fixed colours for special cases.
# These are deliberately NOT part of the continuous colour scale.
ONE_PEAK_COLOR = "black"
ZERO_PEAK_COLOR = "gray"

# Continuous colour map for two-peak points.
COLORMAP = "viridis"

# Maximum difference represented by the colour scale.
# Anything >= this value gets the same maximum colour.
MAX_COLOR_DIFFERENCE = 0.20


# ============================================================
# NATURAL SORTING
# ============================================================

def natural_sort_key(path):
    """
    Natural filename ordering.

    Example:
        experiment_2.csv
        experiment_10.csv

    rather than:
        experiment_10.csv
        experiment_2.csv
    """

    name = path.name.lower()

    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", name)
    ]


# ============================================================
# PROGRESS MANAGEMENT
# ============================================================

def load_progress():
    """
    Load progress from the progress JSON.

    New format:

    {
        "experiment_001.csv": {
            "status": "selected",
            "selected_indices": [12, 25, 31, 48, 72]
        },
        "experiment_002.csv": {
            "status": "skipped",
            "selected_indices": []
        }
    }

    For backwards compatibility, old entries such as:

        "experiment_001.csv": "selected"

    are also accepted.
    """

    if not PROGRESS_FILE.exists():
        return {}

    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

    except Exception as exc:
        print(
            f"Warning: could not read progress file: {exc}"
        )

    return {}


def save_progress(progress):
    """
    Save progress atomically.
    """

    temporary_file = PROGRESS_FILE.with_suffix(".tmp")

    with open(temporary_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)

    temporary_file.replace(PROGRESS_FILE)


def get_progress_status(progress, source_file):
    """
    Return the status stored in the progress dictionary.

    Supports both the new dictionary format and the old
    string format.
    """

    value = progress.get(source_file)

    if isinstance(value, dict):
        return value.get("status", "pending")

    if isinstance(value, str):
        return value

    return "pending"


def get_progress_selected_indices(progress, source_file):
    """
    Return saved selected original row indices for a file.

    Only the new progress format stores the indices.
    """

    value = progress.get(source_file)

    if isinstance(value, dict):
        indices = value.get("selected_indices", [])

        if isinstance(indices, list):
            try:
                return [int(index) for index in indices]
            except (TypeError, ValueError):
                return []

    return []


# ============================================================
# RESULT CSV
# ============================================================

def get_result_source_files():
    """
    Read result.csv and determine which experiments already have
    MAX_SELECTIONS saved bifurcation points.

    A file is considered completed only when it has at least
    MAX_SELECTIONS rows in result.csv.

    This prevents an old one-point-per-file result.csv from being
    incorrectly treated as complete.
    """

    if not RESULT_FILE.exists():
        return set()

    try:
        result = pd.read_csv(RESULT_FILE)

        if SOURCE_COLUMN not in result.columns:
            return set()

        counts = (
            result[SOURCE_COLUMN]
            .dropna()
            .astype(str)
            .value_counts()
        )

        return set(
            counts[
                counts >= MAX_SELECTIONS
            ].index
        )

    except Exception as exc:
        print(
            f"Warning: could not read existing "
            f"{RESULT_FILE.name}: {exc}"
        )

        return set()


def save_selected_row(row, source_file):
    """
    Append one complete original row plus source_file
    to result.csv.
    """

    output_row = row.to_frame().T.copy()

    output_row[SOURCE_COLUMN] = source_file

    if RESULT_FILE.exists():

        output_row.to_csv(
            RESULT_FILE,
            mode="a",
            header=False,
            index=False,
        )

    else:

        output_row.to_csv(
            RESULT_FILE,
            mode="w",
            header=True,
            index=False,
        )


# ============================================================
# DATA LOADING
# ============================================================

def load_experiment(csv_file):
    """
    Load one experiment CSV.
    """

    df = pd.read_csv(csv_file)

    required_columns = [
        X_COLUMN,
        Y_COLUMN,
        *DOA_COLUMNS,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{csv_file.name} is missing required columns: "
            f"{missing}"
        )

    if len(df) == 0:
        raise ValueError(
            f"{csv_file.name} contains no rows."
        )

    # Make trajectory columns numeric.
    df[X_COLUMN] = pd.to_numeric(
        df[X_COLUMN],
        errors="coerce",
    )

    df[Y_COLUMN] = pd.to_numeric(
        df[Y_COLUMN],
        errors="coerce",
    )

    # Make DOA columns numeric.
    for column in DOA_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Keep original row index.
    df = df.copy()
    df["_original_row_index"] = range(len(df))  

    # Only exclude rows that cannot be plotted.
    plot_df = df.dropna(
        subset=[X_COLUMN, Y_COLUMN]
    ).copy()

    if len(plot_df) == 0:
        raise ValueError(
            f"{csv_file.name} has no valid "
            f"{X_COLUMN}/{Y_COLUMN} points."
        )

    return df, plot_df


# ============================================================
# DOA PEAK ANALYSIS
# ============================================================

def analyze_doa_row(row):
    """
    Analyze doa_0 ... doa_119 for one trajectory row.

    Uses scipy.signal.find_peaks with DEFAULT settings.

    Returns a dictionary containing:

        number_of_peaks
        peak_indices
        peak_columns
        peak_values
        difference

    Interpretation:

        0 peaks:
            difference = None

        1 peak:
            difference = None

        >=2 peaks:
            take the two highest peaks by intensity
            and calculate:

                abs(peak1 - peak2)
    """

    doa_values = row[DOA_COLUMNS].to_numpy(
        dtype=float
    )

    # find_peaks does not treat NaN-containing arrays well.
    if not pd.isna(doa_values).all():

        finite_mask = ~pd.isna(doa_values)

        if not finite_mask.all():

            doa_values = doa_values.copy()

            # Replace NaNs by -inf so they cannot become peaks.
            doa_values[~finite_mask] = float("-inf")

    else:

        return {
            "number_of_peaks": 0,
            "peak_indices": [],
            "peak_columns": [],
            "peak_values": [],
            "difference": None,
        }

    # --------------------------------------------------------
    # DEFAULT find_peaks settings
    # --------------------------------------------------------

    peaks, properties = find_peaks(doa_values)

    if len(peaks) == 0:

        return {
            "number_of_peaks": 0,
            "peak_indices": [],
            "peak_columns": [],
            "peak_values": [],
            "difference": None,
        }

    # --------------------------------------------------------
    # Sort peaks by intensity, highest first.
    # --------------------------------------------------------

    peak_values_all = doa_values[peaks]

    order = sorted(
        range(len(peaks)),
        key=lambda i: peak_values_all[i],
        reverse=True,
    )

    sorted_peaks = [
        int(peaks[i])
        for i in order
    ]

    sorted_values = [
        float(peak_values_all[i])
        for i in order
    ]

    # --------------------------------------------------------
    # Exactly one peak
    # --------------------------------------------------------

    if len(sorted_peaks) == 1:

        peak_index = sorted_peaks[0]
        peak_value = sorted_values[0]

        return {
            "number_of_peaks": 1,
            "peak_indices": [peak_index],
            "peak_columns": [
                DOA_COLUMNS[peak_index]
            ],
            "peak_values": [peak_value],
            "difference": None,
        }

    # --------------------------------------------------------
    # Two or more peaks:
    # take the two highest.
    # --------------------------------------------------------

    peak1_index = sorted_peaks[0]
    peak2_index = sorted_peaks[1]

    peak1_value = sorted_values[0]
    peak2_value = sorted_values[1]

    difference = abs(
        peak1_value - peak2_value
    )

    return {
        "number_of_peaks": len(sorted_peaks),
        "peak_indices": [
            peak1_index,
            peak2_index,
        ],
        "peak_columns": [
            DOA_COLUMNS[peak1_index],
            DOA_COLUMNS[peak2_index],
        ],
        "peak_values": [
            peak1_value,
            peak2_value,
        ],
        "difference": difference,
    }


# ============================================================
# INTERACTIVE APPLICATION
# ============================================================

class BifurcationSelector:

    def __init__(self, csv_files):

        self.csv_files = csv_files

        self.progress = load_progress()

        self.saved_files = get_result_source_files()

        self.current_index = 0

        self.df = None
        self.plot_df = None
        self.current_file = None

        # DOA analysis for every plotted row.
        self.doa_analysis = []

        # Continuous colour values.
        self.differences = None

        # ----------------------------------------------------
        # MULTI-SELECTION
        # ----------------------------------------------------
        #
        # These are indices into plot_df.
        #
        # Example:
        #     [12, 25, 31, 48, 72]
        #
        # means five selected points in the current file.
        #
        self.selected_plot_indices = []

        # Matplotlib objects.
        self.fig = None
        self.ax = None
        self.scatter = None
        self.one_peak_scatter = None
        self.zero_peak_scatter = None
        self.selected_scatter = None
        self.info_text = None
        self.colorbar = None

        self.running = True

    # --------------------------------------------------------
    # FILE STATUS
    # --------------------------------------------------------

    def status(self, csv_file):

        name = csv_file.name

        if name in self.saved_files:
            return "selected"

        if get_progress_status(
            self.progress,
            name,
        ) == "skipped":

            return "skipped"

        return "pending"

    # --------------------------------------------------------
    # NEXT PENDING
    # --------------------------------------------------------

    def find_next_pending(self, start_index):

        for i in range(
            start_index,
            len(self.csv_files),
        ):

            if (
                self.status(self.csv_files[i])
                == "pending"
            ):

                return i

        return None

    # --------------------------------------------------------
    # PREVIOUS
    # --------------------------------------------------------

    def find_previous_file(self, start_index):

        if start_index <= 0:
            return None

        return start_index - 1

    # --------------------------------------------------------
    # RESTORE SAVED SELECTIONS
    # --------------------------------------------------------

    def restore_saved_selections(self):
        """
        Restore selected points for the current file.

        The progress JSON stores ORIGINAL row indices.
        We convert those back to indices in plot_df.
        """

        self.selected_plot_indices = []

        if self.current_file is None:
            return

        source_file = self.current_file.name

        saved_original_indices = (
            get_progress_selected_indices(
                self.progress,
                source_file,
            )
        )

        if not saved_original_indices:
            return

        original_to_plot = {}

        for plot_index, (_, row) in enumerate(
            self.plot_df.iterrows()
        ):

            original_index = int(
                row["_original_row_index"]
            )

            original_to_plot[
                original_index
            ] = plot_index

        for original_index in saved_original_indices:

            if original_index in original_to_plot:

                self.selected_plot_indices.append(
                    original_to_plot[original_index]
                )

        # Safety limit.
        self.selected_plot_indices = (
            self.selected_plot_indices[
                :MAX_SELECTIONS
            ]
        )

    # --------------------------------------------------------
    # LOAD CURRENT FILE
    # --------------------------------------------------------

    def load_current_file(self):

        if not self.running:
            return

        csv_file = self.csv_files[
            self.current_index
        ]

        self.current_file = csv_file

        try:

            self.df, self.plot_df = load_experiment(
                csv_file
            )

            # Analyze DOA for every plotted row.
            self.doa_analysis = [
                analyze_doa_row(row)
                for _, row in self.plot_df.iterrows()
            ]

            self.differences = [
                result["difference"]
                for result in self.doa_analysis
            ]

        except Exception as exc:

            print(
                f"\nERROR loading "
                f"{csv_file.name}:"
            )

            print(f"  {exc}")

            self.show_error(exc)

            return

        # ----------------------------------------------------
        # Restore previous selections if available.
        # ----------------------------------------------------

        self.restore_saved_selections()

        self.draw_current_experiment()

    # --------------------------------------------------------
    # DRAW CURRENT EXPERIMENT
    # --------------------------------------------------------

    def draw_current_experiment(self):

        # ----------------------------------------------------
        # Remove previous colorbar BEFORE clearing axes.
        # ----------------------------------------------------

        if self.colorbar is not None:

            try:
                self.colorbar.remove()

            except Exception:
                pass

            self.colorbar = None

        # Now clear the trajectory axes.
        self.ax.clear()

        x = self.plot_df[
            X_COLUMN
        ].to_numpy()

        y = self.plot_df[
            Y_COLUMN
        ].to_numpy()

        # ----------------------------------------------------
        # Connected trajectory.
        # ----------------------------------------------------

        self.ax.plot(
            x,
            y,
            linestyle="-",
            linewidth=1.0,
            alpha=0.6,
            zorder=1,
        )

        # ----------------------------------------------------
        # Prepare marker colours.
        # ----------------------------------------------------

        two_peak_indices = [
            i
            for i, result in enumerate(
                self.doa_analysis
            )
            if result["number_of_peaks"] >= 2
        ]

        one_peak_indices = [
            i
            for i, result in enumerate(
                self.doa_analysis
            )
            if result["number_of_peaks"] == 1
        ]

        zero_peak_indices = [
            i
            for i, result in enumerate(
                self.doa_analysis
            )
            if result["number_of_peaks"] == 0
        ]

        # ----------------------------------------------------
        # Continuous normalization.
        # ----------------------------------------------------

        color_norm = Normalize(
            vmin=0.0,
            vmax=MAX_COLOR_DIFFERENCE,
            clip=True,
        )

        # ----------------------------------------------------
        # Plot TWO-PEAK points.
        # ----------------------------------------------------

        if two_peak_indices:

            two_peak_x = [
                x[i]
                for i in two_peak_indices
            ]

            two_peak_y = [
                y[i]
                for i in two_peak_indices
            ]

            two_peak_values = [
                self.differences[i]
                for i in two_peak_indices
            ]

            self.scatter = self.ax.scatter(
                two_peak_x,
                two_peak_y,
                c=two_peak_values,
                cmap=COLORMAP,
                norm=color_norm,
                s=MARKER_SIZE,
                picker=PICK_RADIUS_POINTS,
                zorder=2,
            )

            # ------------------------------------------------
            # Colorbar.
            # ------------------------------------------------

            self.colorbar = self.fig.colorbar(
                self.scatter,
                ax=self.ax,
                pad=0.02,
            )

            self.colorbar.set_label(
                "Peak intensity difference Δ"
            )

        else:

            # We still need a scatter object so the rest
            # of the code has a consistent structure.

            self.scatter = self.ax.scatter(
                [],
                [],
                s=MARKER_SIZE,
                picker=PICK_RADIUS_POINTS,
                zorder=2,
            )

        # ----------------------------------------------------
        # Plot ONE-PEAK points separately.
        # ----------------------------------------------------

        if one_peak_indices:

            self.one_peak_scatter = self.ax.scatter(
                [
                    x[i]
                    for i in one_peak_indices
                ],
                [
                    y[i]
                    for i in one_peak_indices
                ],
                color=ONE_PEAK_COLOR,
                s=MARKER_SIZE,
                picker=PICK_RADIUS_POINTS,
                zorder=2.5,
                label="1 peak",
            )

        else:

            self.one_peak_scatter = None

        # ----------------------------------------------------
        # Plot ZERO-PEAK points separately.
        # ----------------------------------------------------

        if zero_peak_indices:

            self.zero_peak_scatter = self.ax.scatter(
                [
                    x[i]
                    for i in zero_peak_indices
                ],
                [
                    y[i]
                    for i in zero_peak_indices
                ],
                color=ZERO_PEAK_COLOR,
                s=MARKER_SIZE,
                picker=PICK_RADIUS_POINTS,
                zorder=2.5,
                label="0 peaks",
            )

        else:

            self.zero_peak_scatter = None

        # ----------------------------------------------------
        # Selected-point markers.
        #
        # There can now be up to FIVE simultaneously.
        # ----------------------------------------------------

        self.selected_scatter = self.ax.scatter(
            [],
            [],
            s=SELECTED_MARKER_SIZE,
            marker="o",
            facecolors="none",
            linewidths=2.5,
            edgecolors="red",
            zorder=5,
        )

        # Put the currently selected points onto the plot.
        self.update_selected_markers(
            redraw=False
        )

        # ----------------------------------------------------
        # Labels.
        # ----------------------------------------------------

        file_number = (
            self.current_index + 1
        )

        total_files = len(
            self.csv_files
        )

        current_status = self.status(
            self.current_file
        )

        self.ax.set_xlabel(
            X_COLUMN
        )

        self.ax.set_ylabel(
            Y_COLUMN
        )

        self.ax.set_title(
            f"Experiment "
            f"{file_number}/{total_files}\n"
            f"{self.current_file.name}\n"
            f"Status: {current_status}"
        )

        self.ax.grid(
            True,
            alpha=0.25,
        )

        # ----------------------------------------------------
        # Legend for special cases.
        # ----------------------------------------------------

        if (
            self.one_peak_scatter is not None
            or self.zero_peak_scatter is not None
        ):

            self.ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )

        # ----------------------------------------------------
        # Instructions.
        # ----------------------------------------------------

        self.info_text = self.ax.text(
            0.01,
            0.99,
            self.instructions_text(),
            transform=self.ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.85,
            ),
            zorder=10,
        )

        self.fig.canvas.draw_idle()

    # --------------------------------------------------------
    # INSTRUCTIONS
    # --------------------------------------------------------

    def instructions_text(self):

        selected_count = len(
            self.selected_plot_indices
        )

        if selected_count == MAX_SELECTIONS:
            selection_status = (
                f"Selected: {selected_count}/{MAX_SELECTIONS} ✓"
            )
        else:
            selection_status = (
                f"Selected: {selected_count}/{MAX_SELECTIONS}"
            )

        return (
            "Click a point to select/deselect\n"
            f"{selection_status}\n"
            "\n"
            f"Select exactly {MAX_SELECTIONS} points\n"
            "Enter = Save & Next\n"
            "N / → = Next\n"
            "B / ← = Back\n"
            "S = Skip\n"
            "Q / Esc = Quit"
        )

    # --------------------------------------------------------
    # UPDATE SELECTED MARKERS
    # --------------------------------------------------------

    def update_selected_markers(
        self,
        redraw=True,
    ):
        """
        Update the red rings showing all selected points.
        """

        if self.selected_scatter is None:
            return

        if not self.selected_plot_indices:

            self.selected_scatter.set_offsets(
                np.empty((0, 2))
            )

        else:

            selected_xy = []

            for plot_index in (
                self.selected_plot_indices
            ):

                row = self.plot_df.iloc[
                    plot_index
                ]

                selected_xy.append([
                    row[X_COLUMN],
                    row[Y_COLUMN],
                ])

            self.selected_scatter.set_offsets(
                selected_xy
            )

        if redraw:

            self.fig.canvas.draw_idle()

    # --------------------------------------------------------
    # UPDATE SELECTION INFO
    # --------------------------------------------------------

    def update_selection_info(self):

        if self.info_text is None:
            return

        self.info_text.set_text(
            self.instructions_text()
        )

        self.fig.canvas.draw_idle()

    # --------------------------------------------------------
    # PICK EVENT
    # --------------------------------------------------------

    def on_pick(self, event):

        # Determine which scatter object was clicked.
        artist = event.artist

        valid_artists = [
            self.scatter,
            self.one_peak_scatter,
            self.zero_peak_scatter,
        ]

        if artist not in valid_artists:
            return

        if event.ind is None:
            return

        if len(event.ind) == 0:
            return

        # ----------------------------------------------------
        # We need to map the picked point back to the
        # corresponding index in plot_df.
        # ----------------------------------------------------

        if artist == self.scatter:

            # Two-peak scatter contains only two-peak points.
            candidate_indices = [
                i
                for i, result in enumerate(
                    self.doa_analysis
                )
                if result["number_of_peaks"] >= 2
            ]

        elif artist == self.one_peak_scatter:

            candidate_indices = [
                i
                for i, result in enumerate(
                    self.doa_analysis
                )
                if result["number_of_peaks"] == 1
            ]

        elif artist == self.zero_peak_scatter:

            candidate_indices = [
                i
                for i, result in enumerate(
                    self.doa_analysis
                )
                if result["number_of_peaks"] == 0
            ]

        else:

            return

        # The local indices from event.ind refer to the
        # particular scatter object, not directly to plot_df.
        local_indices = event.ind

        if len(local_indices) == 1:

            local_index = int(
                local_indices[0]
            )

            if local_index >= len(
                candidate_indices
            ):

                return

            plot_index = candidate_indices[
                local_index
            ]

        else:

            # More than one marker may fall within the
            # picker radius. Choose the closest one
            # in screen coordinates.

            mouse_x = event.mouseevent.x
            mouse_y = event.mouseevent.y

            x_data = self.plot_df[
                X_COLUMN
            ].to_numpy()

            y_data = self.plot_df[
                Y_COLUMN
            ].to_numpy()

            closest_plot_index = None
            closest_distance = float("inf")

            for local_index in local_indices:

                local_index = int(
                    local_index
                )

                if local_index >= len(
                    candidate_indices
                ):

                    continue

                candidate_plot_index = (
                    candidate_indices[
                        local_index
                    ]
                )

                display_point = (
                    self.ax.transData.transform(
                        (
                            x_data[
                                candidate_plot_index
                            ],
                            y_data[
                                candidate_plot_index
                            ],
                        )
                    )
                )

                dx = (
                    display_point[0]
                    - mouse_x
                )

                dy = (
                    display_point[1]
                    - mouse_y
                )

                distance = (
                    dx * dx + dy * dy
                )

                if distance < closest_distance:

                    closest_distance = distance

                    closest_plot_index = (
                        candidate_plot_index
                    )

            if closest_plot_index is None:
                return

            plot_index = closest_plot_index

        self.select_point(
            plot_index
        )

    # --------------------------------------------------------
    # SELECT / DESELECT POINT
    # --------------------------------------------------------

    def select_point(self, plot_index):

        plot_index = int(plot_index)

        # ----------------------------------------------------
        # If already selected, clicking it again deselects it.
        # ----------------------------------------------------

        if plot_index in self.selected_plot_indices:

            self.selected_plot_indices.remove(
                plot_index
            )

            print(
                f"\nDeselected point "
                f"{plot_index}."
            )

        # ----------------------------------------------------
        # Otherwise add it to the selection.
        # ----------------------------------------------------

        else:

            if (
                len(self.selected_plot_indices)
                >= MAX_SELECTIONS
            ):

                print(
                    f"\nAlready selected "
                    f"{MAX_SELECTIONS} points."
                )

                print(
                    "Click a selected point "
                    "again to deselect it."
                )

                return

            self.selected_plot_indices.append(
                plot_index
            )

            selection_number = (
                len(self.selected_plot_indices)
            )

            print(
                f"\nSelected point "
                f"{selection_number}/{MAX_SELECTIONS}: "
                f"plot index {plot_index}"
            )

        # ----------------------------------------------------
        # Update visual highlights.
        # ----------------------------------------------------

        self.update_selected_markers(
            redraw=False
        )

        self.update_selection_info()

        # ----------------------------------------------------
        # Show information for the clicked point.
        # ----------------------------------------------------

        self.print_point_details(
            plot_index
        )

        self.fig.canvas.draw_idle()

    # --------------------------------------------------------
    # PRINT POINT DETAILS
    # --------------------------------------------------------

    def print_point_details(
        self,
        plot_index,
    ):
        """
        Print detailed information for one point.

        This does not change the multi-selection.
        """

        row = self.plot_df.iloc[
            plot_index
        ]

        x = row[X_COLUMN]
        y = row[Y_COLUMN]

        # Get original row.
        original_index = int(
            row["_original_row_index"]
        )

        original_row = self.df.iloc[
            original_index
        ]

        # DOA information.
        doa_result = self.doa_analysis[
            plot_index
        ]

        details = []

        # ----------------------------------------------------
        # Selection status.
        # ----------------------------------------------------

        if plot_index in self.selected_plot_indices:

            selection_number = (
                self.selected_plot_indices.index(
                    plot_index
                ) + 1
            )

            details.append(
                f"Selected point: "
                f"{selection_number}/{MAX_SELECTIONS}"
            )

        else:

            details.append(
                "Point is not selected."
            )

        details.append("")

        # ----------------------------------------------------
        # Basic point information.
        # ----------------------------------------------------

        details.append(
            f"Selected row: {original_index}"
        )

        details.append(
            f"{X_COLUMN}: {x}"
        )

        details.append(
            f"{Y_COLUMN}: {y}"
        )

        # ----------------------------------------------------
        # Useful original columns.
        # ----------------------------------------------------

        for column in [
            "time_step",
            "timestamp",
            "phi_in_degrees",
            "angle_0",
            "angle_1",
        ]:

            if column in original_row.index:

                details.append(
                    f"{column}: "
                    f"{original_row[column]}"
                )

        details.append("")

        # ----------------------------------------------------
        # DOA peak information.
        # ----------------------------------------------------

        number_of_peaks = (
            doa_result["number_of_peaks"]
        )

        if number_of_peaks >= 2:

            peak1_index = (
                doa_result["peak_indices"][0]
            )

            peak2_index = (
                doa_result["peak_indices"][1]
            )

            peak1_column = (
                doa_result["peak_columns"][0]
            )

            peak2_column = (
                doa_result["peak_columns"][1]
            )

            peak1_value = (
                doa_result["peak_values"][0]
            )

            peak2_value = (
                doa_result["peak_values"][1]
            )

            difference = (
                doa_result["difference"]
            )

            details.append(
                f"DOA peaks found: "
                f"{number_of_peaks}"
            )

            details.append(
                f"Peak 1: {peak1_column} "
                f"(index {peak1_index})"
            )

            details.append(
                f"  intensity: {peak1_value:.6g}"
            )

            details.append(
                f"Peak 2: {peak2_column} "
                f"(index {peak2_index})"
            )

            details.append(
                f"  intensity: {peak2_value:.6g}"
            )

            details.append(
                f"|Peak 1 - Peak 2|: "
                f"{difference:.6g}"
            )

            details.append(
                "Colour: continuous Δ scale"
            )

        elif number_of_peaks == 1:

            peak_index = (
                doa_result["peak_indices"][0]
            )

            peak_column = (
                doa_result["peak_columns"][0]
            )

            peak_value = (
                doa_result["peak_values"][0]
            )

            details.append(
                "DOA peaks found: 1"
            )

            details.append(
                f"Peak: {peak_column} "
                f"(index {peak_index})"
            )

            details.append(
                f"  intensity: {peak_value:.6g}"
            )

            details.append(
                "Colour: ONE PEAK"
            )

        else:

            details.append(
                "DOA peaks found: 0"
            )

            details.append(
                "Colour: ZERO PEAKS"
            )

        details.append("")

        details.append(
            f"Current selection: "
            f"{len(self.selected_plot_indices)}"
            f"/{MAX_SELECTIONS}"
        )

        print("\n")
        print("=" * 60)
        print(
            f"File: {self.current_file.name}"
        )

        for line in details:
            print(line)

        print("=" * 60)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    def save_current_selection(self):
        """
        Save all five selected points.

        Each point is written as a separate row in result.csv,
        and all five rows receive the same source_file value.

        Returns:
            True  -> save succeeded
            False -> selection was not saved
        """

        selected_count = len(
            self.selected_plot_indices
        )

        if selected_count != MAX_SELECTIONS:

            print(
                f"\nCannot save yet."
            )

            print(
                f"Selected: "
                f"{selected_count}/{MAX_SELECTIONS}"
            )

            print(
                f"Please select exactly "
                f"{MAX_SELECTIONS} points."
            )

            return False

        source_file = (
            self.current_file.name
        )

        # ----------------------------------------------------
        # Prevent duplicate completed files.
        # ----------------------------------------------------

        if source_file in self.saved_files:

            print(
                f"\n{source_file} is already "
                f"present as a completed file "
                f"in {RESULT_FILE.name}."
            )

            return False

        # ----------------------------------------------------
        # Save all five rows.
        # ----------------------------------------------------

        saved_original_indices = []

        print("\n")
        print("=" * 60)
        print(
            f"Saving {MAX_SELECTIONS} points "
            f"from {source_file}"
        )
        print("=" * 60)

        for selection_number, plot_index in enumerate(
            self.selected_plot_indices,
            start=1,
        ):

            row = self.plot_df.iloc[
                plot_index
            ]

            original_index = int(
                row["_original_row_index"]
            )

            original_row = self.df.iloc[
                original_index
            ].copy()

            save_selected_row(
                original_row,
                source_file,
            )

            saved_original_indices.append(
                original_index
            )

            print(
                f"  {selection_number}/"
                f"{MAX_SELECTIONS}: "
                f"original row "
                f"{original_index}"
            )

        # ----------------------------------------------------
        # Mark the file as completed.
        # ----------------------------------------------------

        self.saved_files.add(
            source_file
        )

        self.progress[
            source_file
        ] = {
            "status": "selected",
            "selected_indices": (
                saved_original_indices
            ),
        }

        save_progress(
            self.progress
        )

        print("=" * 60)
        print(
            f"Saved {MAX_SELECTIONS} rows "
            f"from {source_file}"
        )

        print(
            f"Result file: {RESULT_FILE}"
        )

        print("=" * 60)

        return True

    # --------------------------------------------------------
    # NEXT
    # --------------------------------------------------------

    def next_experiment(
        self,
        skip_current=False,
    ):
        """
        Move to the next pending experiment.

        If skip_current=True, the current experiment is marked
        skipped and its temporary selections are discarded.
        """

        if skip_current:

            self.progress[
                self.current_file.name
            ] = {
                "status": "skipped",
                "selected_indices": [],
            }

            save_progress(
                self.progress
            )

        # Clear temporary selections before moving.
        self.selected_plot_indices = []

        next_index = (
            self.current_index + 1
        )

        while (
            next_index
            < len(self.csv_files)
        ):

            if (
                self.status(
                    self.csv_files[
                        next_index
                    ]
                )
                == "pending"
            ):

                self.current_index = (
                    next_index
                )

                self.load_current_file()

                return

            next_index += 1

        print(
            "\nNo more pending experiments."
        )

        self.show_finished()

    # --------------------------------------------------------
    # BACK
    # --------------------------------------------------------

    def previous_experiment(self):

        previous_index = (
            self.find_previous_file(
                self.current_index
            )
        )

        if previous_index is None:

            print(
                "\nAlready at the first "
                "experiment."
            )

            return

        self.current_index = (
            previous_index
        )

        self.load_current_file()

    # --------------------------------------------------------
    # KEYBOARD
    # --------------------------------------------------------

    def on_key(self, event):

        if not self.running:
            return

        key = event.key

        if key is None:
            return

        key = key.lower()

        # ----------------------------------------------------
        # ENTER
        # ----------------------------------------------------

        if key in (
            "enter",
            "return",
        ):

            # Save only if exactly five points are selected.
            saved = (
                self.save_current_selection()
            )

            if saved:

                self.next_experiment()

        # ----------------------------------------------------
        # NEXT
        # ----------------------------------------------------

        elif key in (
            "n",
            "right",
        ):

            # IMPORTANT:
            # N does NOT save selections.
            #
            # It simply moves to the next pending file.
            #
            # Use ENTER to save the five points.

            if self.selected_plot_indices:

                print(
                    "\nWarning: "
                    f"{len(self.selected_plot_indices)} "
                    "selected point(s) will not be saved."
                )

            self.next_experiment()

        # ----------------------------------------------------
        # BACK
        # ----------------------------------------------------

        elif key in (
            "b",
            "left",
        ):

            self.previous_experiment()

        # ----------------------------------------------------
        # SKIP
        # ----------------------------------------------------

        elif key == "s":

            print(
                f"\nSkipped "
                f"{self.current_file.name}"
            )

            self.next_experiment(
                skip_current=True
            )

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        elif key in (
            "q",
            "escape",
        ):

            self.quit()

    # --------------------------------------------------------
    # ERROR SCREEN
    # --------------------------------------------------------

    def show_error(self, exc):

        self.ax.clear()

        self.ax.text(
            0.5,
            0.5,
            (
                f"Error loading:\n\n"
                f"{self.current_file.name}\n\n"
                f"{exc}\n\n"
                "Press N to continue\n"
                "Press B to go back\n"
                "Press Q to quit"
            ),
            transform=self.ax.transAxes,
            ha="center",
            va="center",
        )

        self.ax.set_axis_off()

        self.fig.canvas.draw_idle()

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    def show_finished(self):

        selected_count = len(
            self.saved_files
        )

        skipped_count = sum(
            1
            for file in self.csv_files
            if self.status(file)
            == "skipped"
        )

        pending_count = sum(
            1
            for file in self.csv_files
            if self.status(file)
            == "pending"
        )

        self.ax.clear()

        self.ax.text(
            0.5,
            0.55,
            "All experiments processed",
            transform=self.ax.transAxes,
            ha="center",
            va="center",
            fontsize=18,
        )

        self.ax.text(
            0.5,
            0.40,
            (
                f"Selected files: {selected_count}\n"
                f"Skipped: {skipped_count}\n"
                f"Pending: {pending_count}\n\n"
                f"Points saved per selected file: "
                f"{MAX_SELECTIONS}\n\n"
                f"Results: {RESULT_FILE}\n\n"
                "Press Q or Escape to close."
            ),
            transform=self.ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )

        self.ax.set_axis_off()

        self.fig.canvas.draw_idle()

    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    def quit(self):

        self.running = False

        save_progress(
            self.progress
        )

        print("\n")
        print("=" * 60)
        print("Stopped.")

        print(
            f"Results:  {RESULT_FILE}"
        )

        print(
            f"Progress: {PROGRESS_FILE}"
        )

        print("=" * 60)

        plt.close(self.fig)

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def run(self):

        self.fig, self.ax = plt.subplots(
            figsize=(11, 8),
            constrained_layout=True,
        )

        self.fig.canvas.manager.set_window_title(
            "Interactive Bifurcation Point Selector"
        )

        # Mouse clicks.
        self.fig.canvas.mpl_connect(
            "pick_event",
            self.on_pick,
        )

        # Keyboard controls.
        self.fig.canvas.mpl_connect(
            "key_press_event",
            self.on_key,
        )

        # Start at first pending experiment.
        first_pending = (
            self.find_next_pending(0)
        )

        if first_pending is None:

            self.current_index = 0

            self.load_current_file()

            self.show_finished()

        else:

            self.current_index = (
                first_pending
            )

            self.load_current_file()

        plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    folder = (
        EXPERIMENT_FOLDER
        .expanduser()
        .resolve()
    )

    if not folder.exists():

        print(
            f"ERROR: folder does not exist:\n"
            f"{folder}"
        )

        sys.exit(1)

    if not folder.is_dir():

        print(
            f"ERROR: not a directory:\n"
            f"{folder}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Find experiment CSV files.
    #
    # result.csv is explicitly excluded.
    # --------------------------------------------------------

    csv_files = [
        path
        for path in folder.glob("*.csv")
        if path.name
        != RESULT_FILE.name
    ]

    csv_files.sort(
        key=natural_sort_key
    )

    if not csv_files:

        print(
            f"No CSV files found in:\n"
            f"{folder}"
        )

        sys.exit(1)

    print("=" * 60)
    print(
        "Interactive Bifurcation Point Selector"
    )
    print("=" * 60)

    print(
        f"Folder:       {folder}"
    )

    print(
        f"Experiments:  {len(csv_files)}"
    )

    print(
        f"Points/file:  {MAX_SELECTIONS}"
    )

    print(
        f"Result:       {RESULT_FILE}"
    )

    print(
        f"Progress:     {PROGRESS_FILE}"
    )

    print("=" * 60)

    print(
        "\nSelection workflow:"
    )

    print(
        f"  1. Click {MAX_SELECTIONS} points."
    )

    print(
        "  2. Selected points are highlighted."
    )

    print(
        "  3. Click a selected point again to deselect it."
    )

    print(
        "  4. Press ENTER to save all five points."
    )

    print(
        "  5. The application moves to the next file."
    )

    print("=" * 60)

    app = BifurcationSelector(
        csv_files
    )

    app.run()


if __name__ == "__main__":
    main()