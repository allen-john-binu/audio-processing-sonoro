import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ==================================================
# CONFIGURATION
# ==================================================

# Folder containing the seven CSV files
DATA_FOLDER = Path("./data/SS/")

# Available source-to-sensor distances (cm)
distances = [60, 70, 80, 90, 100, 110, 120]

# True  = Global colour scale across all datasets
# False = Individual colour scale for each dataset
GLOBAL_COLOR_SCALE = False

# Colour map
COLOR_MAP = "viridis"


# ==================================================
# LOAD SEVEN DATASETS
# ==================================================

datasets = {}

for distance in distances:

    csv_path = DATA_FOLDER / f"{distance}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Detect spatial response columns
    spatial_cols = sorted(
        [
            c for c in df.columns
            if re.fullmatch(r"spatial_resp_\d+", c)
        ],
        key=lambda c: int(c.split("_")[-1])
    )

    if not spatial_cols:
        raise ValueError(
            f"No spatial response columns in {csv_path}"
        )

    # Extract data
    times = pd.to_numeric(
        df["timestamp"], errors="coerce"
    ).to_numpy(dtype=float)

    spl = pd.to_numeric(
        df["db_spl"], errors="coerce"
    ).to_numpy(dtype=float)

    response = df[spatial_cols].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)

    # Remove invalid timestamps and SPL values
    valid = np.isfinite(times) & np.isfinite(spl)

    times = times[valid]
    spl = spl[valid]
    response = response[valid]

    # Sort by timestamp
    order = np.argsort(times)

    times = times[order]
    spl = spl[order]
    response = response[order]

    if len(times) == 0:
        raise ValueError(f"No valid measurements in {csv_path}")

    # Convert to elapsed time
    times = times - times[0]

    datasets[distance] = (times, spl, response)

    print(
        f"Loaded {distance}.csv: "
        f"{len(times)} measurements, "
        f"{len(spatial_cols)} angular samples"
    )


# ==================================================
# COLOUR SCALE LIMITS
# ==================================================

# Calculate global min/max using all seven datasets
all_values = np.concatenate([
    data[2].ravel()
    for data in datasets.values()
])

all_values = all_values[np.isfinite(all_values)]

if all_values.size == 0:
    raise ValueError("No valid spatial response values found.")

global_vmin = np.min(all_values)
global_vmax = np.max(all_values)

if global_vmin == global_vmax:
    global_vmax = global_vmin + 1

# Calculate individual limits for each dataset
individual_limits = {}

for distance, (_, _, response) in datasets.items():

    values = response[np.isfinite(response)]

    if values.size == 0:
        individual_limits[distance] = (
            global_vmin, global_vmax
        )
        continue

    vmin = np.min(values)
    vmax = np.max(values)

    if vmin == vmax:
        vmax = vmin + 1

    individual_limits[distance] = (vmin, vmax)


# ==================================================
# SPL SLIDER LIMITS
# ==================================================

all_spl = np.concatenate([
    data[1] for data in datasets.values()
])

spl_min = float(np.min(all_spl))
spl_max = float(np.max(all_spl))

if spl_min == spl_max:
    spl_max = spl_min + 1


# ==================================================
# FIGURE LAYOUT
# ==================================================

fig, axes = plt.subplots(2, 4, figsize=(19, 9))

fig.suptitle(
    "DOA Spatial Response — Different Source Distances - SonoRo speaker",
    fontsize=16
)

# Seven heatmap axes
heat_axes = axes.flatten()[:7]

# Bottom-right panel for slider
control_ax = axes.flatten()[7]

fig.subplots_adjust(
    left=0.07,
    right=0.88,
    top=0.90,
    bottom=0.12,
    wspace=0.32,
    hspace=0.35
)


# ==================================================
# COLOURBARS
# ==================================================

if GLOBAL_COLOR_SCALE:

    colorbar_ax = fig.add_axes([0.91, 0.18, 0.018, 0.65])

    fig.colorbar(
        ScalarMappable(
            norm=Normalize(global_vmin, global_vmax),
            cmap=COLOR_MAP
        ),
        cax=colorbar_ax,
        label="Spatial Response"
    )

else:

    for distance, ax in zip(distances, heat_axes):

        vmin, vmax = individual_limits[distance]

        sm = ScalarMappable(
            norm=Normalize(vmin, vmax),
            cmap=COLOR_MAP
        )

        fig.colorbar(
            sm,
            ax=ax,
            fraction=0.035,
            pad=0.02
        )


# ==================================================
# SPL SLIDER
# ==================================================

control_ax.set_title(
    "SPL CONTROL",
    fontsize=13,
    fontweight="bold",
    pad=15
)

control_ax.axis("off")

slider_ax = control_ax.inset_axes(
    [0.10, 0.45, 0.82, 0.08]
)

spl_slider = Slider(
    ax=slider_ax,
    label="dB SPL",
    valmin=spl_min,
    valmax=spl_max,
    valinit=spl_min,
    valstep=max((spl_max - spl_min) / 200, 0.01)
)


# ==================================================
# HEATMAP UPDATE FUNCTION
# ==================================================

def update(threshold):

    for distance, ax in zip(distances, heat_axes):

        times, spl, response = datasets[distance]

        # Apply common SPL threshold
        keep = spl >= threshold

        ax.clear()

        if not np.any(keep):

            ax.text(
                0.5,
                0.5,
                "No measurements",
                ha="center",
                va="center",
                transform=ax.transAxes
            )

        else:

            filtered_response = response[keep].T

            n_angles = response.shape[1]

            # Shift angular data by 180 degrees
            filtered_response = np.roll(
                filtered_response,
                n_angles // 2,
                axis=0
            )

            # Angular range: -180 to +180 degrees
            y_edges = np.linspace(-180, 180, n_angles + 1)

            # Heatmap column edges
            x_edges = (
                np.arange(filtered_response.shape[1] + 1)
                - 0.5
            )

            # Select colour limits
            if GLOBAL_COLOR_SCALE:

                vmin = global_vmin
                vmax = global_vmax

            else:

                vmin, vmax = individual_limits[distance]

            # Draw heatmap
            ax.pcolormesh(
                x_edges,
                y_edges,
                filtered_response,
                shading="flat",
                cmap=COLOR_MAP,
                vmin=vmin,
                vmax=vmax
            )

            # Time-axis labels
            filtered_times = times[keep]

            tick_count = min(6, len(filtered_times))

            tick_indices = np.unique(
                np.linspace(
                    0,
                    len(filtered_times) - 1,
                    tick_count
                ).round().astype(int)
            )

            ax.set_xticks(tick_indices)

            ax.set_xticklabels([
                f"{filtered_times[i]:.1f}"
                for i in tick_indices
            ])

        # Axis formatting
        ax.set_title(f"{distance} cm", fontsize=12)

        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("DOA angle (degrees)")

        ax.set_ylim(-180, 180)
        ax.set_yticks(np.arange(-180, 181, 45))

    fig.canvas.draw_idle()


# ==================================================
# CONNECT SLIDER
# ==================================================

spl_slider.on_changed(update)

# Initial visualization
update(spl_min)

plt.show()
