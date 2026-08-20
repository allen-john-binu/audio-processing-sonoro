import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider


# =========================================================
# Configuration
# =========================================================

INPUT_FOLDER = Path("./peakCheck/")

# Maximum phi value in your experiment
PHI_MAX = 45.0


# =========================================================
# Load all data
# =========================================================

data = []

for csv_file in sorted(INPUT_FOLDER.glob("*.csv")):

    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row_index, row in enumerate(reader):

            try:
                angle_0 = float(row["angle_0"])
                angle_1 = float(row["angle_1"])

                highest_peak_value = float(
                    row["highest_peak_value"]
                )

                second_peak_value = float(
                    row["second_peak_value"]
                )

                phi_in_degrees = abs(
                    float(row["phi_in_degrees"])
                )

            except (ValueError, TypeError, KeyError):
                continue

            # Ignore invalid peak values
            if highest_peak_value <= 0:
                continue

            if second_peak_value <= 0:
                continue

            # -------------------------------------------------
            # Relative intensity difference
            # -------------------------------------------------

            relative_intensity_db = (
                20
                * np.log10(
                    highest_peak_value
                    / second_peak_value
                )
            )

            # -------------------------------------------------
            # Angular separation
            # -------------------------------------------------

            angular_separation = abs(
                angle_0 - angle_1
            )

            data.append({
                "filename": csv_file.name,
                "row_index": row_index,
                "x": relative_intensity_db,
                "y": angular_separation,
                "phi": phi_in_degrees,
            })


# =========================================================
# Convert to numpy arrays
# =========================================================

if not data:
    raise RuntimeError("No valid data found.")

x = np.array([d["x"] for d in data])
y = np.array([d["y"] for d in data])
phi = np.array([d["phi"] for d in data])


print(f"Total points: {len(data)}")

print("\n|phi_in_degrees|:")
print(f"  min:  {phi.min():.3f}°")
print(f"  max:  {phi.max():.3f}°")
print(f"  mean: {phi.mean():.3f}°")


# =========================================================
# Custom blue -> violet -> red colormap
# =========================================================

blue_violet_red = LinearSegmentedColormap.from_list(
    "blue_violet_red",
    [
        "#2166ac",   # blue
        "#6a3d9a",   # violet
        "#d73027",   # red
    ]
)


# =========================================================
# Create figure
# =========================================================

fig, ax = plt.subplots(figsize=(11, 8))

# Leave room at bottom for slider
plt.subplots_adjust(bottom=0.16)


# =========================================================
# Initial threshold
# =========================================================

initial_threshold = 0.0

mask = phi >= initial_threshold


# =========================================================
# Initial scatter
# =========================================================

scatter = ax.scatter(
    x[mask],
    y[mask],
    c=phi[mask],
    cmap=blue_violet_red,
    vmin=initial_threshold,
    vmax=PHI_MAX,
    s=20,
    alpha=0.75,
)


# =========================================================
# Labels
# =========================================================

ax.set_xlabel(
    "Relative intensity difference \n"
    "20 log₁₀(highest peak / second peak)"
)

ax.set_ylabel(
    "Angular separation between sources (degrees)"
)

ax.set_title(
    "Ring Attractor Decision Landscape"
)

ax.grid(
    True,
    alpha=0.25
)


# =========================================================
# Colorbar
# =========================================================

colorbar = fig.colorbar(
    scatter,
    ax=ax
)

colorbar.set_label(
    "|phi_in_degrees| (degrees)"
)


# =========================================================
# Slider
# =========================================================

slider_ax = fig.add_axes(
    [0.15, 0.055, 0.70, 0.035]
)

threshold_slider = Slider(
    ax=slider_ax,
    label="Minimum |phi|",
    valmin=0.0,
    valmax=PHI_MAX,
    valinit=initial_threshold,
    valstep=1.0,
)


# =========================================================
# Text showing number of points
# =========================================================

count_text = fig.text(
    0.15,
    0.105,
    f"Points: {mask.sum()} / {len(phi)}",
)


# =========================================================
# Slider update function
# =========================================================

def update(threshold):

    # Filter points
    mask = phi >= threshold

    # Update scatter positions
    scatter.set_offsets(
        np.column_stack(
            (x[mask], y[mask])
        )
    )

    # Update colors
    scatter.set_array(
        phi[mask]
    )

    # -----------------------------------------------------
    # Rescale color map:
    #
    # threshold -> blue
    # 45°       -> red
    # -----------------------------------------------------

    scatter.set_clim(
        vmin=threshold,
        vmax=PHI_MAX
    )

    # Update colorbar
    colorbar.update_normal(scatter)

    # Update point count
    count_text.set_text(
        f"Points: {mask.sum()} / {len(phi)}"
    )

    # Update title
    ax.set_title(
        f"Ring Attractor Decision Landscape "
        f"(|phi| ≥ {threshold:.0f}°)"
    )

    fig.canvas.draw_idle()


threshold_slider.on_changed(update)


# =========================================================
# Show
# =========================================================

plt.show()