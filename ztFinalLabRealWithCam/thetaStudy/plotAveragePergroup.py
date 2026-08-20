"""
Using the output of speaker_angle_at_b.py (speaker_angle_at_b_per_type_avg.csv),
group the per-type averaged angles by "group" - the type prefix with its
trailing two-letter code split into a first letter and a second letter, e.g.:

    "09_07_AA" -> group = "09_07_A", second_letter = "A"
    "09_07_AB" -> group = "09_07_A", second_letter = "B"
    ...
    "16_07_EA" -> group = "16_07_E", second_letter = "A"

For each group, plot a line graph of the already-computed avg_angle_deg
(y-axis) against the second letter (x-axis, alphabetically increasing).
Groups don't all have to have the same number of points - e.g. a group
with only EA, EB, EC, ED just plots 4 points (A-D on the x-axis) instead
of 5.

One PNG is saved per group.

Input:
    speaker_angle_at_b_per_type_avg.csv, with (at least) columns:
        prefix, avg_angle_deg

Usage:
    python plot_angle_by_letter_group.py
"""

import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # never try to open a display / show a window
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Config - edit these to match your setup
INPUT_CSV = "./speaker_angle_at_b_per_type_avg.csv"
OUTPUT_DIR = "./angleByGroup/"
# ------------------------------------------------------------------ #

# prefix = <anything>_<two letters>, e.g. "09_07_AA" -> date="09_07", letters="AA"
PREFIX_RE = re.compile(r"^(?P<date>.+)_(?P<letters>[A-Za-z]{2})$")


def parse_prefix(prefix: str):
    """Return (group_key, second_letter) or None if the prefix doesn't match."""
    m = PREFIX_RE.match(prefix)
    if not m:
        return None
    date = m.group("date")
    letters = m.group("letters")
    first_letter, second_letter = letters[0], letters[1]
    group_key = f"{date}_{first_letter}"
    return group_key, second_letter.upper()


def plot_group(group_key: str, sub_df: pd.DataFrame, output_dir: str):
    sub_df = sub_df.sort_values("second_letter")
    x = sub_df["second_letter"].tolist()
    y = sub_df["avg_angle_deg"].tolist()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, y, marker="o", linewidth=2, markersize=6)

    for xi, yi in zip(x, y):
        if pd.notna(yi):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 8), fontsize=8, ha="center")

    ax.set_title(f"{group_key}: average angle at B by condition")
    ax.set_xlabel("Condition (second letter)")
    ax.set_ylabel("Average angle at B (degrees)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{group_key}_angle_by_letter.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    df = pd.read_csv(INPUT_CSV)

    required_cols = ["prefix", "avg_angle_deg"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{INPUT_CSV} is missing required columns: {missing}")

    group_keys = []
    second_letters = []
    for prefix in df["prefix"]:
        parsed = parse_prefix(str(prefix))
        if parsed is None:
            print(f"Skipping prefix that doesn't match <date>_<2 letters> pattern: {prefix}")
            group_keys.append(None)
            second_letters.append(None)
        else:
            group_key, second_letter = parsed
            group_keys.append(group_key)
            second_letters.append(second_letter)

    df["group_key"] = group_keys
    df["second_letter"] = second_letters
    df = df.dropna(subset=["group_key", "second_letter"])

    for group_key, sub_df in sorted(df.groupby("group_key")):
        plot_group(group_key, sub_df, OUTPUT_DIR)


if __name__ == "__main__":
    main()