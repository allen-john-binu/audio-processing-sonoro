import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import argparse
import os

# ── Config ─────────────────────────────────────────────────────────────────────
N_BUMP_RUNS = 50


def parse_group_stats_from_csv(csv_path):
    """
    Parse the group statistics from the ring attractor output CSV.
    
    Expected column structure:
    [timestamp] [bump1...bump50] [group1_len_1...group1_len_50] 
    [group1_angle_1...group1_angle_50] [group2_len_1...group2_len_50] 
    [group2_angle_1...group2_angle_50] [metadata...]
    
    Args:
        csv_path: Path to the output CSV from ring_attractor_modified.py
    
    Returns:
        Dictionary with keys:
        - 'timestamps': array of timestamps
        - 'group1_lengths': (n_timesteps, n_runs) array
        - 'group1_angles': (n_timesteps, n_runs) array
        - 'group2_lengths': (n_timesteps, n_runs) array
        - 'group2_angles': (n_timesteps, n_runs) array
        - 'vicon_angles': array of ground-truth angles
    """
    df = pd.read_csv(csv_path)
    
    n_timesteps = len(df)
    
    # Extract group statistics columns
    bump_cols       = [f"bump{i + 1}" for i in range(N_BUMP_RUNS)]
    group1_len_cols = [f"group1_len_{i + 1}" for i in range(N_BUMP_RUNS)]
    group1_ang_cols = [f"group1_angle_{i + 1}" for i in range(N_BUMP_RUNS)]
    group2_len_cols = [f"group2_len_{i + 1}" for i in range(N_BUMP_RUNS)]
    group2_ang_cols = [f"group2_angle_{i + 1}" for i in range(N_BUMP_RUNS)]
    
    # Convert to numpy arrays
    group1_lengths = df[group1_len_cols].values
    group1_angles  = df[group1_ang_cols].values
    group2_lengths = df[group2_len_cols].values
    group2_angles  = df[group2_ang_cols].values
    vicon_angles   = df["angle"].values
    
    return {
        'timestamps': df["timestamp"].values,
        'group1_lengths': group1_lengths,
        'group1_angles': group1_angles,
        'group2_lengths': group2_lengths,
        'group2_angles': group2_angles,
        'vicon_angles': vicon_angles,
        'n_timesteps': n_timesteps,
        'n_runs': N_BUMP_RUNS
    }


def plot_group_timelines(data, output_path):
    """
    Create a 2×2 dashboard:
    - Top-left: Group1 length timeline (all 50 runs + mean ± std)
    - Top-right: Group2 length timeline (all 50 runs + mean ± std)
    - Bottom-left: Group1 angle timeline (all 50 runs + mean ± std)
    - Bottom-right: Group2 angle timeline (all 50 runs + mean ± std)
    
    Args:
        data: Dictionary from parse_group_stats_from_csv()
        output_path: Path to save the figure
    """
    group1_lengths = data['group1_lengths']
    group1_angles  = data['group1_angles']
    group2_lengths = data['group2_lengths']
    group2_angles  = data['group2_angles']
    n_timesteps    = data['n_timesteps']
    n_runs         = data['n_runs']
    
    # Compute statistics
    timesteps = np.arange(n_timesteps)
    
    # Group1 lengths
    g1_len_mean = np.mean(group1_lengths, axis=1)
    g1_len_std  = np.std(group1_lengths, axis=1)
    g1_len_min  = np.min(group1_lengths, axis=1)
    g1_len_max  = np.max(group1_lengths, axis=1)
    
    # Group2 lengths
    g2_len_mean = np.mean(group2_lengths, axis=1)
    g2_len_std  = np.std(group2_lengths, axis=1)
    g2_len_min  = np.min(group2_lengths, axis=1)
    g2_len_max  = np.max(group2_lengths, axis=1)
    
    # Group1 angles
    g1_ang_mean = np.mean(group1_angles, axis=1)
    g1_ang_std  = np.std(group1_angles, axis=1)
    g1_ang_min  = np.min(group1_angles, axis=1)
    g1_ang_max  = np.max(group1_angles, axis=1)
    
    # Group2 angles
    g2_ang_mean = np.mean(group2_angles, axis=1)
    g2_ang_std  = np.std(group2_angles, axis=1)
    g2_ang_min  = np.min(group2_angles, axis=1)
    g2_ang_max  = np.max(group2_angles, axis=1)
    
    # Create figure (max screen size)
    fig = plt.figure(figsize=(24, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # ── TOP-LEFT: Group1 Length Timeline ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    
    # All 50 individual runs (thin, transparent)
    for run_idx in range(n_runs):
        ax1.plot(timesteps, group1_lengths[:, run_idx], 
                color='C0', alpha=0.1, linewidth=0.5)
    
    # Min-max range (light band)
    ax1.fill_between(timesteps, g1_len_min, g1_len_max, 
                     alpha=0.2, color='C0', label='Min-Max range')
    
    # Mean ± std (darker band)
    ax1.fill_between(timesteps, g1_len_mean - g1_len_std, g1_len_mean + g1_len_std,
                     alpha=0.4, color='C0', label='Mean ± 1 std')
    
    # Mean line
    ax1.plot(timesteps, g1_len_mean, color='C0', linewidth=2.5, label='Mean')
    
    ax1.set_xlabel("Timestep", fontsize=11)
    ax1.set_ylabel("Group Length", fontsize=11)
    ax1.set_title("Group 1: Largest Contiguous Block Length", fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # ── TOP-RIGHT: Group2 Length Timeline ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    
    # All 50 individual runs (thin, transparent)
    for run_idx in range(n_runs):
        ax2.plot(timesteps, group2_lengths[:, run_idx], 
                color='C1', alpha=0.1, linewidth=0.5)
    
    # Min-max range (light band)
    ax2.fill_between(timesteps, g2_len_min, g2_len_max, 
                     alpha=0.2, color='C1', label='Min-Max range')
    
    # Mean ± std (darker band)
    ax2.fill_between(timesteps, g2_len_mean - g2_len_std, g2_len_mean + g2_len_std,
                     alpha=0.4, color='C1', label='Mean ± 1 std')
    
    # Mean line
    ax2.plot(timesteps, g2_len_mean, color='C1', linewidth=2.5, label='Mean')
    
    ax2.set_xlabel("Timestep", fontsize=11)
    ax2.set_ylabel("Group Length", fontsize=11)
    ax2.set_title("Group 2: Second Largest Contiguous Block Length", fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # ── BOTTOM-LEFT: Group1 Angle Timeline ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    
    # All 50 individual runs (thin, transparent)
    for run_idx in range(n_runs):
        ax3.plot(timesteps, group1_angles[:, run_idx], 
                color='C2', alpha=0.1, linewidth=0.5)
    
    # Min-max range (light band)
    ax3.fill_between(timesteps, g1_ang_min, g1_ang_max, 
                     alpha=0.2, color='C2', label='Min-Max range')
    
    # Mean ± std (darker band)
    ax3.fill_between(timesteps, g1_ang_mean - g1_ang_std, g1_ang_mean + g1_ang_std,
                     alpha=0.4, color='C2', label='Mean ± 1 std')
    
    # Mean line
    ax3.plot(timesteps, g1_ang_mean, color='C2', linewidth=2.5, label='Mean')
    
    ax3.set_xlabel("Timestep", fontsize=11)
    ax3.set_ylabel("Angle (degrees)", fontsize=11)
    ax3.set_title("Group 1: Circular Mean Angle", fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-180, 180)
    
    # ── BOTTOM-RIGHT: Group2 Angle Timeline ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    
    # All 50 individual runs (thin, transparent)
    for run_idx in range(n_runs):
        ax4.plot(timesteps, group2_angles[:, run_idx], 
                color='C3', alpha=0.1, linewidth=0.5)
    
    # Min-max range (light band)
    ax4.fill_between(timesteps, g2_ang_min, g2_ang_max, 
                     alpha=0.2, color='C3', label='Min-Max range')
    
    # Mean ± std (darker band)
    ax4.fill_between(timesteps, g2_ang_mean - g2_ang_std, g2_ang_mean + g2_ang_std,
                     alpha=0.4, color='C3', label='Mean ± 1 std')
    
    # Mean line
    ax4.plot(timesteps, g2_ang_mean, color='C3', linewidth=2.5, label='Mean')
    
    ax4.set_xlabel("Timestep", fontsize=11)
    ax4.set_ylabel("Angle (degrees)", fontsize=11)
    ax4.set_title("Group 2: Circular Mean Angle", fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-180, 180)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot → {output_path}")
    plt.close(fig)


def plot_statistics_summary(data, output_path):
    """
    Create a summary statistics figure showing distributions across the experiment.
    
    Args:
        data: Dictionary from parse_group_stats_from_csv()
        output_path: Path to save the figure
    """
    group1_lengths = data['group1_lengths']
    group1_angles  = data['group1_angles']
    group2_lengths = data['group2_lengths']
    group2_angles  = data['group2_angles']
    n_runs         = data['n_runs']
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # Flatten all runs across all timesteps for distribution plots
    g1_len_flat = group1_lengths.flatten()
    g1_ang_flat = group1_angles.flatten()
    g2_len_flat = group2_lengths.flatten()
    g2_ang_flat = group2_angles.flatten()
    
    # Remove zeros for angle distributions (represents "no group")
    g1_ang_nonzero = g1_ang_flat[group1_lengths.flatten() > 0]
    g2_ang_nonzero = g2_ang_flat[group2_lengths.flatten() > 0]
    
    # Top-left: Group1 length histogram
    axes[0, 0].hist(g1_len_flat, bins=30, color='C0', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel("Group Length", fontsize=11)
    axes[0, 0].set_ylabel("Frequency", fontsize=11)
    axes[0, 0].set_title(f"Group 1 Length Distribution\n(μ={np.mean(g1_len_flat):.1f}, σ={np.std(g1_len_flat):.1f})", 
                         fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Top-right: Group2 length histogram
    axes[0, 1].hist(g2_len_flat, bins=30, color='C1', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel("Group Length", fontsize=11)
    axes[0, 1].set_ylabel("Frequency", fontsize=11)
    axes[0, 1].set_title(f"Group 2 Length Distribution\n(μ={np.mean(g2_len_flat):.1f}, σ={np.std(g2_len_flat):.1f})", 
                         fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Bottom-left: Group1 angle histogram
    if len(g1_ang_nonzero) > 0:
        axes[1, 0].hist(g1_ang_nonzero, bins=30, color='C2', alpha=0.7, edgecolor='black', range=(-180, 180))
    axes[1, 0].set_xlabel("Angle (degrees)", fontsize=11)
    axes[1, 0].set_ylabel("Frequency", fontsize=11)
    axes[1, 0].set_title(f"Group 1 Angle Distribution\n(μ={np.mean(g1_ang_nonzero):.1f}°, σ={np.std(g1_ang_nonzero):.1f}°)", 
                         fontsize=11, fontweight='bold')
    axes[1, 0].set_xlim(-180, 180)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Bottom-right: Group2 angle histogram
    if len(g2_ang_nonzero) > 0:
        axes[1, 1].hist(g2_ang_nonzero, bins=30, color='C3', alpha=0.7, edgecolor='black', range=(-180, 180))
    axes[1, 1].set_xlabel("Angle (degrees)", fontsize=11)
    axes[1, 1].set_ylabel("Frequency", fontsize=11)
    axes[1, 1].set_title(f"Group 2 Angle Distribution\n(μ={np.mean(g2_ang_nonzero):.1f}°, σ={np.std(g2_ang_nonzero):.1f}°)", 
                         fontsize=11, fontweight='bold')
    axes[1, 1].set_xlim(-180, 180)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved summary → {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize group statistics from ring attractor simulations."
    )
    parser.add_argument(
        "csv_path",
        help="Path to the output CSV from ring_attractor_modified.py"
    )
    parser.add_argument(
        "--output-dir", "-o", 
        type=str, 
        default=None,
        help="Output directory for plots (default: same directory as CSV)"
    )
    parser.add_argument(
        "--no-summary",
        action='store_true',
        help="Skip summary statistics plot"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.csv_path):
        print(f"ERROR: File not found: {args.csv_path}")
        return
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.csv_path)
        if not args.output_dir:
            args.output_dir = "."
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get base name for output files
    base_name = os.path.splitext(os.path.basename(args.csv_path))[0]
    
    print(f"\nParsing: {args.csv_path}")
    print("=" * 60)
    
    # Parse CSV
    data = parse_group_stats_from_csv(args.csv_path)
    print(f"✓ Loaded {data['n_timesteps']} timesteps × {data['n_runs']} runs")
    
    # Create timeline plot
    timeline_path = os.path.join(args.output_dir, f"{base_name}_group_timelines.png")
    print(f"\nGenerating timeline plot...")
    plot_group_timelines(data, timeline_path)
    
    # Create summary plot
    if not args.no_summary:
        summary_path = os.path.join(args.output_dir, f"{base_name}_group_summary.png")
        print(f"Generating summary statistics plot...")
        plot_statistics_summary(data, summary_path)
    
    print("\n" + "=" * 60)
    print(f"All done! Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()