import pandas as pd
import os
import re
import argparse

# ── Config ─────────────────────────────────────────────────────────────────────
VICON_BASE = "dataFromVicon"
SOUND_BASE = "dataFromSound"
REAL_BASE  = "dataFromReal/doa_results"
OUTPUT_DIR = "processData"

# (vicon_folder, sound_folder, real_folder, vicon_file_prefix, sound_real_file_prefix)
EXPERIMENT_GROUPS = [
    ("expA",  "expA",  "expA",  "expA",  "exA"),
    ("expBA", "expBA", "expBA", "expBA", "exBA"),
    ("expBB", "expBB", "expBB", "expBB", "exBB"),
    ("expBC", "expBC", "expBC", "expBC", "exBC"),
    ("expBD", "expBD", "expBD", "expBD", "exBD"),
    ("expCA", "expCA", "expCA", "expCA", "exCA"),
    ("expCB", "expCB", "expCB", "expCB", "exCB"),
    ("expCC", "expCC", "expCC", "expCC", "exCC"),
    ("expCD", "expCD", "expCD", "expCD", "exCD"),
    ("expDA", "expDA", "expDA", "expDA", "exDA"),
    ("expDB", "expDB", "expDB", "expDB", "exDB"),
    ("expDC", "expDC", "expDC", "expDC", "exDC"),
    ("expDD", "expDD", "expDD", "expDD", "exDD"),
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def is_plain_run_file(filename, file_prefix):
    """Accept only plain run files e.g. exA1.csv — reject _robot, _target1, _target2."""
    pattern = rf"^{re.escape(file_prefix)}\d+\.csv$"
    return bool(re.match(pattern, filename))


def collect_runs():
    """Walk all experiment groups and return a list of run dicts."""
    runs = []
    for (vicon_folder, sound_folder, real_folder,
         vicon_prefix, sound_real_prefix) in EXPERIMENT_GROUPS:

        real_dir  = os.path.join(REAL_BASE,  real_folder)
        sound_dir = os.path.join(SOUND_BASE, sound_folder)
        vicon_dir = os.path.join(VICON_BASE, vicon_folder)

        if not os.path.isdir(real_dir):
            print(f"  [SKIP] Real folder not found: {real_dir}")
            continue

        for filename in sorted(os.listdir(real_dir)):
            if not is_plain_run_file(filename, sound_real_prefix):
                continue

            run_number      = re.search(r"\d+", filename).group()
            vicon_filename  = f"{vicon_prefix}{run_number}.csv"
            output_filename = f"{vicon_folder}_{filename}"

            runs.append({
                "group":    vicon_folder,
                "run_file": filename,
                "vicon":    os.path.join(vicon_dir, vicon_filename),
                "sound":    os.path.join(sound_dir, filename),
                "real":     os.path.join(real_dir,  filename),
                "output":   os.path.join(OUTPUT_DIR, output_filename),
            })
    return runs


# ── Timestamp coverage check ───────────────────────────────────────────────────
def check_timestamp_coverage(real_path, other_path, other_name):
    """
    Compare the master (Real) timestamp range against another file.
    Returns coverage percentage and any warnings about uncovered portions.
    """
    real  = pd.read_csv(real_path,  usecols=["timestamp"])
    other = pd.read_csv(other_path, usecols=["timestamp"])
    real["timestamp"]  = pd.to_datetime(real["timestamp"])
    other["timestamp"] = pd.to_datetime(other["timestamp"])

    master_start = real["timestamp"].min()
    master_end   = real["timestamp"].max()
    other_start  = other["timestamp"].min()
    other_end    = other["timestamp"].max()

    overlap_start = max(master_start, other_start)
    overlap_end   = min(master_end,   other_end)
    has_overlap   = overlap_start <= overlap_end
    master_dur    = master_end - master_start
    overlap_dur   = (overlap_end - overlap_start) if has_overlap else pd.Timedelta(0)
    coverage_pct  = (overlap_dur / master_dur * 100) if master_dur.total_seconds() > 0 else 0.0

    uncovered_start_s = max(0.0, (other_start - master_start).total_seconds())
    uncovered_end_s   = max(0.0, (master_end  - other_end).total_seconds())

    warnings = []
    if not has_overlap:
        warnings.append(f"NO OVERLAP — {other_name} range does not intersect master at all")
    else:
        if uncovered_start_s > 0:
            warnings.append(
                f"Master starts {uncovered_start_s:.2f}s BEFORE {other_name} "
                f"({master_start.strftime('%H:%M:%S.%f')[:-3]} vs {other_start.strftime('%H:%M:%S.%f')[:-3]})"
            )
        if uncovered_end_s > 0:
            warnings.append(
                f"Master ends {uncovered_end_s:.2f}s AFTER {other_name} "
                f"({master_end.strftime('%H:%M:%S.%f')[:-3]} vs {other_end.strftime('%H:%M:%S.%f')[:-3]})"
            )

    return {
        "master_start": master_start, "master_end": master_end,
        "other_start":  other_start,  "other_end":  other_end,
        "coverage_pct": coverage_pct, "has_overlap": has_overlap,
        "warnings":     warnings,
    }


# ── Dry run ────────────────────────────────────────────────────────────────────
def dry_run(runs):
    """Preview all runs: check file existence and timestamp coverage of master vs sources."""
    print("\n" + "=" * 72)
    print("DRY RUN — no files will be written")
    print("=" * 72)

    total           = len(runs)
    n_missing       = 0
    n_no_overlap    = 0
    n_partial       = 0
    n_ready         = 0
    list_no_overlap = []
    list_partial    = []
    list_missing    = []

    for r in runs:
        vicon_ok = os.path.isfile(r["vicon"])
        sound_ok = os.path.isfile(r["sound"])
        real_ok  = os.path.isfile(r["real"])
        files_ok = vicon_ok and sound_ok and real_ok

        print(f"\n{'─' * 72}")
        print(f"  {r['group']} / {r['run_file']}  →  {r['output']}")

        # File existence check
        if not files_ok:
            n_missing += 1
            list_missing.append(f"{r['group']}/{r['run_file']}")
            if not real_ok:
                print(f"  ✗ [MISSING] Real  : {r['real']}")
            if not vicon_ok:
                print(f"  ✗ [MISSING] Vicon : {r['vicon']}")
            if not sound_ok:
                print(f"  ✗ [MISSING] Sound : {r['sound']}")
            continue

        # Timestamp coverage check (only if all files present)
        run_has_no_overlap = False
        run_has_warnings   = False

        for other_path, other_name in [(r["vicon"], "Vicon"), (r["sound"], "Sound")]:
            try:
                info = check_timestamp_coverage(r["real"], other_path, other_name)
                pct  = info["coverage_pct"]
                icon = "✓" if not info["warnings"] else "⚠"

                print(
                    f"  {icon} {other_name:5s} | "
                    f"master [{info['master_start'].strftime('%H:%M:%S')} -> "
                    f"{info['master_end'].strftime('%H:%M:%S')}] | "
                    f"{other_name} [{info['other_start'].strftime('%H:%M:%S')} -> "
                    f"{info['other_end'].strftime('%H:%M:%S')}] | "
                    f"coverage {pct:.1f}%"
                )
                for w in info["warnings"]:
                    print(f"        ⚠  {w}")

                if not info["has_overlap"]:
                    run_has_no_overlap = True
                elif info["warnings"]:
                    run_has_warnings = True

            except Exception as e:
                print(f"  ✗ Could not check timestamps for {other_name}: {e}")
                run_has_no_overlap = True

        if run_has_no_overlap:
            n_no_overlap += 1
            list_no_overlap.append(f"{r['group']}/{r['run_file']}")
        elif run_has_warnings:
            n_partial += 1
            list_partial.append(f"{r['group']}/{r['run_file']}")
        else:
            n_ready += 1

    # Summary
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Total runs            : {total}")
    print(f"  ✓ Fully covered       : {n_ready}")
    print(f"  ⚠ Partially covered   : {n_partial}")
    print(f"  ✗ No overlap          : {n_no_overlap}")
    print(f"  ✗ Missing files       : {n_missing}")

    if list_no_overlap:
        print(f"\n  ✗ No overlap runs:")
        for name in list_no_overlap:
            print(f"      - {name}")

    if list_partial:
        print(f"\n  ⚠ Partially covered runs:")
        for name in list_partial:
            print(f"      - {name}")

    if list_missing:
        print(f"\n  ✗ Missing file runs:")
        for name in list_missing:
            print(f"      - {name}")

    print(f"{'=' * 72}\n")


# ── Merge ──────────────────────────────────────────────────────────────────────
def merge_run(vicon_path, sound_path, real_path):
    """Load, sort, and merge three CSVs using Real as the master timestamp."""
    real  = pd.read_csv(real_path)
    vicon = pd.read_csv(vicon_path)
    sound = pd.read_csv(sound_path)

    real["timestamp"]  = pd.to_datetime(real["timestamp"])
    vicon["timestamp"] = pd.to_datetime(vicon["timestamp"])
    sound["timestamp"] = pd.to_datetime(sound["timestamp"])

    real_sorted  = real.sort_values("timestamp").reset_index(drop=True)
    vicon_sorted = vicon.sort_values("timestamp").reset_index(drop=True)
    sound_sorted = sound.sort_values("timestamp").reset_index(drop=True)

    # Real is master (left) — output rows == len(real)
    merged = pd.merge_asof(
        real_sorted,
        vicon_sorted.rename(columns={"timestamp": "timestamp_vicon"}),
        left_on="timestamp",
        right_on="timestamp_vicon",
        direction="nearest"
    ).drop(columns=["timestamp_vicon"])

    merged = pd.merge_asof(
        merged,
        sound_sorted.rename(columns={"timestamp": "timestamp_sound"}),
        left_on="timestamp",
        right_on="timestamp_sound",
        direction="nearest"
    ).drop(columns=["timestamp_sound"])

    return merged


def run_merge(runs):
    """Execute the merge for all runs, skipping any with missing files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total, success, skipped, failed = len(runs), 0, 0, 0

    for i, r in enumerate(runs, 1):
        prefix = f"[{i}/{total}] {r['group']}/{r['run_file']}"

        missing_files = [k for k in ("vicon", "sound", "real")
                         if not os.path.isfile(r[k])]
        if missing_files:
            print(f"{prefix}  ->  SKIPPED (missing: {', '.join(missing_files)})")
            skipped += 1
            continue

        try:
            merged = merge_run(r["vicon"], r["sound"], r["real"])
            merged.to_csv(r["output"], index=False)
            print(f"{prefix}  ->  OK  ({len(merged)} rows, {len(merged.columns)} cols)")
            success += 1
        except Exception as e:
            print(f"{prefix}  ->  FAILED: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Done — success: {success}  skipped: {skipped}  failed: {failed}")
    print("=" * 50 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Vicon, Sound, and Real CSVs.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview all runs: check file existence and timestamp coverage without writing."
    )
    args = parser.parse_args()

    runs = collect_runs()

    if args.dry_run:
        dry_run(runs)
    else:
        run_merge(runs)