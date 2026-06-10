import pandas as pd
import os
import argparse

# ── Config ─────────────────────────────────────────────────────────────────────
QUALIS_BASE = "dataFromQualis"
SOUND_BASE  = "dataFromSound"
REAL_BASE   = "dataFromReal"
OUTPUT_DIR  = "ztProcessData"

QUALIS_COLS = ["timestamp", "robot_x", "robot_y", "speakerL_x", "speakerL_y", "speakerR_x", "speakerR_y"]


# ── File discovery ─────────────────────────────────────────────────────────────
def collect_runs():
    """
    Discover all CSVs in dataFromReal/ and match by name across the three sources.
    Each experiment shares the same filename across all three folders.
    """
    runs = []

    if not os.path.isdir(REAL_BASE):
        print(f"[ERROR] Real folder not found: {REAL_BASE}")
        return runs

    for filename in sorted(os.listdir(REAL_BASE)):
        if not filename.endswith(".csv"):
            continue

        runs.append({
            "name":   filename,
            "qualis": os.path.join(QUALIS_BASE, filename),
            "sound":  os.path.join(SOUND_BASE,  filename),
            "real":   os.path.join(REAL_BASE,   filename),
            "output": os.path.join(OUTPUT_DIR,  filename),
        })

    return runs


# ── Timestamp coverage check ───────────────────────────────────────────────────
def check_timestamp_coverage(real_path, other_path, other_name):
    """
    Compare the master (Real) timestamp range against another file.
    Returns coverage info and any warnings about uncovered portions.
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
    """Preview all runs: check file existence and timestamp coverage without writing."""
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
        qualis_ok = os.path.isfile(r["qualis"])
        sound_ok  = os.path.isfile(r["sound"])
        real_ok   = os.path.isfile(r["real"])
        files_ok  = qualis_ok and sound_ok and real_ok

        print(f"\n{'─' * 72}")
        print(f"  {r['name']}  →  {r['output']}")

        if not files_ok:
            n_missing += 1
            list_missing.append(r["name"])
            if not real_ok:
                print(f"  ✗ [MISSING] Real   : {r['real']}")
            if not qualis_ok:
                print(f"  ✗ [MISSING] Qualis : {r['qualis']}")
            if not sound_ok:
                print(f"  ✗ [MISSING] Sound  : {r['sound']}")
            continue

        run_has_no_overlap = False
        run_has_warnings   = False

        for other_path, other_name in [(r["qualis"], "Qualis"), (r["sound"], "Sound")]:
            try:
                info = check_timestamp_coverage(r["real"], other_path, other_name)
                pct  = info["coverage_pct"]
                icon = "✓" if not info["warnings"] else "⚠"

                print(
                    f"  {icon} {other_name:6s} | "
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
            list_no_overlap.append(r["name"])
        elif run_has_warnings:
            n_partial += 1
            list_partial.append(r["name"])
        else:
            n_ready += 1

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
def merge_run(qualis_path, sound_path, real_path):
    """Load, sort, and merge three CSVs using Real as the master timestamp."""
    real   = pd.read_csv(real_path)
    qualis = pd.read_csv(qualis_path)
    sound  = pd.read_csv(sound_path)

    real["timestamp"]   = pd.to_datetime(real["timestamp"])
    qualis["timestamp"] = pd.to_datetime(qualis["timestamp"])
    sound["timestamp"]  = pd.to_datetime(sound["timestamp"])

    real_sorted   = real.sort_values("timestamp").reset_index(drop=True)
    qualis_sorted = qualis.sort_values("timestamp").reset_index(drop=True)
    sound_sorted  = sound.sort_values("timestamp").reset_index(drop=True)

    # Merge Qualis (all columns) into Real master
    merged = pd.merge_asof(
        real_sorted,
        qualis_sorted[QUALIS_COLS].rename(columns={"timestamp": "timestamp_qualis"}),
        left_on="timestamp",
        right_on="timestamp_qualis",
        direction="nearest"
    ).drop(columns=["timestamp_qualis"])

    # Merge Sound (all columns) into result
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
        prefix = f"[{i}/{total}] {r['name']}"

        missing_files = [k for k in ("qualis", "sound", "real")
                         if not os.path.isfile(r[k])]
        if missing_files:
            print(f"{prefix}  ->  SKIPPED (missing: {', '.join(missing_files)})")
            skipped += 1
            continue

        try:
            merged = merge_run(r["qualis"], r["sound"], r["real"])
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
    parser = argparse.ArgumentParser(description="Merge Qualis, Sound, and Real CSVs.")
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