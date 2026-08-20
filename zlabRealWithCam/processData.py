import pandas as pd
import os
import argparse

# ── Config ─────────────────────────────────────────────────────────────────────
RING_BASE   = "dataFromRing"
VICON_BASE = "dataFromVicon"
SOUND_BASE = "dataFromSound"
OUTPUT_DIR  = "mergedData"

VICON_COLS = [
    "timestamp",
    "robot_x",
    "robot_y",
    "speakerL_x",
    "speakerL_y",
    "speakerR_x",
    "speakerR_y",
]

SOUND_COLS = [
    "timestamp",
    "left_volume",
    "right_volume",
]



# ── File discovery ─────────────────────────────────────────────────────────────
def collect_runs():
    """
    Discover all CSVs in dataFromRing/ and match by filename
    in dataFromQualis/.
    """

    runs = []

    if not os.path.isdir(RING_BASE):
        print(f"[ERROR] Ring folder not found: {RING_BASE}")
        return runs

    for filename in sorted(os.listdir(RING_BASE)):
        if not filename.endswith(".csv"):
            continue

        runs.append({
            "name": filename,
            "ring": os.path.join(RING_BASE, filename),
            "qualis": os.path.join(VICON_BASE, filename),
            "sound": os.path.join(SOUND_BASE, filename),
            "output": os.path.join(OUTPUT_DIR, filename),
        })

    return runs


# ── Timestamp coverage check ───────────────────────────────────────────────────
def check_timestamp_coverage(master_path, other_path, other_name):
    """
    Compare master (Ring) timestamp range against Qualis.
    """

    master = pd.read_csv(master_path, usecols=["timestamp"])
    other = pd.read_csv(other_path, usecols=["timestamp"])

    master["timestamp"] = pd.to_datetime(master["timestamp"])
    other["timestamp"] = pd.to_datetime(other["timestamp"])

    master_start = master["timestamp"].min()
    master_end = master["timestamp"].max()

    other_start = other["timestamp"].min()
    other_end = other["timestamp"].max()

    overlap_start = max(master_start, other_start)
    overlap_end = min(master_end, other_end)

    has_overlap = overlap_start <= overlap_end

    master_duration = master_end - master_start

    overlap_duration = (
        overlap_end - overlap_start
        if has_overlap
        else pd.Timedelta(0)
    )

    coverage_pct = (
        overlap_duration / master_duration * 100
        if master_duration.total_seconds() > 0
        else 0.0
    )

    uncovered_start_s = max(
        0.0,
        (other_start - master_start).total_seconds()
    )

    uncovered_end_s = max(
        0.0,
        (master_end - other_end).total_seconds()
    )

    warnings = []

    if not has_overlap:
        warnings.append(
            f"NO OVERLAP — {other_name} does not intersect master"
        )
    else:
        if uncovered_start_s > 0:
            warnings.append(
                f"Master starts {uncovered_start_s:.2f}s BEFORE "
                f"{other_name}"
            )

        if uncovered_end_s > 0:
            warnings.append(
                f"Master ends {uncovered_end_s:.2f}s AFTER "
                f"{other_name}"
            )

    return {
        "master_start": master_start,
        "master_end": master_end,
        "other_start": other_start,
        "other_end": other_end,
        "coverage_pct": coverage_pct,
        "has_overlap": has_overlap,
        "warnings": warnings,
    }


# ── Dry run ────────────────────────────────────────────────────────────────────
# ── Dry run ────────────────────────────────────────────────────────────────────
def dry_run(runs):

    print("\n" + "=" * 72)
    print("DRY RUN — no files will be written")
    print("=" * 72)

    total = len(runs)

    fully_covered = []
    partial = []
    no_overlap = []
    missing = []

    for r in runs:

        ring_ok = os.path.isfile(r["ring"])
        qualis_ok = os.path.isfile(r["qualis"])
        sound_ok = os.path.isfile(r["sound"])

        # ── Missing files ──────────────────────────────────────────────────────
        if not ring_ok or not qualis_ok or not sound_ok:

            missing_files = []

            if not ring_ok:
                missing_files.append("Ring")

            if not qualis_ok:
                missing_files.append("Qualis")

            if not sound_ok:
                missing_files.append("Sound")

            missing.append({
                "name": r["name"],
                "files": missing_files,
            })

            continue

        try:

            info = check_timestamp_coverage(
                r["ring"],
                r["qualis"],
                "Qualis"
            )

            result = {
                "name": r["name"],
                "coverage": info["coverage_pct"],
                "warnings": info["warnings"],
                "has_overlap": info["has_overlap"],
            }

            if not info["has_overlap"]:
                no_overlap.append(result)

            elif info["warnings"]:
                partial.append(result)

            else:
                fully_covered.append(result)

        except Exception as e:

            no_overlap.append({
                "name": r["name"],
                "coverage": 0.0,
                "warnings": [f"Failed coverage check: {e}"],
                "has_overlap": False,
            })

    # ── Sort partial overlaps by coverage ──────────────────────────────────────
    partial.sort(key=lambda x: x["name"])

    # ── Report ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("DRY RUN SUMMARY")
    print("=" * 72)

    # Fully covered
    print(f"\n✓ FULLY COVERED ({len(fully_covered)})")

    if fully_covered:
        for r in fully_covered:
            print(
                f"  {r['name']:<40} "
                f"{r['coverage']:>6.1f}%"
            )
    else:
        print("  None")

    # Partial overlap
    print(f"\n⚠ PARTIAL OVERLAP ({len(partial)})")

    if partial:
        for r in partial:

            warning_text = "; ".join(r["warnings"])

            print(
                f"  {r['name']:<40} "
                f"{r['coverage']:>6.1f}%   "
                f"{warning_text}"
            )
    else:
        print("  None")

    # No overlap
    print(f"\n✗ NO OVERLAP ({len(no_overlap)})")

    if no_overlap:
        for r in no_overlap:

            warning_text = "; ".join(r["warnings"])

            print(
                f"  {r['name']:<40} "
                f"{warning_text}"
            )
    else:
        print("  None")

    # Missing files
    print(f"\n✗ MISSING FILES ({len(missing)})")

    if missing:
        for r in missing:
            print(
                f"  {r['name']:<40} "
                f"missing: {', '.join(r['files'])}"
            )
    else:
        print("  None")

    # ── Final counts ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("TOTAL")
    print("=" * 72)

    print(f"Total runs          : {total}")
    print(f"✓ Fully covered     : {len(fully_covered)}")
    print(f"⚠ Partial coverage  : {len(partial)}")
    print(f"✗ No overlap        : {len(no_overlap)}")
    print(f"✗ Missing files     : {len(missing)}")

    print("=" * 72 + "\n")
    
    

# ── Merge ──────────────────────────────────────────────────────────────────────
def merge_run(qualis_path, sound_path, ring_path):

    ring = pd.read_csv(ring_path)
    qualis = pd.read_csv(qualis_path)
    sound = pd.read_csv(sound_path)

    ring["timestamp"] = pd.to_datetime(ring["timestamp"])
    qualis["timestamp"] = pd.to_datetime(qualis["timestamp"])
    sound["timestamp"] = pd.to_datetime(sound["timestamp"])

    ring_sorted = (
        ring
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    qualis_sorted = (
        qualis
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    sound_sorted = (
        sound
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    merged = pd.merge_asof(
        ring_sorted,
        qualis_sorted[VICON_COLS].rename(
            columns={"timestamp": "timestamp_qualis"}
        ),
        left_on="timestamp",
        right_on="timestamp_qualis",
        direction="nearest"
    )
    
    merged = pd.merge_asof(
        merged.sort_values("timestamp"),
        sound_sorted[SOUND_COLS].rename(
            columns={"timestamp": "timestamp_sound"}
        ),
        left_on="timestamp",
        right_on="timestamp_sound",
        direction="nearest"
    )

    # Difference between Ring timestamp and matched Qualis timestamp
    merged["qualis_time_diff_ms"] = (
        merged["timestamp"] - merged["timestamp_qualis"]
    ).abs().dt.total_seconds() * 1000
    merged["sound_time_diff_ms"] = (
        merged["timestamp"] - merged["timestamp_sound"]
    ).abs().dt.total_seconds() * 1000

    merged.drop(
        columns=["timestamp_qualis"],
        inplace=True
    )
    merged.drop(
        columns=["timestamp_sound"],
        inplace=True
    )

    return merged


# ── Execute merge ──────────────────────────────────────────────────────────────
def run_merge(runs):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = len(runs)
    success = 0
    skipped = 0
    failed = 0

    for i, r in enumerate(runs, start=1):

        prefix = f"[{i}/{total}] {r['name']}"

        missing = []

        if not os.path.isfile(r["ring"]):
            missing.append("ring")

        if not os.path.isfile(r["qualis"]):
            missing.append("qualis")

        if missing:
            print(
                f"{prefix} -> SKIPPED "
                f"(missing: {', '.join(missing)})"
            )
            skipped += 1
            continue

        try:

            merged = merge_run(
                r["qualis"],
                r["sound"],
                r["ring"]
            )

            merged.to_csv(
                r["output"],
                index=False
            )

            print(
                f"{prefix} -> OK "
                f"({len(merged)} rows, "
                f"{len(merged.columns)} cols)"
            )

            success += 1

        except Exception as e:

            print(
                f"{prefix} -> FAILED: {e}"
            )

            failed += 1

    print("\n" + "=" * 60)
    print(
        f"Done — success: {success}  "
        f"skipped: {skipped}  "
        f"failed: {failed}"
    )
    print("=" * 60 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Merge Ring (master) with Qualis."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check timestamp coverage only."
    )

    args = parser.parse_args()

    runs = collect_runs()

    if args.dry_run:
        dry_run(runs)
    else:
        run_merge(runs)