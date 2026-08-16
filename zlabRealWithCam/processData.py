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
def dry_run(runs):

    print("\n" + "=" * 72)
    print("DRY RUN — no files will be written")
    print("=" * 72)

    total = len(runs)

    n_ready = 0
    n_partial = 0
    n_no_overlap = 0
    n_missing = 0

    for r in runs:

        print(f"\n{'─' * 72}")
        print(f"{r['name']}")

        ring_ok = os.path.isfile(r["ring"])
        qualis_ok = os.path.isfile(r["qualis"])
        sound_ok = os.path.isfile(r["sound"])

        if not ring_ok or not qualis_ok or not sound_ok:

            n_missing += 1

            if not ring_ok:
                print(f"✗ Missing Ring   : {r['ring']}")

            if not qualis_ok:
                print(f"✗ Missing Qualis : {r['qualis']}")
                
            if not sound_ok:
                print(f"✗ Missing Sound  : {r['sound']}")

            continue

        try:

            info = check_timestamp_coverage(
                r["ring"],
                r["qualis"],
                "Qualis"
            )

            pct = info["coverage_pct"]

            icon = "✓" if not info["warnings"] else "⚠"

            print(
                f"{icon} Coverage {pct:.1f}% | "
                f"Ring [{info['master_start']} -> {info['master_end']}] | "
                f"Qualis [{info['other_start']} -> {info['other_end']}]"
            )

            for w in info["warnings"]:
                print(f"    ⚠ {w}")

            if not info["has_overlap"]:
                n_no_overlap += 1
            elif info["warnings"]:
                n_partial += 1
            else:
                n_ready += 1

        except Exception as e:
            print(f"✗ Failed coverage check: {e}")
            n_no_overlap += 1

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(f"Total runs          : {total}")
    print(f"✓ Fully covered     : {n_ready}")
    print(f"⚠ Partial coverage  : {n_partial}")
    print(f"✗ No overlap        : {n_no_overlap}")
    print(f"✗ Missing files     : {n_missing}")

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