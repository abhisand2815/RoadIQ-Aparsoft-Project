
"""
RoadIQ | YOLO26 Model Weight Downloader
========================================
Downloads official YOLO26 pretrained checkpoints into:
    RoadIQ/weights/

Usage:
    python tools/download_weights.py --preset compact
    python tools/download_weights.py --preset all
    python tools/download_weights.py --preset yolo26

Presets:
    compact : Small selection for initial testing.
    all     : All configured YOLO26 task/size combinations.
    yolo26  : Alias for all YOLO26 checkpoints.

Notes:
    - Existing files are skipped.
    - Failed downloads are reported without stopping the entire run.
    - Custom RoadIQ checkpoints must be supplied separately.
"""

import argparse
import shutil
import sys
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# YOLO26 MODEL CATALOG
# ============================================================

SIZES = ("n", "s", "m", "l", "x")

# Task suffixes commonly provided by Ultralytics YOLO26.
# A checkpoint must actually exist in the installed release.
TASK_SUFFIXES = {
    "Detection": "",
    "Instance Segmentation": "-seg",
    "Pose Estimation": "-pose",
    "Oriented Bounding Boxes": "-obb",
    "Classification": "-cls",
}


def build_yolo26_catalog():
    """Build checkpoint names for the configured YOLO26 tasks."""
    return {
        task: [
            f"yolo26{size}{suffix}.pt"
            for size in SIZES
        ]
        for task, suffix in TASK_SUFFIXES.items()
    }


YOLO26_CATALOG = build_yolo26_catalog()

YOLO26_ALL = list(
    dict.fromkeys(
        filename
        for filenames in YOLO26_CATALOG.values()
        for filename in filenames
    )
)

PRESETS = {
    "compact": [
        "yolo26n.pt",
        "yolo26n-seg.pt",
        "yolo26n-pose.pt",
    ],
    "all": YOLO26_ALL,
    "yolo26": YOLO26_ALL,
}


# ============================================================
# MODEL DOWNLOAD
# ============================================================

def download_model(filename: str) -> bool:
    """
    Download one official YOLO26 checkpoint into weights/.

    Returns True when the file is present after downloading.
    """
    destination = WEIGHTS_DIR / filename

    if destination.is_file() and destination.stat().st_size > 0:
        print(f"[SKIP] {filename} already exists")
        return True

    print(f"\n[DOWNLOAD] {filename}")

    try:
        # Import only the loader needed for this operation.
        from ultralytics import YOLO

        # Ultralytics resolves official model names and downloads
        # missing checkpoints. Set cwd so relative downloads target
        # the project's weights directory.
        import os

        previous_directory = Path.cwd()

        try:
            os.chdir(WEIGHTS_DIR)
            model = YOLO(filename)
        finally:
            os.chdir(previous_directory)

        # Some versions expose the resolved checkpoint location.
        resolved_path = getattr(model, "ckpt_path", None)

        if resolved_path:
            resolved = Path(resolved_path).expanduser()

            if not resolved.is_absolute():
                resolved = WEIGHTS_DIR / resolved

            if (
                resolved.is_file()
                and resolved.resolve() != destination.resolve()
            ):
                shutil.copy2(resolved, destination)

        # Also check for a file created directly in weights/.
        if destination.is_file() and destination.stat().st_size > 0:
            print(f"[OK] {filename} -> {destination}")
            return True

        print(
            f"[WARNING] Ultralytics loaded {filename}, "
            "but the file was not confirmed in weights/."
        )
        print(
            "Check the Ultralytics download/cache location before "
            "assuming the checkpoint was saved locally."
        )
        return False

    except Exception as exc:
        print(f"[FAILED] {filename}: {exc}")
        return False


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download YOLO26 model weights for RoadIQ."
    )

    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS.keys()),
        default="yolo26",
        help=(
            "compact: small selection; "
            "all/yolo26: all configured YOLO26 checkpoints."
        ),
    )

    args = parser.parse_args()
    model_names = list(dict.fromkeys(PRESETS[args.preset]))

    print("=" * 60)
    print("RoadIQ | YOLO26 Weight Downloader")
    print("=" * 60)
    print(f"Preset       : {args.preset}")
    print(f"Target folder: {WEIGHTS_DIR}")
    print(f"Models       : {len(model_names)}")
    print("=" * 60)

    successful = 0
    failed = []

    for index, filename in enumerate(model_names, start=1):
        print(f"\n[{index}/{len(model_names)}] {filename}")

        if download_model(filename):
            successful += 1
        else:
            failed.append(filename)

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Successful / already present: {successful}")
    print(f"Failed or unverified        : {len(failed)}")
    print(f"Destination                 : {WEIGHTS_DIR}")

    if failed:
        print("\nCheck these checkpoints:")
        for filename in failed:
            print(f"  - {filename}")

    print("\nLocal YOLO26 checkpoints:")
    for path in sorted(WEIGHTS_DIR.glob("yolo26*.pt")):
        print(f"  {path.name} ({path.stat().st_size / (1024 * 1024):.1f} MB)")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
