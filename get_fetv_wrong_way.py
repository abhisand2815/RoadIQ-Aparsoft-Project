import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent

DATASETS_DIR = ROOT / "datasets"
OUTPUT_DIR = ROOT / "videos" / "fetv_wrong_way"


def find_fetv_json():
    for file in DATASETS_DIR.rglob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                continue

            for item in data:
                if (
                    isinstance(item, dict)
                    and "clip_name" in item
                    and "answer_violation_type" in item
                ):
                    return file

        except Exception:
            pass

    return None


def find_video(clip_name):
    matches = list(DATASETS_DIR.rglob(clip_name))

    if matches:
        return matches[0]

    return None


def main():

    print("\nSearching FETV dataset...\n")

    if not DATASETS_DIR.exists():
        print("ERROR: datasets folder not found:")
        print(DATASETS_DIR)
        return

    json_file = find_fetv_json()

    if json_file is None:
        print("ERROR: FETV JSON file not found.")
        print("\nJSON files found:")

        for file in DATASETS_DIR.rglob("*.json"):
            print(file)

        return

    print("JSON found:")
    print(json_file)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    count = 0

    print("\nFinding wrong-way videos...\n")

    for item in data:

        violation = str(
            item.get(
                "answer_violation_type",
                ""
            )
        ).strip().lower()

        if violation != "wrong_way":
            continue

        clip_name = item.get("clip_name")

        if not clip_name:
            continue

        source = find_video(clip_name)

        if source is None:
            print(f"Missing: {clip_name}")
            continue

        destination = OUTPUT_DIR / clip_name

        shutil.copy2(
            source,
            destination
        )

        vehicle = item.get(
            "answer_violator_type",
            "unknown"
        )

        print(
            f"Copied: {clip_name} | {vehicle}"
        )

        count += 1

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)

    print(f"Wrong-way videos: {count}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()