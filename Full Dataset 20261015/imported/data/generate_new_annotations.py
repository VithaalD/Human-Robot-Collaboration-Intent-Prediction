from pathlib import Path
from collections import defaultdict, Counter
import json
import random
import sys


# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

IMAGE_ROOT = SCRIPT_DIR / "data" / "image_data"
ANNOTATION_DIR = SCRIPT_DIR / "data" / "annotation"
OUTPUT_JSON = ANNOTATION_DIR / "new_dataset.json"

CLASSES = [
    "INTERACTION",
    "PASSTHRU",
    "WAIT",
]

PARTICIPANTS = [
    "jason",
    "vithaal",
]

# Fixed seed means running this script again gives the same split.
RANDOM_SEED = 42


# ============================================================
# Identify participant
# ============================================================

def get_participant(video_name):

    lower_name = video_name.lower()

    matches = [
        participant
        for participant in PARTICIPANTS
        if participant in lower_name
    ]

    if len(matches) == 0:
        raise ValueError(
            f"Could not identify participant from: {video_name}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple participant names found in: {video_name}"
        )

    return matches[0]


# ============================================================
# Discover processed videos
# ============================================================

def discover_dataset():

    records = []

    for class_name in CLASSES:

        class_dir = IMAGE_ROOT / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        for video_dir in sorted(class_dir.iterdir()):

            if not video_dir.is_dir():
                continue

            n_frames_file = video_dir / "n_frames"

            if not n_frames_file.exists():
                raise FileNotFoundError(
                    f"Missing n_frames: {video_dir}"
                )

            n_frames = int(n_frames_file.read_text().strip())

            participant = get_participant(video_dir.name)

            records.append({
                "video_id": video_dir.name,
                "class": class_name,
                "participant": participant,
                "n_frames": n_frames,
            })

    return records


# ============================================================
# Print dataset audit
# ============================================================

def print_dataset_summary(records):

    print("\n" + "=" * 65)
    print("DATASET AUDIT")
    print("=" * 65)

    print(
        f"{'Class':15}"
        f"{'Jason':>10}"
        f"{'Vithaal':>10}"
        f"{'Total':>10}"
    )

    print("-" * 45)

    for class_name in CLASSES:

        class_records = [
            r for r in records
            if r["class"] == class_name
        ]

        counts = Counter(
            r["participant"] for r in class_records
        )

        print(
            f"{class_name:15}"
            f"{counts['jason']:>10}"
            f"{counts['vithaal']:>10}"
            f"{len(class_records):>10}"
        )

    print("-" * 45)

    participant_counts = Counter(
        r["participant"] for r in records
    )

    print(
        f"{'TOTAL':15}"
        f"{participant_counts['jason']:>10}"
        f"{participant_counts['vithaal']:>10}"
        f"{len(records):>10}"
    )


# ============================================================
# Split one participant/class group
# ============================================================

def split_group(records, rng):
    """
    Split one class + participant group.

    For groups of approximately 9-10 videos this normally gives:

        10 videos -> 7 train, 1 validation, 2 test
         9 videos -> 6 train, 1 validation, 2 test

    Combining Jason + Vithaal therefore keeps each class and
    participant represented in all three subsets.
    """

    records = records.copy()
    rng.shuffle(records)

    n = len(records)

    if n < 3:
        raise ValueError(
            "Need at least 3 videos in every "
            "class/participant group."
        )

    # Roughly 80 / 10 / 10, while guaranteeing validation
    # and test examples.
    n_val = max(1, round(n * 0.10))
    n_test = max(1, round(n * 0.10))


    if n >= 8:
        n_test = 3

    n_train = n - n_val - n_test

    if n_train < 1:
        raise ValueError(
            f"Not enough training samples in group of size {n}."
        )

    train = records[:n_train]
    validation = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]

    return train, validation, test


# ============================================================
# Participant-aware, class-aware split
# ============================================================

def create_split(records):

    rng = random.Random(RANDOM_SEED)

    grouped = defaultdict(list)

    for record in records:
        key = (
            record["class"],
            record["participant"]
        )

        grouped[key].append(record)

    splits = {
        "training": [],
        "validation": [],
        "testing": [],
    }

    for class_name in CLASSES:
        for participant in PARTICIPANTS:

            group = grouped[
                (class_name, participant)
            ]

            if not group:
                raise ValueError(
                    f"No videos found for "
                    f"{class_name}/{participant}"
                )

            train, val, test = split_group(
                group,
                rng
            )

            splits["training"].extend(train)
            splits["validation"].extend(val)
            splits["testing"].extend(test)

    # Sort for easier inspection/debugging.
    for subset in splits:
        splits[subset].sort(
            key=lambda r: (
                r["class"],
                r["participant"],
                r["video_id"],
            )
        )

    return splits


# ============================================================
# Print split statistics
# ============================================================

def print_split_summary(splits):

    print("\n" + "=" * 65)
    print("PROPOSED SPLIT")
    print("=" * 65)

    for subset, records in splits.items():

        print(f"\n{subset.upper()} ({len(records)} videos)")
        print("-" * 65)

        for class_name in CLASSES:

            class_records = [
                r for r in records
                if r["class"] == class_name
            ]

            counts = Counter(
                r["participant"]
                for r in class_records
            )

            print(
                f"{class_name:15} "
                f"Jason={counts['jason']:2}   "
                f"Vithaal={counts['vithaal']:2}   "
                f"Total={len(class_records):2}"
            )


# ============================================================
# Print exact video assignments
# ============================================================

def print_assignments(splits):

    print("\n" + "=" * 65)
    print("VIDEO ASSIGNMENTS")
    print("=" * 65)

    for subset, records in splits.items():

        print(f"\n--- {subset.upper()} ---")

        for record in records:
            print(
                f"{record['class']:12} | "
                f"{record['participant']:7} | "
                f"{record['video_id']}"
            )


# ============================================================
# Build repository-compatible JSON
# ============================================================

def build_annotation_json(splits):

    database = {}

    for subset, records in splits.items():

        for record in records:

            video_id = record["video_id"]

            if video_id in database:
                raise ValueError(
                    f"Duplicate video ID: {video_id}"
                )

            database[video_id] = {
                "subset": subset,
                "annotations": {
                    "label": record["class"]
                }
            }

    return {
        "labels": CLASSES,
        "database": database,
    }


# ============================================================
# Validate the final split
# ============================================================

def validate_split(records, splits):

    original_ids = {
        r["video_id"]
        for r in records
    }

    split_ids = []

    for subset_records in splits.values():
        split_ids.extend(
            r["video_id"]
            for r in subset_records
        )

    # Check for duplicates across subsets.
    if len(split_ids) != len(set(split_ids)):
        raise ValueError(
            "A video appears in more than one subset."
        )

    # Check that nothing disappeared.
    if set(split_ids) != original_ids:
        raise ValueError(
            "Split does not contain exactly the original videos."
        )

    print("\nSplit validation passed.")
    print(f"All {len(original_ids)} videos accounted for exactly once.")


# ============================================================
# Main
# ============================================================

def main():

    print("Image root:")
    print(IMAGE_ROOT)

    records = discover_dataset()

    print_dataset_summary(records)

    if len(records) != 163:
        print(
            f"\nWARNING: Expected 163 videos, "
            f"but discovered {len(records)}."
        )

    splits = create_split(records)

    print_split_summary(splits)

    validate_split(records, splits)

    print_assignments(splits)

    annotation_data = build_annotation_json(
        splits
    )

    ANNOTATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            annotation_data,
            f,
            indent=4
        )

    print("\n" + "=" * 65)
    print("ANNOTATION FILE WRITTEN")
    print("=" * 65)
    print(OUTPUT_JSON)


if __name__ == "__main__":
    main()