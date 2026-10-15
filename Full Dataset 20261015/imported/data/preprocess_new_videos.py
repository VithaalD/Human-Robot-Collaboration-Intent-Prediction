# Jason's file for part 1 of preprocessing video data
#
# preprocess_new_videos.py is frame extraction specifically (currently generalizing)

# note: moving forward, lets standardize calling passthrough video directories are passthrough or passthru
# (either works im just tryna generalize the pipeline eventually so preprocessing becomes even easier)

from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent

SOURCE_ROOT = Path(
    r"C:\Users\qingo\Camera Recordings"
)

OUTPUT_ROOT = SCRIPT_DIR / "data" / "image_data"


CLASS_MAP = {
    "INTERACTION": "INTERACTION",
    "PASSTHRU": "PASSTHRU",
    "WAIT": "WAIT",
}


# --------------------------------------------------
# Process one video
# --------------------------------------------------

def process_video(video_path, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = output_dir / "image_%05d.jpg"

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", "scale=-1:240",
        str(output_pattern),
    ]

    print(f"\nProcessing: {video_path.name}")

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print("  FAILED")
        print(result.stderr)
        return None

    frames = sorted(output_dir.glob("image_*.jpg"))
    n_frames = len(frames)

    if n_frames == 0:
        print("  FAILED: FFmpeg produced no frames.")
        return None

    # Write repository metadata file.
    n_frames_file = output_dir / "n_frames"
    n_frames_file.write_text(str(n_frames))

    print(f"  Extracted {n_frames} frames")

    return n_frames


# --------------------------------------------------
# Main dataset loop
# --------------------------------------------------

def main():
    print("Source root:", SOURCE_ROOT)
    print("Output root:", OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    total_videos = 0
    total_frames = 0
    failures = []

    class_counts = {}

    for source_class, output_class in CLASS_MAP.items():

        source_dir = SOURCE_ROOT / source_class / "videos"
        output_class_dir = OUTPUT_ROOT / output_class

        print("\n" + "=" * 60)
        print(f"{source_class} -> {output_class}")
        print("=" * 60)

        if not source_dir.exists():
            print(f"ERROR: Folder does not exist: {source_dir}")
            sys.exit(1)

        videos = sorted(source_dir.glob("*.avi"))

        class_counts[output_class] = {
            "found": len(videos),
            "processed": 0,
        }

        print(f"Found {len(videos)} videos.")

        for video_path in videos:

            # Example:
            #
            # session_...avi
            #
            # becomes:
            #
            # data/image_data/INTERACTION/session_.../

            video_name = video_path.stem
            output_dir = output_class_dir / video_name

            n_frames = process_video(
                video_path,
                output_dir
            )

            if n_frames is None:
                failures.append(str(video_path))
                continue

            class_counts[output_class]["processed"] += 1
            total_videos += 1
            total_frames += n_frames

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    for class_name, counts in class_counts.items():

        print(
            f"{class_name:12} "
            f"{counts['processed']}/{counts['found']} processed"
        )

    print()
    print(f"Total videos: {total_videos}")
    print(f"Total frames: {total_frames}")
    print(f"Failures:     {len(failures)}")

    if failures:
        print("\nFailed videos:")

        for video in failures:
            print(f"  {video}")


if __name__ == "__main__":
    main()
