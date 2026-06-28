#!/usr/bin/env python3

import argparse
import soundfile as sf
from pathlib import Path
from collections import defaultdict


def get_duration_seconds(path: Path):
    try:
        audio = sf.info(str(path))
    except Exception:
        return None
    return getattr(audio, "duration", None)


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:05.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    stats = defaultdict(lambda: {"count": 0, "total_seconds": 0.0})

    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() != ".wav":
            continue

        try:
            duration = get_duration_seconds(path)
        except Exception:
            continue

        if duration is None:
            continue

        folder = path.parent.relative_to(input_dir)
        folder = str(folder) if str(folder) != "." else "."

        stats[folder]["count"] += 1
        stats[folder]["total_seconds"] += duration

    print(f"{'Folder':<50} {'Files':>8} {'Avg Duration':>15} {'Total Hours':>12}")
    print("-" * 90)

    for folder in sorted(stats):
        count = stats[folder]["count"]
        total_seconds = stats[folder]["total_seconds"]
        avg_seconds = total_seconds / count

        print(
            f"{folder:<50} "
            f"{count:>8} "
            f"{format_duration(avg_seconds):>15} "
            f"{total_seconds / 3600:>12.2f}"
        )


if __name__ == "__main__":
    main()
