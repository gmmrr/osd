from pathlib import Path
import argparse
import soundfile as sf


def format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="Root folder containing wav files")
    args = parser.parse_args()

    root = Path(args.root)
    wav_files = sorted(root.rglob("*.wav"))

    if not wav_files:
        print(f"No .wav files found under: {root}")
        return

    total_seconds = 0.0
    total_files = 0
    by_parent = {}

    for wav_path in wav_files:
        try:
            info = sf.info(str(wav_path))
            duration = info.frames / info.samplerate
        except Exception as e:
            print(f"[ERROR] {wav_path}: {e}")
            continue

        total_seconds += duration
        total_files += 1

        # group by immediate parent folder
        parent = str(wav_path.parent.relative_to(root))
        by_parent.setdefault(parent, {"files": 0, "seconds": 0.0})
        by_parent[parent]["files"] += 1
        by_parent[parent]["seconds"] += duration

    print("Root:", root)
    print("WAV files:", total_files)
    print("Total duration:", format_duration(total_seconds))
    print("Total hours:", round(total_seconds / 3600, 4))

    if total_files > 0:
        print("Average duration:", format_duration(total_seconds / total_files))

    print()
    print("By folder:")
    for parent, stats in sorted(by_parent.items()):
        seconds = stats["seconds"]
        files = stats["files"]
        print(
            f"{parent:40s} "
            f"files={files:6d} "
            f"duration={format_duration(seconds)} "
            f"hours={seconds / 3600:.4f}"
        )


if __name__ == "__main__":
    main()
