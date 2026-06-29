from __future__ import annotations

import sys


def write_progress(index: int, total: int) -> None:
    pct = 100.0 * index / total if total else 100.0
    width = 20
    filled = min(width, max(0, int(round(width * index / total)))) if total else width
    bar = "█" * filled + "░" * (width - filled)
    sys.stdout.write(f"\r   • Progress: |{bar}| {pct:6.2f}% ({index}/{total})")
    sys.stdout.flush()
