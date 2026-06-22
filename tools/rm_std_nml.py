#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.constants import EXT_WAV, KEY_NORM, KEY_STD

STD_SUFFIX = f"_{KEY_STD}{EXT_WAV}"
NML_SUFFIX = f"_{KEY_STD}_{KEY_NORM}{EXT_WAV}"


def is_target_file(path: Path) -> bool:
    name = path.name
    return path.is_file() and (name.endswith(STD_SUFFIX) or name.endswith(NML_SUFFIX))


def find_target_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob(f"*{EXT_WAV}") if is_target_file(p))


def remove_files(files: list[Path]) -> int:
    removed = 0
    for path in files:
        path.unlink()
        removed += 1
        print(f"Removed: {path}")
    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove standardized and normalized wav files under a directory.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Root directory to clean")
    parser.add_argument("--force", action="store_true", help="Delete files without prompting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.input_dir
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")

    targets = find_target_files(root)
    if not targets:
        print(f"⚠️  No standardized/normalized files found in: {root}")
        return

    print(f"Found {len(targets)} file(s) under '{root}'")
    if not args.force:
        answer = input("Delete them? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return

    removed = remove_files(targets)
    print(f"Removed {removed} file(s).")


if __name__ == "__main__":
    main()
