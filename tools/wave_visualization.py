#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import scipy.signal
import soundfile as sf
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox, QFrame, QHBoxLayout, QLabel, QMainWindow, QSizePolicy, QScrollArea, QVBoxLayout, QWidget


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_WAVEFORM_POINTS = 20_000


def display_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def discover_audio_paths(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {input_dir}")
    paths = sorted(path.resolve() for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav")
    if not paths:
        raise FileNotFoundError(f"No WAV files found under: {input_dir}")
    return paths


def load_waveform(audio_path: Path, sample_rate: int, max_points: int) -> tuple[np.ndarray, np.ndarray, int]:
    data, sr = sf.read(str(audio_path), always_2d=True, dtype="float32")
    y = data.mean(axis=1).astype(np.float32, copy=False)
    if y.size and sr != sample_rate:
        g = math.gcd(sample_rate, sr)
        y = scipy.signal.resample_poly(y, sample_rate // g, sr // g).astype(np.float32, copy=False)
        sr = sample_rate
    if not y.size:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32), sr
    duration = float(y.size) / float(sr)
    if y.size > max_points:
        y = y[:: int(math.ceil(y.size / max_points))]
    x = np.linspace(0.0, duration, num=y.size, endpoint=False, dtype=np.float32)
    return x, y, sr


class ClickablePlotWidget(pg.PlotWidget):
    pass


class WavePanel(QWidget):
    def __init__(self, audio_path: Path, waveform_sample_rate: int, waveform_points: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.audio_path = audio_path
        self.waveform_sample_rate = waveform_sample_rate
        self.waveform_points = waveform_points
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        header_row = QWidget()
        header_layout = QHBoxLayout(header_row)
        header_layout.setContentsMargins(2, 0, 2, 0)
        header_layout.setSpacing(8)

        header = QLabel(display_relative(audio_path))
        header.setTextInteractionFlags(Qt.TextSelectableByMouse)
        header.setStyleSheet("color: #6b7280; font-size: 11px; font-weight: 600;")
        header_layout.addWidget(header, 1)
        outer.addWidget(header_row)

        self.plot = ClickablePlotWidget()
        self.plot.setBackground("w")
        self.plot.setMinimumHeight(220)
        self.plot.setMaximumHeight(260)
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Amplitude")
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.getAxis("bottom").setPen(pg.mkPen("#94a3b8"))
        self.plot.getAxis("left").setPen(pg.mkPen("#94a3b8"))
        self.plot.getAxis("bottom").setTextPen(pg.mkPen("#6b7280"))
        self.plot.getAxis("left").setTextPen(pg.mkPen("#6b7280"))
        outer.addWidget(self.plot)

        x, y, sr = load_waveform(audio_path, waveform_sample_rate, waveform_points)
        self.plot.plot(x, y, pen=pg.mkPen((55, 65, 81), width=1.1))
        y_min = float(np.min(y)) if y.size else -1.0
        y_max = float(np.max(y)) if y.size else 1.0
        if abs(y_max - y_min) < 1e-6:
            y_min -= 1.0
            y_max += 1.0
        self.plot.setLimits(xMin=0.0, xMax=max(0.1, float(x[-1]) if x.size else 0.1))
        self.plot.setXRange(0.0, max(0.1, float(x[-1]) if x.size else 0.1), padding=0.01)
        self.plot.setYRange(y_min * 1.15, y_max * 1.15, padding=0.02)

        info = QWidget()
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(6)
        summary = QLabel(f"{audio_path.name} | sample rate={sr} Hz")
        summary.setStyleSheet("color: #111827; font-size: 13px; font-weight: 700;")
        info_layout.addWidget(summary)
        outer.addWidget(info)


class WaveVisualizationWindow(QMainWindow):
    def __init__(self, audio_paths: list[Path], waveform_sample_rate: int, waveform_points: int) -> None:
        super().__init__()
        self.audio_paths = audio_paths
        self.waveform_sample_rate = waveform_sample_rate
        self.waveform_points = waveform_points
        self.panels: list[WavePanel] = []

        self.setWindowTitle("Wave Visualization")
        self.resize(1400, 1000)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)

        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)

        top_label = QLabel("Source audio")
        top_label.setStyleSheet("font-size: 13px; font-weight: 700; color: #111827;")
        top_layout.addWidget(top_label)

        self.audio_combo = QComboBox()
        self.audio_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        for idx, path in enumerate(self.audio_paths):
            self.audio_combo.addItem(path.name, userData=idx)
        self.audio_combo.currentIndexChanged.connect(self._on_audio_changed)
        top_layout.addWidget(self.audio_combo, 1)

        self.group_count_label = QLabel("")
        self.group_count_label.setStyleSheet("color: #6b7280; font-size: 12px;")
        top_layout.addWidget(self.group_count_label)
        root_layout.addWidget(top_bar)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self.scroll_container = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_container)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(18)
        self.scroll_layout.addStretch(1)
        self.scroll_area.setWidget(self.scroll_container)
        root_layout.addWidget(self.scroll_area, 1)

        self.setCentralWidget(root)
        self._load_audio(self.audio_paths[0] if self.audio_paths else None)

    def _clear_panels(self) -> None:
        while self.panels:
            panel = self.panels.pop()
            panel.setParent(None)
            panel.deleteLater()

    def _on_audio_changed(self, index: int) -> None:
        if 0 <= index < len(self.audio_paths):
            self._load_audio(self.audio_paths[index])

    def _load_audio(self, audio_path: Path | None) -> None:
        self._clear_panels()
        if audio_path is None:
            self.group_count_label.setText("No WAV files found.")
            return
        self.group_count_label.setText(f"1 WAV file selected")
        panel = WavePanel(audio_path, self.waveform_sample_rate, self.waveform_points, parent=self.scroll_container)
        self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, panel)
        self.panels.append(panel)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wave visualization.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing WAV files.")
    parser.add_argument("--waveform-sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help=f"Waveform display sample rate (default: {DEFAULT_SAMPLE_RATE}).")
    parser.add_argument("--waveform-points", type=int, default=DEFAULT_WAVEFORM_POINTS, help=f"Maximum number of waveform points (default: {DEFAULT_WAVEFORM_POINTS}).")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    audio_paths = discover_audio_paths(args.input_dir)

    print(f"Scanned {len(audio_paths)} audio file(s) under {args.input_dir}")

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Wave Visualization")
    app.setStyleSheet(
        """
        QWidget {
            background-color: #ffffff;
            color: #111827;
            font-family: "Inter", "Helvetica Neue", sans-serif;
        }
        QComboBox, QSlider, QPushButton, QToolButton {
            font-size: 12px;
        }
        QComboBox {
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            padding: 4px 8px;
            min-height: 24px;
        }
        QComboBox:hover {
            border: 1px solid #94a3b8;
        }
        QComboBox::drop-down {
            border: none;
            width: 24px;
            background: transparent;
        }
        QComboBox::down-arrow {
            width: 10px;
            height: 10px;
        }
        QPushButton, QToolButton {
            border-radius: 10px;
        }
        QToolButton {
            padding: 6px 10px;
        }
        QScrollArea {
            border: none;
        }
        """
    )

    window = WaveVisualizationWindow(
        audio_paths=audio_paths,
        waveform_sample_rate=args.waveform_sample_rate,
        waveform_points=args.waveform_points,
    )
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
