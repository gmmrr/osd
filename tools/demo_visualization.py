#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import scipy.signal
import soundfile as sf
from PySide6.QtCore import QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtMultimedia import QAudioDevice, QAudioOutput, QMediaDevices, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


DEFAULT_WAVEFORM_SAMPLE_RATE = 16_000
DEFAULT_WAVEFORM_POINTS = 5_000
GT_OVERLAP_COLOR = "#f59e0b"
HYP_OVERLAP_COLOR = "#ef4444"
PALETTE = ["#ff0000", "#00a000", "#0057ff", "#00c7d9", "#ff00c8", "#ffd400", "#7a4cff", "#00b894"]


@dataclass(frozen=True)
class Segment:
    speaker: str
    start: float
    end: float


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    duration: float


@dataclass(frozen=True)
class Record:
    uri: str
    audio_path: Path
    duration: float
    speakers: list[str]
    gt_segments: list[Segment]
    gt_overlaps: list[Interval]
    hyp_overlaps: list[Interval]
    hyp_threshold: float | None


def discover_audio_paths(root: Path) -> list[Path]:
    audio_dir = root / "audio"
    search_dir = audio_dir if audio_dir.exists() else root
    paths = sorted(path.resolve() for path in search_dir.glob("*.wav") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No audio files found under: {search_dir}")
    return paths


def load_rttm_segments(root: Path) -> dict[str, list[Segment]]:
    rttm_dir = root / "rttm"
    if not rttm_dir.exists():
        raise FileNotFoundError(f"Missing RTTM directory under: {root}")

    grouped: dict[str, list[Segment]] = {}
    for path in sorted(rttm_dir.glob("*.rttm")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8 or parts[0] != "SPEAKER":
                    continue
                try:
                    uri = parts[1]
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker = parts[7]
                except Exception:
                    continue
                grouped.setdefault(uri, []).append(Segment(speaker=speaker, start=start, end=start + duration))

    for uri in grouped:
        grouped[uri].sort(key=lambda s: (s.start, s.end, s.speaker))
    return grouped


def load_hypothesis_overlaps(path: Path) -> dict[str, tuple[list[Interval], float | None]]:
    paths = sorted(path.glob("*_osd.json")) if path.is_dir() else [path]
    if not paths:
        raise FileNotFoundError(f"No OSD JSON found under: {path}")

    loaded: dict[str, tuple[list[Interval], float | None]] = {}
    for item in paths:
        with item.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        uri = str(raw.get("uri") or item.stem.removesuffix("_osd"))
        threshold = raw.get("overlap_threshold")
        overlaps: list[Interval] = []
        for segment in raw.get("overlaps", []):
            try:
                start = float(segment["start"])
                end = float(segment["end"])
            except Exception:
                continue
            if end > start:
                overlaps.append(Interval(start=start, end=end, duration=end - start))
        loaded[uri] = (sorted(overlaps, key=lambda x: (x.start, x.end)), float(threshold) if threshold is not None else None)
    return loaded


def compute_overlap_intervals(segments: list[Segment]) -> list[Interval]:
    events: list[tuple[float, int, str]] = []
    for segment in segments:
        events.append((segment.start, 1, segment.speaker))
        events.append((segment.end, -1, segment.speaker))
    events.sort(key=lambda item: (item[0], item[1]))

    overlaps: list[Interval] = []
    active = 0
    prev_time: float | None = None
    for time, delta, _ in events:
        if prev_time is not None and time > prev_time and active >= 2:
            overlaps.append(Interval(start=prev_time, end=time, duration=time - prev_time))
        active += delta
        prev_time = time
    return overlaps


def sort_speaker_labels(speakers: list[str]) -> list[str]:
    def key(label: str) -> tuple[int, str]:
        suffix = label.rsplit("_spk", 1)
        if len(suffix) == 2 and suffix[1].isdigit():
            return int(suffix[1]), label
        return 10_000, label

    return sorted(dict.fromkeys(speakers), key=key)


def load_waveform(audio_path: Path, sample_rate: int, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    data, sr = sf.read(str(audio_path), always_2d=True, dtype="float32")
    waveform = data.mean(axis=1).astype(np.float32, copy=False)

    if sr != sample_rate and waveform.size > 0:
        g = math.gcd(sample_rate, sr)
        waveform = scipy.signal.resample_poly(waveform, sample_rate // g, sr // g).astype(np.float32, copy=False)
        sr = sample_rate

    duration = float(waveform.size) / float(sr) if sr > 0 else 0.0
    if waveform.size > max_points:
        stride = int(math.ceil(waveform.size / max_points))
        waveform = waveform[::stride]
    x = np.linspace(0.0, duration, num=max(1, waveform.size), endpoint=False, dtype=np.float32)
    if waveform.size == 0:
        waveform = np.zeros(1, dtype=np.float32)
    return x, waveform


def load_records(ground_truth: Path, hypothesis: Path) -> list[Record]:
    audio_paths = discover_audio_paths(ground_truth)
    gt_segments = load_rttm_segments(ground_truth)
    hyp_overlaps = load_hypothesis_overlaps(hypothesis)

    records: list[Record] = []
    for audio_path in audio_paths:
        uri = audio_path.stem
        if uri not in hyp_overlaps:
            continue
        info = sf.info(str(audio_path))
        segments = gt_segments.get(uri, [])
        speakers = sort_speaker_labels([segment.speaker for segment in segments])
        gt_overlaps = compute_overlap_intervals(segments)
        hyp_items, threshold = hyp_overlaps[uri]
        records.append(
            Record(
                uri=uri,
                audio_path=audio_path,
                duration=float(info.duration),
                speakers=speakers,
                gt_segments=segments,
                gt_overlaps=gt_overlaps,
                hyp_overlaps=hyp_items,
                hyp_threshold=threshold,
            )
        )

    if not records:
        raise RuntimeError("No shared URIs between ground-truth audio and hypothesis JSON.")
    return sorted(records, key=lambda r: r.uri)


class ClickablePlotWidget(pg.PlotWidget):
    seekRequested = Signal(float)

    def mousePressEvent(self, ev: QMouseEvent) -> None:  # type: ignore[override]
        if ev.button() == Qt.LeftButton:
            pos = self.plotItem.vb.mapSceneToView(ev.position())
            self.seekRequested.emit(float(max(0.0, pos.x())))
        super().mousePressEvent(ev)


class TimelinePanel(QWidget):
    seekRequested = Signal(float)

    def __init__(
        self,
        title: str,
        subtitle: str,
        waveform_x: np.ndarray,
        waveform_y: np.ndarray,
        duration: float,
        segments: list[Segment],
        overlaps: list[Interval],
        overlap_color: str,
        show_speakers: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.duration = duration
        self.speaker_buttons: dict[str, QToolButton] = {}
        self.speaker_ranges: dict[str, list[tuple[float, float]]] = {}
        self.colors = {speaker: QColor(PALETTE[i % len(PALETTE)]) for i, speaker in enumerate(sort_speaker_labels([s.speaker for s in segments]))}

        for segment in segments:
            self.speaker_ranges.setdefault(segment.speaker, []).append((segment.start, segment.end))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QLabel(f"{title} | {subtitle}")
        header.setStyleSheet("color: #111827; font-size: 13px; font-weight: 700;")
        layout.addWidget(header)

        self.plot = ClickablePlotWidget()
        self.plot.setBackground("w")
        self.plot.setMinimumHeight(220)
        self.plot.setMaximumHeight(260)
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.seekRequested.connect(self.seekRequested.emit)
        layout.addWidget(self.plot)

        self.plot.plot(waveform_x, waveform_y, pen=pg.mkPen((55, 65, 81), width=1.1))
        y_min = float(np.min(waveform_y)) if waveform_y.size else -1.0
        y_max = float(np.max(waveform_y)) if waveform_y.size else 1.0
        if abs(y_max - y_min) < 1e-6:
            y_min -= 1.0
            y_max += 1.0
        self.plot.setLimits(xMin=0.0, xMax=max(0.1, duration))
        self.plot.setXRange(0.0, max(0.1, duration), padding=0.01)
        self.plot.setYRange(y_min * 1.15, y_max * 1.15, padding=0.02)

        if show_speakers:
            for segment in segments:
                color = QColor(self.colors.get(segment.speaker, QColor("#94a3b8")))
                color.setAlphaF(0.18)
                region = pg.LinearRegionItem(values=(segment.start, segment.end), brush=pg.mkBrush(color), movable=False)
                region.setZValue(-10)
                self.plot.addItem(region)

        color = QColor(overlap_color)
        color.setAlpha(50)
        for overlap in overlaps:
            span = pg.LinearRegionItem(values=(overlap.start, overlap.end), brush=pg.mkBrush(color), movable=False)
            span.setZValue(-20)
            self.plot.addItem(span)

        self.playhead = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen((17, 24, 39), width=2))
        self.playhead.setZValue(50)
        self.plot.addItem(self.playhead)

        if show_speakers and self.speaker_ranges:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            row.addWidget(QLabel("Speakers:"))
            for speaker in sort_speaker_labels(list(self.speaker_ranges)):
                button = QToolButton()
                button.setText(speaker)
                button.setEnabled(False)
                button.setStyleSheet(self._button_style(speaker, False))
                self.speaker_buttons[speaker] = button
                row.addWidget(button)
            row.addStretch(1)
            layout.addLayout(row)

    def _button_style(self, speaker: str, active: bool) -> str:
        color = self.colors.get(speaker, QColor("#6b7280"))
        alpha_bg = 0.28 if active else 0.14
        alpha_border = 0.85 if active else 0.45
        text_color = "#111827" if active else "#374151"
        return (
            "QToolButton {"
            f"background-color: rgba({color.red()}, {color.green()}, {color.blue()}, {alpha_bg});"
            f"border: 1px solid rgba({color.red()}, {color.green()}, {color.blue()}, {alpha_border});"
            "border-radius: 9px; padding: 5px 9px; font-weight: 700;"
            f"color: {text_color};"
            "}"
        )

    def set_playhead(self, time_seconds: float) -> None:
        self.playhead.setPos(max(0.0, min(time_seconds, max(self.duration, 0.1))))
        for speaker, button in self.speaker_buttons.items():
            active = any(start <= time_seconds < end for start, end in self.speaker_ranges.get(speaker, []))
            button.setStyleSheet(self._button_style(speaker, active))


class OSDVisualizationWindow(QMainWindow):
    def __init__(
        self,
        records: list[Record],
        waveform_sample_rate: int,
        waveform_points: int,
        audio_output_device_name: str | None,
    ) -> None:
        super().__init__()
        self.records = records
        self.waveform_sample_rate = waveform_sample_rate
        self.waveform_points = waveform_points
        self.waveform_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.audio_devices: list[QAudioDevice] = []
        self.current_record: Record | None = None
        self.panels: list[TimelinePanel] = []

        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(1.0)
        self.audio_output.setMuted(False)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.positionChanged.connect(self._on_position_changed)
        self.media_player.durationChanged.connect(self._on_duration_changed)
        self.media_player.playbackStateChanged.connect(self._sync_play_button)

        self.setWindowTitle("Demo Visualization")
        self.resize(1400, 950)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)

        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)
        top_layout.addWidget(QLabel("Audio"))
        self.record_combo = QComboBox()
        self.record_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        for index, record in enumerate(self.records):
            self.record_combo.addItem(record.uri, userData=index)
        self.record_combo.currentIndexChanged.connect(self._on_record_changed)
        top_layout.addWidget(self.record_combo, 1)
        top_layout.addWidget(QLabel("Audio output"))
        self.audio_output_combo = QComboBox()
        self.audio_output_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.audio_output_combo.currentIndexChanged.connect(self._on_audio_output_changed)
        top_layout.addWidget(self.audio_output_combo)
        root_layout.addWidget(top)

        self.panel_container = QWidget()
        self.panel_layout = QVBoxLayout(self.panel_container)
        self.panel_layout.setContentsMargins(0, 0, 0, 0)
        self.panel_layout.setSpacing(18)
        self.panel_layout.addStretch(1)
        root_layout.addWidget(self.panel_container, 1)

        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)
        self.play_button = QPushButton("Play")
        self.play_button.setCursor(Qt.PointingHandCursor)
        self.play_button.clicked.connect(self._toggle_playback)
        self.play_button.setStyleSheet(
            "QPushButton {"
            "background-color: #111827; color: white; padding: 8px 16px;"
            "border: none; border-radius: 10px; font-weight: 700;"
            "}"
            "QPushButton:hover { background-color: #1f2937; }"
        )
        controls_layout.addWidget(self.play_button)
        self.time_label = QLabel("00:00.000 / 00:00.000")
        self.time_label.setStyleSheet("font-size: 12px; color: #374151; min-width: 140px;")
        controls_layout.addWidget(self.time_label)
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self._seek_to_slider)
        self.position_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 6px;
                background: #e5e7eb;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #111827;
                border-radius: 3px;
            }
            QSlider::add-page:horizontal {
                background: #d1d5db;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #111827;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            """
        )
        controls_layout.addWidget(self.position_slider, 1)
        root_layout.addWidget(controls)

        self.setCentralWidget(root)
        self._populate_audio_outputs(audio_output_device_name)
        self._load_record(self.records[0] if self.records else None)

    def _get_waveform(self, audio_path: Path) -> tuple[np.ndarray, np.ndarray]:
        key = str(audio_path.resolve())
        if key not in self.waveform_cache:
            self.waveform_cache[key] = load_waveform(audio_path, self.waveform_sample_rate, self.waveform_points)
        return self.waveform_cache[key]

    def _clear_panels(self) -> None:
        while self.panels:
            panel = self.panels.pop()
            panel.setParent(None)
            panel.deleteLater()

    def _load_record(self, record: Record | None) -> None:
        self.current_record = record
        self._clear_panels()
        if record is None:
            return

        waveform_x, waveform_y = self._get_waveform(record.audio_path)
        hyp_panel = TimelinePanel(
            title="OSD Prediction",
            subtitle=f"{record.uri} | threshold={record.hyp_threshold if record.hyp_threshold is not None else '?'} | overlap_segments={len(record.hyp_overlaps)}",
            waveform_x=waveform_x,
            waveform_y=waveform_y,
            duration=record.duration,
            segments=[],
            overlaps=record.hyp_overlaps,
            overlap_color=HYP_OVERLAP_COLOR,
            show_speakers=False,
            parent=self.panel_container,
        )
        hyp_panel.seekRequested.connect(self._seek_from_panel)
        gt_panel = TimelinePanel(
            title="Ground Truth",
            subtitle=f"{record.uri} | duration={record.duration:.2f}s | speakers={len(record.speakers)}",
            waveform_x=waveform_x,
            waveform_y=waveform_y,
            duration=record.duration,
            segments=record.gt_segments,
            overlaps=record.gt_overlaps,
            overlap_color=GT_OVERLAP_COLOR,
            show_speakers=True,
            parent=self.panel_container,
        )
        gt_panel.seekRequested.connect(self._seek_from_panel)
        self.panel_layout.insertWidget(self.panel_layout.count() - 1, gt_panel)
        self.panel_layout.insertWidget(self.panel_layout.count() - 1, hyp_panel)
        self.panels = [gt_panel, hyp_panel]

        self.media_player.setSource(QUrl.fromLocalFile(str(record.audio_path.resolve())))
        self.media_player.setPosition(0)
        self.position_slider.setValue(0)
        self._update_time_label(0, self.media_player.duration())
        self._on_position_changed(0)

    def _toggle_playback(self) -> None:
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def _sync_play_button(self) -> None:
        self.play_button.setText("Pause" if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState else "Play")

    def _on_duration_changed(self, duration_ms: int) -> None:
        self.position_slider.setRange(0, max(0, duration_ms))
        self._update_time_label(self.media_player.position(), duration_ms)

    def _seek_to_slider(self, value: int) -> None:
        self.media_player.setPosition(value)

    def _seek_from_panel(self, time_seconds: float) -> None:
        self.media_player.setPosition(int(max(0.0, time_seconds) * 1000.0))

    def _on_position_changed(self, position_ms: int) -> None:
        duration_ms = max(self.media_player.duration(), 0)
        if not self.position_slider.isSliderDown():
            self.position_slider.setValue(position_ms)
        self._update_time_label(position_ms, duration_ms)
        time_seconds = position_ms / 1000.0
        for panel in self.panels:
            panel.set_playhead(time_seconds)

    def _update_time_label(self, position_ms: int, duration_ms: int) -> None:
        self.time_label.setText(f"{format_time(position_ms / 1000.0)} / {format_time(duration_ms / 1000.0)}")

    def _on_record_changed(self, index: int) -> None:
        if 0 <= index < len(self.records):
            self._load_record(self.records[index])

    def _populate_audio_outputs(self, preferred_device_name: str | None) -> None:
        self.audio_devices = list(QMediaDevices.audioOutputs())
        self.audio_output_combo.blockSignals(True)
        self.audio_output_combo.clear()
        if not self.audio_devices:
            self.audio_output_combo.addItem("No audio output devices found")
            self.audio_output_combo.setEnabled(False)
            self.audio_output.setDevice(QMediaDevices.defaultAudioOutput())
            self.audio_output_combo.blockSignals(False)
            return

        self.audio_output_combo.setEnabled(True)
        default_index = 0
        preferred_index = None
        for index, device in enumerate(self.audio_devices):
            description = device.description() or f"Audio output {index + 1}"
            if device.isDefault():
                description = f"{description} (default)"
                default_index = index
            self.audio_output_combo.addItem(description)
            if preferred_device_name:
                normalized = preferred_device_name.strip().lower()
                if normalized == device.description().strip().lower() or normalized in device.description().strip().lower():
                    preferred_index = index
        target_index = preferred_index if preferred_index is not None else default_index
        self.audio_output_combo.setCurrentIndex(target_index)
        self.audio_output_combo.blockSignals(False)
        self._apply_audio_output_device(target_index)

    def _apply_audio_output_device(self, index: int) -> None:
        if 0 <= index < len(self.audio_devices):
            self.audio_output.setDevice(self.audio_devices[index])

    def _on_audio_output_changed(self, index: int) -> None:
        self._apply_audio_output_device(index)


def format_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes:02d}:{rest:06.3f}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize ground-truth and OSD predictions for the same audio.")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Dataset root containing audio/ and rttm/.")
    parser.add_argument("--hypothesis", type=Path, required=True, help="OSD JSON file or directory containing *_osd.json.")
    parser.add_argument("--audio-output-device", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    records = load_records(args.ground_truth, args.hypothesis)

    print(f"Loaded records: {len(records)}")
    print(f"Ground truth: {args.ground_truth}")
    print(f"Hypothesis  : {args.hypothesis}")

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Demo Visualization")
    app.setStyleSheet(
        """
        QWidget { background-color: #ffffff; color: #111827; font-family: "Inter", "Helvetica Neue", sans-serif; }
        QComboBox, QSlider, QPushButton, QToolButton { font-size: 12px; }
        QComboBox { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 4px 8px; min-height: 24px; }
        QComboBox:hover { border: 1px solid #94a3b8; }
        QComboBox::drop-down { border: none; width: 24px; background: transparent; }
        QPushButton, QToolButton { border-radius: 10px; }
        QToolButton { padding: 6px 10px; }
        """
    )

    window = OSDVisualizationWindow(
        records=records,
        waveform_sample_rate=DEFAULT_WAVEFORM_SAMPLE_RATE,
        waveform_points=DEFAULT_WAVEFORM_POINTS,
        audio_output_device_name=args.audio_output_device,
    )
    window.show()

    timer = QTimer(window)
    timer.timeout.connect(lambda: None)
    timer.start(250)

    try:
        import signal
        signal.signal(signal.SIGINT, lambda *_: app.quit())
    except Exception:
        pass

    app.exec()


if __name__ == "__main__":
    main()
