#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pyqtgraph as pg
import soundfile as sf
from PySide6.QtCore import QUrl, Qt, Signal, QTimer
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtMultimedia import QAudioDevice, QAudioOutput, QMediaDevices, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WAVEFORM_SAMPLE_RATE = 16_000
DEFAULT_WAVEFORM_POINTS = 5_000
DEFAULT_AUDIO_SUFFIX = ".wav"

PALETTE = [
    "#ff0000",  # red
    "#00a000",  # green
    "#0057ff",  # blue
    "#00c7d9",  # cyan
    "#ff00c8",  # magenta
    "#ffd400",  # yellow
    "#7a4cff",  # violet
    "#00b894",  # teal
]
OVERLAP_COLOR = "#f59e0b"
SOURCE_COLOR = "#94a3b8"


@dataclass(frozen=True)
class Segment:
    speaker: str
    start: float
    end: float
    source: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    duration: float
    label: str | None = None


@dataclass
class MixResult:
    uri: str
    media_path: Path
    raw: dict[str, Any]
    speakers: list[str]
    segments: list[Segment]
    overlaps: list[Interval]

    @property
    def label(self) -> str:
        return self.uri


@dataclass
class MixGroup:
    media_path: Path
    results: list[MixResult]

    @property
    def label(self) -> str:
        return self.media_path.name


def display_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def discover_audio_paths(input_dir: Path) -> list[Path]:
    audio_dir = input_dir / "audio"
    if not audio_dir.exists():
        raise FileNotFoundError(f"Missing audio directory under: {input_dir}")
    paths = sorted(path.resolve() for path in audio_dir.glob(f"*{DEFAULT_AUDIO_SUFFIX}") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No audio files found under: {audio_dir}")
    return paths


def load_rttm_segments(input_dir: Path) -> dict[str, list[Segment]]:
    rttm_dir = input_dir / "rttm"
    if not rttm_dir.exists():
        raise FileNotFoundError(f"Missing RTTM directory under: {input_dir}")

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
                end = start + duration
                grouped.setdefault(uri, []).append(Segment(speaker=speaker, start=start, end=end))

    for uri in grouped:
        grouped[uri].sort(key=lambda s: (s.start, s.end, s.speaker))
    return grouped


def compute_overlap_intervals(segments: list[Segment]) -> tuple[list[Interval], int]:
    events: list[tuple[float, int]] = []
    for segment in segments:
        events.append((segment.start, 1))
        events.append((segment.end, -1))
    events.sort(key=lambda item: (item[0], item[1]))

    overlaps: list[Interval] = []
    active = 0
    max_active = 0
    prev_time: float | None = None
    for time, delta in events:
        if prev_time is not None and time > prev_time and active >= 2:
            overlaps.append(Interval(start=prev_time, end=time, duration=time - prev_time))
        active += delta
        max_active = max(max_active, active)
        prev_time = time
    return overlaps, max_active


def sort_speaker_labels(speakers: list[str]) -> list[str]:
    def key(label: str) -> tuple[int, str]:
        suffix = label.rsplit("_spk", 1)
        if len(suffix) == 2 and suffix[1].isdigit():
            return int(suffix[1]), label
        return 10_000, label

    seen: dict[str, None] = {}
    for speaker in speakers:
        seen.setdefault(speaker, None)
    return sorted(seen, key=key)


def build_speaker_color_map(speakers: list[str]) -> dict[str, QColor]:
    return {speaker: QColor(PALETTE[i % len(PALETTE)]) for i, speaker in enumerate(speakers)}


def load_waveform(audio_path: Path, sample_rate: int, max_points: int) -> tuple[np.ndarray, np.ndarray, int]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    try:
        data, sr = sf.read(str(audio_path), always_2d=True, dtype="float32")
        waveform = data.mean(axis=1).astype(np.float32, copy=False)
    except Exception:
        waveform, sr = librosa.load(str(audio_path), sr=None, mono=True)
        waveform = waveform.astype(np.float32, copy=False)

    if sr != sample_rate and waveform.size > 0:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate).astype(np.float32, copy=False)
        sr = sample_rate

    if waveform.size == 0:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32), sr

    duration = float(waveform.size) / float(sr)
    if waveform.size > max_points:
        stride = int(math.ceil(waveform.size / max_points))
        waveform = waveform[::stride]

    x = np.linspace(0.0, duration, num=waveform.size, endpoint=False, dtype=np.float32)
    return x, waveform, sr


class ClickablePlotWidget(pg.PlotWidget):
    seekRequested = Signal(float)

    def mousePressEvent(self, ev: QMouseEvent) -> None:  # type: ignore[override]
        if ev.button() == Qt.LeftButton:
            pos = self.plotItem.vb.mapSceneToView(ev.position())
            self.seekRequested.emit(float(max(0.0, pos.x())))
        super().mousePressEvent(ev)


def load_mix_results_from_dir(input_dir: Path) -> list[MixResult]:
    audio_paths = discover_audio_paths(input_dir)
    segments_by_uri = load_rttm_segments(input_dir)
    results: list[MixResult] = []

    for audio_path in audio_paths:
        uri = audio_path.stem
        segments = segments_by_uri.get(uri, [])
        speakers = sort_speaker_labels([segment.speaker for segment in segments])[:3]
        overlaps, max_active = compute_overlap_intervals(segments)
        info = sf.info(str(audio_path))
        duration = float(info.duration)
        raw = {
            "uri": uri,
            "duration": duration,
            "source_count": len(segments),
            "speaker_count": len(speakers),
            "max_speakers_per_frame": max_active,
        }
        results.append(
            MixResult(
                uri=uri,
                media_path=audio_path.resolve(),
                raw=raw,
                speakers=speakers,
                segments=segments,
                overlaps=overlaps,
            )
        )

    return results


def group_results_by_media(results: list[MixResult]) -> list[MixGroup]:
    grouped: dict[str, MixGroup] = {}
    for result in results:
        key = result.media_path.name
        if key not in grouped:
            grouped[key] = MixGroup(media_path=result.media_path.resolve(), results=[])
        grouped[key].results.append(result)

    groups = list(grouped.values())
    for group in groups:
        group.results.sort(key=lambda r: r.uri)
    groups.sort(key=lambda g: (g.media_path.name, len(g.results)))
    return groups


def choose_default_group(groups: list[MixGroup]) -> MixGroup | None:
    if not groups:
        return None
    return sorted(groups, key=lambda g: (-len(g.results), g.media_path.name))[0]


def load_audio_waveform(audio_path: Path, sample_rate: int, max_points: int) -> tuple[np.ndarray, np.ndarray, int]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    try:
        data, sr = sf.read(str(audio_path), always_2d=True)
        waveform = data.mean(axis=1).astype(np.float32, copy=False)
    except Exception:
        waveform, sr = librosa.load(str(audio_path), sr=None, mono=True)
        waveform = waveform.astype(np.float32, copy=False)

    if sr != sample_rate and waveform.size > 0:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate).astype(np.float32, copy=False)
        sr = sample_rate

    if waveform.size == 0:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32), sr

    total_duration = float(waveform.size) / float(sr)
    if waveform.size > max_points:
        stride = int(math.ceil(waveform.size / max_points))
        waveform = waveform[::stride]

    x = np.linspace(0.0, total_duration, num=waveform.size, endpoint=False, dtype=np.float32)
    return x, waveform, sr


class ResultPanel(QWidget):
    seekRequested = Signal(float)

    def __init__(
        self,
        result: MixResult,
        waveform_x: np.ndarray,
        waveform_y: np.ndarray,
        waveform_duration: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.result = result
        self.waveform_duration = waveform_duration
        self._speaker_colors = build_speaker_color_map(result.speakers)
        self._speaker_buttons: dict[str, QToolButton] = {}
        self._speaker_segments: dict[str, list[tuple[float, float]]] = {}
        for segment in self.result.segments:
            self._speaker_segments.setdefault(segment.speaker, []).append((segment.start, segment.end))

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        header_row = QWidget()
        header_layout = QHBoxLayout(header_row)
        header_layout.setContentsMargins(2, 0, 2, 0)
        header_layout.setSpacing(8)

        header = QLabel(result.label)
        header.setTextInteractionFlags(Qt.TextSelectableByMouse)
        header.setStyleSheet("color: #6b7280; font-size: 11px; font-weight: 600;")
        header_layout.addWidget(header, 1)
        outer.addWidget(header_row)

        self.plot = ClickablePlotWidget()
        self.plot.setBackground("w")
        self.plot.setMinimumHeight(200)
        self.plot.setMaximumHeight(240)
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Amplitude")
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.seekRequested.connect(self.seekRequested.emit)
        outer.addWidget(self.plot)

        self.plot.plot(waveform_x, waveform_y, pen=pg.mkPen((55, 65, 81), width=1.1))

        y_min = float(np.min(waveform_y)) if waveform_y.size else -1.0
        y_max = float(np.max(waveform_y)) if waveform_y.size else 1.0
        if abs(y_max - y_min) < 1e-6:
            y_min -= 1.0
            y_max += 1.0
        self.plot.setLimits(xMin=0.0, xMax=max(0.1, waveform_duration))
        self.plot.setXRange(0.0, max(0.1, waveform_duration), padding=0.01)
        self.plot.setYRange(y_min * 1.15, y_max * 1.15, padding=0.02)

        for segment in self.result.segments:
            color = QColor(self._speaker_colors.get(segment.speaker, QColor(SOURCE_COLOR)))
            color.setAlphaF(0.18)
            region = pg.LinearRegionItem(values=(segment.start, segment.end), brush=pg.mkBrush(color), movable=False)
            region.setZValue(-10)
            self.plot.addItem(region)

        for overlap in self.result.overlaps:
            if overlap.end <= overlap.start:
                continue
            color = QColor(OVERLAP_COLOR)
            color.setAlpha(40)
            span = pg.LinearRegionItem(values=(overlap.start, overlap.end), brush=pg.mkBrush(color), movable=False)
            span.setZValue(-20)
            self.plot.addItem(span)

        self.playhead = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen((17, 24, 39), width=2))
        self.playhead.setZValue(50)
        self.plot.addItem(self.playhead)

        info = QWidget()
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(6)

        meta = self.result.raw
        title = (
            f"{meta.get('uri', 'mix')} | duration={float(meta.get('duration', waveform_duration)):.2f}s | "
            f"segments={int(meta.get('source_count', len(self.result.segments)))} | "
            f"speakers={int(meta.get('speaker_count', len(self.result.speakers)))} | "
            f"max speaker per frame={meta.get('max_speakers_per_frame', meta.get('max_overlap_speaker', '?'))}"
        )
        self.summary_label = QLabel(title)
        self.summary_label.setStyleSheet("color: #111827; font-size: 13px; font-weight: 700;")
        info_layout.addWidget(self.summary_label)

        if self.result.speakers:
            speaker_row = QHBoxLayout()
            speaker_row.setContentsMargins(0, 0, 0, 0)
            speaker_row.setSpacing(6)
            speaker_row.addWidget(QLabel("Speakers:"))
            for speaker in self.result.speakers:
                button = QToolButton()
                button.setText(speaker)
                button.setEnabled(False)
                button.setStyleSheet(self._speaker_button_style(speaker, active=False))
                self._speaker_buttons[speaker] = button
                speaker_row.addWidget(button)
            speaker_row.addStretch(1)
            info_layout.addLayout(speaker_row)

        outer.addWidget(info)
        self.set_active_speakers(set())

    def set_playhead(self, time_seconds: float) -> None:
        self.playhead.setPos(max(0.0, min(time_seconds, max(self.waveform_duration, 0.1))))
        active_speakers = {
            speaker
            for speaker, intervals in self._speaker_segments.items()
            if any(start <= time_seconds < end for start, end in intervals)
        }
        self.set_active_speakers(active_speakers)

    def _speaker_button_style(self, speaker: str, active: bool) -> str:
        color = self._speaker_colors.get(speaker, QColor("#6b7280"))
        alpha_bg = 0.28 if active else 0.14
        alpha_border = 0.85 if active else 0.45
        text_color = "#111827" if active else "#374151"
        return (
            "QToolButton {"
            f"background-color: rgba({color.red()}, {color.green()}, {color.blue()}, {alpha_bg});"
            f"border: 1px solid rgba({color.red()}, {color.green()}, {color.blue()}, {alpha_border});"
            "border-radius: 9px;"
            "padding: 5px 9px;"
            "font-weight: 700;"
            f"color: {text_color};"
            "}"
        )

    def set_active_speakers(self, active_speakers: set[str]) -> None:
        for speaker, button in self._speaker_buttons.items():
            button.setStyleSheet(self._speaker_button_style(speaker, speaker in active_speakers))


class MixVisualizationWindow(QMainWindow):
    def __init__(
        self,
        groups: list[MixGroup],
        waveform_sample_rate: int,
        waveform_points: int,
        audio_output_device_name: str | None = None,
    ) -> None:
        super().__init__()
        self.groups = groups
        self.waveform_sample_rate = waveform_sample_rate
        self.waveform_points = waveform_points
        self.waveform_cache: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        self.current_group: MixGroup | None = None
        self.panels: list[ResultPanel] = []
        self.audio_devices: list[QAudioDevice] = []

        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(1.0)
        self.audio_output.setMuted(False)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.positionChanged.connect(self._on_position_changed)
        self.media_player.durationChanged.connect(self._on_duration_changed)
        self.media_player.playbackStateChanged.connect(self._sync_play_button)

        self.setWindowTitle("Mix Visualization")
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

        self.group_combo = QComboBox()
        self.group_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        for idx, group in enumerate(self.groups):
            self.group_combo.addItem(group.label, userData=idx)
        self.group_combo.currentIndexChanged.connect(self._on_group_changed)
        top_layout.addWidget(self.group_combo, 1)

        audio_label = QLabel("Audio output")
        audio_label.setStyleSheet("font-size: 13px; font-weight: 700; color: #111827;")
        top_layout.addWidget(audio_label)

        self.audio_output_combo = QComboBox()
        self.audio_output_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.audio_output_combo.currentIndexChanged.connect(self._on_audio_output_changed)
        top_layout.addWidget(self.audio_output_combo)

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

        self.bottom_controls = QWidget()
        bottom_layout = QHBoxLayout(self.bottom_controls)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(10)

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
        bottom_layout.addWidget(self.play_button)

        self.time_label = QLabel("00:00.000 / 00:00.000")
        self.time_label.setStyleSheet("font-size: 12px; color: #374151; min-width: 140px;")
        bottom_layout.addWidget(self.time_label)

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
        bottom_layout.addWidget(self.position_slider, 1)

        root_layout.addWidget(self.bottom_controls)
        self.setCentralWidget(root)

        self._load_group(self.groups[0] if self.groups else None)
        self._populate_audio_outputs(audio_output_device_name)

    def _toggle_playback(self) -> None:
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def _sync_play_button(self) -> None:
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setText("Pause")
        else:
            self.play_button.setText("Play")

    def _on_duration_changed(self, duration_ms: int) -> None:
        self.position_slider.setRange(0, max(0, duration_ms))
        self._update_time_label(self.media_player.position(), duration_ms)

    def _seek_to_slider(self, value: int) -> None:
        self.media_player.setPosition(value)

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

    def _on_group_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.groups):
            return
        self._load_group(self.groups[index])

    def _load_group(self, group: MixGroup | None) -> None:
        self.current_group = group
        self._clear_panels()

        if group is None:
            self.group_count_label.setText("No mix records found.")
            return

        self.group_count_label.setText(f"{len(group.results)} mix record(s) for {group.label}")
        waveform_x, waveform_y, sr = self._get_waveform(group.media_path)
        duration = float(waveform_x[-1]) if waveform_x.size else 0.0

        for result in group.results:
            panel = ResultPanel(result, waveform_x, waveform_y, duration, parent=self.scroll_container)
            panel.seekRequested.connect(self._seek_from_panel)
            self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, panel)
            self.panels.append(panel)

        self._set_media_source(group.media_path)
        self.media_player.setPosition(0)
        self.position_slider.setValue(0)
        self._update_time_label(0, self.media_player.duration())
        self._on_position_changed(0)

    def _clear_panels(self) -> None:
        while self.panels:
            panel = self.panels.pop()
            panel.setParent(None)
            panel.deleteLater()

    def _seek_from_panel(self, time_seconds: float) -> None:
        self.media_player.setPosition(int(max(0.0, time_seconds) * 1000.0))

    def _set_media_source(self, media_path: Path) -> None:
        self.media_player.setSource(QUrl.fromLocalFile(str(media_path.resolve())))

    def _get_waveform(self, media_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
        key = str(media_path.resolve())
        if key not in self.waveform_cache:
            self.waveform_cache[key] = load_audio_waveform(
                media_path,
                sample_rate=self.waveform_sample_rate,
                max_points=self.waveform_points,
            )
        return self.waveform_cache[key]

    def _populate_audio_outputs(self, preferred_device_name: str | None = None) -> None:
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
        if index < 0 or index >= len(self.audio_devices):
            return
        try:
            self.audio_output.setDevice(self.audio_devices[index])
        except Exception as exc:
            print(f"Failed to set audio output device: {exc}")
            return
        self.audio_output_combo.setCurrentIndex(index)

    def _on_audio_output_changed(self, index: int) -> None:
        self._apply_audio_output_device(index)


def format_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes:02d}:{rest:06.3f}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mix visualization with per-record panels.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Dataset root containing audio/ and rttm/.",
    )
    parser.add_argument(
        "--waveform-sample-rate",
        type=int,
        default=DEFAULT_WAVEFORM_SAMPLE_RATE,
        help=f"Waveform display sample rate (default: {DEFAULT_WAVEFORM_SAMPLE_RATE}).",
    )
    parser.add_argument(
        "--waveform-points",
        type=int,
        default=DEFAULT_WAVEFORM_POINTS,
        help=f"Maximum number of waveform points per panel (default: {DEFAULT_WAVEFORM_POINTS}).",
    )
    parser.add_argument(
        "--audio-output-device",
        type=str,
        default=None,
        help="Optional audio output device name to preselect.",
    )
    return parser


def select_group(groups: list[MixGroup]) -> MixGroup | None:
    if not groups:
        return None
    return choose_default_group(groups)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    results = load_mix_results_from_dir(args.input_dir)

    groups = group_results_by_media(results)
    selected_group = select_group(groups)

    print(f"Scanned {len(results)} audio file(s) under {args.input_dir / 'audio'}")
    if selected_group is not None:
        print(f"Selected source audio: {selected_group.label} ({len(selected_group.results)} mix record(s))")
    else:
        print("No mix results were found.")

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Mix Visualization")
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

    window = MixVisualizationWindow(
        groups=groups,
        waveform_sample_rate=args.waveform_sample_rate,
        waveform_points=args.waveform_points,
        audio_output_device_name=args.audio_output_device,
    )

    if selected_group is not None:
        index = next((i for i, g in enumerate(groups) if g.media_path.resolve() == selected_group.media_path.resolve()), 0)
        window.group_combo.setCurrentIndex(index)

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
