from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pyqtgraph as pg
import scipy.signal
import soundfile as sf
from PySide6.QtCore import QPointF, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPolygonF
from PySide6.QtMultimedia import QAudioDevice, QAudioOutput, QMediaDevices, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGraphicsRectItem,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QMainWindow,
    QPushButton,
    QProxyStyle,
    QScrollArea,
    QSlider,
    QStyle,
    QVBoxLayout,
    QWidget,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

COLOR_BG = "#ffffff"  # Main app background (demo/mix/osd/vad/wave_visualization.py)
COLOR_TEXT = "#111827"  # Primary text and play button color (demo/mix/osd/vad/wave_visualization.py)
COLOR_TEXT_MUTED = "#9ca3af"  # Secondary text and path note color (demo/mix/osd/vad/wave_visualization.py)
COLOR_TEXT_SUBTLE = "#374151"  # Time text and waveform-adjacent subtle labels (demo/mix/osd/vad/wave_visualization.py)
COLOR_BORDER = "#cbd5e1"  # Default combo-box border (demo/mix/osd/vad/wave_visualization.py)
COLOR_BORDER_HOVER = "#94a3b8"  # Combo-box hover border (demo/mix/osd/vad/wave_visualization.py)
COLOR_BUTTON_HOVER = "#1f2937"  # Play button hover color (demo/mix/osd/vad/wave_visualization.py)
COLOR_SLIDER_GROOVE = "#e5e7eb"  # Playback slider base track (demo/mix/osd/vad/wave_visualization.py)
COLOR_SLIDER_FILL = "#111827"  # Playback slider filled track and handle (demo/mix/osd/vad/wave_visualization.py)
COLOR_SLIDER_REST = "#d1d5db"  # Playback slider remaining track (demo/mix/osd/vad/wave_visualization.py)
COLOR_PLOT_BG = "#ffffff"  # Normal waveform panel background (demo/mix/osd/wave_visualization.py)
COLOR_PLOT_BG_SUBTLE = "#f8fafc"  # Gray-tone waveform panel background (demo/mix/osd/vad/wave_visualization.py)
COLOR_WAVEFORM = "#374151"  # Waveform line color (demo/mix/osd/vad/wave_visualization.py)
COLOR_PLAYHEAD = "#111827"  # Playback cursor line (demo/mix/osd/vad/wave_visualization.py)
COLOR_MONO_INTERVAL = "#afafaf"  # Single-color interval regions for OSD/VAD annotations (demo/osd/vad_visualization.py)
COLOR_AXIS = "#94a3b8"  # Wave plot axis lines (demo/mix/osd/vad/wave_visualization.py)
COLOR_AXIS_TEXT = "#6b7280"  # Wave plot axis text (demo/mix/osd/vad/wave_visualization.py)

WAVEFORM_SAMPLE_RATE = 16_000
WAVEFORM_POINTS_PER_SECOND = 2_000
COMBO_MIN_WIDTH = 140
TOP_ROW_SPACING = 16
PANEL_SPACING = 14
PLOT_PADDING = 8

SPEAKER_PALETTE = [
    "#ff0000",
    "#00a000",
    "#0057ff",
    "#00c7d9",
    "#ff00c8",
    "#ffd400",
    "#7a4cff",
    "#00b894",
]

VISUALIZATION_STYLESHEET = f"""
QWidget {{
    background-color: {COLOR_BG};
    color: {COLOR_TEXT};
}}
QComboBox, QSlider, QPushButton, QToolButton {{
    font-size: 12px;
}}
QComboBox {{
    background-color: {COLOR_BG};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 4px 8px;
    min-height: 24px;
}}
QComboBox:hover {{
    border: 1px solid {COLOR_BORDER_HOVER};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
    background: transparent;
}}
QComboBox::down-arrow {{
    width: 10px;
    height: 10px;
}}
QComboBox QAbstractItemView {{
    background-color: {COLOR_BG};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    margin-top: -1px;
    padding: 0px;
    outline: 0px;
    show-decoration-selected: 0;
    selection-background-color: #e5e7eb;
    selection-color: {COLOR_TEXT};
}}
QComboBox QAbstractItemView::item {{
    min-height: 20px;
    padding-top: 2px;
    padding-right: 12px;
    padding-bottom: 2px;
    padding-left: 12px;
}}
QComboBox QAbstractItemView::item:selected {{
    background: #e5e7eb;
}}
QComboBox QAbstractItemView::item:hover {{
    background: #f3f4f6;
}}
QPushButton, QToolButton {{
    border-radius: 10px;
}}
QToolButton {{
    padding: 6px 10px;
}}
QScrollArea {{
    border: none;
}}
"""

PLAY_BUTTON_STYLESHEET = f"""
QPushButton {{
    background-color: {COLOR_TEXT};
    color: white;
    padding: 8px 16px;
    border: none;
    border-radius: 10px;
    font-weight: 700;
}}
QPushButton:hover {{
    background-color: {COLOR_BUTTON_HOVER};
}}
"""

SLIDER_STYLESHEET = f"""
QSlider::groove:horizontal {{
    height: 6px;
    background: {COLOR_SLIDER_GROOVE};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {COLOR_SLIDER_FILL};
    border-radius: 3px;
}}
QSlider::add-page:horizontal {{
    background: {COLOR_SLIDER_REST};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {COLOR_SLIDER_FILL};
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
"""

TIME_LABEL_STYLESHEET = f"font-size: 12px; color: {COLOR_TEXT_SUBTLE}; min-width: 140px;"
TOP_LABEL_STYLESHEET = f"font-size: 13px; font-weight: 700; color: {COLOR_TEXT};"
PANEL_INFO_STYLESHEET = f"color: {COLOR_TEXT}; font-size: 12px; font-weight: 700;"
META_NOTE_STYLESHEET = f"color: {COLOR_TEXT_MUTED}; font-size: 12px; font-weight: 400; margin-left: 12px;"


def label_html(text: str) -> str:
    return f'<span style="color: {COLOR_TEXT_SUBTLE}; font-size: 12pt; font-weight: 400;">{text}</span>'


@dataclass(frozen=True)
class TimeInterval:
    start: float
    end: float


@dataclass(frozen=True)
class SpeakerInterval:
    speaker: str
    start: float
    end: float


@dataclass(frozen=True)
class SpeakerCountInterval:
    start: float
    end: float
    count: int


@dataclass(frozen=True)
class OSDAnnotation:
    json_path: Path
    intervals: list[TimeInterval]
    onset: float | None
    offset: float | None


@dataclass
class VisualizationPanelSpec:
    tags: list[str]
    note: str | None = None
    colored_speakers: list[SpeakerInterval] = field(default_factory=list)
    overlay_intervals: list[TimeInterval] = field(default_factory=list)
    overlay_color: str = COLOR_MONO_INTERVAL
    use_subtle_background: bool = False
    speakers: list[str] = field(default_factory=list)
    speaker_count_intervals: list[SpeakerCountInterval] = field(default_factory=list)
    max_speaker_count: int = 0
    always_show_speaker_count_strip: bool = False


@dataclass
class AudioVisualizationRecord:
    audio_path: Path
    duration: float
    panels: list[VisualizationPanelSpec]
    dropdown_label: str | None = None


def make_panel(**kwargs) -> VisualizationPanelSpec:
    return VisualizationPanelSpec(**kwargs)


def make_record(audio_path: Path, panels: list[VisualizationPanelSpec], dropdown_label: str | None = None) -> AudioVisualizationRecord:
    return AudioVisualizationRecord(
        audio_path=audio_path,
        duration=audio_duration(audio_path),
        panels=panels,
        dropdown_label=dropdown_label,
    )


def sort_records(records: list[AudioVisualizationRecord]) -> list[AudioVisualizationRecord]:
    return sorted(records, key=lambda record: record.dropdown_label or "")


def display_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def require_directory(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise NotADirectoryError(path)
    return path.resolve()


def audio_duration(audio_path: Path) -> float:
    return float(sf.info(str(audio_path)).duration)


def find_wav_paths(root: Path, *, prefer_audio_dir: bool = False, recursive: bool = False) -> list[Path]:
    base_dir = require_directory(root)
    search_dir = base_dir / "audio" if prefer_audio_dir and (base_dir / "audio").is_dir() else base_dir
    iterator = search_dir.rglob("*.wav") if recursive else search_dir.glob("*.wav")
    paths = sorted(path.resolve() for path in iterator if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No WAV files found under: {search_dir}")
    return paths


def load_json_object(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_waveform(audio_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
    waveform = data.mean(axis=1).astype(np.float32, copy=False)

    if sample_rate != WAVEFORM_SAMPLE_RATE and waveform.size > 0:
        divisor = math.gcd(WAVEFORM_SAMPLE_RATE, sample_rate)
        waveform = scipy.signal.resample_poly(
            waveform,
            WAVEFORM_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        ).astype(np.float32, copy=False)
        sample_rate = WAVEFORM_SAMPLE_RATE

    duration = float(waveform.size) / float(sample_rate) if sample_rate > 0 else 0.0
    target_points = max(1, int(math.ceil(duration * WAVEFORM_POINTS_PER_SECOND)))
    if waveform.size > target_points:
        stride = int(math.ceil(waveform.size / target_points))
        waveform = waveform[::stride]

    if waveform.size == 0:
        waveform = np.zeros(1, dtype=np.float32)

    x_axis = np.linspace(0.0, duration, num=max(1, waveform.size), endpoint=False, dtype=np.float32)
    return x_axis, waveform


def sort_speakers(speakers: Iterable[str]) -> list[str]:
    def sort_key(label: str) -> tuple[int, str]:
        prefix, sep, suffix = label.rpartition("_spk")
        if sep and suffix.isdigit():
            return int(suffix), label
        return 10_000, label

    return sorted(dict.fromkeys(speakers), key=sort_key)


def build_speaker_count_intervals(
    speaker_segments: list[SpeakerInterval],
    duration: float,
) -> tuple[list[SpeakerCountInterval], int]:
    events: list[tuple[float, int, str]] = []
    for segment in speaker_segments:
        start = max(0.0, segment.start)
        end = min(duration, segment.end)
        if end > start:
            events.append((start, 1, segment.speaker))
            events.append((end, -1, segment.speaker))
    events.sort(key=lambda item: item[0])

    active: Counter[str] = Counter()
    intervals: list[SpeakerCountInterval] = []
    cursor = 0.0
    event_index = 0

    while event_index < len(events):
        event_time = events[event_index][0]
        if event_time > cursor:
            intervals.append(SpeakerCountInterval(cursor, event_time, len(active)))
        while event_index < len(events) and events[event_index][0] == event_time:
            _, delta, speaker = events[event_index]
            active[speaker] += delta
            if active[speaker] <= 0:
                del active[speaker]
            event_index += 1
        cursor = event_time

    if cursor < duration:
        intervals.append(SpeakerCountInterval(cursor, duration, len(active)))

    merged: list[SpeakerCountInterval] = []
    for interval in intervals:
        if merged and merged[-1].count == interval.count:
            previous = merged[-1]
            merged[-1] = SpeakerCountInterval(previous.start, interval.end, interval.count)
        else:
            merged.append(interval)

    return merged, max((interval.count for interval in merged), default=0)


def format_optional_float_tag(name: str, value: float | None, precision: int = 2) -> str:
    return f"{name}={value:.{precision}f}" if value is not None else f"{name}=?"


def load_rttm_directory(rttm_dir: Path) -> dict[str, list[SpeakerInterval]]:
    rttm_dir = require_directory(rttm_dir)
    segments_by_uri: dict[str, list[SpeakerInterval]] = {}

    for rttm_path in sorted(rttm_dir.glob("*.rttm")):
        with rttm_path.open("r", encoding="utf-8") as f:
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
                segments_by_uri.setdefault(uri, []).append(
                    SpeakerInterval(speaker=speaker, start=start, end=start + duration)
                )

    for uri in segments_by_uri:
        segments_by_uri[uri].sort(key=lambda item: (item.start, item.end, item.speaker))
    return segments_by_uri


def load_rttm_segments(dataset_root: Path) -> dict[str, list[SpeakerInterval]]:
    return load_rttm_directory(dataset_root / "rttm")


def load_osd_annotations(path: Path) -> dict[str, OSDAnnotation]:
    json_paths = sorted(path.glob("*_osd.json")) if path.is_dir() else [path]
    if not json_paths:
        raise FileNotFoundError(f"No OSD JSON found under: {path}")

    annotations: dict[str, OSDAnnotation] = {}
    for json_path in json_paths:
        data = load_json_object(json_path)
        uri = str(data.get("uri") or json_path.stem.removesuffix("_osd"))
        intervals: list[TimeInterval] = []
        for segment in data.get("overlaps", []):
            try:
                start = float(segment["start"])
                end = float(segment["end"])
            except Exception:
                continue
            if end > start:
                intervals.append(TimeInterval(start=start, end=end))
        annotations[uri] = OSDAnnotation(
            json_path=json_path.resolve(),
            intervals=sorted(intervals, key=lambda item: (item.start, item.end)),
            onset=float(data["onset"]) if data.get("onset") is not None else None,
            offset=float(data["offset"]) if data.get("offset") is not None else None,
        )
    return annotations


def speaker_palette_color(index: int) -> QColor:
    return QColor(SPEAKER_PALETTE[index % len(SPEAKER_PALETTE)])


def build_speaker_color_map(speakers: list[str]) -> dict[str, QColor]:
    return {speaker: speaker_palette_color(index) for index, speaker in enumerate(speakers)}


def populate_audio_output_combo(combo: QComboBox, audio_output: QAudioOutput) -> list[QAudioDevice]:
    audio_devices = list(QMediaDevices.audioOutputs())
    combo.blockSignals(True)
    combo.clear()
    if not audio_devices:
        combo.addItem("No audio output devices found")
        combo.setEnabled(False)
        audio_output.setDevice(QMediaDevices.defaultAudioOutput())
    else:
        combo.setEnabled(True)
        default_index = 0
        for index, device in enumerate(audio_devices):
            label = device.description() or f"Audio output {index + 1}"
            if device.isDefault():
                label = f"{label} (default)"
                default_index = index
            combo.addItem(label)
        combo.setCurrentIndex(default_index)
        audio_output.setDevice(audio_devices[default_index])
    combo.blockSignals(False)
    return audio_devices


class StyledComboBox(QComboBox):
    class _NonNativePopupStyle(QProxyStyle):
        def styleHint(
            self,
            hint: QStyle.StyleHint,
            option=None,
            widget=None,
            returnData=None,
        ) -> int:
            if hint == QStyle.StyleHint.SH_ComboBox_Popup:
                return 0
            return super().styleHint(hint, option, widget, returnData)

    class _PopupListView(QListView):
        def paintEvent(self, event) -> None:  # type: ignore[override]
            super().paintEvent(event)

            scrollbar = self.verticalScrollBar()
            if scrollbar.maximum() <= scrollbar.minimum():
                return

            painter = QPainter(self.viewport())
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(COLOR_TEXT))

            width = self.viewport().width()
            arrow_width = 10
            arrow_height = 6
            center_x = width / 2.0

            if scrollbar.value() > scrollbar.minimum():
                top_arrow = QPolygonF(
                    [
                        QPointF(center_x, 4.0),
                        QPointF(center_x - arrow_width / 2.0, 4.0 + arrow_height),
                        QPointF(center_x + arrow_width / 2.0, 4.0 + arrow_height),
                    ]
                )
                painter.drawPolygon(top_arrow)

            if scrollbar.value() < scrollbar.maximum():
                bottom_y = float(max(0, self.viewport().height() - 4))
                bottom_arrow = QPolygonF(
                    [
                        QPointF(center_x, bottom_y),
                        QPointF(center_x - arrow_width / 2.0, bottom_y - arrow_height),
                        QPointF(center_x + arrow_width / 2.0, bottom_y - arrow_height),
                    ]
                )
                painter.drawPolygon(bottom_arrow)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(COMBO_MIN_WIDTH)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.setMaxVisibleItems(12)
        self.setStyle(self._NonNativePopupStyle(self.style()))
        self.setView(self._PopupListView())
        self.view().setFrameShape(QFrame.Shape.NoFrame)
        self.view().setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view().setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def showPopup(self) -> None:  # type: ignore[override]
        super().showPopup()
        view = self.view()
        current_index = self.model().index(self.currentIndex(), 0)
        if current_index.isValid():
            view.setCurrentIndex(current_index)
            view.scrollTo(current_index, QListView.ScrollHint.PositionAtCenter)
        popup = view.window()
        popup.move(self.mapToGlobal(self.rect().bottomLeft()))
        popup.setMinimumWidth(max(self.width(), COMBO_MIN_WIDTH))



def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:06.3f}"


class ClickablePlotWidget(pg.PlotWidget):
    seek_requested = Signal(float)

    def _emit_seek(self, event: QMouseEvent) -> None:
        position = self.plotItem.vb.mapSceneToView(event.position())
        self.seek_requested.emit(float(max(0.0, position.x())))

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self._emit_seek(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.buttons() & Qt.LeftButton:
            self._emit_seek(event)
        super().mouseMoveEvent(event)


class ClickableSlider(QSlider):
    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton and self.orientation() == Qt.Horizontal:
            value = QStyle.sliderValueFromPosition(
                self.minimum(),
                self.maximum(),
                int(event.position().x()),
                max(1, self.width()),
            )
            self.setValue(value)
            self.sliderMoved.emit(value)
        super().mousePressEvent(event)


class VisualizationPanelWidget(QWidget):
    seek_requested = Signal(float)

    def __init__(
        self,
        panel_spec: VisualizationPanelSpec,
        waveform_x: np.ndarray,
        waveform_y: np.ndarray,
        duration: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.duration = duration
        self.speakers = panel_spec.speakers
        self.speaker_index = {speaker: index for index, speaker in enumerate(self.speakers)}
        self.speaker_colors = build_speaker_color_map(self.speakers)
        self.show_speaker_count_strip = bool(
            panel_spec.speaker_count_intervals
            and panel_spec.max_speaker_count > 0
            and (panel_spec.always_show_speaker_count_strip or panel_spec.max_speaker_count > 2)
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        tags_label = QLabel(" | ".join(panel_spec.tags))
        tags_label.setStyleSheet(PANEL_INFO_STYLESHEET)
        header_layout.addWidget(tags_label, 0)

        if panel_spec.note:
            note_label = QLabel(panel_spec.note)
            note_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            note_label.setStyleSheet(META_NOTE_STYLESHEET)
            header_layout.addWidget(note_label, 0)

        header_layout.addStretch(1)
        if self.speakers:
            header_layout.addWidget(self._build_speaker_legend(), 0, Qt.AlignmentFlag.AlignRight)
        layout.addWidget(header)

        plot_background = COLOR_PLOT_BG_SUBTLE if panel_spec.use_subtle_background else COLOR_PLOT_BG
        self.plot = ClickablePlotWidget()
        self.plot.setBackground(plot_background)
        self.plot.setMinimumHeight(240 if self.show_speaker_count_strip else 220)
        self.plot.setMaximumHeight(280 if self.show_speaker_count_strip else 260)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.setContentsMargins(PLOT_PADDING, PLOT_PADDING, PLOT_PADDING, PLOT_PADDING)
        self.plot.plotItem.layout.setContentsMargins(PLOT_PADDING, PLOT_PADDING, PLOT_PADDING, PLOT_PADDING)
        self.plot.showGrid(x=True, y=True, alpha=0.5)
        self.plot.setLabel("bottom", label_html("Time (s)"))
        self.plot.setLabel("left", label_html("Amplitude"))
        self.plot.setMouseEnabled(x=False, y=False)

        bottom_axis = self.plot.getAxis("bottom")
        left_axis = self.plot.getAxis("left")
        bottom_axis.setPen(pg.mkPen(COLOR_AXIS))
        left_axis.setPen(pg.mkPen(COLOR_AXIS))
        bottom_axis.setTextPen(pg.mkPen(COLOR_AXIS_TEXT))
        left_axis.setTextPen(pg.mkPen(COLOR_AXIS_TEXT))
        left_axis.enableAutoSIPrefix(False)
        self.plot.seek_requested.connect(self.seek_requested.emit)
        layout.addWidget(self.plot)

        self.plot.plot(waveform_x, waveform_y, pen=pg.mkPen(COLOR_WAVEFORM, width=1.1))
        self._set_plot_ranges(waveform_y, duration)
        view_y_min, view_y_max = self.plot.getViewBox().viewRange()[1]
        self._add_speaker_regions(panel_spec, view_y_min, view_y_max)
        self._add_overlay_regions(panel_spec)
        self._add_speaker_count_strip(panel_spec, plot_background)

        self.playhead = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen(COLOR_PLAYHEAD, width=2))
        self.playhead.setZValue(50)
        self.plot.addItem(self.playhead)

    def _set_plot_ranges(self, waveform_y: np.ndarray, duration: float) -> tuple[float, float]:
        y_min = float(np.min(waveform_y)) if waveform_y.size else -1.0
        y_max = float(np.max(waveform_y)) if waveform_y.size else 1.0
        if abs(y_max - y_min) < 1e-6:
            y_min -= 1.0
            y_max += 1.0
        self.plot.setLimits(xMin=0.0, xMax=max(0.1, duration))
        self.plot.setXRange(0.0, max(0.1, duration), padding=0.01)
        self.plot.setYRange(y_min * 1.15, y_max * 1.15, padding=0.02)
        return y_min, y_max

    def _add_speaker_regions(self, panel_spec: VisualizationPanelSpec, y_min: float, y_max: float) -> None:
        if not panel_spec.colored_speakers:
            return

        speaker_count = max(1, len(self.speakers))
        lane_height = (y_max - y_min) / speaker_count
        for interval in panel_spec.colored_speakers:
            speaker_index = self.speaker_index.get(interval.speaker, 0)
            lane_bottom = y_min + speaker_index * lane_height
            rect = QGraphicsRectItem(interval.start, lane_bottom, max(0.0, interval.end - interval.start), lane_height)
            color = QColor(self.speaker_colors[interval.speaker])
            color.setAlphaF(0.18)
            rect.setBrush(pg.mkBrush(color))
            rect.setPen(pg.mkPen(None))
            rect.setZValue(-10)
            self.plot.addItem(rect)

        overlap_intervals = self._speaker_overlap_intervals(panel_spec.colored_speakers)
        if overlap_intervals:
            self._add_full_height_regions(overlap_intervals, panel_spec.overlay_color, z_value=-5)

    def _add_overlay_regions(self, panel_spec: VisualizationPanelSpec) -> None:
        if not panel_spec.overlay_intervals:
            return
        self._add_full_height_regions(panel_spec.overlay_intervals, panel_spec.overlay_color, z_value=-20)

    def _add_speaker_count_strip(
        self,
        panel_spec: VisualizationPanelSpec,
        background_color: str,
    ) -> None:
        if not self.show_speaker_count_strip:
            return

        self.speaker_count_strip = pg.ViewBox(enableMenu=False)
        self.speaker_count_strip.setMouseEnabled(x=False, y=False)
        self.speaker_count_strip.setYRange(0.0, 1.0, padding=0.0)
        self.speaker_count_strip.setLimits(yMin=0.0, yMax=1.0)
        self.speaker_count_strip.setXLink(self.plot.getViewBox())
        self.plot.plotItem.layout.addItem(self.speaker_count_strip, 4, 1)
        self.plot.plotItem.layout.setRowFixedHeight(4, 14)
        self.plot.plotItem.layout.setRowSpacing(3, 6)

        max_count = panel_spec.max_speaker_count
        background = QColor(background_color)
        for interval in panel_spec.speaker_count_intervals:
            background_ratio = (max_count - interval.count) / max_count
            color = QColor(
                round(background.red() * background_ratio),
                round(background.green() * background_ratio),
                round(background.blue() * background_ratio),
            )
            rect = QGraphicsRectItem(
                interval.start,
                0.0,
                max(0.0, interval.end - interval.start),
                1.0,
            )
            rect.setBrush(pg.mkBrush(color))
            rect.setPen(pg.mkPen(None))
            self.speaker_count_strip.addItem(rect)

    def _speaker_overlap_intervals(self, intervals: list[SpeakerInterval]) -> list[TimeInterval]:
        events: list[tuple[float, int]] = []
        for interval in intervals:
            if interval.end > interval.start:
                events.append((interval.start, 1))
                events.append((interval.end, -1))

        events.sort(key=lambda item: (item[0], item[1]))
        active = 0
        overlap_start: float | None = None
        overlaps: list[TimeInterval] = []

        for time, delta in events:
            prev_active = active
            active += delta
            if prev_active < 2 and active >= 2:
                overlap_start = time
            elif prev_active >= 2 and active < 2 and overlap_start is not None and time > overlap_start:
                overlaps.append(TimeInterval(start=overlap_start, end=time))
                overlap_start = None

        return overlaps

    def _add_full_height_regions(self, intervals: list[TimeInterval], color_value: str, *, z_value: float) -> None:
        color = QColor(color_value)
        color.setAlpha(80)
        for interval in intervals:
            region = pg.LinearRegionItem(
                values=(interval.start, interval.end),
                brush=pg.mkBrush(color),
                pen=pg.mkPen(None),
                hoverPen=pg.mkPen(None),
                movable=False,
            )
            region.setZValue(z_value)
            self.plot.addItem(region)

    def _build_speaker_legend(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        for speaker in self.speakers:
            speaker_index = self.speaker_index[speaker]
            entry = QWidget()
            entry_layout = QHBoxLayout(entry)
            entry_layout.setContentsMargins(0, 0, 0, 0)
            entry_layout.setSpacing(6)

            dot = QLabel()
            dot.setFixedSize(12, 12)
            dot.setStyleSheet(f"background-color: {self.speaker_colors[speaker].name()}; border-radius: 6px;")
            entry_layout.addWidget(dot)

            speaker_label = QLabel(speaker)
            speaker_label.setStyleSheet(f"font-size: 12px; color: {COLOR_TEXT_MUTED}; font-weight: 400;")
            entry_layout.addWidget(speaker_label)

            layout.addWidget(entry)

        return widget

    def set_playhead(self, time_seconds: float) -> None:
        self.playhead.setPos(max(0.0, min(time_seconds, max(self.duration, 0.1))))


class VisualizationWindow(QMainWindow):
    def __init__(self, window_title: str, records: list[AudioVisualizationRecord]) -> None:
        super().__init__()
        self.records = records
        self.panels: list[VisualizationPanelWidget] = []
        self.waveform_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.audio_devices: list[QAudioDevice] = []

        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(1.0)
        self.audio_output.setMuted(False)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.positionChanged.connect(self._on_position_changed)
        self.media_player.durationChanged.connect(self._on_duration_changed)
        self.media_player.playbackStateChanged.connect(self._sync_play_button)

        self.setWindowTitle(window_title)
        self.resize(1400, 1000)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(14)

        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(TOP_ROW_SPACING)

        audio_label = QLabel("Audio")
        audio_label.setStyleSheet(TOP_LABEL_STYLESHEET)
        top_layout.addWidget(audio_label)

        self.record_combo = StyledComboBox()
        for index, record in enumerate(self.records):
            label = record.dropdown_label or display_relative(record.audio_path)
            self.record_combo.addItem(label, userData=index)
        self.record_combo.currentIndexChanged.connect(self._on_record_changed)
        top_layout.addWidget(self.record_combo, 1)

        device_group = QWidget()
        device_layout = QHBoxLayout(device_group)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.setSpacing(10)

        device_label = QLabel("Device")
        device_label.setStyleSheet(TOP_LABEL_STYLESHEET)
        device_layout.addWidget(device_label)

        self.audio_output_combo = StyledComboBox()
        self.audio_output_combo.currentIndexChanged.connect(self._on_audio_output_changed)
        device_layout.addWidget(self.audio_output_combo)
        top_layout.addWidget(device_group, 0, Qt.AlignmentFlag.AlignRight)
        root_layout.addWidget(top_row)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.scroll_container = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_container)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(PANEL_SPACING)
        self.scroll_layout.addStretch(1)
        self.scroll_area.setWidget(self.scroll_container)
        root_layout.addWidget(self.scroll_area, 1)

        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)

        self.play_button = QPushButton("Play")
        self.play_button.setCursor(Qt.PointingHandCursor)
        self.play_button.setStyleSheet(PLAY_BUTTON_STYLESHEET)
        self.play_button.clicked.connect(self._toggle_playback)
        controls_layout.addWidget(self.play_button)

        self.time_label = QLabel("00:00.000 / 00:00.000")
        self.time_label.setStyleSheet(TIME_LABEL_STYLESHEET)
        controls_layout.addWidget(self.time_label)

        self.position_slider = ClickableSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.setStyleSheet(SLIDER_STYLESHEET)
        self.position_slider.sliderMoved.connect(self._seek_to_slider)
        controls_layout.addWidget(self.position_slider, 1)
        root_layout.addWidget(controls)

        self.setCentralWidget(root)
        self._populate_audio_outputs()
        self._load_record(0 if self.records else None)

    def _populate_audio_outputs(self) -> None:
        self.audio_devices = populate_audio_output_combo(self.audio_output_combo, self.audio_output)

    def _get_waveform(self, audio_path: Path) -> tuple[np.ndarray, np.ndarray]:
        key = str(audio_path.resolve())
        if key not in self.waveform_cache:
            self.waveform_cache[key] = load_waveform(audio_path)
        return self.waveform_cache[key]

    def _clear_panels(self) -> None:
        while self.panels:
            panel = self.panels.pop()
            panel.setParent(None)
            panel.deleteLater()

    def _load_record(self, index: int | None) -> None:
        self._clear_panels()
        if index is None or not (0 <= index < len(self.records)):
            return

        record = self.records[index]
        waveform_x, waveform_y = self._get_waveform(record.audio_path)
        for panel_spec in record.panels:
            panel = VisualizationPanelWidget(panel_spec, waveform_x, waveform_y, record.duration, parent=self.scroll_container)
            panel.seek_requested.connect(self._seek_from_panel)
            self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, panel)
            self.panels.append(panel)

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
        label = "Pause" if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState else "Play"
        self.play_button.setText(label)

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
        self.time_label.setText(f"{format_timestamp(position_ms / 1000.0)} / {format_timestamp(duration_ms / 1000.0)}")

    def _on_record_changed(self, index: int) -> None:
        self._load_record(index)

    def _on_audio_output_changed(self, index: int) -> None:
        if 0 <= index < len(self.audio_devices):
            self.audio_output.setDevice(self.audio_devices[index])


def run_visualization(window_title: str, records: list[AudioVisualizationRecord]) -> None:
    created_app = QApplication.instance() is None
    app = QApplication.instance() or QApplication([])
    app.setApplicationName(window_title)
    app.setStyleSheet(VISUALIZATION_STYLESHEET)

    window = VisualizationWindow(window_title, records)
    window.show()

    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(250)

    def cleanup() -> None:
        timer.stop()
        window.media_player.stop()
        window.media_player.setSource(QUrl())

    def quit_gracefully() -> None:
        cleanup()
        window.close()
        app.exit(0)

    old_sigint_handler = None
    try:
        import signal

        old_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: QTimer.singleShot(0, quit_gracefully))
    except Exception:
        pass

    exit_code = 0
    try:
        exit_code = app.exec()
    finally:
        cleanup()
        window.close()
        try:
            import signal

            if old_sigint_handler is not None:
                signal.signal(signal.SIGINT, old_sigint_handler)
        except Exception:
            pass
    if created_app:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(int(exit_code))
