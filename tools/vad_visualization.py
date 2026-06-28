from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
import scipy.signal
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
    QVBoxLayout,
    QWidget,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_ROOTS = (
    REPO_ROOT / "data" / "short_test",
    REPO_ROOT / "data" / "nonverbal_speech_38k",
    REPO_ROOT / "data" / "nonverbal_tts",
    REPO_ROOT / "data" / "ami",
)
DEFAULT_WAVEFORM_SAMPLE_RATE = 16_000
DEFAULT_WAVEFORM_POINTS = 5_000
VAD_COLOR = "#afafaf"


@dataclass(frozen=True)
class Segment:
    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class JsonResult:
    json_path: Path
    media_path: Path
    speakers: list[str]
    segments: list[Segment]
    threshold: float | None
    raw: dict[str, Any]

    @property
    def label(self) -> str:
        return display_relative(self.json_path)


@dataclass
class MediaGroup:
    media_path: Path
    results: list[JsonResult]

    @property
    def label(self) -> str:
        return self.media_path.name


def display_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def resolve_media_path(json_data: dict[str, Any], json_path: Path) -> Path:
    for key in ("input", "media", "audio", "path"):
        raw = json_data.get(key)
        if not raw:
            continue

        candidate = Path(str(raw)).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()

        for base in (json_path.parent, REPO_ROOT, *DEFAULT_JSON_ROOTS):
            resolved = (base / candidate).resolve()
            if resolved.exists():
                return resolved

        return (json_path.parent / candidate).resolve()

    # Fallback: infer from the JSON file name.
    inferred = json_path.with_suffix(".wav")
    if inferred.exists():
        return inferred.resolve()
    return inferred.resolve()


def discover_json_paths(json_roots: list[Path]) -> list[Path]:
    json_paths: list[Path] = []
    seen: set[Path] = set()
    for json_root in json_roots:
        if not json_root.exists():
            continue
        for path in json_root.rglob("*.json"):
            if not path.is_file():
                continue
            if "_vad" not in path.stem:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            json_paths.append(resolved)
    return sorted(json_paths, key=lambda p: p.as_posix())


def load_result(json_path: Path) -> JsonResult:
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a VAD JSON object, got {type(raw).__name__} in {json_path}")

    media_path = resolve_media_path(raw, json_path)
    segments: list[Segment] = []
    for seg in raw.get("segments", []):
        try:
            start = float(seg["start"])
            end = float(seg["end"])
        except Exception:
            continue
        speaker = str(seg.get("speaker") or "VAD")
        segments.append(Segment(speaker=speaker, start=start, end=end))

    segments.sort(key=lambda s: (s.start, s.end, s.speaker))

    threshold: float | None = None
    parameters = raw.get("parameters")
    if isinstance(parameters, dict):
        try:
            threshold = float(parameters.get("threshold"))
        except Exception:
            threshold = None
    if threshold is None:
        try:
            threshold = float(raw.get("threshold"))
        except Exception:
            threshold = None

    return JsonResult(
        json_path=json_path.resolve(),
        media_path=media_path,
        speakers=["VAD"] if segments else [],
        segments=segments,
        threshold=threshold,
        raw=raw,
    )


def group_results_by_media(results: list[JsonResult]) -> list[MediaGroup]:
    grouped: dict[str, MediaGroup] = {}
    for result in results:
        key = result.media_path.name
        if key not in grouped:
            grouped[key] = MediaGroup(media_path=result.media_path.resolve(), results=[])
        grouped[key].results.append(result)

    groups = list(grouped.values())
    for group in groups:
        group.results.sort(key=sort_result_by_threshold)
    groups.sort(key=lambda g: (g.media_path.name, len(g.results)))
    return groups


def choose_default_group(groups: list[MediaGroup]) -> MediaGroup | None:
    if not groups:
        return None
    return sorted(groups, key=lambda g: (-len(g.results), g.media_path.name))[0]


def sort_result_by_threshold(result: JsonResult) -> tuple[bool, float, str]:
    threshold = result.threshold
    if threshold is None:
        return True, float("inf"), result.json_path.as_posix()
    return False, threshold, result.json_path.as_posix()


def load_audio_waveform(
    audio_path: Path,
    sample_rate: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    data, sr = sf.read(str(audio_path), always_2d=True)
    waveform = data.mean(axis=1).astype(np.float32, copy=False)

    if sr != sample_rate and waveform.size > 0:
        g = math.gcd(sample_rate, sr)
        waveform = scipy.signal.resample_poly(waveform, sample_rate // g, sr // g).astype(np.float32, copy=False)
        sr = sample_rate

    if waveform.size == 0:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32), sr

    total_duration = float(waveform.size) / float(sr)

    if waveform.size > max_points:
        stride = int(math.ceil(waveform.size / max_points))
        waveform = waveform[::stride]

    x = np.linspace(0.0, total_duration, num=waveform.size, endpoint=False, dtype=np.float32)
    return x, waveform, sr


class ClickablePlotWidget(pg.PlotWidget):
    seekRequested = Signal(float)

    def mousePressEvent(self, ev: QMouseEvent) -> None:  # type: ignore[override]
        if ev.button() == Qt.LeftButton:
            pos = self.plotItem.vb.mapSceneToView(ev.position())
            self.seekRequested.emit(float(max(0.0, pos.x())))
        super().mousePressEvent(ev)


class ResultPanel(QWidget):
    seekRequested = Signal(float)

    def __init__(
        self,
        result: JsonResult,
        waveform_x: np.ndarray,
        waveform_y: np.ndarray,
        waveform_duration: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.result = result
        self.waveform_duration = waveform_duration

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
        header.setStyleSheet(
            "color: #6b7280; font-size: 11px; font-weight: 600;"
        )
        header_layout.addWidget(header, 1)

        threshold_label = QLabel(self._format_threshold_label(result.threshold))
        threshold_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        threshold_label.setStyleSheet(
            "color: #374151; font-size: 11px; font-weight: 600;"
            "background-color: #eef2f7; border: 1px solid #dbe3ee;"
            "border-radius: 999px; padding: 2px 8px;"
        )
        threshold_label.setVisible(result.threshold is not None)
        header_layout.addWidget(threshold_label, 0, Qt.AlignRight)

        outer.addWidget(header_row)

        self.plot = ClickablePlotWidget()
        self.plot.setBackground("#f8fafc")
        self.plot.setMinimumHeight(180)
        self.plot.setMaximumHeight(220)
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Amplitude")
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.seekRequested.connect(self.seekRequested.emit)
        outer.addWidget(self.plot)

        self.wave_item = self.plot.plot(
            waveform_x,
            waveform_y,
            pen=pg.mkPen((55, 65, 81), width=1.1),
        )
        self.plot.setLimits(xMin=0.0, xMax=max(0.1, float(waveform_x[-1]) if waveform_x.size else 0.1))
        self.plot.setXRange(0.0, max(0.1, float(waveform_x[-1]) if waveform_x.size else 0.1), padding=0.01)
        y_min = float(np.min(waveform_y)) if waveform_y.size else -1.0
        y_max = float(np.max(waveform_y)) if waveform_y.size else 1.0
        if abs(y_max - y_min) < 1e-6:
            y_min -= 1.0
            y_max += 1.0
        self.plot.setYRange(y_min * 1.15, y_max * 1.15, padding=0.02)

        for region_span in self.result.segments:
            if region_span.end <= region_span.start:
                continue
            region = pg.LinearRegionItem(
                values=(region_span.start, region_span.end),
                brush=pg.mkBrush(self._region_color_with_alpha(0.18)),
                movable=False,
            )
            region.setZValue(-10)
            self.plot.addItem(region)

        self.playhead = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=False,
            pen=pg.mkPen((17, 24, 39), width=2),
        )
        self.playhead.setZValue(50)
        self.plot.addItem(self.playhead)

    @property
    def result_duration(self) -> float:
        if self.result.segments:
            return max(seg.end for seg in self.result.segments)
        return self.waveform_duration

    def _region_color_with_alpha(self, alpha: float) -> QColor:
        color = QColor(VAD_COLOR)
        color.setAlphaF(max(0.0, min(1.0, alpha)))
        return color

    def _format_threshold_label(self, threshold: float | None) -> str:
        if threshold is None:
            return ""
        return f"threshold: {threshold:.2f}"

    def set_playhead(self, time_seconds: float) -> None:
        self.playhead.setPos(max(0.0, min(time_seconds, max(self.waveform_duration, self.result_duration))))


class DiarizationVisualization(QMainWindow):
    def __init__(
        self,
        groups: list[MediaGroup],
        json_roots: list[Path],
        waveform_sample_rate: int,
        waveform_points: int,
        audio_output_device_name: str | None = None,
    ) -> None:
        super().__init__()
        self.groups = groups
        self.json_roots = json_roots
        self.waveform_sample_rate = waveform_sample_rate
        self.waveform_points = waveform_points
        self.waveform_cache: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        self.current_group: MediaGroup | None = None
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
        self.media_player.errorOccurred.connect(self._on_media_error)

        self.setWindowTitle("Diarization Visualization")
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

    def _on_media_error(self, error: object, error_string: str) -> None:
        if error_string:
            print(f"Media error: {error_string}")

    def _on_duration_changed(self, duration_ms: int) -> None:
        self.position_slider.setRange(0, max(0, duration_ms))
        self._update_slider_style(self.media_player.position(), duration_ms)
        self._update_time_label(self.media_player.position(), duration_ms)

    def _seek_to_slider(self, value: int) -> None:
        self.media_player.setPosition(value)

    def _on_position_changed(self, position_ms: int) -> None:
        duration_ms = max(self.media_player.duration(), 0)
        if not self.position_slider.isSliderDown():
            self.position_slider.setValue(position_ms)
        self._update_slider_style(position_ms, duration_ms)
        self._update_time_label(position_ms, duration_ms)
        time_seconds = position_ms / 1000.0
        for panel in self.panels:
            panel.set_playhead(time_seconds)

    def _update_time_label(self, position_ms: int, duration_ms: int) -> None:
        self.time_label.setText(f"{format_time(position_ms / 1000.0)} / {format_time(duration_ms / 1000.0)}")

    def _update_slider_style(self, position_ms: int, duration_ms: int) -> None:
        if duration_ms <= 0:
            return
        self.position_slider.setStyleSheet(
            f"""
            QSlider::groove:horizontal {{
                height: 6px;
                background: #e5e7eb;
                border-radius: 3px;
            }}
            QSlider::sub-page:horizontal {{
                background: #111827;
                border-radius: 3px;
            }}
            QSlider::add-page:horizontal {{
                background: #d1d5db;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: #111827;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
            """
        )

    def _on_group_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.groups):
            return
        self._load_group(self.groups[index])

    def _load_group(self, group: MediaGroup | None) -> None:
        self.current_group = group
        self._clear_panels()

        if group is None:
            self.group_count_label.setText("No JSON results found.")
            return

        self.group_count_label.setText(f"{len(group.results)} JSON result(s) for {group.label}")
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
    parser = argparse.ArgumentParser(description="Diarization visualization with per-JSON panels.")
    parser.add_argument(
        "--json-root",
        type=Path,
        nargs="+",
        default=list(DEFAULT_JSON_ROOTS),
        help=(
            "Root directory or directories to recursively scan for JSON files "
            f"(default: {', '.join(str(p) for p in DEFAULT_JSON_ROOTS)})."
        ),
    )
    parser.add_argument(
        "--media",
        type=Path,
        default=None,
        help="Optional explicit source audio path. If omitted, the first source with the most JSON results is used.",
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
        help=(
            "Optional audio output device name to preselect. "
            "If omitted, the default system output is used."
        ),
    )
    return parser


def select_group(groups: list[MediaGroup], media: Path | None) -> MediaGroup | None:
    if not groups:
        return None
    if media is None:
        return choose_default_group(groups)

    media_name = media.name
    for group in groups:
        if group.media_path.name == media_name:
            return group

    return choose_default_group(groups)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    json_paths = discover_json_paths(args.json_root)
    results = [load_result(path) for path in json_paths]
    groups = group_results_by_media(results)
    selected_group = select_group(groups, args.media)

    root_list = ", ".join(str(path) for path in args.json_root)
    print(f"Scanned {len(json_paths)} JSON file(s) under {root_list}")
    if selected_group is not None:
        print(f"Selected source audio: {selected_group.label} ({len(selected_group.results)} JSON result(s))")
    else:
        print("No diarization JSON results were found.")

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Diarization Visualization")
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

    window = DiarizationVisualization(
        groups=groups,
        json_roots=args.json_root,
        waveform_sample_rate=args.waveform_sample_rate,
        waveform_points=args.waveform_points,
        audio_output_device_name=args.audio_output_device,
    )

    if selected_group is not None:
        index = next((i for i, g in enumerate(groups) if g.media_path.resolve() == selected_group.media_path.resolve()), 0)
        window.group_combo.setCurrentIndex(index)

    window.show()

    # Keep Qt responsive to Ctrl+C in terminal.
    timer = QTimer(window)
    timer.timeout.connect(lambda: None)
    timer.start(250)

    def _quit_on_sigint(*_: object) -> None:
        app.quit()

    try:
        import signal

        signal.signal(signal.SIGINT, _quit_on_sigint)
    except Exception:
        pass

    app.exec()


if __name__ == "__main__":
    main()
