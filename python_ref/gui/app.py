"""PyQt6 + matplotlib annotation window — a thin shell over SpectrogramSession.

Three selection tools are wired to the engine: rectangle (drag), magnetic ridge (click
start + end), and smart energy brush (drag to paint, optional dB threshold). The view can
be switched live (STFT/CQT/chroma/scalogram); switching re-analyses the audio and clears
layers (their masks are view-specific). All selections work in display coordinates, so the
same gestures apply in every view.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from PyQt6 import QtWidgets

from ..params import STFT
from ..transforms import TRANSFORMS
from .render import SpectrogramRenderer
from .session import SpectrogramSession


class AnnotatorWindow(QtWidgets.QMainWindow):
    def __init__(self, session: SpectrogramSession):
        super().__init__()
        self.session = session
        self.renderer = SpectrogramRenderer(session.transform)
        self._selection: tuple[float, float, float, float] | None = None
        self._ridge_start: tuple[float, float] | None = None
        self._brushing = False
        self._brush_pts: list[tuple[float, float]] = []
        self.setWindowTitle(f"Anots — {Path(session.source_file).name or 'untitled'}")

        canvas = self._build_canvas()
        self.setCentralWidget(self._wrap_with_controls(canvas))
        self._draw_spectrogram()

    # --- construction --------------------------------------------------------

    def _build_canvas(self) -> FigureCanvasQTAgg:
        self.figure = Figure(figsize=(10, 5))
        self.ax = self.figure.add_subplot(111)
        canvas = FigureCanvasQTAgg(self.figure)
        self.selector = RectangleSelector(
            self.ax, self._on_select, useblit=True, interactive=True,
            button=[1], minspanx=1e-3, minspany=1.0,
        )
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("button_release_event", self._on_release)
        return canvas

    def _wrap_with_controls(self, canvas: FigureCanvasQTAgg) -> QtWidgets.QWidget:
        self.view_box = QtWidgets.QComboBox()
        self.view_box.addItems(sorted(TRANSFORMS))
        self.view_box.setCurrentText(self.session.transform.name)
        self.view_box.currentTextChanged.connect(self._on_view_changed)

        self.tool_box = QtWidgets.QComboBox()
        self.tool_box.addItems(["rectangle", "ridge", "brush"])
        self.tool_box.currentTextChanged.connect(self._on_tool_changed)

        self.label_edit = QtWidgets.QLineEdit(placeholderText="label (optional)")
        self.confidence = QtWidgets.QDoubleSpinBox(minimum=0.0, maximum=1.0, singleStep=0.1, value=1.0)
        self.radius = QtWidgets.QSpinBox(minimum=1, maximum=64, value=8)
        self.threshold_on = QtWidgets.QCheckBox("dB ≥")
        self.threshold_db = QtWidgets.QDoubleSpinBox(minimum=-120.0, maximum=0.0, value=-40.0)

        open_btn = QtWidgets.QPushButton("Open audio…")
        open_btn.clicked.connect(self._open_audio)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(open_btn)
        top.addWidget(QtWidgets.QLabel("View:")); top.addWidget(self.view_box)
        top.addWidget(QtWidgets.QLabel("Tool:")); top.addWidget(self.tool_box)
        top.addWidget(QtWidgets.QLabel("Label:")); top.addWidget(self.label_edit, 1)
        top.addWidget(QtWidgets.QLabel("Conf:")); top.addWidget(self.confidence)

        # Right-hand layer panel: list + delete (multi-layer management).
        self.layer_list = QtWidgets.QListWidget()
        del_btn = QtWidgets.QPushButton("Delete layer")
        del_btn.clicked.connect(self._delete_layer)
        panel = QtWidgets.QVBoxLayout()
        panel.addWidget(QtWidgets.QLabel("Layers"))
        panel.addWidget(self.layer_list, 1)
        panel.addWidget(del_btn)
        panel_w = QtWidgets.QWidget(); panel_w.setLayout(panel); panel_w.setMaximumWidth(240)

        middle = QtWidgets.QHBoxLayout()
        middle.addWidget(canvas, 1)
        middle.addWidget(panel_w)

        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(QtWidgets.QLabel("Brush r:")); bottom.addWidget(self.radius)
        bottom.addWidget(self.threshold_on); bottom.addWidget(self.threshold_db)
        for text, slot in (("Add layer", self._add_layer),
                           ("Play selection", self._play_selection),
                           ("Export…", self._export)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            bottom.addWidget(button)

        self.status = QtWidgets.QLabel("Rectangle: drag a box, then Add layer. (No label? one is auto-named.)")
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(middle, 1)
        layout.addLayout(bottom)
        layout.addWidget(self.status)
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        return container

    # --- drawing -------------------------------------------------------------

    def _draw_spectrogram(self) -> None:
        self.renderer.draw(self.ax, self.session.coeffs, self.session.sr, self.session.duration_s)
        self.figure.tight_layout()

    def _redraw(self) -> None:
        self.ax.clear()
        self._draw_spectrogram()
        self.figure.canvas.draw_idle()

    # --- view / tool selection ----------------------------------------------

    @property
    def tool(self) -> str:
        return self.tool_box.currentText()

    def _open_audio(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open audio", "", "Audio (*.wav *.flac *.ogg *.aiff *.aif);;All files (*)")
        if not path:
            return
        self.session = build_session(path, self.view_box.currentText())
        self.renderer = SpectrogramRenderer(self.session.transform)
        self.setWindowTitle(f"Anots — {Path(path).name}")
        self._selection = self._ridge_start = None
        self._refresh_layers()
        self.status.setText(f"Loaded {Path(path).name}.")

    def _on_view_changed(self, name: str) -> None:
        s = self.session
        self.session = SpectrogramSession(s.y, s.sr, source_file=s.source_file,
                                          transform=TRANSFORMS[name]())
        self.renderer = SpectrogramRenderer(self.session.transform)
        self._selection = self._ridge_start = None
        self._refresh_layers()
        self.status.setText(f"View: {name} (layers cleared — masks are view-specific).")

    def _on_tool_changed(self, name: str) -> None:
        self.selector.set_active(name == "rectangle")
        self._ridge_start = None
        self._brushing = False
        hint = {"rectangle": "drag a box, then Add layer.",
                "ridge": "click a start point, then an end point.",
                "brush": "drag to paint; tick 'dB ≥' to grab only loud bins."}[name]
        self.status.setText(f"{name.capitalize()}: {hint}")

    # --- matplotlib interactions --------------------------------------------

    def _on_select(self, press, release) -> None:  # rectangle (RectangleSelector)
        self._selection = (press.xdata, release.xdata, press.ydata, release.ydata)
        t0, t1, y0, y1 = self._selection
        self.status.setText(f"Selection: {t0:.3f}–{t1:.3f} s, y {y0:.1f}–{y1:.1f}")

    def _on_press(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if self.tool == "ridge":
            if self._ridge_start is None:
                self._ridge_start = (event.xdata, event.ydata)
                self.status.setText("Ridge: now click the end point.")
            else:
                t0, y0 = self._ridge_start
                self._commit("ridge", lambda lbl, c: self.session.add_ridge_display(
                    t0, y0, event.xdata, event.ydata, lbl, confidence=c))
                self._ridge_start = None
        elif self.tool == "brush":
            self._brushing = True
            self._brush_pts = [(event.xdata, event.ydata)]

    def _on_motion(self, event) -> None:
        if self._brushing and event.inaxes is self.ax and event.xdata is not None:
            self._brush_pts.append((event.xdata, event.ydata))

    def _on_release(self, event) -> None:
        if self.tool != "brush" or not self._brushing:
            return
        self._brushing = False
        if event.xdata is not None:
            self._brush_pts.append((event.xdata, event.ydata))
        pts = self._brush_pts
        thr = self.threshold_db.value() if self.threshold_on.isChecked() else None
        self._commit("brush", lambda lbl, c: self.session.add_brush(
            pts, lbl, radius=self.radius.value(), threshold_db=thr, confidence=c))

    # --- committing layers ---------------------------------------------------

    def _add_layer(self) -> None:  # rectangle commit (drag + button)
        if self._selection is None:
            self.status.setText("Draw a rectangle first.")
            return
        t0, t1, y0, y1 = self._selection
        self._commit("rectangle", lambda lbl, c: self.session.add_rectangle_display(
            t0, t1, y0, y1, lbl, confidence=c))
        self._selection = None

    def _commit(self, tool: str, make_layer) -> None:
        # Label is optional — auto-name so a gesture always produces a visible layer.
        label = self.label_edit.text() or f"layer_{len(self.session.layers) + 1}"
        layer = make_layer(label, self.confidence.value())
        self._refresh_layers()
        self.status.setText(f"Added '{layer.label}' via {tool} ({len(self.session.layers)} layers).")

    def _delete_layer(self) -> None:
        row = self.layer_list.currentRow()
        if row < 0:
            row = len(self.session.layers) - 1  # default to the most recent
        if 0 <= row < len(self.session.layers):
            removed = self.session.layers.pop(row)
            self._refresh_layers()
            self.status.setText(f"Removed '{removed.label}' ({len(self.session.layers)} left).")
        else:
            self.status.setText("No layer to delete.")

    def _refresh_layers(self) -> None:
        """Redraw the spectrogram with every current layer overlaid, and sync the list."""
        self.ax.clear()
        self._draw_spectrogram()
        for layer in self.session.layers:
            self._draw_overlay(layer)
        self.figure.canvas.draw_idle()
        self.layer_list.clear()
        for i, layer in enumerate(self.session.layers):
            self.layer_list.addItem(f"{i}: {layer.label} [{layer.tool_used.value}]")

    def _draw_overlay(self, layer) -> None:
        active_t = np.where(layer.mask.any(axis=0))[0]
        active_f = np.where(layer.mask.any(axis=1))[0]
        if not len(active_t) or not len(active_f):
            return
        sr, tf = self.session.sr, self.session.transform
        x0, x1 = tf.col_to_time(active_t[0], sr), tf.col_to_time(active_t[-1], sr)
        y0, y1 = tf.row_to_display_y(active_f[0], sr), tf.row_to_display_y(active_f[-1], sr)
        self.ax.add_patch(plt_rect((x0, y0), x1 - x0, y1 - y0, layer.color))

    def _play_selection(self) -> None:
        if not self.session.layers:
            self.status.setText("No layer to play.")
            return
        sd.play(self.session.reconstruct_layer(self.session.layers[-1]), self.session.sr)

    def _export(self) -> None:
        out = QtWidgets.QFileDialog.getExistingDirectory(self, "Export to")
        if out:
            anns = self.session.export(out)
            self.status.setText(f"Exported {len(anns)} clips to {out}")


def plt_rect(xy, w, h, color):
    from matplotlib.patches import Rectangle

    return Rectangle(xy, w, h, fill=False, edgecolor=color, linewidth=1.5)


def build_session(path: str, view: str = "stft") -> SpectrogramSession:
    """Load ``path`` into a session backed by the named transform view (design §4.3)."""
    if view not in TRANSFORMS:
        raise SystemExit(f"unknown view {view!r}; choose from {', '.join(TRANSFORMS)}")
    transform = TRANSFORMS[view]()  # each view carries its own locked default params
    return SpectrogramSession.load(path, STFT, transform=transform)


def run(path: str, view: str = "stft") -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = AnnotatorWindow(build_session(path, view))
    window.resize(1100, 700)
    window.show()
    return app.exec()
