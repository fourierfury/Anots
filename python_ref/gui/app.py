"""PyQt6 + matplotlib annotation window — a thin shell over SpectrogramSession."""

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
from .render import SpectrogramRenderer
from .session import SpectrogramSession


class AnnotatorWindow(QtWidgets.QMainWindow):
    def __init__(self, session: SpectrogramSession):
        super().__init__()
        self.session = session
        self.renderer = SpectrogramRenderer(session.transform)
        self._selection: tuple[float, float, float, float] | None = None
        self.setWindowTitle(f"Anots — {Path(session.source_file).name or 'untitled'}")

        canvas = self._build_canvas()
        self.setCentralWidget(self._wrap_with_controls(canvas))
        self._draw_spectrogram()

    def _build_canvas(self) -> FigureCanvasQTAgg:
        self.figure = Figure(figsize=(10, 5))
        self.ax = self.figure.add_subplot(111)
        canvas = FigureCanvasQTAgg(self.figure)
        self.selector = RectangleSelector(
            self.ax, self._on_select, useblit=True, interactive=True,
            button=[1], minspanx=1e-3, minspany=1.0,
        )
        return canvas

    def _wrap_with_controls(self, canvas: FigureCanvasQTAgg) -> QtWidgets.QWidget:
        self.label_edit = QtWidgets.QLineEdit(placeholderText="label")
        self.confidence = QtWidgets.QDoubleSpinBox(minimum=0.0, maximum=1.0, singleStep=0.1, value=1.0)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Label:"))
        bar.addWidget(self.label_edit, 1)
        bar.addWidget(QtWidgets.QLabel("Conf:"))
        bar.addWidget(self.confidence)
        for text, slot in (("Add layer", self._add_layer),
                           ("Play selection", self._play_selection),
                           ("Export…", self._export)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            bar.addWidget(button)

        self.status = QtWidgets.QLabel("Drag a rectangle on the spectrogram.")
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(canvas, 1)
        layout.addLayout(bar)
        layout.addWidget(self.status)
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        return container

    def _draw_spectrogram(self) -> None:
        self.renderer.draw(self.ax, self.session.coeffs, self.session.sr, self.session.duration_s)
        self.figure.tight_layout()

    def _on_select(self, press, release) -> None:
        self._selection = (press.xdata, release.xdata, press.ydata, release.ydata)
        t0, t1, f0, f1 = self._selection
        self.status.setText(f"Selection: {t0:.3f}–{t1:.3f} s, {f0:.0f}–{f1:.0f} Hz")

    def _add_layer(self) -> None:
        if self._selection is None or not self.label_edit.text():
            self.status.setText("Need a selection and a label.")
            return
        t0, t1, f0, f1 = self._selection
        layer = self.session.add_rectangle(
            t0, t1, f0, f1, self.label_edit.text(), confidence=self.confidence.value()
        )
        self._overlay(layer)
        self.status.setText(f"Added '{layer.label}' ({len(self.session.layers)} layers).")

    def _overlay(self, layer) -> None:
        active_t = np.where(layer.mask.any(axis=0))[0]
        active_f = np.where(layer.mask.any(axis=1))[0]
        if not len(active_t) or not len(active_f):
            return
        sr, tf = self.session.sr, self.session.transform
        x0, x1 = tf.col_to_time(active_t[0], sr), tf.col_to_time(active_t[-1], sr)
        y0, y1 = tf.row_to_display_y(active_f[0], sr), tf.row_to_display_y(active_f[-1], sr)
        self.ax.add_patch(plt_rect((x0, y0), x1 - x0, y1 - y0, layer.color))
        self.figure.canvas.draw_idle()

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


def run(path: str) -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = AnnotatorWindow(SpectrogramSession.load(path, STFT))
    window.resize(1100, 650)
    window.show()
    return app.exec()
