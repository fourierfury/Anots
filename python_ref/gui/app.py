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
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from PyQt6 import QtCore, QtWidgets

from ..params import STFT
from ..transforms import TRANSFORMS
from .render import SpectrogramRenderer, energy_bounds, occupied_band
from .session import SpectrogramSession


class AnnotatorWindow(QtWidgets.QMainWindow):
    def __init__(self, session: SpectrogramSession | None = None):
        super().__init__()
        self.session = session
        self.renderer = SpectrogramRenderer(session.transform) if session else None
        self._selection: tuple[float, float, float, float] | None = None
        self._ridge_start: tuple[float, float] | None = None
        self._brushing = False
        self._brush_pts: list[tuple[float, float]] = []
        self._cursor = None
        self._play_dur = 0.0
        self._clock = QtCore.QElapsedTimer()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._advance_cursor)
        title = Path(session.source_file).name if session else "open an audio file"
        self.setWindowTitle(f"Anots — {title}")

        canvas = self._build_canvas()
        self.setCentralWidget(self._wrap_with_controls(canvas))
        if session:
            self._draw_spectrogram()
            self._auto_frequency_view()
        else:
            self._show_placeholder()

    def _show_placeholder(self) -> None:
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Open audio…  to begin", ha="center", va="center",
                     transform=self.ax.transAxes, fontsize=14, color="0.4")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.figure.canvas.draw_idle()

    def _require_session(self) -> bool:
        if self.session is None:
            self.status.setText("Open an audio file first (Open audio…).")
            return False
        return True

    # --- construction --------------------------------------------------------

    def _build_canvas(self) -> FigureCanvasQTAgg:
        self.figure = Figure(figsize=(10, 5))
        self.ax = self.figure.add_subplot(111)
        canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(canvas, self)  # pan / zoom-rect / home, all views
        self.selector = RectangleSelector(
            self.ax, self._on_select, useblit=True, interactive=True,
            button=[1], minspanx=1e-3, minspany=1.0,
        )
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("button_release_event", self._on_release)
        return canvas

    def _nav_active(self) -> bool:
        """True while the toolbar is in pan/zoom mode — selection tools must stand down."""
        return bool(getattr(self.toolbar, "mode", ""))

    def _wrap_with_controls(self, canvas: FigureCanvasQTAgg) -> QtWidgets.QWidget:
        self.view_box = QtWidgets.QComboBox()
        self.view_box.addItems(sorted(TRANSFORMS))
        self.view_box.setCurrentText(self.session.transform.name if self.session else "stft")
        self.view_box.currentTextChanged.connect(self._on_view_changed)

        self.tool_box = QtWidgets.QComboBox()
        self.tool_box.addItems(["rectangle", "lasso", "ridge", "brush"])
        self.tool_box.currentTextChanged.connect(self._on_tool_changed)

        self.label_edit = QtWidgets.QLineEdit(placeholderText="label (optional)")
        self.confidence = QtWidgets.QDoubleSpinBox(minimum=0.0, maximum=1.0, singleStep=0.1, value=1.0)
        self.radius = QtWidgets.QSpinBox(minimum=1, maximum=64, value=8)
        self.threshold_on = QtWidgets.QCheckBox("dB ≥")
        self.threshold_db = QtWidgets.QDoubleSpinBox(minimum=-120.0, maximum=0.0, value=-40.0)

        open_btn = QtWidgets.QPushButton("Open audio…")
        open_btn.clicked.connect(self._open_audio)
        fit_btn = QtWidgets.QPushButton("Fit to content")
        fit_btn.clicked.connect(self._fit_to_content)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(open_btn)
        top.addWidget(QtWidgets.QLabel("View:")); top.addWidget(self.view_box)
        top.addWidget(fit_btn)
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

        canvas_col = QtWidgets.QVBoxLayout()
        canvas_col.addWidget(self.toolbar)       # zoom/pan controls sit above the plot
        canvas_col.addWidget(canvas, 1)
        canvas_col_w = QtWidgets.QWidget(); canvas_col_w.setLayout(canvas_col)
        middle = QtWidgets.QHBoxLayout()
        middle.addWidget(canvas_col_w, 1)
        middle.addWidget(panel_w)

        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(QtWidgets.QLabel("Brush r:")); bottom.addWidget(self.radius)
        bottom.addWidget(self.threshold_on); bottom.addWidget(self.threshold_db)
        for text, slot in (("Add layer", self._add_layer),
                           ("▶ Play all", self._play_all),
                           ("▶ Play selection", self._play_selection),
                           ("■ Stop", self._stop_playback),
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
        self._refresh_layers(preserve_view=False)
        self._auto_frequency_view()
        self.status.setText(f"Loaded {Path(path).name} — frequency auto-fit to its 99% band (Home = full range).")

    def _on_view_changed(self, name: str) -> None:
        if self.session is None:
            return
        s = self.session
        self.session = SpectrogramSession(s.y, s.sr, source_file=s.source_file,
                                          transform=TRANSFORMS[name]())
        self.renderer = SpectrogramRenderer(self.session.transform)
        self._selection = self._ridge_start = None
        self._refresh_layers(preserve_view=False)
        self._auto_frequency_view()
        self.status.setText(f"View: {name} (layers cleared; frequency auto-fit to occupied band).")

    def _on_tool_changed(self, name: str) -> None:
        self.selector.set_active(name == "rectangle")
        self._ridge_start = None
        self._brushing = False
        hint = {"rectangle": "drag a box, then Add layer.",
                "lasso": "drag to outline any shape; release to fill it.",
                "ridge": "click a start point, then an end point.",
                "brush": "drag to paint; tick 'dB ≥' to grab only loud bins."}[name]
        self.status.setText(f"{name.capitalize()}: {hint}")

    # --- matplotlib interactions --------------------------------------------

    def _on_select(self, press, release) -> None:  # rectangle (RectangleSelector)
        self._selection = (press.xdata, release.xdata, press.ydata, release.ydata)
        t0, t1, y0, y1 = self._selection
        self.status.setText(f"Selection: {t0:.3f}–{t1:.3f} s, y {y0:.1f}–{y1:.1f}")

    def _on_press(self, event) -> None:
        if self.session is None or self._nav_active() or event.inaxes is not self.ax or event.xdata is None:
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
        elif self.tool in ("brush", "lasso"):
            self._brushing = True
            self._brush_pts = [(event.xdata, event.ydata)]

    def _on_motion(self, event) -> None:
        if self._nav_active():
            return
        if self._brushing and event.inaxes is self.ax and event.xdata is not None:
            self._brush_pts.append((event.xdata, event.ydata))

    def _on_release(self, event) -> None:
        if self.tool not in ("brush", "lasso") or not self._brushing:
            return
        self._brushing = False
        if event.xdata is not None:
            self._brush_pts.append((event.xdata, event.ydata))
        pts = self._brush_pts
        if self.tool == "lasso":
            self._commit("lasso", lambda lbl, c: self.session.add_lasso(pts, lbl, confidence=c))
        else:
            thr = self.threshold_db.value() if self.threshold_on.isChecked() else None
            self._commit("brush", lambda lbl, c: self.session.add_brush(
                pts, lbl, radius=self.radius.value(), threshold_db=thr, confidence=c))

    # --- committing layers ---------------------------------------------------

    def _add_layer(self) -> None:  # rectangle commit (drag + button)
        if not self._require_session():
            return
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
        if not self._require_session():
            return
        row = self.layer_list.currentRow()
        if row < 0:
            row = len(self.session.layers) - 1  # default to the most recent
        if 0 <= row < len(self.session.layers):
            removed = self.session.layers.pop(row)
            self._refresh_layers()
            self.status.setText(f"Removed '{removed.label}' ({len(self.session.layers)} left).")
        else:
            self.status.setText("No layer to delete.")

    def _refresh_layers(self, preserve_view: bool = True) -> None:
        """Redraw the spectrogram with every layer's TRUE mask overlaid, and sync the list.

        ``preserve_view`` keeps the current zoom/pan across a redraw (adding/deleting a
        layer shouldn't reset your zoom); a view switch or new file passes ``False`` to
        reset to the full extent.
        """
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()
        self._cursor = None  # the playhead line is discarded by ax.clear(); recreate on next play
        self._draw_spectrogram()
        self._draw_mask_overlay()
        if preserve_view:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        self.figure.canvas.draw_idle()
        self.layer_list.clear()
        for i, layer in enumerate(self.session.layers):
            self.layer_list.addItem(f"{i}: {layer.label} [{layer.tool_used.value}]")

    def _auto_frequency_view(self) -> None:
        """Auto-set the frequency axis to the audio's occupied band on load (99% energy).

        This is the "detect the frequency range automatically" behaviour: a low-frequency
        clip opens focused on its band instead of wasting the axis up to Nyquist. Time axis
        stays full; the toolbar's Home shows the full Nyquist range; the user can zoom freely.
        Works in any view (the band is found in coefficient rows, mapped to display units).
        """
        if self.session is None:
            return
        band = occupied_band(np.abs(self.session.coeffs))
        if band is None:
            return
        lo, hi = band
        sr, tf = self.session.sr, self.session.transform
        _, _, y0d, y1d = self.renderer.extent(self.session.coeffs, sr, self.session.duration_s)
        ys = sorted((tf.row_to_display_y(lo, sr), tf.row_to_display_y(hi, sr)))
        mar = 0.08 * (ys[1] - ys[0] + 1.0)
        self.ax.set_ylim(max(y0d, ys[0] - mar), min(y1d, ys[1] + mar))
        self.figure.canvas.draw_idle()

    def _fit_to_content(self) -> None:
        """Zoom both axes to where the energy actually lives (design-grade 'fit to content').

        Works in every view: energy bounds are found in coefficient space, then mapped to
        display units through the transform, so a low-frequency clip stops wasting the
        whole high-frequency axis.
        """
        if not self._require_session():
            return
        bounds = energy_bounds(self.session.transform.display_db(self.session.coeffs))
        if bounds is None:
            self.status.setText("No signal to fit to.")
            return
        r_lo, r_hi, c_lo, c_hi = bounds
        sr, tf = self.session.sr, self.session.transform
        x0d, x1d, y0d, y1d = self.renderer.extent(self.session.coeffs, sr, self.session.duration_s)
        ys = sorted((tf.row_to_display_y(r_lo, sr), tf.row_to_display_y(r_hi, sr)))
        xs = sorted((tf.col_to_time(c_lo, sr), tf.col_to_time(c_hi + 1, sr)))
        ymar, xmar = 0.05 * (ys[1] - ys[0] + 1.0), 0.02 * (xs[1] - xs[0] + 1e-9)
        self.ax.set_ylim(max(y0d, ys[0] - ymar), min(y1d, ys[1] + ymar))
        self.ax.set_xlim(max(x0d, xs[0] - xmar), min(x1d, xs[1] + xmar))
        self.figure.canvas.draw_idle()
        self.status.setText("Fit to content. (Use the toolbar's Home to reset full view.)")

    def _draw_mask_overlay(self) -> None:
        """Composite every layer's feathered mask into one translucent RGBA image.

        Drawing the mask itself (not a bounding box) is what makes each tool's real shape
        visible — brush dabs, the ridge's curved tube, lasso polygons — in the layer colour.
        """
        if not self.session.layers:
            return
        from matplotlib.colors import to_rgb

        h, w = self.session.coeffs.shape
        rgba = np.zeros((h, w, 4), dtype=np.float64)
        for layer in self.session.layers:
            a = 0.55 * np.clip(layer.mask, 0.0, 1.0)        # feathered edges -> soft alpha
            col = np.array(to_rgb(layer.color))
            base_a = rgba[..., 3]
            new_a = a + base_a * (1.0 - a)                  # over-compositing
            denom = np.where(new_a > 0, new_a, 1.0)
            for k in range(3):
                rgba[..., k] = (col[k] * a + rgba[..., k] * base_a * (1.0 - a)) / denom
            rgba[..., 3] = new_a
        extent = self.renderer.extent(self.session.coeffs, self.session.sr, self.session.duration_s)
        self.ax.imshow(rgba, extent=extent, origin="lower", aspect="auto",
                       interpolation="nearest", zorder=5)

    # --- playback + moving playhead -----------------------------------------

    def _play_all(self) -> None:
        if not self._require_session():
            return
        self._play(np.asarray(self.session.y, dtype=np.float32), self.session.duration_s, "full file")

    def _play_selection(self) -> None:
        if not self._require_session():
            return
        if not self.session.layers:
            self.status.setText("No layer to play — draw one first.")
            return
        clip = self.session.reconstruct_layer(self.session.layers[-1]).astype(np.float32)
        self._play(clip, len(clip) / self.session.sr, f"'{self.session.layers[-1].label}'")

    def _play(self, audio: np.ndarray, duration: float, what: str) -> None:
        try:
            import sounddevice as sd
            sd.stop()
            sd.play(audio, self.session.sr)
        except Exception as exc:  # no output device, backend missing, etc. — report, don't crash
            self.status.setText(f"Cannot play audio ({type(exc).__name__}: {exc}).")
            return
        self.status.setText(f"Playing {what} ({duration:.2f}s)…")
        self._start_playhead(duration)

    def _stop_playback(self) -> None:
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self._timer.stop()
        if self._cursor is not None:
            self._cursor.set_visible(False)
            self.figure.canvas.draw_idle()
        self.status.setText("Stopped.")

    def _start_playhead(self, duration: float) -> None:
        self._play_dur = duration
        self._clock.restart()
        self._timer.start(40)          # ~25 fps cursor updates
        self._advance_cursor()

    def _advance_cursor(self) -> None:
        t = self._clock.elapsed() / 1000.0
        if t >= self._play_dur:
            self._stop_playback()
            return
        if self._cursor is None:
            self._cursor = self.ax.axvline(t, color="red", linewidth=1.2, zorder=10)
        else:
            self._cursor.set_xdata([t, t])
            self._cursor.set_visible(True)
        self.figure.canvas.draw_idle()

    def _export(self) -> None:
        if not self._require_session():
            return
        out = QtWidgets.QFileDialog.getExistingDirectory(self, "Export to")
        if out:
            anns = self.session.export(out)
            self.status.setText(f"Exported {len(anns)} clips to {out}")


def build_session(path: str, view: str = "stft") -> SpectrogramSession:
    """Load ``path`` into a session backed by the named transform view (design §4.3)."""
    if view not in TRANSFORMS:
        raise SystemExit(f"unknown view {view!r}; choose from {', '.join(TRANSFORMS)}")
    transform = TRANSFORMS[view]()  # each view carries its own locked default params
    return SpectrogramSession.load(path, STFT, transform=transform)


def run(path: str | None = None, view: str = "stft") -> int:
    app = QtWidgets.QApplication(sys.argv)
    session = build_session(path, view) if path else None  # no path -> open empty, use "Open audio…"
    window = AnnotatorWindow(session)
    window.resize(1100, 700)
    window.show()
    return app.exec()
