"""The ``Transform`` abstraction: one transform domain's analysis, reconstruction,
and coordinate mapping.

This is the seam every view (STFT today; CQT, chroma, scalogram next) plugs into.
It is deliberately **Qt-free and matplotlib-free**: it carries only the math that the
native Rust + C++ engine must reproduce within ``params.TOLERANCE`` (design §17).
All rendering (colormaps, axes, tick labels) lives in the GUI's renderer, never here —
so the ground-truth-bearing core stays on the same side of the boundary as the port.

Each domain declares its reconstruction ``accuracy_class`` honestly ("exact" vs
"approximate", design §11); a reference tool that misreports its fidelity is not a
reference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..annotation.model import ExtractionMode


class Transform(ABC):
    #: short identifier recorded in each annotation's ``view`` field
    name: str = "transform"
    #: "exact" | "approximate" — the reconstruction fidelity class (design §11)
    accuracy_class: str = "exact"
    #: human-readable y-axis meaning (the renderer turns this into a label)
    y_label: str = "Frequency (Hz)"

    @abstractmethod
    def forward(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Analyze a signal into a coefficient matrix, shape ``(n_rows, n_cols)``."""

    @abstractmethod
    def display_db(self, coeffs: np.ndarray) -> np.ndarray:
        """dB-scaled magnitude matrix of ``coeffs`` for display (pure numpy)."""

    @abstractmethod
    def reconstruct(
        self,
        y: np.ndarray,
        coeffs: np.ndarray,
        mask: np.ndarray,
        mode: ExtractionMode | str = ExtractionMode.POSITIVE,
        sr: int | None = None,
        length: int | None = None,
    ) -> np.ndarray:
        """Reconstruct a time-domain clip from a selection ``mask``.

        Both the original signal ``y`` and this domain's ``coeffs`` are provided:
        STFT inverts its own coeffs, while domains whose reconstruction is defined
        on the original signal (CQT-via-STFT here; chroma's comb filter, §11) use
        ``y``.
        """

    # --- coordinate mapping: display units <-> coefficient indices ---

    @abstractmethod
    def time_to_col(self, t: float, sr: int) -> int:
        """Seconds -> column (frame) index."""

    @abstractmethod
    def col_to_time(self, col: int, sr: int) -> float:
        """Column (frame) index -> seconds."""

    @abstractmethod
    def value_to_row(self, hz: float, sr: int) -> int:
        """Frequency in Hz -> row index in this domain's coefficient matrix."""

    @abstractmethod
    def row_to_value(self, row: int, sr: int) -> float:
        """Row index -> the y-axis value (Hz) at that row."""

    @abstractmethod
    def provenance(self) -> dict:
        """Transform parameters recorded in each annotation (design §10.1)."""

    def layer_params(self, mask: np.ndarray, sr: int) -> dict:
        """Per-selection view metadata merged into the annotation's transform_params.

        Default: none. Chroma overrides this to record the selected pitch-class set
        (design §4.3.5), since a chroma selection has no single frequency band.
        """
        return {}

    # --- display hooks (the renderer reads these; defaults preserve STFT behavior) ---

    def y_extent(self, coeffs: np.ndarray, sr: int) -> tuple[float, float]:
        """(low, high) for the displayed y-axis. STFT: a continuous Hz axis."""
        return (self.row_to_value(0, sr), self.row_to_value(coeffs.shape[0] - 1, sr))

    def y_ticks(self, coeffs: np.ndarray, sr: int) -> tuple[list, list] | None:
        """(positions, labels) to override y ticks, or None for default ticks."""
        return None

    def row_to_display_y(self, row: int, sr: int) -> float:
        """A row's position in the displayed y-axis (Hz for STFT; see CQT override)."""
        return self.row_to_value(row, sr)

    def display_y_to_row(self, y_display: float, sr: int) -> int:
        """Inverse of :meth:`row_to_display_y`: a displayed y-value -> coefficient row.

        This is what a GUI selection needs. The default assumes the display y-axis IS the
        value axis (Hz), so it routes through :meth:`value_to_row`. Log-scale views whose
        display axis is a bin/band index (CQT, chroma, scalogram) override this to round the
        index directly — their y-axis is not Hz, so ``value_to_row`` would be wrong.
        """
        return self.value_to_row(y_display, sr)
