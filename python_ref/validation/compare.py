"""Numerical comparison primitives for the DSP Validation Protocol.

The core rule: ``max(abs(reference - other)) < TOLERANCE``. The tolerance is the
floating-point precision threshold and is never relaxed to make a test pass — a
larger deviation always indicates a real bug. When a comparison fails, the result
object carries enough information (where the worst divergence is) to start
debugging, and the offending arrays can be dumped to disk for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..params import TOLERANCE


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of comparing a reference array against another implementation."""

    passed: bool
    max_abs_diff: float
    tolerance: float
    #: Flat index of the largest deviation (``None`` if shapes mismatch).
    worst_index: tuple[int, ...] | None
    shape_reference: tuple[int, ...]
    shape_other: tuple[int, ...]
    message: str

    def __bool__(self) -> bool:  # allow `if result:`
        return self.passed


def max_abs_diff(reference: np.ndarray, other: np.ndarray) -> float:
    """Maximum elementwise absolute difference between two complex/real arrays.

    Complex inputs are compared by complex magnitude of the difference, which
    captures both magnitude and phase divergence in one number.
    """
    a = np.asarray(reference)
    b = np.asarray(other)
    if a.shape != b.shape:
        raise ValueError(
            f"shape mismatch: reference {a.shape} vs other {b.shape}"
        )
    return float(np.max(np.abs(a - b)))


def compare(
    reference: np.ndarray,
    other: np.ndarray,
    tolerance: float = TOLERANCE,
    label: str = "output",
) -> ComparisonResult:
    """Compare two arrays and return a structured result (does not raise)."""
    a = np.asarray(reference)
    b = np.asarray(other)

    if a.shape != b.shape:
        return ComparisonResult(
            passed=False,
            max_abs_diff=float("inf"),
            tolerance=tolerance,
            worst_index=None,
            shape_reference=a.shape,
            shape_other=b.shape,
            message=(
                f"{label}: shape mismatch — reference {a.shape} vs other {b.shape}"
            ),
        )

    diff = np.abs(a - b)
    mad = float(np.max(diff)) if diff.size else 0.0
    worst = (
        tuple(int(i) for i in np.unravel_index(int(np.argmax(diff)), diff.shape))
        if diff.size
        else None
    )
    passed = mad < tolerance
    status = "OK" if passed else "FAIL"
    return ComparisonResult(
        passed=passed,
        max_abs_diff=mad,
        tolerance=tolerance,
        worst_index=worst,
        shape_reference=a.shape,
        shape_other=b.shape,
        message=(
            f"{label}: {status} — max|diff|={mad:.3e} "
            f"(tol={tolerance:.0e}) worst@{worst}"
        ),
    )


def assert_matches(
    reference: np.ndarray,
    other: np.ndarray,
    tolerance: float = TOLERANCE,
    label: str = "output",
    dump_dir: str | Path | None = None,
) -> ComparisonResult:
    """Assert two arrays match within ``tolerance``; raise ``AssertionError`` if not.

    On failure, optionally dump both arrays to ``dump_dir`` as ``.npy`` files for
    offline inspection, then raise with a descriptive message.
    """
    result = compare(reference, other, tolerance=tolerance, label=label)
    if not result.passed:
        if dump_dir is not None:
            d = Path(dump_dir)
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"{label}_reference.npy", np.asarray(reference))
            np.save(d / f"{label}_other.npy", np.asarray(other))
        raise AssertionError(result.message)
    return result
