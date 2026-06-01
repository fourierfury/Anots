"""Validation harness.

Implements the DSP Validation Protocol: every output from an alternative
implementation (the future native core) must match the Python reference within
``params.TOLERANCE``. See :func:`max_abs_diff` and :func:`assert_matches`.
"""

from .compare import ComparisonResult, assert_matches, max_abs_diff

__all__ = ["max_abs_diff", "assert_matches", "ComparisonResult"]
