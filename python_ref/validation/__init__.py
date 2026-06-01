"""Validation harness: native-core output must match the reference within TOLERANCE."""

from .compare import ComparisonResult, assert_matches, compare, max_abs_diff

__all__ = ["max_abs_diff", "compare", "assert_matches", "ComparisonResult"]
