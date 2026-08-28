"""Landmark-defined polygon ROI definitions used by the delivery estimator.

The original project contains a larger visualization script with the same
constant. The heart-rate estimator only needs these indices, so the delivery
package keeps this file intentionally small while preserving the exact ROI
definitions used by the source implementation.
"""

from __future__ import annotations


ROI_POLYGON_INDICES = [
    ("1", [78, 55, 56, 73, 57, 52, 0, 1, 2, 3, 4, 5, 6, 80]),
    ("2", [79, 58, 63, 76, 62, 61, 32, 31, 30, 29, 28, 27, 26, 81]),
    ("3", [6, 7, 8, 9, 10, 47, 82, 80]),
    ("4", [26, 25, 24, 23, 22, 51, 83, 81]),
    ("5", [78, 43, 79, 81, 83, 51, 50, 49, 48, 47, 82, 80]),
]
