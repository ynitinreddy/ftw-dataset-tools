"""3x3 spatial blocks of FTW chips and stable per-block hash scores."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def _infer_grid_step(values: pd.Series) -> int:
    """Infer the spacing between adjacent grid cells from a series of coordinates.

    FTW grid IDs encode easting/northing as multiples of the grid's km_size
    (e.g. 0, 2, 4, 6... for km_size=2), not as sequential integers. Block grouping
    must divide by this step size rather than by 1, or blocks end up lopsided.

    Coordinates are 0-aligned multiples of km_size, so the GCD of the gaps
    recovers the step even when no two adjacent cells are populated (sparse
    coverage such as 0, 6, 10 still yields 2), where the smallest gap would
    overestimate it.

    Returns 0 when the axis has fewer than two distinct values (e.g. a single
    column or row of chips), meaning the spacing cannot be observed.
    """
    unique_sorted = np.sort(values.unique())
    if len(unique_sorted) < 2:
        return 0
    return int(np.gcd.reduce(np.diff(unique_sorted)))


def mgrs_squares(chip_ids: pd.Series) -> pd.Series:
    """The MGRS 100 km square of each chip, e.g. ftw-36NXF6658 -> 36NXF."""
    return chip_ids.astype(str).str[4:-4]


def chip_block_ids(chip_ids: pd.Series, km_size: float | None = None) -> pd.Series:
    """Group chips into 3x3 blocks of grid cells within their MGRS 100 km square.

    IDs follow ftw-<zone><band><grid><EENN>, e.g. ftw-36NXF6658. ``km_size`` sets
    the cell spacing; when None it is inferred from the IDs present, which can
    overestimate it on sparse chip sets.
    """
    chip_ids = chip_ids.astype(str)
    min_length = chip_ids.str.len().min()
    if min_length < 13:
        raise ValueError(
            f"Invalid chip ID format: IDs must be at least 13 characters (e.g., 'ftw-36NXF6658'). "
            f"Found chip ID with length {min_length}"
        )

    if not all(chip_ids.str.startswith("ftw-")):
        invalid_ids = chip_ids[~chip_ids.str.startswith("ftw-")].tolist()
        raise ValueError(
            f"Invalid chip ID format: IDs must start with 'ftw-'. "
            f"Found invalid IDs: {invalid_ids[:5]}"
        )

    try:
        eastings = chip_ids.str[-4:-2].astype(int)
        northings = chip_ids.str[-2:].astype(int)
    except (ValueError, TypeError) as e:
        raise ValueError(
            "Invalid chip ID format: Unable to extract numeric easting/northing from last "
            f"4 characters. Expected format: ftw-<zone><band><grid><EENN>. Error: {e}"
        ) from e

    # np.gcd treats an unobservable axis (0) as neutral; 1 is the last resort.
    grid_step = int(km_size or 0) or (
        int(np.gcd(_infer_grid_step(eastings), _infer_grid_step(northings))) or 1
    )
    block_east = (eastings // grid_step) // 3
    block_north = (northings // grid_step) // 3
    return mgrs_squares(chip_ids) + "_" + block_east.astype(str) + "_" + block_north.astype(str)


def block_scores(block_ids: pd.Series, salt: str) -> pd.Series:
    """A stable score in [0, 1) per block, the same on every machine and run."""

    def score(block_id: str) -> float:
        digest = hashlib.blake2b(f"{salt}:{block_id}".encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") / 2**64

    return block_ids.map({block_id: score(block_id) for block_id in block_ids.unique()})
