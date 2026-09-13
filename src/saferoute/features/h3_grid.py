"""H3 hexagonal grid assignment and grid-cell enumeration.

Cell size is a configured resolution (config/base.yaml: spatial.h3_resolution),
default 9 — average edge length ~174m, chosen as the closest standard H3
resolution to the PRD's "~100m" target (H3 does not offer an exact 100m
resolution; res 10 averages ~66m edge, res 9 ~174m — res 9 is used because
Chicago's crime density at hourly bins is too sparse per-cell at res 10 to
support reliable label estimation, see README §Spatial Resolution).
"""
from __future__ import annotations

import h3
import pandas as pd


def assign_h3_cells(df: pd.DataFrame, resolution: int, lat_col: str = "latitude", lon_col: str = "longitude") -> pd.DataFrame:
    """Assign an H3 cell id to each row given lat/lon columns.

    Rows with missing or out-of-range coordinates are dropped.
    """
    df = df.copy()
    valid = df[lat_col].between(-90, 90) & df[lon_col].between(-180, 180)
    df = df[valid].copy()
    df["h3_cell"] = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(df[lat_col], df[lon_col])
    ]
    return df


def enumerate_grid_cells(bbox: list[float], resolution: int) -> list[str]:
    """Enumerate all H3 cells covering a (south, west, north, east) bounding box."""
    south, west, north, east = bbox
    polygon = h3.LatLngPoly(
        [(south, west), (south, east), (north, east), (north, west)]
    )
    return list(h3.h3shape_to_cells(polygon, resolution))


def cell_centroid(cell: str) -> tuple[float, float]:
    """Return (lat, lon) centroid of an H3 cell."""
    return h3.cell_to_latlng(cell)
