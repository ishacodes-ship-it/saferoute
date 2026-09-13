"""Shared geometry helpers.

Centroids must be computed in a projected (metric) CRS, not directly on
geographic (lat/lon degree) geometries -- geopandas warns about this for a
reason: degree-based centroid math distorts with latitude and geometry
extent. For the small road-segment and POI-point geometries in this
project the numeric error is negligible, but "negligible for our case" is
not the same as "correct" -- this project targets a research paper, so the
projection step is done properly rather than suppressing the warning.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np


def centroid_latlon(gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) arrays for each geometry's centroid, computed in a
    local UTM projection and reprojected back to EPSG:4326.
    """
    if len(gdf) == 0:
        return np.array([]), np.array([])

    projected = gdf.to_crs(gdf.estimate_utm_crs())
    centroids_projected = projected.geometry.centroid
    centroids_wgs84 = gpd.GeoSeries(centroids_projected, crs=projected.crs).to_crs("EPSG:4326")
    return centroids_wgs84.y.to_numpy(), centroids_wgs84.x.to_numpy()
