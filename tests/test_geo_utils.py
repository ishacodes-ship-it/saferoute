import geopandas as gpd
from shapely.geometry import LineString, Point

from saferoute.features.geo_utils import centroid_latlon


def test_centroid_of_point_is_itself():
    gdf = gpd.GeoDataFrame(geometry=[Point(-87.6298, 41.8781)], crs="EPSG:4326")
    lat, lon = centroid_latlon(gdf)
    assert abs(lat[0] - 41.8781) < 1e-4
    assert abs(lon[0] - (-87.6298)) < 1e-4


def test_centroid_of_line_is_midpoint_ish():
    gdf = gpd.GeoDataFrame(geometry=[LineString([(-87.63, 41.88), (-87.62, 41.88)])], crs="EPSG:4326")
    lat, lon = centroid_latlon(gdf)
    assert abs(lat[0] - 41.88) < 0.01
    assert -87.63 < lon[0] < -87.62


def test_empty_geodataframe_returns_empty_arrays():
    gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    lat, lon = centroid_latlon(gdf)
    assert len(lat) == 0
    assert len(lon) == 0
