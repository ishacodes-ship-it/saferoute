import pandas as pd

from saferoute.features.h3_grid import assign_h3_cells, cell_centroid, enumerate_grid_cells


def test_assign_h3_cells_basic():
    df = pd.DataFrame({"latitude": [41.8781, 41.8800], "longitude": [-87.6298, -87.6250]})
    out = assign_h3_cells(df, resolution=9)
    assert "h3_cell" in out.columns
    assert out["h3_cell"].notna().all()
    assert len(out) == 2


def test_assign_h3_cells_drops_invalid_coords():
    df = pd.DataFrame({"latitude": [41.8781, 999.0], "longitude": [-87.6298, -87.6250]})
    out = assign_h3_cells(df, resolution=9)
    assert len(out) == 1


def test_enumerate_grid_cells_nonempty_and_covers_bbox():
    bbox = [41.87, -87.63, 41.89, -87.61]
    cells = enumerate_grid_cells(bbox, resolution=9)
    assert len(cells) > 0
    for cell in cells:
        lat, lon = cell_centroid(cell)
        assert 41.85 <= lat <= 41.91
        assert -87.65 <= lon <= -87.59


def test_cell_centroid_roundtrip():
    df = pd.DataFrame({"latitude": [41.8781], "longitude": [-87.6298]})
    out = assign_h3_cells(df, resolution=9)
    cell = out["h3_cell"].iloc[0]
    lat, lon = cell_centroid(cell)
    assert abs(lat - 41.8781) < 0.01
    assert abs(lon - (-87.6298)) < 0.01
