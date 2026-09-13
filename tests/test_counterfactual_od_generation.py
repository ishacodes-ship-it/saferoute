import networkx as nx
import pandas as pd

from saferoute.benchmark.counterfactual import generate_od_pairs_near_incidents
from saferoute.features.h3_grid import cell_centroid, enumerate_grid_cells


def _small_grid_graph():
    """A connected line graph whose nodes sit at real coordinates covering
    a couple of H3 cells, so nearest_nodes/random-walk logic has somewhere
    to go."""
    g = nx.MultiDiGraph()
    lats = [41.880 + 0.001 * i for i in range(10)]
    lon = -87.630
    for i, lat in enumerate(lats):
        g.add_node(i, x=lon, y=lat)
    for i in range(9):
        g.add_edge(i, i + 1, key=0, length=100)
        g.add_edge(i + 1, i, key=0, length=100)
    g.graph["crs"] = "EPSG:4326"
    return g


def test_generate_od_pairs_uses_h3_cell_and_time_bin_not_lat_lon():
    """Regression test: this function used to reference row['latitude'],
    row['longitude'], and row['date'] -- columns that do not exist on the
    dense (h3_cell, time_bin) feature table rows it's actually called with
    in scripts/run_benchmark.py, silently producing 0 OD pairs every time."""
    cells = enumerate_grid_cells([41.87, -87.64, 41.90, -87.62], resolution=9)
    incidents_df = pd.DataFrame({
        "h3_cell": cells[:5],
        "time_bin": pd.date_range("2023-06-01", periods=5, freq="1h"),
    })
    graph = _small_grid_graph()

    pairs = generate_od_pairs_near_incidents(incidents_df, graph, n_pairs=5, walk_hops=3, seed=0, min_od_distance_m=150.0)
    assert len(pairs) > 0
    for p in pairs:
        assert p.incident_cell in cells
        assert isinstance(p.incident_time, pd.Timestamp)
        assert p.source in graph.nodes
        assert p.target in graph.nodes


def test_generate_od_pairs_respects_minimum_distance():
    """Regression test for the pilot-run finding: a fixed-hop-count walk on a
    dense street grid (short edges) produced OD pairs only 30-110m apart,
    making avoidance geometrically impossible for every method. The walk
    must now continue until each endpoint is actually >= min_od_distance_m
    from the incident location."""
    import math

    cells = enumerate_grid_cells([41.87, -87.64, 41.90, -87.62], resolution=9)
    incidents_df = pd.DataFrame({
        "h3_cell": cells[:5],
        "time_bin": pd.date_range("2023-06-01", periods=5, freq="1h"),
    })
    graph = _small_grid_graph()

    pairs = generate_od_pairs_near_incidents(incidents_df, graph, n_pairs=5, walk_hops=3, seed=0, min_od_distance_m=300.0)
    for p in pairs:
        lat, lon = cell_centroid(p.incident_cell)
        for node in (p.source, p.target):
            node_lat, node_lon = graph.nodes[node]["y"], graph.nodes[node]["x"]
            dlat, dlon = math.radians(node_lat - lat), math.radians(node_lon - lon)
            a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat)) * math.cos(math.radians(node_lat)) * math.sin(dlon / 2) ** 2
            dist = 2 * 6_371_000 * math.asin(math.sqrt(a))
            assert dist >= 300.0 - 1.0  # small float tolerance


def test_generate_od_pairs_empty_incidents_returns_empty():
    graph = _small_grid_graph()
    incidents_df = pd.DataFrame({"h3_cell": [], "time_bin": []})
    pairs = generate_od_pairs_near_incidents(incidents_df, graph, n_pairs=5, walk_hops=3, seed=0)
    assert pairs == []
