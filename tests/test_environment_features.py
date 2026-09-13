import networkx as nx

from saferoute.features.environment_features import compute_lighting_by_cell


def _graph_with_no_lit_tags():
    """Regression case: a graph where NOT ONE edge carries an OSM `lit` tag --
    a real, observed situation for a small bbox extract, not a hypothetical."""
    g = nx.MultiDiGraph()
    g.add_node(0, x=-87.63, y=41.88)
    g.add_node(1, x=-87.629, y=41.881)
    from shapely.geometry import LineString

    g.add_edge(0, 1, key=0, geometry=LineString([(-87.63, 41.88), (-87.629, 41.881)]), length=100)
    g.graph["crs"] = "EPSG:4326"
    return g


def _graph_with_mixed_lit_tags():
    # The two edges are placed ~5km apart (0.05 deg) to guarantee they fall
    # into DIFFERENT H3 res-9 cells (~174m edge length) -- close-together
    # points would land in the same cell and average out to 0.5, making the
    # test unable to distinguish "correctly computed 0.5" from "fallback 0.5".
    g = nx.MultiDiGraph()
    g.add_node(0, x=-87.63, y=41.88)
    g.add_node(1, x=-87.6295, y=41.8805)
    g.add_node(2, x=-87.58, y=41.93)
    g.add_node(3, x=-87.5795, y=41.9305)
    from shapely.geometry import LineString

    g.add_edge(0, 1, key=0, geometry=LineString([(-87.63, 41.88), (-87.6295, 41.8805)]), length=100, lit="yes")
    g.add_edge(2, 3, key=0, geometry=LineString([(-87.58, 41.93), (-87.5795, 41.9305)]), length=100, lit="no")
    g.graph["crs"] = "EPSG:4326"
    return g


def test_lighting_falls_back_to_default_when_no_edge_has_lit_tag():
    """Regression test: this used to return NaN for every cell (grouped.mean()
    of an all-NaN group is NaN, and the old `len(grouped) else 0.5` guard
    didn't catch that), which propagated into every downstream feature row
    and silently emptied the training set after dropna()."""
    graph = _graph_with_no_lit_tags()
    result = compute_lighting_by_cell(graph, cells=["some_cell_not_in_graph"], resolution=9)
    assert result.notna().all()
    assert (result == 0.5).all()


def test_lighting_computed_correctly_when_tags_present():
    graph = _graph_with_mixed_lit_tags()
    import saferoute.features.h3_grid as h3_grid

    # Get the actual cells the graph's edges fall into so we can check their values.
    from saferoute.features.geo_utils import centroid_latlon
    import osmnx as ox

    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index()
    lat, lon = centroid_latlon(edges)
    cells = [h3_grid.assign_h3_cells(__import__("pandas").DataFrame({"latitude": [la], "longitude": [lo]}), 9)["h3_cell"].iloc[0] for la, lo in zip(lat, lon)]

    result = compute_lighting_by_cell(graph, cells=cells, resolution=9)
    assert result.notna().all()
    # one lit=yes edge (1.0) and one lit=no edge (0.0) -- neither should be the NaN-fallback 0.5
    assert set(result.round(4).tolist()) <= {0.0, 1.0}
