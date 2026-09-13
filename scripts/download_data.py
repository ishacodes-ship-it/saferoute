#!/usr/bin/env python
"""M0 step 1: download all raw data sources into data/raw/.

Usage: python scripts/download_data.py [--config config/base.yaml] [--override config/pilot.yaml]
"""
import argparse
import logging

from saferoute.data import ingest_census, ingest_crime, ingest_osm, ingest_weather

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", default=None)
    parser.add_argument("--skip", nargs="*", default=[], choices=["crime", "osm", "weather", "census"])
    args = parser.parse_args()

    from saferoute.config import DEFAULT_CONFIG_PATH, load_config

    cfg = load_config(args.config or DEFAULT_CONFIG_PATH, args.override)

    if "crime" not in args.skip:
        df = ingest_crime.fetch_crime_data(cfg)
        out = cfg.path("data_raw") / f"{cfg['city']['name']}_crime_raw.parquet"
        df.to_parquet(out, index=False)
        logging.info("crime -> %s", out)

    if "osm" not in args.skip:
        graph = ingest_osm.fetch_osm_graph(cfg)
        import osmnx as ox
        out = cfg.path("data_raw") / f"{cfg['city']['name']}_walk_network.graphml"
        ox.save_graphml(graph, out)
        logging.info("osm -> %s", out)

    if "weather" not in args.skip:
        df = ingest_weather.fetch_weather_data(cfg)
        out = cfg.path("data_raw") / f"{cfg['city']['name']}_weather_hourly.parquet"
        df.to_parquet(out, index=False)
        logging.info("weather -> %s", out)

    if "census" not in args.skip:
        gdf = ingest_census.fetch_census_data(cfg)
        out = cfg.path("data_raw") / f"{cfg['city']['name']}_census_tracts.geojson"
        gdf.to_file(out, driver="GeoJSON")
        logging.info("census -> %s", out)


if __name__ == "__main__":
    main()
