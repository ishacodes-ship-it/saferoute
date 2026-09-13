"""Download tract-level ACS demographics for the city's county from the Census API.

Free, public, no key required for the ACS 5-year endpoint at low request
volume (a key raises the rate limit — set CENSUS_API_KEY if needed). Used
only by the Phase 2 fairness audit (src/saferoute/fairness); not part of the
risk model's features.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from saferoute.config import Config, load_config

logger = logging.getLogger(__name__)

ACS_URL_TEMPLATE = "https://api.census.gov/data/{year}/acs/acs5"

# FIPS state=17 (Illinois), county=031 (Cook County, contains the city of Chicago).
# Documented here rather than in config/base.yaml because it's specific to the
# Chicago MVP city choice; a different city config needs its own FIPS codes.
STATE_FIPS = "17"
COUNTY_FIPS = "031"

# TIGER/Line tract boundary shapefile (cartographic, generalized) for the same year.
TIGER_URL_TEMPLATE = (
    "https://www2.census.gov/geo/tiger/GENZ{year}/shp/cb_{year}_{state}_tract_500k.zip"
)


def fetch_census_data(cfg: Config) -> gpd.GeoDataFrame:
    """Fetch ACS variables per tract and join to tract boundary geometry."""
    src = cfg["data_sources"]["census"]
    year = src["year"]
    api_key = os.environ.get("CENSUS_API_KEY", "")

    params = {
        "get": ",".join(["NAME"] + src["variables"]),
        "for": "tract:*",
        "in": f"state:{STATE_FIPS} county:{COUNTY_FIPS}",
    }
    if api_key:
        params["key"] = api_key

    logger.info("Fetching ACS %d tract data for state=%s county=%s", year, STATE_FIPS, COUNTY_FIPS)
    resp = requests.get(ACS_URL_TEMPLATE.format(year=year), params=params, timeout=60)
    resp.raise_for_status()
    rows = resp.json()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    for var in src["variables"]:
        df[var] = pd.to_numeric(df[var], errors="coerce")
    df["GEOID"] = df["state"] + df["county"] + df["tract"]

    tiger_url = TIGER_URL_TEMPLATE.format(year=year, state=STATE_FIPS)
    logger.info("Fetching TIGER tract boundaries from %s", tiger_url)
    tracts = gpd.read_file(tiger_url)
    tracts = tracts.rename(columns={"GEOID": "GEOID"})[["GEOID", "geometry"]]

    merged = tracts.merge(df, on="GEOID", how="inner")
    logger.info("Merged %d census tracts with geometry", len(merged))
    return merged


def main(config_path: str | None = None) -> Path:
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(config_path) if config_path else load_config()
    gdf = fetch_census_data(cfg)
    out_path = cfg.path("data_raw") / f"{cfg['city']['name']}_census_tracts.geojson"
    gdf.to_file(out_path, driver="GeoJSON")
    logger.info("Wrote %s (%d tracts)", out_path, len(gdf))
    return out_path


if __name__ == "__main__":
    main()
