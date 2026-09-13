"""Download Chicago crime incidents from the City of Chicago Socrata portal.

Source: data.cityofchicago.org, dataset ijzp-q8t2 ("Crimes - 2001 to Present").
Free, public, no API key required (an app token raises the rate limit but is
optional — see SODAPY_APP_TOKEN in config/base.yaml).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from sodapy import Socrata

from saferoute.config import Config, load_config

logger = logging.getLogger(__name__)

# Columns we actually use downstream — pulling only these keeps payload size
# and PII exposure minimal (data minimization, PRD §6).
SELECT_FIELDS = [
    "id",
    "date",
    "primary_type",
    "description",
    "location_description",
    "arrest",
    "domestic",
    "latitude",
    "longitude",
]

PAGE_SIZE = 50_000


def fetch_crime_data(cfg: Config) -> pd.DataFrame:
    """Fetch all crime records in the configured date range and bounding box.

    Paginates through the Socrata API (`PAGE_SIZE` rows per request) until
    exhausted. Rows missing lat/long are dropped — they cannot be assigned
    an H3 cell.
    """
    src = cfg["data_sources"]["crime"]
    token = os.environ.get(src["app_token_env"])
    client = Socrata(src["domain"], token, timeout=60)

    where_clause = (
        f"date >= '{src['date_start']}T00:00:00' "
        f"AND date <= '{src['date_end']}T23:59:59' "
        f"AND latitude IS NOT NULL AND longitude IS NOT NULL"
    )

    frames = []
    offset = 0
    while True:
        logger.info("Fetching crime records offset=%d", offset)
        results = client.get(
            src["dataset_id"],
            select=",".join(SELECT_FIELDS),
            where=where_clause,
            order="date",
            limit=PAGE_SIZE,
            offset=offset,
        )
        if not results:
            break
        frames.append(pd.DataFrame.from_records(results))
        offset += PAGE_SIZE
        if len(results) < PAGE_SIZE:
            break

    if not frames:
        raise RuntimeError(
            "No crime records returned — check date_start/date_end and dataset_id in config/base.yaml"
        )

    df = pd.concat(frames, ignore_index=True)
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    df["date"] = pd.to_datetime(df["date"])

    south, west, north, east = cfg["city"]["bbox"]
    in_bbox = (
        (df["latitude"] >= south)
        & (df["latitude"] <= north)
        & (df["longitude"] >= west)
        & (df["longitude"] <= east)
    )
    df = df[in_bbox].reset_index(drop=True)
    logger.info("Fetched %d crime records within bbox and date range", len(df))
    return df


def main(config_path: str | None = None) -> Path:
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(config_path) if config_path else load_config()
    df = fetch_crime_data(cfg)
    out_path = cfg.path("data_raw") / f"{cfg['city']['name']}_crime_raw.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(df))
    return out_path


if __name__ == "__main__":
    main()
