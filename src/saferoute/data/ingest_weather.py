"""Download historical hourly weather for the city from Open-Meteo's free archive API.

Open-Meteo's archive endpoint requires no API key and has no documented rate
limit for reasonable non-commercial use, which is why it replaces the PRD's
suggested OpenWeatherMap (free tier capped at 1,000 calls/day — too tight for
bulk historical backfill of a multi-year training window). Same purpose:
a weather context feature for the risk model.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from saferoute.config import Config, load_config

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather_data(cfg: Config) -> pd.DataFrame:
    """Fetch hourly weather at the city's bbox centroid for the configured date range."""
    south, west, north, east = cfg["city"]["bbox"]
    lat = (south + north) / 2
    lon = (west + east) / 2
    src = cfg["data_sources"]["weather"]

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": cfg["data_sources"]["crime"]["date_start"],
        "end_date": cfg["data_sources"]["crime"]["date_end"],
        "hourly": ",".join(src["variables"]),
        "timezone": cfg["city"]["timezone"],
    }
    logger.info("Fetching weather archive for lat=%.4f lon=%.4f", lat, lon)
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()["hourly"]

    df = pd.DataFrame(payload)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "timestamp"})
    logger.info("Fetched %d hourly weather records", len(df))
    return df


def main(config_path: str | None = None) -> Path:
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(config_path) if config_path else load_config()
    df = fetch_weather_data(cfg)
    out_path = cfg.path("data_raw") / f"{cfg['city']['name']}_weather_hourly.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(df))
    return out_path


if __name__ == "__main__":
    main()
