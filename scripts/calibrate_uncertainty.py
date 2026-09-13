#!/usr/bin/env python
"""Fit the MAPIE conformal wrapper around the best model and report empirical coverage."""
import argparse

from saferoute.models.uncertainty import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", default=None)
    args = parser.parse_args()
    run(args.config, args.override)
