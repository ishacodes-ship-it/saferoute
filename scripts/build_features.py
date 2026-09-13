#!/usr/bin/env python
"""M0 step 2: build the joined feature table and train/val/test/spatial_holdout splits."""
import argparse

from saferoute.features.build_features import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", default=None)
    args = parser.parse_args()
    run(args.config, args.override)
