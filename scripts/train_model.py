#!/usr/bin/env python
"""M1: train and compare candidate risk models, save the best artifact + feature importance."""
import argparse

from saferoute.models.train_risk_model import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", default=None)
    args = parser.parse_args()
    run(args.config, args.override)
