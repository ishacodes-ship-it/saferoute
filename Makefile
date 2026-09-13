.PHONY: setup test pilot full clean-results

CONFIG ?= config/base.yaml

setup:
	python3.12 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

test:
	pytest -q

# Full paper-scale pipeline (config/base.yaml). Long-running: real Chicago
# crime data 2018-2023 + full-city OSM graph. Not run automatically by CI.
full:
	python scripts/download_data.py --config $(CONFIG)
	python scripts/build_features.py --config $(CONFIG)
	python scripts/train_model.py --config $(CONFIG)
	python scripts/calibrate_uncertainty.py --config $(CONFIG)
	python scripts/run_routing_ablations.py --config $(CONFIG)
	python scripts/run_benchmark.py --config $(CONFIG)
	python scripts/run_explainability_eval.py --config $(CONFIG)
	python scripts/run_fairness_audit.py --config $(CONFIG)
	python scripts/run_feedback_simulation.py --config $(CONFIG)
	python scripts/generate_figures.py --config $(CONFIG)
	python scripts/generate_tables.py --config $(CONFIG)

# Small pilot run for end-to-end validation -- NOT a paper-scale result.
pilot:
	python scripts/download_data.py --override config/pilot.yaml --skip census
	python scripts/build_features.py --override config/pilot.yaml
	python scripts/train_model.py --override config/pilot.yaml
	python scripts/calibrate_uncertainty.py --override config/pilot.yaml
	python scripts/run_routing_ablations.py --override config/pilot.yaml
	python scripts/run_benchmark.py --override config/pilot.yaml --n-incidents 20
	python scripts/run_explainability_eval.py --override config/pilot.yaml
	python scripts/run_feedback_simulation.py --override config/pilot.yaml --n-cells 20 --n-agents 100 --n-steps 20
	python scripts/generate_figures.py --override config/pilot.yaml
	python scripts/generate_tables.py --override config/pilot.yaml

clean-results:
	rm -rf results/tables/* results/figures/* results/logs/* results/models/*
	touch results/tables/.gitkeep results/figures/.gitkeep results/logs/.gitkeep
