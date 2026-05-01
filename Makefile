.PHONY: install download lint format test run run-exp2 tune figures mlflow-ui clean

install:
	pip install -e ".[dev]"

download:
	python scripts/download_data.py

lint:
	ruff check src/ scripts/ tests/
	black --check src/ scripts/ tests/

format:
	black src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

test:
	pytest tests/ -v

tune:
	python scripts/tune_hyperparams.py

run:
	python scripts/run_experiment.py --experiment 1

run-exp2:
	python scripts/run_experiment.py --experiment 2

figures:
	python scripts/make_figures.py

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

clean:
	rm -rf outputs/figures/ .pytest_cache/ **/__pycache__/ *.egg-info
