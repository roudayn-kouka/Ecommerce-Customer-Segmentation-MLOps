.PHONY: install lint format test pipeline train plots api docker airflow-init airflow-up airflow-down clean

PYTHON ?= python

install:            ## Install runtime + dev dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt
	pre-commit install

lint:               ## Static checks
	ruff check .
	ruff format --check .

format:             ## Auto-format the code
	ruff check --fix .
	ruff format .

test:               ## Unit tests with coverage
	$(PYTHON) -m pytest --cov --cov-report=term

pipeline:           ## Full pipeline (download -> features -> model)
	$(PYTHON) -m src.pipeline

train:              ## Pipeline + training on the local dataset (no download)
	$(PYTHON) -m src.pipeline --skip-download

plots:              ## Cluster interpretation plots
	$(PYTHON) -m src.plots

api:                ## Serve the model locally on http://localhost:8000/docs
	uvicorn api.main:app --reload --port 8000

docker:             ## Build the API Docker image
	docker build -f api/Dockerfile -t customer-segmentation-api .

airflow-init:
	docker compose up airflow-init

airflow-up:
	docker compose up -d

airflow-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
