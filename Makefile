.PHONY: help install test train api dashboard clean all

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	python -m venv venv
	source venv/bin/activate && pip install --upgrade pip
	source venv/bin/activate && pip install -r requirements.txt

start-infra: ## Start infrastructure services
	docker-compose up -d

stop-infra: ## Stop infrastructure services
	docker-compose down

restart-infra: ## Restart infrastructure services
	docker-compose restart

train: ## Train ML models
	source venv/bin/activate && python src/models/trainer.py

api: ## Start API server
	source venv/bin/activate && python src/api/main.py

dashboard: ## Start Streamlit dashboard
	source venv/bin/activate && streamlit run src/dashboard/app.py

streaming: ## Start streaming pipeline
	source venv/bin/activate && python src/streaming/pipeline.py

test: ## Run unit tests
	source venv/bin/activate && pytest tests/ -v --cov=src --cov-report=html

test-unit: ## Run unit tests only
	source venv/bin/activate && pytest tests/ -v -m "unit"

test-integration: ## Run integration tests only
	source venv/bin/activate && pytest tests/ -v -m "integration"

load-test: ## Run load tests
	source venv/bin/activate && locust -f tests/locustfile.py --host=http://localhost:8000

lint: ## Run code linting
	source venv/bin/activate && flake8 src/ tests/

format: ## Format code with black
	source venv/bin/activate && black src/ tests/

clean: ## Clean generated files
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf data/processed/
	rm -rf models/*.pkl
	rm -rf models/*.json
	rm -rf spark-warehouse/
	rm -rf metastore_db/
	rm -rf mlflow-artifacts/
	rm -rf mlflow.db

all: clean install start-infra train ## Full setup: clean, install, start infra, train models

dev: start-infra ## Start development environment
	@echo "Infrastructure started. Run 'make api' to start the API server."

prod: ## Start production environment
	docker-compose -f docker-compose.prod.yml up -d
