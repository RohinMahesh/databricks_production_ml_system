env:
	conda create -n databricks_production_ml_system python=3.10.12

install-deps:
	python -m pip install -r requirements.txt

test:
	pytest --log-cli-level=DEBUG --cov=. --cov-report=html

lint:
	isort .
	black .