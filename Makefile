PYTHON ?= python
VENV = .venv

export TAG = $(shell git rev-parse --short=8 HEAD)

CODE = app

.PHONY: pretty lint

init:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install poetry
	$(VENV)/bin/poetry install -v

run: run-label
	PYTHONPATH=. $(VENV)/bin/python -m streamlit run app/main.py

run-label:
	cd script && ../$(VENV)/bin/python -m flask run --host 10.129.0.9 &

up:
	docker-compose up app

lint:
	black --target-version py36 --check --skip-string-normalization $(CODE)
	flake8 --jobs 4 --statistics $(CODE)
	pylint --jobs 4 --rcfile=setup.cfg $(CODE)
	mypy $(CODE)

pretty:
	black --target-version py36 --skip-string-normalization $(CODE)
	isort --apply --recursive $(CODE)
	unify --in-place --recursive $(CODE)
