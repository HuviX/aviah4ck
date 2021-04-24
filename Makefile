PYTHON ?= python
VENV = .venv

CODE = app

.PHONY: pretty lint

init:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install poetry
	$(VENV)/bin/poetry install -v

run:
	PYTHONPATH=. $(VENV)/bin/python -m streamlit run app/main.py

lint:
	black --target-version py36 --check --skip-string-normalization $(CODE)
	flake8 --jobs 4 --statistics $(CODE)
	pylint --jobs 4 --rcfile=setup.cfg $(CODE)
	mypy $(CODE)

pretty:
	black --target-version py36 --skip-string-normalization $(CODE)
	isort --apply --recursive $(CODE)
	unify --in-place --recursive $(CODE)
