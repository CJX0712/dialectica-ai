# Dialectica · 常用目标
# 作者: 晨星
PY ?= python

.PHONY: install install-prod verify test lint serve bench lock emoji clean

install:
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

install-prod:
	$(PY) -m pip install -r requirements-prod.txt

test:
	$(PY) -m pytest tests -q

lint:
	$(PY) -m ruff check src tests

verify:
	$(PY) scripts/verify.py

serve:
	$(PY) -m dialectica.cli serve

bench:
	$(PY) scripts/report.py

lock:
	$(PY) scripts/make_lock.py

emoji:
	$(PY) scripts/scan_emoji.py

clean:
	-rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
