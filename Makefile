.PHONY: test
test:
	pytest -v --ignore=sandbox/ --cov=./ --cov-report=html --cov-config=.coveragerc test/