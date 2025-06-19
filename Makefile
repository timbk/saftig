SHELL := /bin/bash

all: doc test linter

test:
	coverage run -m unittest discover .
coverage:
	coverage report
cweb:
	coverage html && open htmlcov/index.html

linter:
	./tooling/run_linter.sh

doc: doc/source/* doc/*
	cd doc/ && $(MAKE) html

view: doc
	open doc/build/html/index.html

.PHONY: all, doc, view, test, linter, coverage, cweb
