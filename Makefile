SHELL := /bin/bash

all: doc testing linter

test:
	cd testing && python testing.py

linter:
	./tooling/run_linter.sh

doc: doc/source/* doc/*
	cd doc/ && $(MAKE) html

view: doc
	open doc/build/html/index.html

.PHONY: all, doc, view, test, linter
