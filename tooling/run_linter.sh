#!/bin/sh
pylint --rcfile=pylint.rc --fail-under=9 $(git ls-files 'saftig/*.py')
