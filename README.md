# saftig – Static & Adaptive Filtering Techniques In Gravitation-wave-research
Static &amp; Adaptive Filtering Techniques

Implementations of different filtering techniques for the prediction of a correlated signal component from witness signals.
The main goal is to provide a unified interface for the different filtering techniques.

## Terminology

* Witness signal w: One or multiple sensors that are used to make a prediction
* Target signal s: The goal for the prediction

## Useful commands
```bash
make # run linter, testing and generate documentation
make test # run just the tests
make view # build and open documentation

# run an individual test
python -m unittest testing.test_wf
```

