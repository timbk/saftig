# saftig – Static & Adaptive Filtering Techniques In Gravitational-wave-research

![Test status](https://github.com/timbk/saftig/actions/workflows/testing.yml/badge.svg)
![Linting status](https://github.com/timbk/saftig/actions/workflows/pylint.yml/badge.svg)

Python implementations of different static and adaptive filtering techniques for the prediction of a correlated signal component from witness signals.
The main goal is to provide a unified interface for the different filtering techniques.

## Features

Static:
* Wiener Filter (WF)

Adaptive
* Updating Wiener Fitler (UWF)
* Least-Mean-Squares Filter (LMS)

Non-Linear:
* Experimental non-linear LMS Filter variant (PolynomialLMS)

## Install

From pypi: `pip install saftig`

From repository: `pip install .`

From repository (editable): `make ie`

## Minimal example

```python
>>> import saftig as sg
>>>
>>> # generate data
>>> n_channel = 2
>>> witness, target = sg.TestDataGenerator([0.1]*n_channel).generate(int(1e5))
>>>
>>> # instantiate the filter and apply it
>>> filt = sg.LMSFilter(n_filter=128, idx_target=0, n_channel=n_channel)
>>> filt.condition(witness, target)
>>> prediction = filt.apply(witness, target) # check on the data used for conditioning
>>>
>>> # success
>>> sg.RMS(target-prediction) / sg.RMS(prediction)
0.08221177645361015
```

## Terminology

* Witness signal w: One or multiple sensors that are used to make a prediction
* Target signal s: The goal for the prediction
