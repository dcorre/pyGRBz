# Bayesian photometric redshift code dedicated to Gamma-Ray Bursts in Python

* Free software: MIT license
* Documentation: https://pyGRBz.readthedocs.io.

Release status
--------------

[![PyPI version](https://badge.fury.io/py/pyGBRz.svg)](https://badge.fury.io/py/pyGRBz)
![Supported Python versions](https://img.shields.io/pypi/pyversions/pyGBRz.svg)


Development status
--------------------

[![Build Status](https://travis-ci.com/dcorre/pyGBRz.svg?branch=master)](https://travis-ci.com/dcorre/pyGRBz)
[![codecov](https://codecov.io/gh/dcorre/pyGBRz/branch/master/graphs/badge.svg)](https://codecov.io/gh/dcorre/pyGBRz/branch/master)
[![Documentation Status](https://readthedocs.org/projects/pygrbz/badge/?version=latest)](https://pygrbz.readthedocs.io/en/latest/?badge=latest)

Features
--------
* pyGRBz is a bayesian photometric redshift code dedicated to Gamma-Ray Bursts developed in Python3.
* It uses either the Ensemblesampler or PTsampler from the [emcee]() package.
* The photometric redshift is assessed using spectral signature imprinted on the GRB afterglow Spectral Energy Distribution (SED): Lyman series absorption features, Lyman break and/or the presence of the 2175 Angstroms bump in the extinction law used to model the amount of extinction in the host galaxy.
* The input can be either a GRB afterglow SED or multi-wavelength light curve.
* In case, a light curve is provided, each band is fitted independtly with a template model (either a power law or a broken power law). The fit is performed using the [iminuit] package.
* 3 physical parameters are assessed: the redhsift, the amount of dust extinction in the V band, the spectral slope. 
* 4 extinction laws can be used to model extinction in the host galaxy: MW, SMC, LMC from [Pei+92](http://adsabs.harvard.edu/abs/1992ApJ...395..130P) and no dust.
* When the input is a light curve, the temporal slope for each band is also fitted.
* Have a look to the notebook(s) to see how it works.


Installation
------------
See the doc: https://pyGRBz.readthedocs.io.

Credits
-------

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.

