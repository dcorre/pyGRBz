#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
        'Click>=6.0',
        'numpy',
        'scipy',
        'matplotlib',
        'jupyter',
        'astropy',
        'iminuit',
        # 'emcee<3.0',
        'emcee',
        'corner',
        'sfdmap',
        'extinction',
        'cython',
        'pyGRBaglow']

setup_requirements = ['pytest-runner', 'numpy']

test_requirements = ['pytest', ]

setup(
    author="David Corre",
    author_email='david.corre.fr@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Python Bayesian photometric redshift code dedicated"
                "to Gamma-Ray Bursts",
    entry_points={
        'console_scripts': [
            'pyGRBz=pyGRBz.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=False,
    keywords=['pyGRBz', 'Gamma-Ray Burst', 'redshift photometric',
              'Bayesian', 'MCMC'],
    name='pyGRBz',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/dcorre/pyGRBz',
    download_url='https://github.com/dcorre/pyGRBz/archive/v0.1.0.tar.gz',
    version='0.1.0',
    zip_safe=False,
)
