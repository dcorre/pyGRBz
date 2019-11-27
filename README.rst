======
pyGRBz
======


.. image:: https://img.shields.io/pypi/v/pygrbz.svg
        :target: https://pypi.python.org/pypi/pygrbz

.. image:: https://img.shields.io/travis/dcorre/pygrbz.svg
        :target: https://travis-ci.org/dcorre/pygrbz

.. image:: https://readthedocs.org/projects/pygrbz/badge/?version=latest
        :target: https://pygrbz.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Bayesian photometric redshift code dedicated to Gamma-Ray Bursts in Python


DESCRIPTION
--------

pyGRBz is a bayesian photometric redshift code dedicated to Gamma-Ray Bursts developed in Python3. It uses either the Ensemblesampler or PTsampler from the emcee package.
The code can take either a GRB light curve or SED as an input. In the case of light curves, each band is fitted separately with a single power law or a broken power law using the iminuit package. As a by-product of the SED fitting, the amount of attenuation in the host galaxy
 


* Free software: MIT license
* Documentation: https://pygrbz.readthedocs.io.


Features
--------

* TODO



Tutorials
--------

In the $pyGRBz_DIR directory you will find a folder named "notebooks", it contains tutorials and examples for the corresponding module. In order to test that the installation went correctly you can execute the one named "Tuto_photoz.ipynb".

To do so, open a terminal and place yourself in the $pyGRBz_DIR directory and type::

    jupyter notebook


It will open a new tab in your internet browser. Select the folder notebooks and then the file named **Tuto_photoz**.  A new tab opens.  
It is a different version of ipython, composed of cells you can separately execute. You can execute each cell by pressing shift+enter. A * will appear inside the brackets [] on the left side of the cell, meaning that the cell is running. Once it is done a number appears and you can execute the next cell. If one cell is running you can not execute other ones, you need to wait  that a number appears inside the brackets, for instance [1] if it is the first cell you execute, then you can go for next cell and so on.  

If you do not see any error message below one cell, it means that all dependencies correctly installed.

Some tutorials about jupyter notebook:

http://bebi103.caltech.edu.s3-website-us-east-1.amazonaws.com/2015/tutorials/t0b_intro_to_jupyter_notebooks.html
http://jupyter-notebook.readthedocs.io/en/latest/
https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook#gs.F4FgfvY

Inside the cells it is pure python, so you can write inside as you would do in your favorite python editor.  




Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
