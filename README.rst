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


Dependencies
--------


**1.   PREREQUISITES**

You need to have git and anaconda to perform the installation. Git is installed per default on Linux, but not on windows. 


1.1    Windows only :
       - Download git at https://git-scm.com/download/win
       - Run setup (keep default configurations)

       You can now use your git from the cmd or the graphic client !  

  
1.2.    Download Miniconda for python 3 here: https://conda.io/miniconda.html   
    |  (If you want to download a complete python environment (~1.5GB) you can download Anaconda (instead of Miniconda) for Python 3 here: https://www.continuum.io/downloads)    
    |

1.3.    Create an environment variable, pyGRBz_DIR, corresponding to the directory where you want to install the project.
       | Add a directory for the pyGRBz package  (change the path name accordingly to yours).
    
       * **For Ubuntu**::
       
                 Open the .bashrc file and add the following line at the end:
                 export pyGRBz_DIR="/home/dcorre/code/pyGRBz"
       
      
       Then type: *source .bashrc* in the terminal to take the changes into account  

       * **For windows**::
        
                  Right click on My Computer -> Properties -> Advanced System settings -> Environment Variables
                  Add variable pyGRBz_DIR with value "/home/dcorre/code/pyGRBz" (change the path name accordingly to yours)
       
       
 
       Then close and open the terminal again to update the modifications (for windows only)  
       
       * **For Mac**::

            To do




**2. INSTALLATION OF PROJECT DEPENDENCIES**


2.1.    **Create anaconda environment**

It will create locally a python environment in which you can install and specify the libraries we need.  
The advantage is that it will avoid you to change the libraries version you are using for other projects.  
   
From terminal::

     conda create --name pyGRBz python=3  numpy scipy matplotlib jupyter astropy


You can use an other environment name than *pyGRBz*.
   
Once it is installed, type in a terminal::

      source activate pyGRBz
   
It will activate the created environment. You can see that (GRB_photoZ) is added in front of your ID in your terminal. If you type "conda list" you can check which libraries and version are installed. When you want to exit this environment type "source deactivate".  


2.2    **Install other dependencies with pip:**

Write in a terminal::

      pip install iminuit emcee corner sfdmap extinction hjson


To take into account the extinction processes at place in the host galaxy and in the IGM you need to install the following module:   
    - **los_extinction**: Line of sight extinction  




Install pyGRBz
--------

You must have an account on the LAM gitlab : https://gitlab.lam.fr/svom-colibri/simulations/pyGRBz
    
If you don't, send me an email to david.corre@lam.fr  

**1. Clone the project**

**If you are using Windows, we remind you to download git (See "Get git for Windows" section 1)**
   
Cloning the project means that you will get a dynamically synchronized version. Yyou will be able to update your version as soon as a new version is available (with the command : "git pull").


In a terminal, type::

     git clone git@gitlab.lam.fr:svom-colibri/simulations/pyGRBz.git pyGRBz


This creates a pyGRBz/ folder containing the project (including a .git/ subfolder for synchronization with the git repository)



**2. INSTALLATION**   
   
Open a terminal go to $pyGRBz_DIR directory and type::

    python setup.py develop





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
