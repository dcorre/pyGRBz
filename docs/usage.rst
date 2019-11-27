#####
Usage
#####


Fit 1 GRB SED
=============

Import pyGRBz package:     

.. code:: python

    import os 
    from pyGRBz.pyGRBz import GRB_photoZ

Instantiate the class 
 
.. code:: python

    photoz = GRB_photoZ(output_dir='/results/Tuto/SED/')

Load the SED of GRB050904 stored in data/sed/

.. code:: python

    >>> photoz.load_data(data_dir='/data/sed/',data_name=['GRB050904'])
    
    
    Observations:
    Name      time_since_burst band  mag  mag_err  zp phot_sys detection telescope
    --------- ---------------- ---- ----- ------- --- -------- --------- ---------
    GRB050904                1   Ks  20.0    0.07   -       AB         1     isaac
    GRB050904                1    H 20.37    0.07   -       AB         1     isaac
    GRB050904                1    J  20.7    0.06   -       AB         1     isaac
    GRB050904                1    z  21.8     0.2   -       AB         1     fors2
    GRB050904                1    I 24.45     0.2   -       AB         1     fors2
    GRB050904                1    R  23.9    0.05   -       AB         0     cafos
    GRB050904                1    V  24.6    0.05   -       AB         0     laica


    Info about data:
    name      type  RA_J2000   DEC_J2000   ... beta_inf beta_X beta_X_sup beta_X_inf
    --------- ---- ---------- ------------ ... -------- ------ ---------- ----------
    GRB050904  sed 0h54m50.6s +14d05m04.5s ...      0.3    -99        -99        -99
    Formatting the data: apply Galactice correction if needed and compute fluxes in Jansky

Now data need to be corrected for galactic extinction if needed and fluxes expressed in microJy.

.. code:: python

    >>> photoz.formatting()

    SEDS formatted:
    Name      time_since_burst band  mag  ... ext_mag      flux          flux_err   
                                          ...            microJy         microJy    
    --------- ---------------- ---- ----- ... ------- -------------- ---------------
    GRB050904                1    V  24.6 ...     0.0  0.52480746025 0.0241682766933
    GRB050904                1    R  23.9 ...     0.0            1.0 0.0460517018599
    GRB050904                1    I 24.45 ...     0.0 0.602559586074  0.110995577643
    GRB050904                1    z  21.8 ...     0.0  6.91830970919   1.27439974441
    GRB050904                1    J  20.7 ...     0.0  19.0546071796   1.05299650667
    GRB050904                1    H 20.37 ...     0.0  25.8226019063   1.66484466993
    GRB050904                1   Ks  20.0 ...     0.0   36.307805477   2.34085072622



The next Extract the SED at a given time.

In case the input data is already a SED. This function has to be run in order to have the right formatting for the follwing computations

.. code:: python

    photoz.extract_sed(model='SPL',method='ReddestBand')

Create flat priors

.. code:: python

    priors=dict(z=[0,11],Av=[0,2],beta=[0,2],norm=[0,10])

    # Run the MCMC algorithm.
    # Select the extinction law to used: 'smc', 'lmc', 'mw', 'nodust'
    # Nthreads: number of threads to use in case of parallelisation
    # nwalkers: number of walkers
    # Nsteps1: number of steps for the first burn-in phase
    # Nsteps2: number of steps for the second burn-in phase
    # Nsteps3: number of steps for the production run
    # Select to add dust, gas in host and our galaxy
    # Select IGM transmission method: 'Madau' or 'Meiksin'


    photoz.fit(ext_law='smc',Nthreads=4,sampler_type='ensemble',nwalkers=30,Nsteps1=300,Nsteps2=1000,nburn=300,Host_dust=True,Host_gas=False,MW_dust=False,MW_gas=False,DLA=False,igm_att='Meiksin',clean_data=False,plot_all=False,plot_deleted=False,priors=priors)
