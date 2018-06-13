=====
Usage
=====

To use pyGRBz in a project::

.. plot::

    from pyGRBz.pyGRBz import GRB_photoZ

    # Load module
    photoz = GRB_photoZ(output_dir=os.getenv('pyGRBz_DIR')+'/pyGRBz/results/Tuto/SED/')

    # Load the GRB SED stored in data/sed/
    photoz.load_data(data_dir='data/sed/',data_name=['GRB050904'])

    # Format data in order to apply galactic estinction and calculates the flux in Jansky to each observations
    photoz.formatting()


    # Extract the SED at a given time.
    # First the data are fitted either with a single power law (SPL) or a broken power law (BPL)
    # Secondly the time at which to extract the SED can be either 'fixed' (needs to give through time_SED in seconds) or 
    # computed to be the time at which the flux is maximum in the reddest band ('ReddestBand')

    # In case the input data is already a SED. THis function has to run in order to have the right
    # formatting for the follwing computations

    photoz.extract_sed(model='SPL',method='ReddestBand')

    # Create flat priors
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
