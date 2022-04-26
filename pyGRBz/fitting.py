#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
import emcee
import numpy as np
from scipy.special import erf
from multiprocessing import Pool
from astropy.table import Table, vstack
from astropy.io import ascii
from pyGRBz.plotting import (
    plot_mcmc_evolution,
    plot_triangle,
    plot_mcmc_fit,
)

# Import cythonised code if present
try:
    from pyGRBz.fluxes_cy import compute_model_integrated_flux
except:
    from pyGRBz.fluxes import compute_model_integrated_flux


def residuals(params, kind="mag"):
    """Calculate the residuals, observations - models"""

    # Adapt the number of parameters in fonction of dust model and gas extinction
    if Host_gas_g:
        if ext_law_g == "nodust":
            z, beta, norm, NHx = params
            Av = 0
        else:
            z, beta, norm, Av, NHx = params
    else:
        NHx = 0
        if ext_law_g == "nodust":
            z, beta, norm = params
            Av = 0
        else:
            z, beta, norm, Av = params

    # Calculate the Flux in microJansky for the given set of parameters and a
    # tt0 = time.time()
    flux_model = compute_model_integrated_flux(
        wavelength_g,
        sys_response,
        F0,
        wvl0,
        norm,
        beta,
        z,
        Av,
        NHx,
        ext_law_g,
        Host_dust_g,
        Host_gas_g,
        igm_att_g,
    )
    # tt1 = time.time()
    # print ("Total time integrated flux: {:.2e}s".format(tt1-tt0))
    # for i in range(len(flux_obs)):
    #    print (flux_obs[i], flux_model[i], fluxerr_obs[i])

    return (flux_obs - flux_model) / fluxerr_obs


def lnprior(params):
    """Set the allowed parameter range. Return the lnPrior"""

    # Adapt the number of parameters in fonction of dust model and gas extinction
    if Host_gas_g:
        if ext_law_g == "nodust":
            z, beta, norm, NHx = params
            Av = 0
        else:
            z, beta, norm, Av, NHx = params
    else:
        if ext_law_g == "nodust":
            z, beta, norm = params
            Av = 0
        else:
            z, beta, norm, Av = params

    # So far only flat prior implemented
    # If the current parameter value is outside the allowed range,
    # it is set to -inf,
    # meaning that the probability to have this value is 0
    # Otherwise to 0, meaning probability to have this value == 1
    if not priors_g["z"][0] < z < priors_g["z"][1]:
        return -np.inf
    if not priors_g["beta"][0] < beta < priors_g["beta"][1]:
        return -np.inf
    if not priors_g["norm"][0] < norm < priors_g["norm"][1]:
        return -np.inf
    if ext_law_g != "nodust":
        if not priors_g["Av"][0] < Av < priors_g["Av"][1]:
            return -np.inf
    if Host_gas_g is True:
        if not priors_g["NHx"][0] < NHx < priors_g["NHx"][1]:
            return -np.inf

    return 0.0


def lnlike(params):
    """Calculate the log likelihood. Return the lnLikelihood"""
    kind = "flux"
    # Calculate the residuals: (obs - model)/obs_err for each band
    res = residuals(params, kind=kind)

    # cumulative distribution function of the residuals
    if kind == "flux":
        residuals_cdf = 0.5 * (1 + erf(-res / np.sqrt(2)))
    elif kind == "mag":
        residuals_cdf = 0.5 * (1 + erf(res / np.sqrt(2)))

    # Survival function
    residuals_edf = 1 - residuals_cdf
    # residuals pdf
    residuals_pdf = -0.5 * res**2

    # detect is 1 if detections and 0 if no detection
    mask = detection_flag == 1
    lnlik = np.sum(residuals_pdf[mask])
    if (~mask).any():
        lnlik += np.sum(np.log(residuals_edf[~mask]))

    return lnlik


def chi2_comp(params):
    """Calculate the chi square associated to a flux. Return the chi square"""
    kind = "flux"
    # Calculate the residuals: (obs - model)/obs_err for each band
    res = residuals(params, kind=kind)

    # cumulative distribution function of the residuals
    if kind == "flux":
        residuals_cdf = 0.5 * (1 + erf(-res / np.sqrt(2)))
    elif kind == "mag":
        residuals_cdf = 0.5 * (1 + erf(res / np.sqrt(2)))

    # Survival function
    residuals_edf = 1 - residuals_cdf
    # residuals pdf
    residuals_pdf = -0.5 * res**2

    # detect is 1 if detections and 0 if no detection
    mask = detection_flag == 1
    lnlik = np.sum(residuals_pdf[mask])
    chi2 = -2 * lnlik

    return chi2


def lnlik_C(yerr):
    """constant term of the log likelihood expression"""
    lnC = -0.5 * len(yerr) * np.log(2 * np.pi * yerr**2)
    return lnC


def lnprob(params):
    """Add lnPrior and lnLikelihood"""

    # Get the lnPrior
    lp = lnprior(params)
    # Get the lnLikelihood
    lnlik = lnlike(params)

    #  Check whether it is finite
    if not np.isfinite(lp):
        return -np.inf
    if not np.isfinite(lnlik):
        return -np.inf

    return lp + lnlik


def dof(params, y):
    """Calculate the number of degrees of freedom"""
    n = len(y)
    k = len(params)
    dof = n - k
    return dof


def Likelihood(yerr, lnlik):
    """Compute the Likelihood from lnlik and lnlik_C"""
    L = np.exp(-2 * (lnlik_C(yerr) + lnlik))
    return L


def AIC(k, yerr, best_lnlik):
    """AIC criteria"""
    val = 2 * k - 2 * (lnlik_C(yerr) + best_lnlik)
    return val


def AICc(k, yerr, best_lnlik):
    """AICc criteria"""
    n = len(yerr)
    _AIC = AIC(k, yerr, best_lnlik)
    AICc = _AIC + 2 * k * (k + 1) / (n - k - 1)
    return AICc


###
def BIC(params, y, Host_gas_g, ext_law_g):
    """BIC criteria"""
    if Host_gas_g:
        if ext_law_g == "nodust":
            Av = 0
            z = params["z"]
            beta = params["beta"]
            norm = params["norm"]
            NHx = params["NHx"]
            parameters = z, beta, norm, NHx
        else:
            z = params["z"]
            beta = params["beta"]
            Av = params["Av"]
            norm = params["norm"]
            NHx = params["NHx"]
            parameters = z, beta, norm, Av, NHx
    else:
        NHx = 0
        if ext_law_g == "nodust":
            Av = 0
            z = params["z"]
            beta = params["beta"]
            norm = params["norm"]
            parameters = z, beta, norm
        else:
            z = params["z"]
            beta = params["beta"]
            Av = params["Av"]
            norm = params["norm"]
            parameters = z, beta, norm, Av

    chi2 = chi2_comp(parameters)
    k = len(parameters)
    N = len(y)
    bic = chi2 + k * np.log(N)
    return bic


###


def BIC2(params, chi2, y, Host_gas_g, ext_law_g):
    """BIC criteria"""
    if Host_gas_g:
        if ext_law_g == "nodust":
            Av = 0
            z = params["z"]
            beta = params["beta"]
            norm = params["norm"]
            NHx = params["NHx"]
            parameters = z, beta, norm, NHx
        else:
            z = params["z"]
            beta = params["beta"]
            Av = params["Av"]
            norm = params["norm"]
            NHx = params["NHx"]
            parameters = z, beta, norm, Av, NHx
    else:
        NHx = 0
        if ext_law_g == "nodust":
            Av = 0
            z = params["z"]
            beta = params["beta"]
            norm = params["norm"]
            parameters = z, beta, norm
        else:
            z = params["z"]
            beta = params["beta"]
            Av = params["Av"]
            norm = params["norm"]
            parameters = z, beta, norm, Av

    k = len(parameters)
    N = len(y)
    bic = chi2 + k * np.log(N)
    return bic


def reduced_chi2(chi2, y):
    N = len(y)
    chi2_reduced = chi2 / (N - 1)
    return chi2_reduced


def find_maximum_redshift(sed, mask_det):
    """
    Compute maximum redhift in case of non detection.
    Try to decrease the redshift parameter space based on the detection in the
    filter having the lowest effective wavelength (i.e. bluest)
    Assume that no flux will be observed belwo Lyman break at 912 angstroms
    Take 10% of the value to be safer?
    """

    # Sed is already sorted by ascending effective wavelength
    # Make another check to avoid X-ray data
    mask_no_X = (mask_det) & (sed["eff_wvl"] > 1000)
    if mask_no_X.any():
        wvl_cutoff = (
            float(sed["eff_wvl"][mask_no_X][0])
            + float(sed["band_width"][mask_no_X][0]) / 2
        )
        priors_g["z"][1] = (wvl_cutoff / 912) - 1
        print(
            "Bluest band detection: %s/%s with eff_wvl=%.0f and bandwidth=%.0f (Angstroms).\n"
            % (
                sed["telescope"][mask_det][0],
                sed["band"][mask_det][0],
                sed["eff_wvl"][mask_det][0],
                sed["band_width"][mask_det][0],
            )
        )
        print(
            "Assuming no flux can be observed below Lyman break "
            "at 912 Angstroms\n"
            "--> maximum allowed redshift is %.2f.\n" % (priors_g["z"][1])
        )
        print(
            "This value is used to constrain the redshift parameter "
            "space in the analysis below."
        )


def set_initial_values(nwalkers, ndim):
    """Initial values for walkers"""
    starting_guesses = np.random.rand(nwalkers, ndim)

    # Initial values for redshift taken between
    # priors['z'][0] and priors['z'][1]
    starting_guesses[:, 0] *= priors_g["z"][1] - priors_g["z"][0]
    starting_guesses[:, 0] += priors_g["z"][0]
    # Initial values for spectral slope taken between
    # priors['beta'][0] and priors['beta'][1]
    starting_guesses[:, 1] *= priors_g["beta"][1] - priors_g["beta"][0]
    starting_guesses[:, 1] += priors_g["beta"][0]
    # Initial values for normalisation factor taken between
    # priors['norm'][0] and priors['norm'][1]
    starting_guesses[:, 2] *= priors_g["norm"][1] - priors_g["norm"][0]
    starting_guesses[:, 2] += priors_g["norm"][0]
    if ext_law_g != "nodust":
        # Initial values for Av taken between
        # priors['Av'][0] and priors['Av'][1]
        starting_guesses[:, 3] *= priors_g["Av"][1] - priors_g["Av"][0]
        starting_guesses[:, 3] += priors_g["Av"][0]
    if Host_gas_g is True:
        if ext_law_g == "nodust":
            idx = 3
        else:
            idx = 4
        # Initial values for NHx taken between
        # priors['NHx'][0] and priors['NHx'][1]
        starting_guesses[:, idx] *= priors_g["NHx"][1] - priors_g["NHx"][0]
        starting_guesses[:, idx] += priors_g["NHx"][0]

    return starting_guesses


def get_data(sed, grb_info, mask):
    """Get data specific to the GRB that are used in the MCMC process"""
    # If redshift and Av are provided in the data file use them
    try:
        z_sim = float(np.asscalar(grb_info["z"][mask]))
    except:
        z_sim = -99
    try:
        Av_sim = float(np.asscalar(grb_info["Av_host"][mask]))
    except:
        Av_sim = -99
    try:
        beta_sim = float(np.asscalar(grb_info["beta"][mask]))
    except:
        beta_sim = -99
    print("z_lit: {0:.2f}   Av_lit: {1:.2f}".format(z_sim, Av_sim))

    # Normalisation values chosed to be the ones of the reddest band
    detection_flag = np.array(sed["detection"], dtype=int)
    mask = detection_flag == 1
    F0 = float(sed["flux_corr"][np.argmax(sed["eff_wvl"][mask])])
    wvl0 = float(sed["eff_wvl"][np.argmax(sed["eff_wvl"][mask])])
    # print ("Reference wavelength: {:.2f}".format(wvl0))
    # print ("Reference Flux: {:.2f}".format(F0))

    # Substract the galctic extinction
    # flux_corr = sed["flux"] - sed["ext_mag"]
    flux_obs = np.array(sed["flux_corr"], dtype=np.float64)

    # eff_wvl = np.array(sed["eff_wvl"])
    fluxerr_obs = np.array(sed["flux_corr_err"], dtype=np.float64)
    # flux_corr_err = np.array(sed["flux_err"])
    sys_response = np.array(sed["sys_response"], dtype=np.float64)

    return (
        z_sim,
        Av_sim,
        beta_sim,
        F0,
        wvl0,
        detection_flag,
        flux_obs,
        fluxerr_obs,
        sys_response,
    )


def sampler_run(
    nwalkers, ndim, starting_guesses, Nthreads, Nsteps1, Nsteps2, std_gaussianBall
):
    """Run the MCMC sampler"""

    with Pool(Nthreads) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            lnprob,
            moves=[
                (emcee.moves.DEMove(), 0.8),
                (emcee.moves.DESnookerMove(), 0.2),
            ],
            pool=pool,
        )

        # First run: burn-in
        pos = starting_guesses
        if Nsteps1 > 0:
            print("Running burn-in")
            pos, prob, state = sampler.run_mcmc(pos, Nsteps1, progress=True)
            sampler.reset()

        # Second run: run used for the statisctics
        # Takes the values of the last steps of the burn-in run as
        # initial values
        if Nsteps2 > 0:
            print("Running production")
            if Nsteps1 > 0:
                print(
                    "Nsteps1 > 0 --> Initial values are drawn from a "
                    "Gaussian distribution with means equal to the "
                    "values returning the best chi2 during first run "
                    "and std of %.2e" % std_gaussianBall
                )
                # Start from a gaussian centered on values returning
                # the best chi2 for the first run
                p = pos[np.unravel_index(np.nanargmax(prob), prob.shape)]
                pos = [
                    p + std_gaussianBall * np.random.randn(ndim)
                    for i in range(nwalkers)
                ]
            sampler.run_mcmc(pos, Nsteps2, progress=True)

    print(
        "\nAutocorrelation time: {0:.2f} steps".format(
            sampler.get_autocorr_time(quiet=True)[0]
        )
    )
    print("\n")

    # Store the chains
    chain = sampler.chain
    lnproba = sampler.lnprobability
    acceptance_fraction = sampler.acceptance_fraction

    #  Free memory
    del sampler

    return chain, lnproba, acceptance_fraction


def do_results_plots(
    chains,
    chains_del,
    lnproba,
    result_1_SED,
    best_chi2_val,
    nburn,
    ndim,
    Av_sim,
    z_sim,
    beta_sim,
    sed,
    plot,
    plot_all,
    plot_deleted,
    output_dir,
    filename_suffix,
):
    """Create plots to analyse MCMC results"""

    # Create evolution plot
    plot_mcmc_evolution(
        chains,
        chains_del,
        nburn,
        ndim,
        ext_law_g,
        Host_gas_g,
        Av_sim,
        z_sim,
        sed["Name"][0],
        plot,
        plot_deleted,
        output_dir=output_dir,
        filename_suffix=filename_suffix,
        priors=priors_g,
    )

    chains_post_burn = chains[:, nburn:, :].copy()

    # Create the triangle plot
    if ext_law_g == "nodust" and Host_gas_g is False:
        samplesTriangle = chains_post_burn
    elif ext_law_g != "nodust" and Host_gas_g is False:
        # Change orders to have norm at the bottom
        # of triangle plot
        samplesTriangle = chains_post_burn.copy()
        # Put Av as second
        samplesTriangle[:, :, 1] = chains_post_burn[:, :, 3]
        # Put beta as third
        samplesTriangle[:, :, 2] = chains_post_burn[:, :, 1]
        # Put norm as fourth
        samplesTriangle[:, :, 3] = chains_post_burn[:, :, 2]
    elif ext_law_g == "nodust" and Host_gas_g is True:
        # Change orders to have norm at the bottom
        # of triangle plot
        samplesTriangle = chains_post_burn.copy()
        # Put beta as second
        samplesTriangle[:, :, 1] = chains_post_burn[:, :, 1]
        # Put NHx as third
        samplesTriangle[:, :, 2] = chains_post_burn[:, :, 3]
        # Put norm as fourth
        samplesTriangle[:, :, 3] = chains_post_burn[:, :, 2]
    elif ext_law_g != "nodust" and Host_gas_g is True:
        # Change orders to have norm at the bottom
        # of triangle plot
        samplesTriangle = chains_post_burn.copy()
        # Put Av as second
        samplesTriangle[:, :, 1] = chains_post_burn[:, :, 3]
        # Put beta as third
        samplesTriangle[:, :, 2] = chains_post_burn[:, :, 1]
        # Put NHx as fourth
        samplesTriangle[:, :, 3] = chains_post_burn[:, :, 4]
        # Put norm as fifth
        samplesTriangle[:, :, 4] = chains_post_burn[:, :, 2]

    plot_triangle(
        samplesTriangle.reshape((-1, ndim)),
        ndim,
        z_sim,
        ext_law_g,
        Host_gas_g,
        Av_sim,
        beta_sim,
        sed["Name"][0],
        plot,
        plot_deleted,
        filename_suffix=filename_suffix,
        output_dir=output_dir,
        priors=priors_g,
    )

    plot_mcmc_fit(
        result_1_SED,
        ndim,
        best_chi2_val,
        sed,
        wavelength_g,
        chains_post_burn.reshape((-1, ndim)),
        plot_all,
        plot,
        ext_law_g,
        Host_dust_g,
        Host_gas_g,
        MW_dust_g,
        MW_gas_g,
        DLA_g,
        igm_att_g,
        output_dir=output_dir,
        filename_suffix=filename_suffix,
    )


def mcmc(
    seds,
    grb_info,
    wavelength,
    plot,
    Nsteps1=300,
    Nsteps2=1000,
    nwalkers=30,
    Nthreads=1,
    nburn=300,
    ext_law="smc",
    clean_data=False,
    plot_all=False,
    plot_deleted=False,
    Host_dust=True,
    Host_gas=False,
    MW_dust=False,
    MW_gas=False,
    DLA=False,
    igm_att="Meiksin",
    output_dir="results/test/",
    filename_suffix="",
    std_gaussianBall=1e-2,
    priors=dict(z=[0, 11], Av=[0, 5], beta=[0, 3], NHx=[0.1, 100], norm=[0.8, 5]),
    adapt_z=True,
):
    """Compute the MCMC algorithm"""

    # Set global variables
    # Need it for speeding multiproccesing
    # Could use self instance in a class but there were some problems
    # with non pickable stuff
    global flux_obs
    global fluxerr_obs
    global detection_flag
    global wavelength_g
    global F0
    global wvl0
    global sys_response
    global ext_law_g
    global Host_dust_g
    global Host_gas_g
    global MW_dust_g
    global MW_gas_g
    global DLA_g
    global igm_att_g
    global priors_g

    # Initialise what can be at this stage
    ext_law_g = ext_law
    Host_dust_g = Host_dust
    Host_gas_g = Host_gas
    MW_dust_g = MW_dust
    MW_gas_g = MW_gas
    DLA_g = DLA
    igm_att_g = igm_att
    priors_g = priors
    wavelength_g = wavelength

    results = []

    # if not os.path.exists(output_dir): os.makedirs(output_dir)

    #  Check input parameters
    if Nsteps2 < nburn:
        print("ERROR: Nsteps2 < nburn: there will be no values to estimate")
        sys.exit(1)

    #  Adapt number of parameters in fonction of the selected dust model
    if ext_law == "nodust" and Host_gas is False:
        ndim = 3
    elif (ext_law != "nodust" and Host_gas is False) or (
        ext_law == "nodust" and Host_gas is True
    ):
        ndim = 4
    elif ext_law != "nodust" and Host_gas is True:
        ndim = 5

    # Compute the initial values for the parameters
    # Initialise the ndim array "starting_guesses" with
    # random values between 0 and 1
    starting_guesses = set_initial_values(nwalkers, ndim)

    list_notdetected = []

    for counter, sed in enumerate(seds.group_by("Name").groups):
        # Reset priors in case they were modified, i.e with find_maximum_redshift()
        priors_g = priors

        # Sort by ascending wavelength
        sed.sort(["eff_wvl"])

        print(
            "\n\nFit {:d}/{:d} \t Object: {:s} \n".format(
                counter + 1, len(grb_info), sed["Name"][0]
            )
        )

        # Check that there is at least a detection in one band
        mask_det = sed["detection"] == 1
        if mask_det.any():
            mask = grb_info["name"] == sed["Name"][0]

            # If adapt_z is True try to reduce parameter space for the redshift
            # based on non detection in blue bands
            if adapt_z:
                find_maximum_redshift(sed, mask_det)

            # Get data related to the current GRB SED
            # These data are global so can be used in other methods within this script
            (
                z_sim,
                Av_sim,
                beta_sim,
                F0,
                wvl0,
                detection_flag,
                flux_obs,
                fluxerr_obs,
                sys_response,
            ) = get_data(sed, grb_info, mask)

            # Run the MCMC sampler
            chains, lnproba, acceptance_fraction = sampler_run(
                nwalkers,
                ndim,
                starting_guesses,
                Nthreads,
                Nsteps1,
                Nsteps2,
                std_gaussianBall,
            )

            # Clean the chains. Normally no need to with this version
            if clean_data:
                chains_corr, chains_del, lnproba_corr, index_removed = clean_chains(
                    chains,
                    lnproba,
                    acceptance_fraction,
                    nburn,
                    acceptance_frac_lim=0.05,
                )
            else:
                chains_corr = chains
                lnproba_corr = lnproba
                chains_del = None

            #  Keep only data after discarding the "nburn" first steps

            # If less than (nwalkers-5) chains remains after the "chains
            # cleaning", do not compute statistics and ends here
            if clean_data and (nwalkers - len(index_removed)) < 5:
                print(
                    "WARNING: After cleaning the chains, "
                    "only %d chains are left." % len(chains_corr[:, 0, 0]),
                    " A minimum of 5 walkers is required. "
                    "Either increase the number of walkers or "
                    "adapt the priors.",
                )
                continue
            else:
                # Compute statistics
                result_1_SED, best_chi2_val = compute_statistics(
                    chains_corr,
                    lnproba_corr,
                    acceptance_fraction,
                    nburn,
                    ndim,
                    sed,
                    z_sim,
                    Av_sim,
                    flux_obs,
                    fluxerr_obs,
                )

                # Create plots
                do_results_plots(
                    chains_corr,
                    chains_del,
                    lnproba_corr,
                    result_1_SED,
                    best_chi2_val,
                    nburn,
                    ndim,
                    Av_sim,
                    z_sim,
                    beta_sim,
                    sed,
                    plot,
                    plot_all,
                    plot_deleted,
                    output_dir,
                    filename_suffix,
                )

                # If detections, write a result file
                ascii.write(
                    result_1_SED,
                    output_dir
                    + str(sed["Name"][0])
                    + "/best_fits_"
                    + ext_law
                    + filename_suffix
                    + ".dat",
                    overwrite=True,
                )

                # Write fluxes for best fit
                save_best_fit_fluxes(
                    best_chi2_val,
                    output_dir,
                    ext_law,
                    str(sed["Name"][0]),
                    filename_suffix,
                )

                results.append(result_1_SED)
        else:
            # No detections
            print("No detection in all bands for %s " % sed["Name"][0])
            list_notdetected.append(sed["Name"][0])
    print("\nList of GRB not detected: {}\n".format(list_notdetected))

    # Write the grb params in an ascii file
    if list_notdetected:
        test_list = []
        for name in list_notdetected:
            test_list.append(grb_info[grb_info["name"] == name])
        ascii.write(
            np.array(list_notdetected),
            output_dir + "notdetected_" + ext_law + filename_suffix + ".dat",
            overwrite=True,
        )

    # If detections, write a result file
    if results:
        results = vstack(results)
        ascii.write(
            results,
            output_dir + "best_fits_all_" + ext_law + filename_suffix + ".dat",
            overwrite=True,
        )


def save_best_fit_fluxes(best_chi2_val, output_dir, ext_law, GRB_name, filename_suffix):
    """Store fluxes of the best fit"""
    # Calculate the Flux in microJansky for the given set of parameters and a
    flux_model = compute_model_integrated_flux(
        wavelength_g,
        sys_response,
        F0,
        wvl0,
        best_chi2_val["norm"],
        best_chi2_val["beta"],
        best_chi2_val["z"],
        best_chi2_val["Av"],
        best_chi2_val["NHx"],
        ext_law_g,
        Host_dust_g,
        Host_gas_g,
        igm_att_g,
    )
    fluxes_data = [flux for flux in flux_model]
    labels = [f"Flux_model_band_{i+1}" for i in range(len(flux_model))]
    for flux, err, det in zip(flux_obs, fluxerr_obs, detection_flag):
        fluxes_data = fluxes_data + [flux, err, det]
    for i in range(len(flux_model)):
        labels = labels + [
            f"Flux_obs_band_{i+1}",
            f"Fluxerr_obs_band_{i+1}",
            f"detection_obs_band_{i+1}",
        ]
    # Write fluxes for best fit
    ascii.write(
        np.array(fluxes_data),
        output_dir
        + GRB_name
        + "/best_fit_fluxes_"
        + ext_law
        + filename_suffix
        + ".dat",
        names=labels,
        overwrite=True,
    )


def clean_chains(chains, lnproba, chains_acc_frac, nburn, acceptance_frac_lim=0.15):
    """
    Clean the mcmc chains.
    So far just remove the chain with a low acceptance fraction
    and nan probability after burn-in phase.
    """
    print_corr = True

    index2remove = []

    nwalkers = len(chains[:, 0, 0])

    for walker in range(nwalkers):

        # Remove all walker with low acceptance fraction
        if chains_acc_frac[walker] < acceptance_frac_lim:
            index2remove.append(walker)
            if print_corr:
                print(
                    "Walker %d removed: mean acceptance fraction of %.2f < %.2f"
                    % (walker, chains_acc_frac[walker], acceptance_frac_lim)
                )
        else:
            # Searching for non finite probability after the burn-in phase
            # mask = np.isfinite(lnproba[walker][nburn:]) is False
            mask = np.isfinite(lnproba[walker][nburn:])
            mask = ~mask
            if mask.any():
                index2remove.append(walker)
                if print_corr:
                    print(
                        "Walker %d removed: " % walker,
                        "non finite probability found after burn-in phase",
                    )

    chains_corr = chains.copy()
    lnproba_corr = lnproba.copy()

    if not index2remove:
        print("No walker removed for statistical analysis")
        chains_del = None
    else:
        # delete possible duplicates in index2remove
        index2remove = sorted(set(index2remove), reverse=True)
        print("\n%d/%d walkers removed" % (len(index2remove), nwalkers))
        for i in index2remove:
            chains_corr = np.delete(chains_corr, i, 0)
            lnproba_corr = np.delete(lnproba_corr, i, 0)

        chains_del = chains.copy()
        for i in range(nwalkers):
            i2 = nwalkers - 1 - i
            if i2 not in index2remove:
                chains_del = np.delete(chains_del, i2, 0)

    return chains_corr, chains_del, lnproba_corr, index2remove


def return_bestlnproba(lnproba, chains):
    """Extract the values of the parameter for which the likelihood is min"""

    idx = np.unravel_index(np.nanargmax(lnproba), lnproba.shape)
    chi2 = -2 * lnproba[idx]

    # Print the parameters for the best likelihood
    print("\nBest fit:")
    best_val = {}
    if ext_law_g == "nodust" and Host_gas_g is False:
        print(
            "z: {:.3f}  beta: {:.3f}  Norm: {:.3f}     chi2: {:.3f}".format(
                chains[idx][0], chains[idx][1], chains[idx][2], chi2
            )
        )
        best_val["z"] = chains[idx][0]
        best_val["beta"] = chains[idx][1]
        best_val["norm"] = chains[idx][2]
        best_val["Av"] = 0
        best_val["NHx"] = -1
    elif ext_law_g != "nodust" and Host_gas_g is False:
        print(
            "z: {:.3f}  Av: {:.3f}  beta: {:.3f}  Norm: {:.3f}     chi2: {:.3f}".format(
                chains[idx][0], chains[idx][3], chains[idx][1], chains[idx][2], chi2
            )
        )
        best_val["z"] = chains[idx][0]
        best_val["beta"] = chains[idx][1]
        best_val["norm"] = chains[idx][2]
        best_val["Av"] = chains[idx][3]
        best_val["NHx"] = -1
    elif ext_law_g == "nodust" and Host_gas_g is True:
        print(
            "z: {:.3f}  beta: {:.3f}  NHx: {:.3f}  Norm: {:.3f}     chi2: {:.3f}".format(
                chains[idx][0], chains[idx][1], chains[idx][3], chains[idx][2], chi2
            )
        )
        best_val["z"] = chains[idx][0]
        best_val["beta"] = chains[idx][1]
        best_val["norm"] = chains[idx][2]
        best_val["Av"] = 0
        best_val["NHx"] = chains[idx][3]
    elif ext_law_g != "nodust" and Host_gas_g is True:
        print(
            "z: {:.3f}  Av: {:.3f}  beta: {:.3f}  NHx: {:.3f}  Norm: {:.3f}     chi2: {:.3f}".format(
                chains[idx][0],
                chains[idx][3],
                chains[idx][1],
                chains[idx][2],
                chains[idx][4],
                chi2,
            )
        )
        best_val["z"] = chains[idx][0]
        best_val["beta"] = chains[idx][1]
        best_val["norm"] = chains[idx][2]
        best_val["Av"] = chains[idx][3]
        best_val["NHx"] = chains[idx][4]
    return best_val


def compute_statistics(
    chains,
    lnproba,
    acceptance_fraction,
    nburn,
    ndim,
    sed,
    z_sim,
    Av_sim,
    flux_obs,
    fluxerr_obs,
):
    """Compute statistics for the current run"""

    #  Compute mean acceptance fraction
    mean_acceptance_fraction = np.mean(acceptance_fraction)
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(acceptance_fraction)))

    # Do not consider burn-in phase
    chains_post_burn = chains[:, nburn:, :]
    lnproba_post_burn = lnproba[:, nburn:]

    mask_nan = np.isfinite(lnproba_post_burn)

    best_chi2_val = return_bestlnproba(lnproba_post_burn, chains_post_burn)
    sum_proba = np.sum(np.exp(lnproba_post_burn[mask_nan]))
    mean_proba = np.mean(np.exp(lnproba_post_burn[mask_nan]))
    best_chi2 = (
        -2
        * lnproba_post_burn[
            np.unravel_index(np.nanargmax(lnproba_post_burn), lnproba_post_burn.shape)
        ]
    )

    # BIC computation
    bic = BIC(best_chi2_val, flux_obs, Host_gas_g, ext_law_g)
    bic2 = BIC2(best_chi2_val, best_chi2, flux_obs, Host_gas_g, ext_law_g)

    # Reduced chi2
    chi2_reduced = reduced_chi2(best_chi2, flux_obs)

    print("\nMean Proba: %.2e" % mean_proba)
    print("Sum Proba: %.2e" % sum_proba)

    params_mcmc = map(
        lambda v: (
            v[3],
            v[4] - v[3],
            v[3] - v[2],
            v[5] - v[3],
            v[3] - v[1],
            v[6] - v[3],
            v[3] - v[0],
        ),
        zip(
            *np.percentile(
                chains_post_burn.reshape((-1, ndim)),
                [0.15, 2.5, 16, 50, 84, 97.5, 99.85],
                axis=0,
            )
        ),
    )

    if ext_law_g == "nodust" and Host_gas_g is False:
        z_mcmc, beta_mcmc, norm_mcmc = params_mcmc
        Av_mcmc = [0, 0, 0, 0, 0, 0, 0]
        NHx_mcmc = [0, 0, 0, 0, 0, 0, 0]

    elif ext_law_g != "nodust" and Host_gas_g is False:
        z_mcmc, beta_mcmc, norm_mcmc, Av_mcmc = params_mcmc
        NHx_mcmc = [0, 0, 0, 0, 0, 0, 0]

    elif ext_law_g == "nodust" and Host_gas_g is True:
        z_mcmc, beta_mcmc, norm_mcmc, NHx_mcmc = params_mcmc
        Av_mcmc = [0, 0, 0, 0, 0, 0, 0]

    elif ext_law_g != "nodust" and Host_gas_g is True:
        z_mcmc, beta_mcmc, norm_mcmc, Av_mcmc, NHx_mcmc = params_mcmc

    perc = ["68%", "95%", "99%"]
    sigs = [1, 2, 3]

    for i in range(3):
        print("\n{} - {:d} sigma:".format(perc[i], sigs[i]))
        print(
            "z: {:.3f} +{:.3f} -{:.3f}".format(
                z_mcmc[0], z_mcmc[1 + 2 * i], z_mcmc[2 + 2 * i]
            )
        )
        if ext_law_g != "nodust":
            print(
                "Av: {:.3f} +{:.3f} -{:.3f}".format(
                    Av_mcmc[0], Av_mcmc[1 + 2 * i], Av_mcmc[2 + 2 * i]
                )
            )
        print(
            "Beta: {:.3f} +{:.3f} -{:.3f}".format(
                beta_mcmc[0], beta_mcmc[1 + 2 * i], beta_mcmc[2 + 2 * i]
            )
        )
        if Host_gas_g is True:
            print(
                "NHx: {:.3f} +{:.3f} -{:.3f}".format(
                    NHx_mcmc[0], NHx_mcmc[1 + 2 * i], NHx_mcmc[2 + 2 * i]
                )
            )
        print(
            "norm: {:.3f} +{:.3f} -{:.3f}".format(
                norm_mcmc[0], norm_mcmc[1 + 2 * i], norm_mcmc[2 + 2 * i]
            )
        )

    # number of bands
    nb_bands = len(sed["band"])
    # Number of band with a signal
    nb_detected = sum(sed["detection"])
    #  corresponding bands
    band_detected_list = sed[sed["detection"] == 1]["band"]
    band_detected = ""
    for i in band_detected_list:
        band_detected = band_detected + str(i)

    # Save results
    results_current_run = [
        [sed["Name"][0]],
        [z_sim],
        [Av_sim],
        [ext_law_g],
        [best_chi2_val["z"]],
        [best_chi2_val["Av"]],
        [best_chi2_val["beta"]],
        [best_chi2_val["norm"]],
        [best_chi2_val["NHx"]],
        [z_mcmc[0]],
        [z_mcmc[1]],
        [z_mcmc[2]],
        [z_mcmc[0]],
        [z_mcmc[3]],
        [z_mcmc[4]],
        [z_mcmc[0]],
        [z_mcmc[5]],
        [z_mcmc[6]],
        [Av_mcmc[0]],
        [Av_mcmc[1]],
        [Av_mcmc[2]],
        [Av_mcmc[0]],
        [Av_mcmc[3]],
        [Av_mcmc[4]],
        [Av_mcmc[0]],
        [Av_mcmc[5]],
        [Av_mcmc[6]],
        [beta_mcmc[0]],
        [beta_mcmc[1]],
        [beta_mcmc[2]],
        [beta_mcmc[0]],
        [beta_mcmc[3]],
        [beta_mcmc[4]],
        [beta_mcmc[0]],
        [beta_mcmc[5]],
        [beta_mcmc[6]],
        [norm_mcmc[0]],
        [norm_mcmc[1]],
        [norm_mcmc[2]],
        [norm_mcmc[0]],
        [norm_mcmc[3]],
        [norm_mcmc[4]],
        [norm_mcmc[0]],
        [norm_mcmc[5]],
        [norm_mcmc[6]],
        [NHx_mcmc[0]],
        [NHx_mcmc[1]],
        [NHx_mcmc[2]],
        [NHx_mcmc[0]],
        [NHx_mcmc[3]],
        [NHx_mcmc[4]],
        [NHx_mcmc[0]],
        [NHx_mcmc[5]],
        [NHx_mcmc[6]],
        [best_chi2],
        [mean_proba],
        [sum_proba],
        [mean_acceptance_fraction],
        [nb_bands],
        [nb_detected],
        [band_detected],
        [bic],
        [bic2],
        [chi2_reduced],
    ]
    result_1_SED = Table(
        results_current_run,
        names=[
            "name",
            "z_sim",
            "Av_host_sim",
            "ext_law",
            "best_z",
            "best_Av",
            "best_slope",
            "best_scaling",
            "best_NHx",
            "zphot_68",
            "zphot_68_sup",
            "zphot_68_inf",
            "zphot_95",
            "zphot_95_sup",
            "zphot_95_inf",
            "zphot_99",
            "zphot_99_sup",
            "zphot_99_inf",
            "Av_68",
            "Av_68_sup",
            "Av_68_inf",
            "Av_95",
            "Av_95_sup",
            "Av_95_inf",
            "Av_99",
            "Av_99_sup",
            "Av_99_inf",
            "beta_68",
            "beta_68_sup",
            "beta_68_inf",
            "beta_95",
            "beta_95_sup",
            "beta_95_inf",
            "beta_99",
            "beta_99_sup",
            "beta_99_inf",
            "norm_68",
            "norm_68_sup",
            "norm_68_inf",
            "norm_95",
            "norm_95_sup",
            "norm_95_inf",
            "norm_99",
            "norm_99_sup",
            "norm_99_inf",
            "NHx_68",
            "NHx_68_sup",
            "NHx_68_inf",
            "NHx_95",
            "NHx_95_sup",
            "NHx_95_inf",
            "NHx_99",
            "NHx_99_sup",
            "NHx_99_inf",
            "best_chi2",
            "mean_proba",
            "sum_proba",
            "mean_acc",
            "nb_bands",
            "nb_detection",
            "band_detected",
            "bic",
            "bic2",
            "chi2_reduced",
        ],
        dtype=(
            "S10",
            "f8",
            "f8",
            "S10",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "f8",
            "i2",
            "i2",
            "S10",
            "f8",
            "f8",
            "f8",
        ),
    )
    return result_1_SED, best_chi2_val
