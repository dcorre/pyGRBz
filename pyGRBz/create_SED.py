#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import erf
import sys
import time
from multiprocessing import Pool
from iminuit import Minuit, describe
from iminuit.util import make_func_code
import emcee
from astropy.table import Table, vstack
from astropy.io import ascii
from pyGRBz.utils import mag2Jy, Jy2Mag
from pyGRBz.fluxes import compute_model_flux, BPL_lc, SPL_lc
from pyGRBz.plotting import (
    plot_lc_fit_check,
    plot_sed,
    plot_mcmc_evolution,
    plot_triangle,
    plot_mcmc_fit,
)


class Chi2Functor_lc:
    def __init__(self, f, t, y, yerr):
        # def __init__(self,f,wvl,y):
        self.f = f
        self.t = t
        self.y = y
        self.yerr = yerr
        f_sig = describe(f)
        # this is how you fake function
        # signature dynamically
        # docking off independent variable
        self.func_code = make_func_code(f_sig[1:])
        # this keeps np.vectorize happy
        self.func_defaults = None
        # print (make_func_code(f_sig[1:]))

    def __call__(self, *arg):
        # notice that it accept variable length
        # positional arguments
        chi2 = sum(
            ((y - self.f(t, *arg)) ** 2 / yerr ** 2)
            for t, y, yerr in zip(self.t, self.y, self.yerr)
        )
        # chi2 = sum((y-self.f(wvl,*arg))**2 for wvl,y in zip(self.wvl,self.y))
        return chi2


def fit_lc(observations, grb_info, model, method="best", print_level=0):
    """
    Fit the lightcurve in order to get a flux and its uncertainty at each time
    The fit is performed for each band separetely
    """
    grb_ref = []
    band_list = []
    telescope_list = []
    nb_obs = []
    F0_list = []
    norm_list = []
    alpha_list = []
    alpha1_list = []
    alpha2_list = []
    t1_list = []
    t0_list = []
    s_list = []
    chi2_list = []

    # Go through each grb
    for obs_table in observations.group_by("Name").groups:
        mask = grb_info["name"] == obs_table["Name"][0]

        # Fit light curve for each band of a given telescope
        for band_table in obs_table.group_by(["telescope", "band"]).groups.keys:
            mask2 = (obs_table["band"] == band_table["band"]) & (
                obs_table["telescope"] == band_table["telescope"]
            )
            time = obs_table["time_since_burst"][mask2]
            y = obs_table[mask2]["flux_corr"]
            yerr_ = obs_table[mask2]["flux_corr_err"]

            # -------Guess initial values-----------
            F0_guess = y[0]

            # Search for extremum
            idx = np.argmax(y)
            if (idx < len(y) - 1) and (idx > 0):
                t1_guess = time[idx]
                limit_t1_guess = (0.1 * t1_guess, 10 * t1_guess)
            else:
                idx = np.argmin(y)
                if (idx > 0) and (idx < len(y) - 1):
                    t1_guess = time[idx]
                    limit_t1_guess = (0.1 * t1_guess, 10 * t1_guess)
                else:
                    t1_guess = time[0]
                    limit_t1_guess = (0, None)
            norm_guess = 1

            if model == "BPL":
                chi2_func = Chi2Functor_lc(BPL_lc, time, y, yerr_)
                kwdarg = dict(
                    F0=F0_guess,
                    norm=norm_guess,
                    alpha1=-0.5,
                    alpha2=0.5,
                    t1=t1_guess,
                    s=1,
                )

            elif model == "SPL":
                chi2_func = Chi2Functor_lc(SPL_lc, time, y, yerr_)
                kwdarg = dict(
                    F0=F0_guess,
                    norm=norm_guess,
                    alpha=1,
                    t0=t1_guess,
                )
            # print (describe(chi2_func))
            else:
                sys.exit(
                    'Error: "%s" model for fitting the' % model,
                    "light curve unknown." ' It should be either "BPL" or "SPL"',
                )

            m = Minuit(chi2_func, **kwdarg)
            # assign print_level
            m.print_level = print_level

            # Fixed parameters
            m.fixed["F0"] = True
            m.fixed["norm"] = True
            if model == "BPL":
                m.fixed["alpha1"] = False
                m.fixed["alpha2"] = False
                m.fixed["t1"] = False
                m.fixed["s"] = False
            elif model == "SPL":
                m.fixed["alpha"] = False
                m.fixed["t0"] = True

            # Set limits to parameter
            m.limits["norm"] = (0.1, 10)
            if model == "BPL":
                m.limits["alpha1"] = (-3, 0)
                m.limits["alpha2"] = (0, 3)
                m.limits["t1"] = (0, None)
                m.limits["s"] = (0.01, 20)
            elif model == "BPL":
                m.limits["alpha"] = (-10, 10)
                m.limits["t0"] = (0, None)

            m.strategy = 1
            # m.migrad(nsplit=1,precision=1e-10)
            m.migrad()
            # print (band)
            if print_level > 0:
                print("Valid Minimum: %s " % str(m.migrad_ok()))
                print(
                    "Is the covariance matrix accurate:" "%s" % str(m.matrix_accurate())
                )

            grb_ref.append(grb_info["name"][mask][0])
            band_list.append(band_table["band"])
            telescope_list.append(band_table["telescope"])
            nb_obs.append(len(y))
            F0_list.append(m.values["F0"])
            norm_list.append(m.values["norm"])
            chi2_list.append(m.fval)
            if model == "SPL":
                alpha_list.append(m.values["alpha"])
                t0_list.append(m.values["t0"])
            elif model == "BPL":
                alpha1_list.append(m.values["alpha1"])
                alpha2_list.append(m.values["alpha2"])
                t1_list.append(m.values["t1"])
                s_list.append(m.values["s"])

        if method == "best":
            min_obs = 2

            for i in range(len(nb_obs)):
                # If few points take the parameters of the fit of the band
                # with most observations. It assumes achromatic evolution
                if nb_obs[i] < min_obs:

                    # Find band with most observations
                    idx_max_obs = np.argmax(nb_obs)
                    best_band = band_list[np.argmax(nb_obs)]

                    if model == "SPL":
                        alpha_list[i] = alpha_list[idx_max_obs]

                    elif model == "BPL":
                        alpha1_list[i] = alpha1_list[idx_max_obs]
                        alpha2_list[i] = alpha2_list[idx_max_obs]
                        t1_list[i] = t1_list[idx_max_obs]
                        s_list[i] = s_list[idx_max_obs]

    # create astropy table as output
    if model == "BPL":
        lc_fit_params = Table(
            [
                grb_ref,
                telescope_list,
                band_list,
                F0_list,
                norm_list,
                alpha1_list,
                alpha2_list,
                t1_list,
                s_list,
                chi2_list,
            ],
            names=[
                "name",
                "telescope",
                "band",
                "F0",
                "norm",
                "alpha1",
                "alpha2",
                "t1",
                "s",
                "chi2",
            ],
        )
    elif model == "SPL":
        lc_fit_params = Table(
            [
                grb_ref,
                telescope_list,
                band_list,
                F0_list,
                norm_list,
                alpha_list,
                t0_list,
                chi2_list,
            ],
            names=["name", "telescope", "band", "F0", "norm", "alpha", "t0", "chi2"],
        )
    """
    if method == 'best':
        # If few points take the parameters of the best fit.
        # It assumes achromatic evolution
        mask = lc_params['band'][np.argmax(lc_params['chi2'])]
    """

    return lc_fit_params


def extract_seds(
    observations,
    grb_info,
    plot=True,
    model="PL",
    method="ReddestBand",
    time_SED=1,
    output_dir="results/",
    filename_suffix="",
):
    """
    Extracts the SED at a given time for the given lightcurves
    """
    # Sort data by ascending eff. wavelength
    observations.sort(["Name", "eff_wvl", "time_since_burst"])
    # If data already in sed format
    mask_sed = grb_info["type"] == "sed"
    if mask_sed.any():
        mask_sed2 = np.array([False] * len(observations["Name"]))
        for i in range(np.sum(mask_sed)):
            mask_sed2[mask_sed2 == False] = (
                observations["Name"][~mask_sed2] == grb_info["name"][mask_sed][i]
            )
        seds = observations[mask_sed2].copy()

    # If data in light curve format
    mask_lc = grb_info["type"] == "lc"
    if mask_lc.any():
        mask_lc2 = np.array([False] * len(observations["Name"]))
        for i in range(np.sum(mask_lc)):
            mask_lc2[mask_lc2 == False] = (
                observations["Name"][~mask_lc2] == grb_info["name"][mask_lc][i]
            )

        lc_fit_params = fit_lc(observations[mask_lc2], grb_info[mask_lc], model)
        # print (lc_fit_params)
        if plot:
            plot_lc_fit_check(
                observations[mask_lc2],
                grb_info[mask_lc],
                lc_fit_params,
                model,
                plot,
                output_dir=output_dir,
                filename_suffix=filename_suffix,
            )

        name_sed = []
        band_list = []
        band_width_list = []
        sys_response_list = []
        wvl_eff = []
        tel_name = []
        sed_flux = []
        sed_flux_err = []
        mag_ext_list = []
        sed_flux_corr = []
        sed_fluxerr_corr = []
        time_sed_list = []
        flux_unit = []
        zp = []
        detection = []
        convert_dict = {"photometry_system": "AB"}

        for obs_table in observations[mask_lc2].group_by("Name").groups:

            if method == "ReddestBand":
                # Find reddest band
                reddest_band = obs_table["band"][np.argmax(obs_table["eff_wvl"])]
                print("reddest band: %s" % reddest_band)

                # Maximum flux in reddest band in the observation set
                idx = np.argmax(
                    obs_table[obs_table["band"] == reddest_band]["flux_corr"]
                )
                time_sed = obs_table[obs_table["band"] == reddest_band][
                    "time_since_burst"
                ][idx]

            elif method == "fixed":
                # Extract the sed at the given time
                time_sed = time_SED

            # print (time_sed)

            for tel in obs_table.group_by(["telescope", "band"]).groups.keys:
                min_obs = 1
                # print (tel)
                # print (obs_table[obs_table['band'] == tel['band']])
                # Do not use bands with only one point.
                # Can be used if achromatic assumption
                if len(obs_table[obs_table["band"] == tel["band"]]) <= min_obs:
                    continue

                mask2 = (
                    (lc_fit_params["name"] == obs_table["Name"][0])
                    & (lc_fit_params["band"] == tel["band"])
                    & (lc_fit_params["telescope"] == tel["telescope"])
                )
                mask3 = (obs_table["band"] == tel["band"]) & (
                    obs_table["telescope"] == tel["telescope"]
                )

                if model == "BPL":
                    flux = BPL_lc(
                        time_sed,
                        float(lc_fit_params["F0"][mask2]),
                        float(lc_fit_params["norm"][mask2]),
                        float(lc_fit_params["alpha1"][mask2]),
                        float(lc_fit_params["alpha2"][mask2]),
                        float(lc_fit_params["t1"][mask2]),
                        float(lc_fit_params["s"][mask2]),
                    )
                elif model == "SPL":
                    flux = SPL_lc(
                        time_sed,
                        float(lc_fit_params["F0"][mask2]),
                        float(lc_fit_params["t0"][mask2]),
                        float(lc_fit_params["norm"][mask2]),
                        float(lc_fit_params["alpha"][mask2]),
                    )

                # Estimate the error with the closest data point
                idx = np.argmin((time_sed - obs_table["time_since_burst"][mask3]) ** 2)
                fluxerr = obs_table["flux_corr_err"][mask3][idx]

                #  Check whether it is a detection or upper limit at this time
                if time_sed > obs_table["time_since_burst"][mask3][idx]:
                    detected = obs_table["detection"][mask3][idx]
                else:
                    detected = obs_table["detection"][mask3][idx - 1]

                name_sed.append(obs_table["Name"][0])
                band_list.append(tel["band"])
                band_width_list.append(obs_table["band_width"][mask3][0])
                sys_response_list.append(obs_table["sys_response"][mask3][0])
                wvl_eff.append(obs_table["eff_wvl"][mask3][0])
                sed_flux_corr.append(flux)
                sed_fluxerr_corr.append(fluxerr)
                time_sed_list.append(time_sed)
                # sed_mag.append(Jy2Mag(convert_dict, flux * 1e-6))
                # sed_magerr.append(2.5 / ((flux) * np.log(10)) * fluxerr)
                sed_flux.append(Jy2Mag(convert_dict, flux * 1e-6))
                sed_flux_err.append(2.5 / ((flux) * np.log(10)) * fluxerr)
                mag_ext_list.append(obs_table["ext_mag"][mask3][0])
                flux_unit.append(obs_table["flux_unit"][mask3][0])
                zp.append(obs_table["zp"][mask3][0])
                tel_name.append(obs_table["telescope"][mask3][0])
                # detection.append(obs_table['detection'][mask3][idx])
                detection.append(detected)

        # create astropy table
        seds_extracted = Table(
            [
                name_sed,
                time_sed_list,
                band_list,
                sed_flux,
                sed_flux_err,
                zp,
                flux_unit,
                detection,
                tel_name,
                wvl_eff,
                band_width_list,
                sys_response_list,
                mag_ext_list,
                sed_flux_corr,
                sed_fluxerr_corr,
            ],
            names=[
                "Name",
                "time_since_burst",
                "band",
                "flux",
                "flux_err",
                "zp",
                "flux_unit",
                "detection",
                "telescope",
                "eff_wvl",
                "band_width",
                "sys_response",
                "ext_mag",
                "flux_corr",
                "flux_corr_err",
            ],
        )
        seds_extracted["time_since_burst"].unit = "s"
        # seds['flux'].unit='microJy'
        # seds['flux_err'].unit='microJy'
        seds_extracted["eff_wvl"].unit = "Angstrom"
        seds_extracted["band_width"].unit = "Angstrom"

        # dealing with non detection
        mask = seds_extracted["detection"] == -1
        if mask.any():
            seds_extracted["flux_corr_err"][mask] = seds_extracted["flux_corr"][mask]
            seds_extracted["flux_corr"][mask] = seds_extracted["flux_corr"][mask]
            seds_extracted["flux"][mask] = Jy2Mag(
                convert_dict, seds_extracted["flux_corr"][mask] * 1e-6
            )
            seds_extracted["flux_err"][mask] = (
                seds_extracted["flux_corr_err"][mask]
                * 2.5
                / np.log(10)
                / seds_extracted["flux_corr"][mask]
            )

    if mask_sed.any() and mask_lc.any():
        seds = vstack([seds, seds_extracted], join_type="outer")
    elif mask_lc.any() and not mask_sed.any():
        seds = seds_extracted.copy()
    # print ("extracted seds")
    # print (seds)
    seds.sort(["Name", "eff_wvl"])

    plot_sed(
        seds,
        grb_info,
        plot,
        model,
        output_dir=output_dir,
        filename_suffix=filename_suffix,
    )
    return seds
