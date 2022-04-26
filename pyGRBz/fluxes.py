#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import time
import numpy as np
from pyGRBz.extinction_correction import sed_extinction


def SPL_lc(t, F0, t0, norm, alpha):
    """Temporal evolution as single power law for Light Curve"""
    return norm * F0 * (t / t0) ** (-alpha)


def BPL_lc(t, F0, norm, alpha1, alpha2, t1, s):
    """Temporal evolution as broken power law for Light Curve"""
    return (
        norm
        * F0
        * ((t / t1) ** (-s * alpha1) + (t / t1) ** (-s * alpha2)) ** (-1.0 / s)
    )


def SPL_sed(wvl, F0, wvl0, norm, beta):
    """Spectral evolution as single power law for SED"""
    return norm * F0 * (wvl / wvl0) ** beta


def BPL_sed(wvl, F0, norm, beta1, beta2, wvl1, s):
    """Spectral evolution as broken power law for SED"""
    return (
        norm
        * F0
        * ((wvl / wvl1) ** (s * beta1) + (wvl / wvl1) ** (s * beta2)) ** (1.0 / s)
    )


def compute_model_flux(
    wvl, F0, wvl0, norm, beta, z, Av, NHx, ext_law, Host_dust, Host_gas, igm_att
):
    """Compute model flux over filter bands"""

    Flux = SPL_sed(wvl, F0, wvl0, norm, beta) * sed_extinction(
        wvl,
        z,
        Av,
        NHx=NHx,
        ext_law=ext_law,
        Host_dust=Host_dust,
        Host_gas=Host_gas,
        igm_att=igm_att,
    )
    return Flux


def compute_model_integrated_flux(
    wavelength,
    sys_response,
    F0,
    wvl0,
    norm,
    beta,
    z,
    Av,
    NHx,
    ext_law,
    Host_dust,
    Host_gas,
    igm_att,
):
    """Compute and integrate model flux over filter bands"""
    # tt0 = time.time()
    flux = compute_model_flux(
        wavelength,
        F0,
        wvl0,
        norm,
        beta,
        z,
        Av,
        NHx,
        ext_law,
        Host_dust,
        Host_gas,
        igm_att,
    )

    integrated_flux = np.trapz(flux * sys_response, x=wavelength, axis=1) / (
        np.trapz(sys_response, x=wavelength, axis=1)
    )

    # dwvl = np.gradient(wavelength)
    # dwvl = np.ediff1d(wavelength)
    # dwvl = np.append(dwvl, dwvl[-1])
    # integrated_flux = np.sum(flux[:-1] * sys_response[:,:-1] *dwvl, axis=1) / (
    #        np.sum(sys_response[:,:-1]*dwvl, axis=1)
    # )

    # tt1 = time.time()

    # print ("Summing flux in {:.2e}".format(tt1-tt0))
    # print ("Summing flux 2 in {:.2e}".format(tt2-tt1))
    return integrated_flux
