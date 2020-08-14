#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d
from pyGRBz import constants as cc
import imp


def resample(x, y, x_new, y_min=None, y_max=None):
    """
    Extrapole a given set of data to the new desired range x_new

    Parameters:
    ------------
    x : array
        x-axis values

    y : array
        y-axis values

    x_new : array
        new x-axis range

    y_min : float (optional)
        set the lowest value of y to y_min

    y_max : float (optional)
        set the highest value of y to y_max

    Return
    ---------

    y_new : array
        y values corresponding to the x_new values
    """

    xx = []
    yy = []

    if x_new[0] <= x[0]:
        xx.append(x_new[0] - (x_new[1] - x_new[0]))
        yy.append(0.0)  # set y to 0 outside the given x range

    xx.extend(x)
    yy.extend(y)

    if x_new[-1] >= x[-1]:
        # First creation of a new point equal to zero,
        # with the same step as before
        xx.append(x[-1] + (x[-1] - x[-2]))
        yy.append(0.0)
        # Then set y=0 until the end
        xx.append(x_new[-1] + (x_new[-1] - x_new[-2]))
        yy.append(0.0)

    xx = np.array(xx)
    yy = np.array(yy)

    f = interp1d(xx, yy, kind="linear")
    y_new = f(x_new)

    if y_min is not None and y_max is not None:
        for i in range(len(y_new)):
            if y_new[i] < y_min:
                y_new[i] = y_min
            if y_new[i] > y_max:
                y_new[i] = y_max

    return y_new


def lambda_to_nu(wavelength):
    """Convert wavelength (A) to frequency (Hz)

    Parameters
    ----------
    wavelength: float or array of floats
        The wavelength(s) in Angstrom.

    Returns
    -------
    nu: float or array of floats
        The frequency(ies) in Hz.

    """
    return cc.c_light_m_s / (wavelength * 1.0e-10)


def nu_to_lambda(frequency):
    """Convert frequency (Hz) to wavelength (A)

    Parameters
    ----------
    frequency: float or numpy.array of floats
        The frequency(ies) in Hz.

    Returns
    -------
    wavelength: float or numpy.array of floats
        The wavelength(s) in Angstrom

    """
    return 1.0e-10 * cc.c_light_m_s / frequency


def flambda_to_fnu(wavelength, flambda):
    """
    Convert a Flambda vs lambda spectrum to Fv vs lambda

    Parameters
    ----------
    wavelength: list-like of floats
        The wavelengths in A.
    flambda: list-like of floats
        Flambda flux density in erg/s/cm2/A

    Returns
    -------
    fnu: array of floats
        The Fν flux density in erg/s/cm2/Hz

    """
    # wavelength = np.array(wavelength, dtype=float)
    # flambda = np.array(flambda, dtype=float)

    # Factor 1e-10 is to switch from A to m (only one because the other A
    # wavelength goes with the Flambda in erg/s/cm2/A
    fnu = flambda * wavelength * wavelength / cc.c_light_m_s * 1e-10

    return fnu


def fnu_to_flambda(wavelength, fnu):
    """
    Convert a Fv vs lambda spectrum to Flambda vs lambda

    Parameters
    ----------
    wavelength: list-like of floats
        The wavelengths in A.
    fnu: array of floats
        The Fν flux density in erg/s/cm2/Hz

    Returns
    -------
    flambda: list-like of floats
        Flambda flux density in erg/s/cm2/A

    """
    # wavelength = np.array(wavelength, dtype=float)
    # fnu = np.array(fnu, dtype=float)

    # Factor 1e10 is to switch from nm to m (only one because the other nm
    # wavelength goes with the Flambda in erg/s/cm2/A).
    flambda = fnu / wavelength / wavelength * cc.c_light_m_s * 1e10

    return flambda


def flambda_to_fJy(wavelength, flambda):
    """
    Convert a Flambda vs lambda spectrum to FJy vs lambda

    Parameters
    ----------
    wavelength: list-like of floats
        The wavelengths in Angstrom
    flambda: list-like of floats
        Flambda flux density in erg/s/cm2/A

    Returns
    -------
    fJy: array of floats
        The FJy flux density in Jy

    """
    # wavelength = np.array(wavelength, dtype=float)
    # flambda = np.array(flambda, dtype=float)

    # Factor 1e+23 is to switch from erg/s/cm2/Hz to Jy
    # Factor 1e-10 is to switch from A to m (only one because the other nm
    # wavelength goes with the Flambda in erg/s/cm2/A).
    fJy = 1e23 * 1e-10 * flambda * wavelength * wavelength / cc.c_light_m_s

    return fJy


def fJy_to_flambda(wavelength, fJy):
    """
    Convert a FJy vs lamnda spectrum to Flambda vs lambda

    Parameters
    ----------
    wavelength: list-like of floats
        The wavelengths in Angstrom.
    fJy: list-like of floats
        The FJy flux density in Jy

    Returns
    -------
    flambda: array of floats
        Flambda flux density in erg/cm2/s/A

    """
    # wavelength = np.array(wavelength, dtype=float)
    # fJy = np.array(fJy, dtype=float)

    # Factor 1e-23 is to switch from Jy to erg/cm2/s/Hz
    # Factor 1e+10 is to switch from A to m
    flambda = 1e-23 * 1e10 * fJy / (wavelength * wavelength) * cc.c_light_m_s

    return flambda


def fJy_to_fnu(fJy):
    """
    Convert a FJy vs lambda spectrum to Flambda vs lambda

    Parameters
    ----------
    fJy: list-like of floats
        The Fν flux density in Jy

    Returns
    -------
    fnu: array of floats
        Fv flux density in erg/s/cm2/Hz

    """
    # fJy = np.array(fJy, dtype=float)

    fnu = 1e23 * fJy

    return fnu


def fnu_to_fJy(fnu):
    """
    Convert a Fν vs lambda spectrum to FJy vs lambda

    Parameters
    ----------
    fJY: list-like of floats
        The Fν flux density in Jy

    Returns
    -------
    fnu: array of floats
        Fv flux density in erg/s/cm2/Hz

    """
    # wavelength = np.array(wavelength, dtype=float)
    # fnu = np.array(fJy, dtype=float)

    # Factor 1e-29 is to switch from Jy to W/m²/Hz
    # Factor 1e+9 is to switch from m to nm
    fJy = 1e-23 * fnu

    return fJy


def flambda_to_fph(wavelength, flambda):
    """
    Convert a Flambda vs lambda spectrum to Fph vs lambda

    Parameters
    ----------
    wavelength: list-like of floats
        The wavelengths in Angstrom
    flambda: list-like of floats
        Flambda flux density in erg/s/cm2/A

    Returns
    -------
    fph: array of floats
        The Fph flux density in ph/s/cm2/A

    """
    # wavelength = np.array(wavelength, dtype=float)
    # flambda = np.array(flambda, dtype=float)

    # 1e-10 to convert from Angstrom to m
    joules_per_photon = cc.h_c / (wavelength * 1e-10)  # J/ph

    # Factor 1e-7 is to switch from erg/s/cm2/A to J/s/cm2/A

    fph = flambda * 1e-7 / joules_per_photon

    return fph


def fJy_to_fph(wavelength, fJy):
    """
    Convert a FJy vs lamnda spectrum to Flambda vs lambda

    Parameters
    ----------
    wavelength: list-like of floats
        The wavelengths in Angstrom.
    fJy: list-like of floats
        The FJy flux density in Jy

    Returns
    -------
    flambda: array of floats
        Flambda flux density in erg/cm2/s/A

    """
    # wavelength = np.array(wavelength, dtype=float)
    # fJy = np.array(fJy, dtype=float)

    # Factor 1e-23 is to switch from Jy to erg/cm2/s/Hz
    # Factor 1e+10 is to switch from A to m
    fph = fJy * 1e-30 / (wavelength * cc.H_planck)

    return fph


def mag2Jy(info_dict, Mag):
    """Converts a magnitude into flux density in Jy

    Parameters
    -----------
    info_dict: dictionary

    Mag: array or float
        AB or vega magnitude

    Returns
    -------
    fluxJy: array or float
        flux density in Jy

    """
    if info_dict["photometry_system"] == "AB":
        # 1e23 to convert from  erg/s/cm2/Hz to Jansky
        fluxJy = (10 ** (-(Mag + 48.6) / 2.5)) * 1e23

    return fluxJy


def Jy2Mag(info_dict, fluxJy):
    """Converts flux density in Jy into magnitude

    Parameters
    ----------
    info_dict: dictionary

    fluxJy: array or float
        flux density in Jy

    Returns
    -------
    mag : array or float
        magnitude

    """
    if info_dict["photometry_system"] == "AB":
        Mag = -2.5 * (np.log10(fluxJy) - 23.0) - 48.6

    return Mag


def angles_conversion(angle, unit1, unit2):
    """ Returns a degree in radian
    Parameters
    ---------
    angle: float or array

    unit1: string
        unit of the input angle,
        either 'deg', 'rad', 'arcmin' or 'arcsec'

    unit2: string
        unit of the outut angle,
        either 'deg', 'rad', 'arcmin' or 'arcsec'
    Returns
    -------
    angle_conv: float or array
    """

    if unit1 not in ["deg", "rad", "arcmin", "arcsec"]:
        raise ValueError(
            'incorrect units for unit1\n'
            'Correct units are "deg","rad","arcmin" and "arcsec"'
        )
    if unit2 not in ["deg", "rad", "arcmin", "arcsec"]:
        raise ValueError(
            'incorrect units for unit2\n'
            'Correct units are "deg","rad","arcmin" and "arcsec"'
        )

    if unit1 == "deg" and unit2 == "rad":
        angle_conv = angle * np.pi / 180
    elif unit1 == "deg" and unit2 == "arcmin":
        angle_conv = angle * 60.0
    elif unit1 == "deg" and unit2 == "arcsec":
        angle_conv = angle * 3600.0
    elif unit1 == "rad" and unit2 == "deg":
        angle_conv = angle * 180.0 / np.pi
    elif unit1 == "rad" and unit2 == "arcmin":
        angle_conv = angle * 180.0 / np.pi * 60
    elif unit1 == "rad" and unit2 == "arcsec":
        angle_conv = angle * 180.0 / np.pi * 3600
    elif unit1 == "arcmin" and unit2 == "deg":
        angle_conv = angle / 60.0
    elif unit1 == "arcmin" and unit2 == "rad":
        angle_conv = angle / 60.0 * np.pi / 180
    elif unit1 == "arcmin" and unit2 == "arcsec":
        angle_conv = angle * 60.0
    elif unit1 == "arcsec" and unit2 == "deg":
        angle_conv = angle / 3600.0
    elif unit1 == "arcsec" and unit2 == "rad":
        angle_conv = angle / 3600.0 * np.pi / 180
    elif unit1 == "arcsec" and unit2 == "arcmin":
        angle_conv = angle / 60.0
    elif unit1 == unit2:
        angle_conv = angle

    return angle_conv


def plot_colorfilter(band):
    """ Associate a color to a given filter for nice plotting
    Parameters
    ----------
    band: string
        filter band ie 'u','g',...

    Returns
    -------
    band_color: string
        color associated with the band filter ie 'u' with blue
    """

    if band == "u" or band == "B":
        color_band = "purple"
    elif band == "g" or band == "V":
        color_band = "mediumspringgreen"
    elif band == "r" or band == "R":
        color_band = "red"
    elif band == "i" or band == "I":
        color_band = "orange"
    elif band == "zs":
        color_band = "salmon"
    elif band == "z" or band == "Z":
        color_band = "grey"
    elif band == "y":
        color_band = "chocolate"
    elif band == "Y":
        color_band = "orange"
    elif band == "J":
        color_band = "maroon"
    elif band == "H":
        color_band = "black"
    elif band == "K" or band == "Ks":
        color_band = "green"

    return color_band


def mean_efficiency_passband(info_dict, wavelength, passband):
    """ Computes the mean transmission of a given passband

    Parameters
    -----------

    info_dict: dictionnary

    wavelength : array
        wavelengths in angstrom
    passband : array
        transmission of the passband (between 0 and 1)

    Returns
    --------
    mean_trans_passband: float
        mean transmission of the given passband

    """
    cuton_wvl = info_dict["cuton_wvl"]
    cutoff_wvl = info_dict["cutoff_wvl"]
    w = np.where((wavelength > cuton_wvl) & (wavelength < cutoff_wvl))
    mean_trans_passband = np.mean(passband[w])
    return mean_trans_passband


def column(matrix, i):
    return [row[i] for row in matrix]


def fun_trapz(x, y, dx=None):
    """
    Compute the trapeze integration using np.dot instead of np.trapz.
    It runs about 2 times faster
    """

    if dx is None:
        dx = np.diff(x)
    return np.dot(dx, y[1:] + y[:-1]) * 0.5


def convAB(wavelength, filter_trans):
    """ Compute the AB magnitude of Vega for the given filter band
    """
    wavelength_vega, flux_vega = sed_vega()  # in (A, erg/s/cm2/A)
    f = interp1d(wavelength_vega, flux_vega, kind="linear")
    flux_vega_resampled = f(wavelength)
    flux_vega_resampled = flambda_to_fJy(wavelength,
                                         flux_vega_resampled)
    convAB = -2.5 * np.log10(np.sum(flux_vega_resampled *
                                    filter_trans / wavelength *
                                    np.gradient(wavelength),
                                    axis=0) /
                             np.sum(3631 * filter_trans / wavelength *
                                    np.gradient(wavelength), axis=0))
    return convAB


def sed_vega():
    """
    This function stores the SED of Vega, used for calculation of magnitudes
    on the Vega system. The Vega SED used is taken from Bohlin 2007
    (http://adsabs.harvard.edu/abs/2007ASPC..364..315B), and is taken
    from the STScI CALSPEC library
    (http://www.stsci.edu/hst/observatory/cdbs/calspec.html).

    Parameters
    ----------
    wvl: array
         wavelength in nm at which the flux has to be computed

    Return
    ------
    wavelength: array
                wavelength in Angstrom

    Flux: array
          Flux of Vega in erg/cm2/s/A

    """
    # If code arrived here, it means that the pyGRBz package
    # has already been installed so no need to use try except to check
    _, path, _ = imp.find_module("pyGRBz")
    VEGA_PATH = path + "/data/bohlin2006_Vega.dat"  # from HST CALSPEC

    inFile = open(VEGA_PATH, "r")
    lines = inFile.readlines()

    wavelength = []
    flux = []
    for line in lines:
        if line[0] != "#" and len(line) > 3:
            bits = line.split()
            flux.append(float(bits[1]))
            wavelength.append(float(bits[0]))

    wavelength = np.array(wavelength)
    flux = np.array(flux, dtype=np.float64)

    return [wavelength, flux]
