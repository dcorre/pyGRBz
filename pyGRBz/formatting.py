# -*- coding: utf-8 -*-
import numpy as np
from pyGRBz.utils import mag2Jy, convAB
from astropy.table import Table, Column
from pyGRBz.extinction_correction import correct_MW_ext
from pyGRBz.io_grb import load_telescope_transmissions
from pyGRBz.utils import resample
from scipy.interpolate import interp1d

import imp


def set_wavelength(data, path):
    """First initialisation of the wavelength array"""
    wvl_min = 1e6
    wvl_max = 0.0
    wvl_min_X = 1e6
    wvl_max_X = 0.0
    X_data = False
    # Sort the telescope used
    for tel in data.group_by(["telescope", "band"]).groups.keys:
        # Import the filter throughput curve only once if filter used several
        # times (for a light curve for instance)

        # Import the throughput curve
        wvl, trans = load_telescope_transmissions(
            {"telescope": tel["telescope"], "band": tel["band"], "path": path},
            None,
            resamp=False,
        )
        wvl = np.array(wvl)
        trans = np.array(trans)
        mask = trans > 0.02
        wvl_start = np.min(wvl[mask])
        wvl_end = np.max(wvl[mask])
        if wvl_start > 800:
            if wvl_start < wvl_min:
                wvl_min = wvl_start
            if wvl_end > wvl_max:
                wvl_max = wvl_end
        # Xray domain
        else:
            X_data = True
            if wvl_start < wvl_min_X:
                wvl_min_X = wvl_start
            if wvl_end > wvl_max_X:
                wvl_max_X = wvl_end

    # Take 20% to be sure as it is not resampled yet and will be optimised later
    wavelength = np.arange(0.8 * wvl_min, 1.2 * wvl_max, 50).astype(np.float)
    # if Xray data
    if X_data:
        wvl_X = np.arange(0.8 * wvl_min_X, 1.2 * wvl_max_X, 1)
        wavelength = np.append(wvl_X, wavelength)

    return wavelength


def load_sys_response(data, path, wvl_step, wvl_step_X):
    """Load the system throuput curves for each filter in the data

    Returns
    -------
    sys_rep: astropy Table
    """
    wavelength = set_wavelength(data, path)

    # works for constant wvl_step
    dwvl = np.gradient(wavelength)

    sys_res = []
    tel_name = []
    tel_band = []
    wvl_eff = []
    width = []
    zp = []

    # Sort the telescope used
    for tel in data.group_by(["telescope", "band"]).groups.keys:
        # Import the filter throughput curve only once if filter used several
        # times (for a light curve for instance)
        tel_name.append(tel["telescope"])
        tel_band.append(tel["band"])

        # Import the throughput curve
        _, filter_trans = load_telescope_transmissions(
            {"telescope": tel["telescope"], "band": tel["band"], "path": path},
            wavelength,
        )

        # calculate the effective wavelength
        a = np.trapz(wavelength * filter_trans, wavelength)
        b = np.trapz(filter_trans, wavelength)
        wvl_eff.append(a / b)

        # Calculate the width of the band
        mask = filter_trans > 0.02 * max(filter_trans)

        width.append(wavelength[mask][-1] - wavelength[mask][0])

        sys_res.append(filter_trans)

    sys_res_table = Table(
        [tel_name, tel_band, wvl_eff, width, sys_res],
        names=["telescope", "band", "wvl_eff", "band_width", "sys_response"],
    )

    # Sort the table by telescope names and ascending eff. wavelength
    sys_res_table.sort(["telescope", "wvl_eff"])

    new_wvl, sys_res_table = optimise_wvl_range(
        sys_res_table, wavelength, wvl_step=50, wvl_step_X=10
    )
    return new_wvl, sys_res_table


def optimise_wvl_range(system_response, wavelength, wvl_step, wvl_step_X):
    """Keep only wavelength of interest"""

    new_wvl = []
    wvl_interval = []
    wvl_min = 1e6
    wvl_max = 0.0
    wvl_min_X = 1e6
    wvl_max_X = 0.0
    X_data = False
    # Get wvl_start and wvl_end for each filter
    for sys_res in system_response:
        wvl_start = 0.97 * (sys_res["wvl_eff"] - sys_res["band_width"] / 2)
        wvl_end = 1.03 * (sys_res["wvl_eff"] + sys_res["band_width"] / 2)
        # UV-VIsible-IR domain
        if wvl_start > 800:
            if wvl_start < wvl_min:
                wvl_min = wvl_start
            if wvl_end > wvl_max:
                wvl_max = wvl_end
        # Xray domain
        else:
            X_data = True
            if wvl_start < wvl_min_X:
                wvl_min_X = wvl_start
            if wvl_end > wvl_max_X:
                wvl_max_X = wvl_end

    # Create new wavlength range with 50 angstroms step
    new_wvl = np.arange(wvl_min, wvl_max + wvl_step, wvl_step)
    if X_data:
        # Create wavlength range for X data with step of 10 angstroms
        wvl_range_X = np.arange(wvl_min_X, wvl_max_X + wvl_step_X, wvl_step_X)
        new_wvl = np.append(wvl_range_X, new_wvl)

    # new_response = np.zeros((len(sys_res), len(new_wvl)))
    new_response = []
    # import matplotlib.pyplot as plt
    # plt.figure()
    # Compute filter transmission associated to new wavelength
    for i, sys_res in enumerate(system_response):
        trans = resample(wavelength, system_response["sys_response"][i], new_wvl)
        f = interp1d(wavelength, sys_res["sys_response"], kind="linear")
        trans = f(new_wvl)
        new_response.append(trans)
        # plt.plot(new_wvl, trans)
    # plt.show()

    # Remove wavelength outside filter passband
    sys_res_total = np.sum(new_response, axis=0)
    mask = sys_res_total > 0.001
    new_wvl_cut = new_wvl[mask]
    new_response_cut = []
    for res in new_response:
        new_response_cut.append(res[mask])

    del system_response["sys_response"]
    system_response["sys_response"] = new_response_cut
    return new_wvl_cut, system_response


def formatting_data(
    data, system_response, grb_info, wavelength, dustrecalib="yes", thres_err=0.02
):
    """ """
    try:
        _, path_dust_map, _ = imp.find_module("pyGRBaglow")
    except:
        print("path to pyGRBaglow can not be found.")

    dustmapdir = path_dust_map + "/galactic_dust_maps"

    #  Add filter info to data (throughut curve,eff. wvl and width)
    col_band_width = Column(name="band_width", data=np.zeros(len(data)))
    col_band_effwvl = Column(name="eff_wvl", data=np.zeros(len(data)))
    col_band_zp = Column(name="zeropoint", data=np.zeros(len(data)))
    col_band_sysres = Column(
        name="sys_response", data=np.zeros((len(data), len(wavelength)))
    )
    data.add_columns([col_band_effwvl, col_band_width, col_band_zp, col_band_sysres])

    for table in data.group_by(["telescope", "band"]).groups.keys:
        # print (table)
        mask1 = data["telescope"] == table["telescope"]
        mask1[mask1 == True] = data[mask1]["band"] == table["band"]
        mask2 = system_response["telescope"] == table["telescope"]
        mask2[mask2 == True] = system_response[mask2]["band"] == table["band"]
        # print (system_response[mask3][mask4]['sys_response'])
        # print (system_response[mask3][mask4]['sys_response'][0])

        width = []
        effwvl = []
        sys_res = []
        for i in range(np.sum(mask2)):
            width.append(system_response[mask2]["band_width"][0])
            effwvl.append(system_response[mask2]["wvl_eff"][0])
            sys_res.append(system_response[mask2]["sys_response"][0])
        data["band_width"][mask1] = width
        data["eff_wvl"][mask1] = effwvl
        data["sys_response"][mask1] = sys_res

    # Convert vega magnitudes in AB if needed
    mask1 = data["flux_unit"] == "vega"
    if mask1.any():
        # print ('some vega')

        #  If a Vega-AB correction is present in the file use this value

        if "ABcorr" in data.colnames:
            mask2 = (data["flux_unit"] == "vega") & (~data["ABcorr"].mask)
            if mask2.any():
                # print ('AB corr')
                for table in data[mask2]:
                    mask3 = mask2.copy()

                    mask3[mask3 == True] = data[mask3]["Name"] == table["Name"]
                    mask3[mask3 == True] = (
                        data[mask3]["telescope"] == table["telescope"]
                    )
                    mask3[mask3 == True] = data[mask3]["band"] == table["band"]
                    mask3[mask3 == True] = (
                        data[mask3]["time_since_burst"] == table["time_since_burst"]
                    )

                    newABmag = table["flux"] + table["ABcorr"]
                    photsys = "AB"
                    # substitute the vega magnitudes by AB ones
                    data["flux"][mask3] = newABmag
                    data["flux_unit"][mask3] = photsys

            #  When no AB correction is given in input file, compute it
            mask2 = (data["flux_unit"] == "vega") & (data["ABcorr"].mask)
            if mask2.any():

                # print ('convAB')
                for table in data[mask2]:
                    mask3 = mask2.copy()

                    mask3[mask3 == True] = data[mask3]["Name"] == table["Name"]
                    mask3[mask3 == True] = (
                        data[mask3]["telescope"] == table["telescope"]
                    )
                    mask3[mask3 == True] = data[mask3]["band"] == table["band"]
                    mask3[mask3 == True] = (
                        data[mask3]["time_since_burst"] == table["time_since_burst"]
                    )

                    newABmag = table["flux"] + convAB(wavelength, table["sys_response"])
                    photsys = "AB"
                    # substitute the vega magnitudes by AB ones
                    data["flux"][mask3] = newABmag
                    data["flux_unit"][mask3] = photsys

        else:
            # print ('convAB')
            for table in data[mask1]:
                mask3 = mask1.copy()

                mask3[mask3 == True] = data[mask3]["Name"] == table["Name"]
                mask3[mask3 == True] = data[mask3]["telescope"] == table["telescope"]
                mask3[mask3 == True] = data[mask3]["band"] == table["band"]
                mask3[mask3 == True] = (
                    data[mask3]["time_since_burst"] == table["time_since_burst"]
                )

                newABmag = table["flux"] + convAB(wavelength, table["sys_response"])
                photsys = "AB"
                # substitute the vega magnitudes by AB ones
                data["flux"][mask3] = newABmag
                data["flux_unit"][mask3] = photsys

    # Correct for galactic extinction
    data = correct_MW_ext(
        data, grb_info, wavelength, dustmapdir=dustmapdir, recalibration=dustrecalib
    )

    # Add Flux to the seds
    flux = np.zeros(len(data))
    flux_err = np.zeros(len(data))

    mask = data["flux_unit"] == "AB"
    convert_dict = {"photometry_system": "AB"}
    flux[mask] = mag2Jy(convert_dict, data["flux"][mask] - data["ext_mag"][mask]) * 1e6
    # flux_err[mask] = np.array(abs(flux[mask] * -0.4 * np.log(10) * data["flux_err"][mask]))
    flux_err[mask] = np.sqrt(
        abs(flux[mask] * np.log(10) / 2.5 * data["flux_err"][mask])
    )

    mask = data["flux_unit"] == "Jy"
    # Need to add dust correction and express in microJy
    flux[mask] = data["flux"][mask] * 10 ** (0.4 * data["ext_mag"][mask]) * 1e6
    flux_err[mask] = data["flux_err"][mask] * 1e6

    mask = data["flux_unit"] == "mJy"
    # Need to add dust correction and express in microJy
    flux[mask] = data["flux"][mask] * 10 ** (0.4 * data["ext_mag"][mask]) * 1e3
    flux_err[mask] = data["flux_err"][mask] * 1e3

    mask = data["flux_unit"] == "microJy"
    # Need to add dust correction
    flux[mask] = data["flux"][mask] * 10 ** (0.4 * data["ext_mag"][mask])
    flux_err[mask] = data["flux_err"][mask]

    # Check that flux error is at least 2% of the flux, otherwise the fit is given
    # poor results
    for i in range(len(flux)):
        if flux_err[i] / flux[i] < thres_err:
            flux_err[i] = thres_err * flux[i]

    col_flux = Column(name="flux_corr", data=flux, unit="microJy")
    col_flux_err = Column(name="flux_corr_err", data=flux_err, unit="microJy")

    data.add_columns([col_flux, col_flux_err])

    data.sort(["Name", "eff_wvl"])
    return data
