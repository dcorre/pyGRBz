#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import time
import sys
import numpy as np
from astropy.table import Column
from pyGRBaglow.igm import madau, dla

# Import cythonised code if present
try:
    from pyGRBaglow.igm_cy import meiksin
except:
    from pyGRBaglow.igm import meiksin
try:
    from pyGRBaglow.reddening_cy import Pei92, sne, gas_absorption
except:
    from pyGRBaglow.reddening import Pei92, sne, gas_absorption


def sed_extinction(
    wavelength,
    z,
    Av,
    NHx=1,
    ext_law="smc",
    Host_dust=True,
    Host_gas=False,
    igm_att="Meiksin",
):
    """
    Computes the total transmission per wavelength at a given redshift.
    It can include different mechanism: dust extinction, photoelectric
    absorption both in the host galaxy and in our galaxy; presence of a DLA,
    and photon absorbed or scattered in the IGM in the line of sigth.

    Parameters
    ----------
    wavelength: array
        wavelength in angstroms

    z: float
        redshift

    Av: float
        amount of extinction in the V band

    ext_law: string
        name of the extinction law to use:
        'MW', 'LMC', 'SMC' or 'nodust'

    Host_dust: boolean
        whether to apply dust extinction from the host galaxy

    Host_gas: boolean
        whether to apply photoelectric absorption by metals

    igm_att: string
        name of the IGM extinction model to use:
        'Meiksin' or 'Madau'

    Returns
    -------
    trans_tot: array
        total transmission on the line of sight
    """
    # tt00 = time.time()
    # Make sure wavelength is an array
    wavelength = np.atleast_1d(wavelength)

    Trans_tot = np.ones(len(wavelength), dtype=np.float64)
    # tt0 = time.time()
    # ------------------------------------
    # Calculate extinction along the los
    # -------------------------------------
    if ext_law != "nodust" and Host_dust:
        # Transmission due to host galaxy reddening
        if ext_law == "sne":
            Trans_tot *= sne(wavelength, Av, z, Xcut=True)[1]
        else:
            Trans_tot *= Pei92(wavelength, Av, z, ext_law=ext_law, Xcut=True)[1]
        # tt1 = time.time()

    if Host_gas:
        # Transmission due to host galaxy gas extinction
        Trans_tot *= gas_absorption(wavelength, z, NHx=NHx)
        # tt2 = time.time()

    if igm_att.lower() == "meiksin":
        # tt2 = time.time()
        # Add IGM attenuation using Meiksin 2006 model
        Trans_tot *= meiksin(wavelength / 10.0, z, unit="nm", Xcut=True)
        # tt3 = time.time()
    elif igm_att.lower() == "madau":
        # Add IGM attenuation using Madau 1995 model
        Trans_tot *= madau(wavelength, z, Xcut=True)
        # tt3 = time.time()
    else:
        sys.exit(
            "Model %s for IGM attenuation not known\n"
            'It should be either "Madau" or "Meiksin" ' % igm_att
        )

    # print ("Extinction time dust: {:.2e}s".format(tt1-tt0))
    # print ("Extinction time gas: {:.2e}s".format(tt2-tt1))
    # print ("Extinction time igm: {:.2e}s".format(tt3-tt2))
    # print ("Extinction time total: {:.2e}s".format(tt3-tt00))

    return Trans_tot


def correct_MW_ext(
    data, grb_info, wavelength, recalibration="no", dustmapdir="Default"
):
    """
    Correct the data from extinction occuring in the line of sight in
    the Milky Way.
    """
    # from dustmap import SFD98Map
    from sfdmap import SFDMap
    from astropy.coordinates import SkyCoord
    import extinction
    import pkg_resources

    # from scipy.interpolate import interp1d

    if dustmapdir == "Default":
        #  Get the path to the directory where the dust maps are
        dustmapdir = pkg_resources.resource_filename("pyGRBaglow", "galactic_dust_maps")

    #  Add column in data for the galctic extinction in mag
    col_band_extmag = Column(name="ext_mag", data=np.zeros((len(data))))
    data.add_columns([col_band_extmag])
    # print (data)
    for grb in grb_info.group_by(["name"]).groups:
        mask = data["Name"] == grb["name"]

        if "MW_corrected" in grb.colnames and grb["MW_corrected"][0].lower() == "no":
            # load dust map
            # If original value from  Schlegel, Finkbeiner & Davis (1998)
            # use scaling=1.0
            # For using the recalibration by Schlafly & Finkbeiner (2011)
            # use sacling = 0.86 (default)
            if recalibration == "no":
                m = SFDMap(mapdir=dustmapdir, scaling=1.0)
            elif recalibration == "yes":
                m = SFDMap(mapdir=dustmapdir)

            try:
                #  Coordinates in hmsdms
                grb_coord = SkyCoord(grb["RA_J2000"], grb["DEC_J2000"], frame="icrs")
            except:
                # else assumes that it is given in degrees
                grb_coord = SkyCoord(
                    grb["RA_J2000"], grb["DEC_J2000"], frame="icrs", unit="deg"
                )

            # transfrom coord in (RA, Dec) in degrees in the ICRS
            # (e.g.,"J2000") system
            # Return the E(B-V) reddening in magnitude from the Schlegel map
            # Schlegel et al. 1998 (ApJ 500, 525)
            # E_BV=m.get_ebv((grb_coord.ra,grb_coord.dec))
            E_BV = m.ebv((grb_coord.ra, grb_coord.dec))
            print(
                "\nReddening along the line of sight of"
                "{0:s}: E(B-V) = {1:.3f}\n".format(grb["name"][0], E_BV[0])
            )
            # Compute the extinction according to Cardelli 1989
            Rv = 3.1
            A_lambda = extinction.ccm89(wavelength, Rv * E_BV, Rv)
            # f = interp1d(wavelength, A_lambda, kind="linear")

            dwvl = np.gradient(wavelength)

            # Go line per line in masked data
            # Need to rewrite this part, I do not remember what the masks do.
            for table in data[mask]:
                mask1 = mask.copy()
                mask1[mask1 == True] = data[mask1]["Name"] == table["Name"]
                mask1[mask1 == True] = data[mask1]["telescope"] == table["telescope"]
                mask1[mask1 == True] = data[mask1]["band"] == table["band"]
                mask1[mask1 == True] = (
                    data[mask1]["time_since_burst"] == table["time_since_burst"]
                )

                # Compute extinction in magnitudes for each passband
                extinction_mag = float(
                    -2.5
                    * np.log10(
                        np.sum(
                            table["sys_response"] * np.exp(-A_lambda / 1.086) * dwvl,
                            axis=0,
                        )
                    )
                    + 2.5 * np.log10(np.sum(table["sys_response"] * dwvl, axis=0))
                )
                print(
                    "Galactic extinction in band {0:s} {1:s}: {2:.3f} AB mag".format(
                        table["telescope"], table["band"][0], extinction_mag
                    )
                )
                # Fill the data 'ext_mag' column
                data["ext_mag"][mask1] = extinction_mag

    data.sort(["Name", "eff_wvl"])
    return data
