#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os

from pyGRBz.formatting import load_sys_response, formatting_data
from pyGRBz.io_grb import load_observations, load_info_observations
from pyGRBz.create_SED import extract_seds
from pyGRBz.fitting import mcmc
from pyGRBz.plotting import plot_zphot
import imp
import warnings

warnings.filterwarnings("ignore")


class GRB_photoZ:
    """
    Class to manipulate the different photoZ modules.

    """

    def __init__(
        self,
        wvl_step=100,
        wvl_step_X=10,
        thres_err=0.02,
        plot=True,
        output_dir="/results/",
    ):
        """
        Class Constructor.

        """

        try:
            _, path, _ = imp.find_module("pyGRBz")
        except:
            print("path to pyGRBz can not be found.")

        self.wvl_step = wvl_step
        self.wvl_step_X = wvl_step_X
        # If ratio flux_err/flux < thres_err then set flux_err=thres_err*flux
        self.thres_err = thres_err
        self.plot = plot
        self.path = path
        self.output_dir = self.path + output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self, data_dir="/data/sed/", data_name="GRB130606A"):
        """Load observations, either a light curve or sed

        Returns
        -------
        data: astropy Table

        """
        data_dir = self.path + data_dir

        if not isinstance(data_name, list):
            self.filenames = [data_dir + data_name + ".txt"]
        else:
            self.filenames = []
            for i, grbname in enumerate(data_name):
                self.filenames.append(data_dir + grbname + ".txt")

                #  Create directories for results
                if not os.path.exists(self.output_dir + grbname):
                    os.makedirs(self.output_dir + grbname)

        # print (self.filenames)

        # Load observations
        self.data = load_observations(self.filenames)
        print("\nObservations:\n {}\n".format(self.data))

        self.grb_info = load_info_observations(self.filenames)
        print("\nInfo about data:\n {}\n".format(self.grb_info))

        # Load the system througput curves
        self.wavelength, self.system_response = load_sys_response(
            self.data,
            path=self.path,
            wvl_step=self.wvl_step,
            wvl_step_X=self.wvl_step_X,
        )

    def formatting(self, dustrecalib="yes"):
        """set the data in the right format"""

        # Add the fluxes to the seds and convert vega in AB
        # magnitudes if needed
        self.data = formatting_data(
            self.data,
            self.system_response,
            self.grb_info,
            self.wavelength,
            dustrecalib=dustrecalib,
            thres_err=self.thres_err,
        )
        print("\nSEDS formatted:\n {}\n".format(self.data))

    def extract_sed(
        self,
        model="SPL",
        method="ReddestBand",
        time_SED=1,
        output_dir="/results/",
        filename_suffix="",
    ):
        """Extract the SED from LC if needed"""

        self.seds = extract_seds(
            self.data,
            self.grb_info,
            plot=self.plot,
            model=model,
            method=method,
            time_SED=time_SED,
            output_dir=self.output_dir,
            filename_suffix=filename_suffix,
        )
        print(
            "\nSEDS:\n {}\n".format(
                self.seds[
                    "Name", "time_since_burst", "band", "flux", "flux_err", "flux_unit"
                ]
            )
        )

    def fit(
        self,
        ext_law="smc",
        Nthreads=1,
        nwalkers=30,
        Nsteps1=300,
        Nsteps2=1000,
        nburn=300,
        Host_dust=True,
        Host_gas=False,
        igm_att="Meiksin",
        clean_data=False,
        plot_all=False,
        plot_deleted=False,
        priors=dict(z=[0, 11], Av=[0, 2], beta=[0, 2], NHx=[0.1, 100], norm=[0, 10]),
        filename_suffix="",
        std_gaussianBall=1e-2,
        adapt_z=True,
    ):
        """Run the MCMC code"""
        results = mcmc(
            self.seds,
            self.grb_info,
            self.wavelength,
            self.plot,
            Nthreads=Nthreads,
            ext_law=ext_law,
            nwalkers=nwalkers,
            Nsteps1=Nsteps1,
            Nsteps2=Nsteps2,
            nburn=nburn,
            std_gaussianBall=std_gaussianBall,
            Host_dust=Host_dust,
            Host_gas=Host_gas,
            igm_att=igm_att,
            clean_data=clean_data,
            plot_all=plot_all,
            plot_deleted=plot_deleted,
            priors=priors,
            output_dir=self.output_dir,
            filename_suffix=filename_suffix,
            adapt_z=adapt_z,
        )
        print(results)

    def plot_zsim_zphot(
        self,
        input_file="best_fits_mcmc",
        output_suffix="testagain",
        sigma=2,
        input_dir="/results/",
        output_dir="/plots/",
        plot=True,
    ):

        plot_zphot(
            input_file,
            output_suffix,
            sigma,
            input_dir=self.path + input_dir,
            output_dir=self.path + output_dir,
            plot=True,
        )


if __name__ == "__main__":
    data_name = ["GRB130606A"]  # ,'GRB130215A']

    fit = GRB_photoZ()
    fit.load_data(data_name=data_name)
    fit.formatting()
    fit.extract_sed()
    fit.fit()
