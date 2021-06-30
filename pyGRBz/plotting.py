#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from astropy.io import ascii
import corner
from pyGRBz.models import Flux_template, SPL_lc, BPL_lc


def plot_lc_fit_check(
    observations,
    grb_info,
    lc_fit_params,
    model,
    plot,
    output_dir="/results/",
    filename_suffix="",
):
    """Plot the fitting light curves"""

    # Go through each grb observations
    for obs_table in observations.group_by("Name").groups:
        # If redshift and Av are provided in the data file use them
        try:
            z_sim = grb_info["z"][obs_table["Name"][0] == grb_info["name"]][0]
        except:
            z_sim = -99
        try:
            mask = obs_table["Name"][0] == grb_info["name"]
            Av_sim = grb_info["Av_host"][mask][0]
        except:
            Av_sim = -99

        # Set color for plots
        cmap = plt.get_cmap("rainbow")
        mask = lc_fit_params["name"] == obs_table["Name"][0]
        color_len = len(lc_fit_params["band"][mask]) + 1
        colors = [cmap(i) for i in np.linspace(0, 1, color_len)]

        plt.figure()
        # sort observations by eff. wavelength, telescope and band.
        # keep just one time
        for i, band_table in enumerate(
            obs_table.group_by(["eff_wvl", "telescope", "band"]).groups.keys
        ):
            # print (band_table)
            # select all observations with same band and telescope
            mask = (obs_table["band"] == band_table["band"]) & (
                obs_table["telescope"] == band_table["telescope"]
            )
            time = obs_table["time_since_burst"][mask]
            # print (time)
            # exptime = obs_table['exptime'][mask]
            xerr = np.ones(len(time)) * 15
            y = obs_table[mask]["flux_corr"]
            # print (y)
            yerr_ = obs_table[mask]["flux_corr_err"]

            # Select the fit parameters for the corresponding band and
            # telescope
            mask2 = (
                (lc_fit_params["name"] == obs_table["Name"][0])
                & (lc_fit_params["band"] == band_table["band"])
                & (lc_fit_params["telescope"] == band_table["telescope"])
            )
            # print (i,obs_table['detection'][mask])

            for t in range(len(time)):
                plt.errorbar(
                    time[t],
                    y[t],
                    xerr=xerr[t],
                    yerr=yerr_[t],
                    uplims=1 - obs_table["detection"][mask][t],
                    label=band_table["telescope"] + " " + band_table["band"],
                    color=colors[i],
                )
            time_fit = np.linspace(time[0], time[-1], 100)
            # print (lc_fit_params['F0'][mask2])
            if model == "BPL":
                plt.plot(
                    time_fit,
                    BPL_lc(
                        time_fit,
                        float(lc_fit_params["F0"][mask2]),
                        float(lc_fit_params["norm"][mask2]),
                        float(lc_fit_params["alpha1"][mask2]),
                        float(lc_fit_params["alpha2"][mask2]),
                        float(lc_fit_params["t1"][mask2]),
                        float(lc_fit_params["s"][mask2]),
                    ),
                    label=band_table["telescope"] + " " + band_table["band"],
                    color=colors[i],
                )
            elif model == "SPL":
                plt.plot(
                    time_fit,
                    SPL_lc(
                        time_fit,
                        float(lc_fit_params["F0"][mask2]),
                        float(lc_fit_params["t0"][mask2]),
                        float(lc_fit_params["norm"][mask2]),
                        float(lc_fit_params["alpha"][mask2]),
                    ),
                    label=band_table["telescope"] + " " + band_table["band"],
                    color=colors[i],
                )
        # plt.gca().invert_yaxis()
        # plt.xlim(obs_table['time_since_burst'][0]-60,
        #          obs_table['time_since_burst'][-1]+90)
        # plt.ylim(0,230)
        # plt.xscale('log')
        # print (time[0],z_sim,Av_sim)
        plt.title(
            "Light curve from T-To=%.0f to T-To=%.0f sec \n z=%.2f, Av_host=%.2f \n %s"
            % (
                np.min(obs_table["time_since_burst"]),
                np.max(obs_table["time_since_burst"]),
                float(z_sim),
                float(Av_sim),
                obs_table["Name"][0],
            )
        )
        plt.xlabel(r"T-T$_{0}$ [seconds]")
        plt.ylabel(r"Flux [$\mu$Jy]")
        # plt.axvline(305,color='red',lw=3)
        # do not duplicate legends and only one point in the label
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), numpoints=1, loc="best")
        plt.grid(True)
        os.makedirs(os.path.join(output_dir, str(obs_table["Name"][0])), exist_ok=True)
        plt.tight_layout()
        plt.savefig(
            output_dir
            + str(obs_table["Name"][0])
            + "/lc_fit_"
            + model
            + filename_suffix
            + ".png"
        )
        if plot:
            plt.show()
        #plt.close("all")


def plot_sed(seds, grb_info, plot, model, output_dir="/results/", filename_suffix=""):
    """Plot the extracted SED"""
    for sed in seds.group_by("Name").groups:
        plt.figure()
        # If redshift and Av are provided in the data file use them
        try:
            z = float(grb_info["z"][grb_info["name"] == sed["Name"][0]][0])
        except:
            z = -99
        try:
            mask = grb_info["name"] == sed["Name"][0]
            Av_host = float(grb_info["Av_host"][mask][0])
        except:
            Av_host = -99

        # Set color for plots
        cmap = plt.get_cmap("rainbow")
        colors = [cmap(i) for i in np.linspace(0, 1, len(sed["band"]) + 1)]

        for i, band_table in enumerate(
            sed.group_by(
                [
                    "eff_wvl",
                    "telescope",
                    "band",
                    "band_width",
                    "flux_corr",
                    "flux_corr_err",
                    "detection",
                ]
            ).groups.keys
        ):
            if band_table["detection"] == 0:
                # arbitrary, only for visualisation
                yerr = 0.2 * band_table["flux_corr"]
            else:
                yerr = band_table["flux_corr_err"]
            plt.errorbar(
                band_table["eff_wvl"],
                band_table["flux_corr"],
                xerr=band_table["band_width"] / 2,
                yerr=yerr,
                uplims=1 - band_table["detection"],
                label=band_table["telescope"] + " " + band_table["band"],
                color=colors[i],
            )
        plt.xlabel(r"$\lambda$ [angstroms]")
        # plt.ylabel(r'Flux ($\mu$Jy)')
        plt.ylabel("Flux (microJy)")
        # plt.xlim(sed["eff_wvl"][0] / 2, sed["eff_wvl"][-1] * 1.5)
        # plt.xlim(0, sed["eff_wvl"][-1] * 1.5)
        # plt.ylim(min(sed["flux_corr"]) - 3, max(sed["flux_corr"]) + 3)
        # plt.ylim(max(sed['mag'])+3,min(sed['mag'])-3)
        plt.title(
            "%s SED extracted at T-To=%.0f sec \n z=%.2f, Av_host=%.2f"
            % (sed["Name"][0], sed["time_since_burst"][0], z, Av_host)
        )
        plt.grid(True)
        plt.legend(numpoints=1)
        plt.yscale("log")
        plt.xscale("log")
        plt.tight_layout()
        plt.savefig(
            output_dir
            + str(sed["Name"][0])
            + "/extracted_sed_"
            + model
            + filename_suffix
            + ".png"
        )
        if plot:
            plt.show()
        #plt.close("all")


def plot_mcmc_evolution(
    samples_corr,
    samples_del,
    nburn,
    ndim,
    ext_law,
    Host_gas,
    Av_sim,
    z_sim,
    name,
    plot,
    plot_deleted,
    output_dir="/test/",
    filename_suffix="",
    priors=dict(z=[0, 11], Av=[0, 2], beta=[0, 2], NHx=[0.1, 100], norm=[0, 10]),
):
    """Plot the chains evolution"""
    i = 0
    fig, axes = plt.subplots(ndim, 1, sharex=True)
    axes[i].plot(samples_corr[:, :, 0].T, color="grey", alpha=0.6)
    axes[i].axvline(x=nburn, linewidth=1, linestyle="--", color="red")
    axes[i].set_ylabel("z")
    axes[i].set_ylim(priors["z"][0], priors["z"][1])
    i += 1
    if ext_law != "nodust":
        axes[i].plot(samples_corr[:, :, 3].T, color="grey", alpha=0.6)
        axes[i].axvline(x=nburn, linewidth=1, linestyle="--", color="red")
        axes[i].set_ylabel(r"$ A_{V} $")
        axes[i].set_ylim(priors["Av"][0], priors["Av"][1])
        i += 1
    axes[i].plot(samples_corr[:, :, 1].T, color="grey", alpha=0.6)
    axes[i].axvline(x=nburn, linewidth=1, linestyle="--", color="red")
    axes[i].set_ylabel(r"$ \beta $")
    axes[i].set_ylim(priors["beta"][0], priors["beta"][1])
    i += 1
    if Host_gas is True:
        if ext_law == "nodust":
            idx = 3
        else:
            idx = 4
        axes[i].plot(samples_corr[:, :, idx].T, color="grey", alpha=0.6)
        axes[i].axvline(x=nburn, linewidth=1, linestyle="--", color="red")
        axes[i].set_ylabel(r"$ NH_{x} $")
        axes[i].set_ylim(priors["NHx"][0], priors["NHx"][1])
        i += 1

    axes[i].plot(samples_corr[:, :, 2].T, color="grey", alpha=0.6)
    axes[i].axvline(x=nburn, linewidth=1, linestyle="--", color="red")
    axes[i].set_ylabel("norm")
    axes[i].set_xlabel("steps")
    # axes[i].set_ylim(priors["norm"][0], priors["norm"][1])

    # axes[i].set_ylim(0,2)
    i = 0
    if plot_deleted and samples_del is not None:
        axes[i].plot(samples_del[:, :, 0].T, color="red", alpha=0.1)
        i += 1
        if ext_law != "nodust":
            axes[i].plot(samples_del[:, :, 3].T, color="red", alpha=0.1)
            i += 1
        axes[i].plot(samples_del[:, :, 1].T, color="red", alpha=0.1)
        i += 1
        if Host_gas is True:
            if ext_law == "nodust":
                idx = 3
            else:
                idx = 4
            axes[i].plot(samples_del[:, :, idx].T, color="red", alpha=0.1)
            i += 1
        axes[i].plot(samples_del[:, :, 2].T, color="red", alpha=0.1)

    fig.gca().annotate(
        "zsim: %.2f    Av_sim: %.2f " % (z_sim, Av_sim),
        xy=(0.5, 1.0),
        xycoords="figure fraction",
        xytext=(0, -5),
        textcoords="offset points",
        ha="center",
        va="top",
    )
    plt.tight_layout()
    fig.savefig(
        output_dir
        + str(name)
        + "/evolution_steps_"
        + ext_law
        + filename_suffix
        + ".png"
    )
    if plot:
        plt.show()
    #plt.close(fig)
    #plt.close("all")


def plot_triangle(
    samples,
    ndim,
    z_sim,
    ext_law,
    Host_gas,
    Av_sim,
    beta_sim,
    name,
    plot,
    plot_deleted,
    filename_suffix="",
    output_dir="/test/",
    priors=dict(z=[0, 11], Av=[0, 2], beta=[0, 2], NHx=[0.1, 100], norm=[0, 10]),
):
    """ Plot the triangle """
    if ext_law == "nodust" and Host_gas is False:
        labels = ["z", "beta", "norm"]
        truths = [z_sim, beta_sim, -1]
        range_val = [
            (priors["z"][0], priors["z"][1]),
            (priors["Av"][0], priors["Av"][1]),
            1.0,
        ]
    elif ext_law != "nodust" and Host_gas is False:
        labels = ["z", "Av", "beta", "norm"]
        truths = [z_sim, Av_sim, -1, -1]
        range_val = [
            (priors["z"][0], priors["z"][1]),
            (priors["Av"][0], priors["Av"][1]),
            (priors["beta"][0], priors["beta"][1]),
            # (priors["norm"][0], priors["norm"][1])
            1.0,
        ]
    elif ext_law == "nodust" and Host_gas is True:
        labels = ["z", "beta", "NHx", "norm"]
        truths = [z_sim, -1, -1, -1]
        range_val = [
            (priors["z"][0], priors["z"][1]),
            (priors["beta"][0], priors["beta"][1]),
            (priors["NHx"][0], priors["NHx"][1]),
            # (priors["norm"][0], priors["norm"][1])
            1.0,
        ]
    elif ext_law != "nodust" and Host_gas is True:
        labels = ["z", "Av", "beta", "NHx", "norm"]
        truths = [z_sim, Av_sim, -1, -1, -1]
        range_val = [
            (priors["z"][0], priors["z"][1]),
            (priors["Av"][0], priors["Av"][1]),
            (priors["beta"][0], priors["beta"][1]),
            (priors["NHx"][0], priors["NHx"][1]),
            # (priors["norm"][0], priors["norm"][1])
            1.0,
        ]

    fig = corner.corner(
        samples,
        labels=labels,
        truths=truths,
        bins=50,
        range=range_val,
        levels=[0.68, 0.95, 0.99],
        plot_contours=True,
        fill_contours=True,
        plot_datapoints=False,
        color="blue",
        scale_hist=True,
    )

    fig.gca().annotate(
        "zsim: %.2f    Av_sim: %.2f " % (z_sim, Av_sim),
        xy=(0.5, 1.0),
        xycoords="figure fraction",
        xytext=(0, -5),
        textcoords="offset points",
        ha="center",
        va="top",
    )
    plt.tight_layout()
    fig.savefig(
        output_dir + str(name) + "/triangle_plot_" + ext_law + filename_suffix + ".png"
    )
    if plot:
        plt.show()

    #plt.close(fig)
    #plt.close("all")


def plot_mcmc_fit(
    results,
    ndim,
    best_chi2_val,
    sed,
    wavelength,
    samples,
    plot_all,
    plot,
    ext_law,
    Host_dust,
    Host_gas,
    MW_dust,
    MW_gas,
    DLA,
    igm_att,
    output_dir="/test/",
    filename_suffix="",
):
    """
    Plot the flux template corresponding to the best fit of the mcmc
    """
    z_sim = results["z_sim"][0]
    Av_sim = results["Av_host_sim"][0]
    # print (sed)
    mask = sed["detection"] == 1
    F0 = sed["flux_corr"][np.argmax(sed["eff_wvl"][mask])]
    wvl0 = sed["eff_wvl"][np.argmax(sed["eff_wvl"][mask])]

    z_fit = results["zphot_68"][0]
    Av_fit = results["Av_68"][0]
    beta_fit = results["beta_68"][0]
    norm_fit = results["norm_68"][0]
    NHx_fit = results["NHx_68"][0]

    z_minL = best_chi2_val["z"]
    beta_minL = best_chi2_val["beta"]
    norm_minL = best_chi2_val["norm"]
    Av_minL = best_chi2_val["Av"]
    NHx_minL = best_chi2_val["NHx"]

    print("\nFor best SED plot:")
    print(
        "- Median values PDF: {:.3f} {:.3f} {:.3f} {:.3f}".format(
            z_fit, Av_fit, beta_fit, norm_fit
        )
    )

    print(
        "- Best fit: {:.3f} {:.3f} {:.3f} {:.3f}".format(
            z_minL, Av_minL, beta_minL, norm_minL
        )
    )

    flux_median = Flux_template(
        wavelength,
        F0,
        wvl0,
        norm_fit,
        beta_fit,
        z_fit,
        Av_fit,
        NHx_fit,
        ext_law,
        Host_dust,
        Host_gas,
        MW_dust,
        MW_gas,
        DLA,
        igm_att,
    )
    flux_minL = Flux_template(
        wavelength,
        F0,
        wvl0,
        norm_minL,
        beta_minL,
        z_minL,
        Av_minL,
        NHx_minL,
        ext_law,
        Host_dust,
        Host_gas,
        MW_dust,
        MW_gas,
        DLA,
        igm_att,
    )
    plt.figure()
    plt.plot(wavelength, flux_median, label="median", ls="--", lw=1.5, color="blue")
    plt.plot(wavelength, flux_minL, label="best fit", lw=1.5, color="green")

    if plot_all:
        if ext_law == "nodust" and Host_gas is False:
            for (z, beta, norm) in samples:
                plt.plot(
                    wavelength,
                    Flux_template(wavelength, F0, wvl0, norm, beta, z, 0, 0, ext_law),
                    color="k",
                    alpha=0.01,
                )
        elif ext_law != "nodust" and Host_gas is False:
            for (z, beta, norm, Av) in samples:
                plt.plot(
                    wavelength,
                    Flux_template(wavelength, F0, wvl0, norm, beta, z, Av, 0, ext_law),
                    color="k",
                    alpha=0.01,
                )
        elif ext_law == "nodust" and Host_gas is True:
            for (z, beta, norm, NHx) in samples:
                plt.plot(
                    wavelength,
                    Flux_template(wavelength, F0, wvl0, norm, beta, z, 0, NHx, ext_law),
                    color="k",
                    alpha=0.01,
                )
        elif ext_law != "nodust" and Host_gas is True:
            for (z, beta, norm, Av, NHx) in samples:
                plt.plot(
                    wavelength,
                    Flux_template(
                        wavelength, F0, wvl0, norm, beta, z, Av, NHx, ext_law
                    ),
                    color="k",
                    alpha=0.01,
                )

    # Set color for plots
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, len(sed) + 1)]

    for i in range(len(sed["band"])):
        # mask = seds['band'] == band
        if sed["detection"][i] == 0:
            yerr = 0.3
        else:
            yerr = sed["flux_corr_err"][i]
        plt.errorbar(
            sed["eff_wvl"][i],
            sed["flux_corr"][i],
            xerr=sed["band_width"][i] / 2,
            yerr=yerr,
            uplims=1 - sed["detection"][i],
            color=colors[i],
            lw=1.5,
        )
        plt.annotate(
            sed["band"][i],
            xy=(sed["eff_wvl"][i], max(sed["flux_corr"]) * 1.3),
            color=colors[i],
            fontsize=16,
            horizontalalignment="right",
            verticalalignment="bottom",
        )
    plt.xlabel(r"Observed wavelength [angstroms]", fontsize=14)
    # plt.ylabel(r'Flux ($\mu$Jy)')
    plt.ylabel(r"Flux [$\mu$Jy]", fontsize=14)
    # plt.xlim(sed["eff_wvl"][0] / 2, sed["eff_wvl"][-1] * 1.3)
    plt.ylim(min(sed["flux_corr"]) * 0.1, max(sed["flux_corr"]) * 2.3)
    # plt.ylim(1,400)
    plt.title(
        "SED extracted at T-To=%.0f sec\n z_sim=%.2f, Av_sim=%.2f"
        % (float(sed["time_since_burst"][0]), float(z_sim), float(Av_sim))
    )
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True)
    name_tel = sed["telescope"][0]
    if name_tel not in ["ratir", "grond", "gft"]:
        name_tel = "mix"
    plt.legend(
        loc="lower right",
        numpoints=1,
        title="%s\n (%s data)" % (sed["Name"][0], name_tel),
        fontsize=16,
    )
    # set the legend title size
    plt.setp(plt.gca().get_legend().get_title(), fontsize="16")
    plt.tight_layout()
    plt.savefig(
        output_dir
        + str(sed["Name"][0])
        + "/bestfit_"
        + ext_law
        + filename_suffix
        + ".png"
    )
    if plot:
        plt.show()
    #plt.close("all")


def plot_zphot(
    input_file,
    output_suffix,
    sigma,
    input_dir="/results/",
    output_dir="/plots/",
    plot=True,
):
    """
    Plots zphot vs zsim.
    No inputs required, the plot is saved in plots/ (default)

    Parameters
    ----------


    Returns
    -------

    """

    filename = "%s%s" % (input_dir, input_file)
    models = ascii.read(filename + ".dat")
    models.sort("z_sim")
    zsim = models["z_sim"]
    Avsim = models["Av_host_sim"]

    if sigma == 1:
        zphot = models["zphot_68"]
        z_sup = zphot + models["zphot_68_sup"]
        z_inf = zphot - models["zphot_68_inf"]
    elif sigma == 2:
        zphot = models["zphot_95"]
        z_sup = zphot + models["zphot_95_sup"]
        z_inf = zphot - models["zphot_95_inf"]
    elif sigma == 3:
        zphot = models["zphot_99"]
        z_sup = zphot + models["zphot_99_sup"]
        z_inf = zphot - models["zphot_99_inf"]

    nb_detections = models["nb_detection"]
    nb_bands = models["nb_bands"]

    # binning data with same number of points
    nb_data_per_bin = 1
    counter = 0
    bin_means = []
    zsim_bin = []
    bin_inf_means = []
    bin_sup_means = []
    append_mean = bin_means.append
    append_inf = bin_inf_means.append
    append_sup = bin_sup_means.append
    append_sim = zsim_bin.append

    zsim_mean = 0
    zphot_mean = 0
    zinf_mean = 0
    zsup_mean = 0
    for i in range(len(models)):
        if counter < nb_data_per_bin:
            zsim_mean += zsim[i]
            zphot_mean += zphot[i]
            zinf_mean += z_inf[i]
            zsup_mean += z_sup[i]
            counter += 1
            # print (counter,i,zsim_mean)
        elif counter == nb_data_per_bin:
            append_mean(zphot_mean / nb_data_per_bin)
            append_sim(zsim_mean / nb_data_per_bin)
            append_inf(zinf_mean / nb_data_per_bin)
            append_sup(zsup_mean / nb_data_per_bin)
            counter = 0
            zsim_mean = 0
            zphot_mean = 0
            zinf_mean = 0
            zsup_mean = 0

    bin_means = np.array(bin_means)
    zsim_bin = np.array(zsim_bin)
    bin_inf_means = np.array(bin_inf_means)
    bin_sup_means = np.array(bin_sup_means)

    # Set Av to -0.1 when not available from literature.
    # It allows to keep a readable colorbar
    for k in range(len(Avsim)):
        if Avsim[k] == -99:
            Avsim[k] = -0.1

    plt.figure()
    # If the number is higher than 16 this has to be extended.
    markers = [
        "x",
        "s",
        "*",
        "v",
        "o",
        "^",
        "D",
        "<",
        ">",
        "8",
        "p",
        "h",
        "H",
        "d",
        "P",
        "X",
    ]

    # Fake data for generating legend
    for i, mar in enumerate(markers[: np.max(nb_bands)]):
        plt.scatter(-1, -1, marker=mar, label=str(i + 1), color="black")
    plt.legend(
        loc="upper left",
        scatterpoints=1,
        ncol=2,
        fontsize=8,
        title="Nb of bands\n with detection",
    )
    x = np.arange(0, 12, 1)
    for i, nb_det in enumerate(nb_detections):
        plt.scatter(
            [zsim[i]],
            [zphot[i]],
            c=[Avsim[i]],
            cmap=plt.cm.jet,
            marker=markers[int(nb_det - 1)],
            vmin=min(Avsim),
            vmax=max(Avsim),
            s=50,
        )
    plt.fill_between(list(zsim), list(z_inf), list(z_sup), color="blue", alpha=0.3)
    plt.plot(x, x, ls="--", color="red", lw=0.5)
    plt.xlabel(r"$z_{sim}$", fontsize=15)
    plt.ylabel(r"$z_{phot}$", fontsize=15)
    plt.xlim(0, 11)
    plt.xticks(np.arange(0, 12, 1))
    plt.yticks(np.arange(0, 12, 1))
    plt.ylim(0, 11)
    plt.grid(True)
    plt.colorbar().set_label("     Av", rotation=0)
    plt.savefig(output_dir + "zphot" + output_suffix + ".png")
    if plot:
        plt.show()

    plt.figure()
    # Fake data for generating legend
    for i, mar in enumerate(markers[: np.max(nb_bands)]):
        plt.scatter(-1, -1, marker=mar, label=str(i + 1), color="black")
    plt.legend(
        loc="upper left",
        scatterpoints=1,
        ncol=2,
        fontsize=8,
        title="Nb of bands\n with detection",
    )

    x = np.arange(0, 12, 1)
    for i, nb_det in enumerate(nb_detections):
        plt.scatter(
            [zsim[i]],
            [zphot[i] - zsim[i]],
            c=[Avsim[i]],
            cmap=plt.cm.jet,
            vmin=min(Avsim),
            vmax=max(Avsim),
            marker=markers[int(nb_det - 1)],
            s=50,
        )
    plt.fill_between(
        list(zsim),
        list(np.array(z_inf) - np.array(zsim)),
        list(np.array(z_sup) - np.array(zsim)),
        color="blue",
        alpha=0.3,
    )
    plt.xlabel(r"$z_{sim}$", fontsize=15)
    plt.ylabel(r"$z_{phot}-z_{sim}$", fontsize=15)
    plt.xlim(0, 11)
    plt.xticks(np.arange(0, 12, 1))
    plt.yticks(np.arange(-6, 6, 1))
    plt.ylim(-6, 6)
    plt.grid(True)
    plt.colorbar().set_label("     Av", rotation=0)
    plt.savefig(output_dir + "deltaz_abs" + output_suffix + ".png")
    if plot:
        plt.show()

    plt.figure()
    # Fake data for generating legend
    for i, mar in enumerate(markers[: np.max(nb_bands)]):
        plt.scatter(-1, -1, marker=mar, label=str(i + 1), color="black")
    plt.legend(
        loc="upper left",
        scatterpoints=1,
        ncol=2,
        fontsize=8,
        title="Nb of bands\n with detection",
    )

    x = np.arange(0, 12, 1)
    for i, nb_det in enumerate(nb_detections):
        plt.scatter(
            [zsim[i]],
            [(zphot[i] - zsim[i]) / (1 + zsim[i])],
            c=[Avsim[i]],
            cmap=plt.cm.jet,
            vmin=min(Avsim),
            vmax=max(Avsim),
            marker=markers[int(nb_det - 1)],
            s=50,
        )
    plt.fill_between(
        list(zsim),
        list((np.array(z_inf) - np.array(zsim)) / (1 + np.array(zsim))),
        list((np.array(z_sup) - np.array(zsim)) / (1 + np.array(zsim))),
        color="blue",
        alpha=0.3,
    )
    plt.xlabel(r"$z_{sim}$", fontsize=15)
    plt.ylabel(r"$(z_{phot}-z_{sim})/(1+z)$", fontsize=15)
    plt.xlim(0, 11)
    plt.xticks(np.arange(0, 12, 1))
    plt.yticks(np.arange(-1, 2, 0.2))
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.colorbar().set_label("     Av", rotation=0)
    plt.savefig(output_dir + "deltaz_rel" + output_suffix + ".png")
    if plot:
        plt.show()


def plot_zphot_comp():
    """"""

    path = "results/"
    filename_1 = path + "best_fits_tot_68_emp" + ".dat"
    filename_2 = path + "best_fits_tot_68_emp_sansy" + ".dat"

    color_file = ["blue", "red", "green"]
    label = ["no y band", "y band"]

    for i, filename in enumerate([filename_2, filename_1]):
        results = ascii.read(filename)
        results.sort("z")
        zsim = results["z"]
        zphot = results["zphot"]
        z_sup = zphot + results["zphot_sup"]
        z_inf = zphot - results["zphot_inf"]

        x = np.arange(0, 12, 1)
        # plt.scatter(zsim,zphot, label='best fit data',s=1,
        #             facecolor='0.7', lw = 0.8)
        plt.fill_between(
            list(zsim),
            list(z_inf),
            list(z_sup),
            color=color_file[i],
            alpha=0.3,
            label=label[i],
        )

    plt.plot(x, x, ls="--", color="red", lw=0.5)
    plt.legend(loc="upper left")
    plt.xlabel(r"$z_{sim}$")
    plt.ylabel(r"$z_{phot}$")
    plt.xlim(0, 11)
    plt.xticks(np.arange(0, 12, 1))
    plt.yticks(np.arange(0, 12, 1))
    plt.ylim(0, 11)
    plt.grid(True)
    plt.savefig("plots/comp_zphot_emp_68.png")
    plt.show()
