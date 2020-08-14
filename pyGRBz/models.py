# -*- coding: utf-8 -*-
from pyGRBz.extinction_correction import sed_extinction


def SPL_lc(t, F0, t0, norm, alpha):
    return norm * F0 * (t / t0) ** (-alpha)


def BPL_lc(t, F0, norm, alpha1, alpha2, t1, s):
    return (
        norm
        * F0
        * ((t / t1) ** (-s * alpha1) +
           (t / t1) ** (-s * alpha2)) ** (-1.0 / s)
    )


def SPL_sed(wvl, F0, wvl0, norm, beta):
    return norm * F0 * (wvl / wvl0) ** beta


def BPL_sed(wvl, F0, norm, beta1, beta2, wvl1, s):
    return (
        norm
        * F0
        * ((wvl / wvl1) ** (s * beta1) +
           (wvl / wvl1) ** (s * beta2)) ** (1.0 / s)
    )


def template1(wvl, t, F0, wvl0, t0, norm, alpha, beta, z, Av, ext_law,
              Host_dust, Host_gas, MW_dust, MW_gas, DLA, igm_att):
    return (
        norm
        * F0
        * (t / t0) ** (-alpha)
        * (wvl / wvl0) ** beta
        * sed_extinction(wvl, z, Av, ext_law=ext_law, Host_dust=Host_dust,
                         Host_gas=Host_gas, MW_dust=MW_dust,
                         MW_gas=MW_gas, DLA=DLA, igm_att=igm_att)
    )


def template2(wvl, t, F0, wvl0, norm, alpha1, alpha2, t1, s, beta,
              z, Av, ext_law, Host_dust, Host_gas, MW_dust, MW_gas, DLA,
              igm_att):
    Flux = (
        norm
        * F0
        * ((t / t1) ** (-s * alpha1) +
           (t / t1) ** (-s * alpha2)) ** (-1.0 / s)
        * (wvl / wvl0) ** beta
        * sed_extinction(wvl, z, Av, ext_law=ext_law, Host_dust=Host_dust,
                         Host_gas=Host_gas, MW_dust=MW_dust,
                         MW_gas=MW_gas, DLA=DLA, igm_att=igm_att)
    )
    return Flux


def Flux_template(wvl, F0, wvl0, norm, beta, z, Av, ext_law,
                  Host_dust, Host_gas, MW_dust, MW_gas, DLA, igm_att):
    Flux = SPL_sed(wvl, F0, wvl0, norm, beta) * sed_extinction(
        wvl,
        z,
        Av,
        ext_law=ext_law,
        Host_dust=Host_dust,
        Host_gas=Host_gas,
        MW_dust=MW_dust,
        MW_gas=MW_gas,
        DLA=DLA,
        igm_att=igm_att,
    )
    return Flux


def SPL(wvl, t, F0, wvl0, t0, norm, beta, alpha):
    return norm * F0 * (wvl / wvl0) ** beta * (t / t0) ** (-alpha)


def BPL(wvl, t, F0, wvl0, t0, norm, beta, alpha1, alpha2, s):
    return (
        norm
        * F0
        * (wvl / wvl0) ** beta
        * ((t / t0) ** (-s * alpha1) +
           (t / t0) ** (-s * alpha2)) ** (-1.0 / s)
    )


"""
def Flux_template(wvl,t,F0,wvl0,t0,norm,beta,alpha,z,Av):
    return SPL(wvl,t,F0,wvl0,t0,norm,beta,alpha) * sed_extinction(wvl,z,Av)
"""
