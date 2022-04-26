#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
from libc.math cimport pow
import cython
from cython.parallel import prange, parallel
from pyGRBaglow.reddening_cy import Pei92, sne, gas_absorption
from pyGRBaglow.igm_cy import meiksin


cdef double SPL_sed_float(double wvl, double F0, double wvl0, double norm,
                          double beta) nogil:
    """Spectral evolution as single power law for SED"""
    return norm * F0 * pow(wvl / wvl0, beta)


cpdef SPL_sed(double[:] wvl, double F0, double wvl0, double norm, double beta):
    """Spectral evolution as single power law for SED"""
    cdef Py_ssize_t i
    cdef Py_ssize_t N = len(wvl)
    cdef double[:] flux = np.zeros(N, dtype=np.float64)
    
    for i in prange(N, nogil=True, num_threads=1):
        flux[i] = SPL_sed_float(wvl[i], F0, wvl0, norm, beta)
    return flux


cdef double BPL_sed_float(double wvl, double F0, double norm, double beta1,
                          double beta2, double wvl1, double s) nogil:
    """Spectral evolution as broken power law for SED"""
    cdef double flux
    flux = norm * F0 * pow(pow(wvl / wvl1, s * beta1) + pow(wvl / wvl1, s * beta2), 1.0 / s)
    
    return flux


cpdef BPL_sed(double[:] wvl, double F0, double norm, double beta1,
              double beta2, double wvl1, double s):
    """Spectral evolution as single power law for SED"""
    cdef Py_ssize_t i
    cdef Py_ssize_t N = len(wvl)
    cdef double[:] flux = np.zeros(N, dtype=np.float64)
    
    for i in prange(N, nogil=True, num_threads=1):
        flux[i] = BPL_sed_float(wvl[i], F0, norm, beta1, beta2, wvl1, s)
    return flux


cpdef compute_model_flux(
    double[:] wvl,
    double F0,
    double wvl0,
    double norm,
    double beta,
    double z,
    double Av,
    double NHx,
    str ext_law="smc",
    bint Host_dust=True,
    bint Host_gas=True,
    str igm_att="meiksin"
):
    """Compute model flux at each given wavelength, with extincton."""
    cdef Py_ssize_t i
    cdef Py_ssize_t N = len(wvl)
    cdef double[:] flux = SPL_sed(wvl, F0, wvl0, norm, beta)
    cdef double[:] Trans_igm = meiksin(wvl, z, unit="angstroms", Xcut=True)
    cdef double[:] Trans_dust = np.empty(N, dtype=np.float64)
    cdef double[:] Trans_gas = np.empty(N, dtype=np.float64)

    if Host_dust:
        if ext_law == "sne":
            Trans_dust = sne(wvl, Av, z, Xcut=True)[1]
        else:
            Trans_dust = Pei92(wvl, Av, z, ext_law=ext_law, Xcut=True)[1]
    if Host_gas:
        Trans_gas = gas_absorption(wvl, z , NHx=NHx)
    
    for i in prange(N, nogil=True, num_threads=1):
        flux[i] = flux[i] * Trans_igm[i]
        if Host_dust:
            flux[i] = flux[i] * Trans_dust[i]
        if Host_gas:    
            flux[i] = flux[i] * Trans_gas[i]
    
    return flux
    

def compute_model_integrated_flux(
    double[:] wvl,
    double[:, ::1] sys_response,
    double F0,
    double wvl0,
    double norm,
    double beta,
    double z,
    double Av,
    double NHx,
    str ext_law="smc",
    bint Host_dust=True,
    bint Host_gas=True,
    str igm_att="meiksin"
):
    """Integrate model flux through a system response using trapezoid method"""
    cdef Py_ssize_t i, j
    cdef Py_ssize_t N = len(wvl)
    cdef Py_ssize_t N2 = sys_response.shape[0]
    cdef double dwvl
    cdef double[:] flux_int = np.zeros(N2, dtype=np.float64)
    cdef double[:] a=np.zeros(N2, dtype=np.float64)
    cdef double[:] b=np.zeros(N2, dtype=np.float64)
    cdef double[:] flux = compute_model_flux(wvl, F0, wvl0, norm, beta, z, Av, NHx,
                                                 ext_law, Host_dust, Host_gas, igm_att
                                                )

    for i in prange(N-1, nogil=True, num_threads=1):
        for j in range(N2):
            dwvl = wvl[i+1]-wvl[i]
            a[j] = a[j] + (sys_response[j,i+1] + sys_response[j,i])/2. * dwvl
            b[j] = b[j] + (sys_response[j,i+1] * flux[i+1] + sys_response[j,i] * flux[i])/2. * dwvl
        
    for j in range(N2):
        flux_int[j] = b[j] / a[j]

    return flux_int

