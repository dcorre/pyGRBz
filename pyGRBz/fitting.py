# -*- coding: utf-8 -*-
import numpy as np 
from scipy.special import erf
from scipy.stats import chi2
import os
import sys
import iminuit
from iminuit import Minuit,describe
from iminuit.util import make_func_code
import emcee
from astropy.table import Table, vstack
from astropy.io import ascii
from .utils import mag2Jy,Jy2Mag
from .models import Flux_template, BPL_lc, SPL_lc
from .plotting import plot_lc_fit_check, plot_sed, plot_mcmc_evolution, plot_triangle, plot_mcmc_fit


class Chi2Functor_lc:
   def __init__(self,f,t,y,yerr):
   #def __init__(self,f,wvl,y):
       self.f = f
       self.t = t
       self.y = y
       self.yerr = yerr
       f_sig = describe(f)
       #this is how you fake function 
       #signature dynamically
       self.func_code = make_func_code(f_sig[1:])#docking off independent variable
       self.func_defaults = None #this keeps np.vectorize happy
       #print (make_func_code(f_sig[1:]))
   def __call__(self,*arg):
       #notice that it accept variable length
       #positional arguments
       chi2 = sum(((y-self.f(t,*arg))**2/yerr**2) for t,y,yerr in zip(self.t,self.y,self.yerr))
       #chi2 = sum((y-self.f(wvl,*arg))**2 for wvl,y in zip(self.wvl,self.y))
       return chi2



def fit_lc(observations,grb_info,model,method='best'):
    """ Fit the lightcurve in order to get a flux and its uncertainty at each time
        The fit is performed for each band separetely
    """
    grb_ref=[]
    band_list=[] 
    telescope_list=[] 
    F0_list=[]
    norm_list=[]
    alpha_list=[]
    alpha1_list=[]
    alpha2_list=[]
    t1_list=[]
    t0_list=[]
    s_list=[]
    chi2_list=[]

    # Go through each grb
    for obs_table in observations.group_by('Name').groups:
        mask = grb_info['name'] == obs_table['Name'][0]

        # Check whether it is a light curve or a sed
        z_sim = grb_info['z'][mask]
        Av_sim = grb_info['Av_host'][mask]

        # Fit light curve for each band of a given telescope
        for band_table  in obs_table.group_by(['telescope','band']).groups.keys:
            mask2 = (obs_table['band'] == band_table['band']) & (obs_table['telescope'] == band_table['telescope'])
            time = obs_table['time_since_burst'][mask2]#obs_table[mask]['Obs_time'] +obs_table['t_since_burst'][0]
            #print (time)

            y=obs_table[mask2]['flux']
            #print (y)
            yerr_=obs_table[mask2]['flux_err']
            #print(yerr_)

            # -------Guess initial values-----------
            F0_guess=y[0]
            #print (F0_guess)
            
            # Search for extremum 
            #argrelextrema(, np.greater)
            idx=np.argmax(y)
            if (idx < len(y)-1) and (idx >0) :
                t1_guess=time[idx]
                limit_t1_guess=(0.1*t1_guess,10*t1_guess)
            else:
                idx=np.argmin(y)
                if (idx>0) and (idx<len(y)-1):
                    t1_guess=time[idx]
                    limit_t1_guess=(0.1*t1_guess,10*t1_guess)
                else:
                    t1_guess = time[0]
                    limit_t1_guess=(0,None)
            #t1_guess=lc_fit[np.argmax(lc_fit[:,2]),0]
            #print (t1_guess)
            norm_guess=1

            if model == 'BPL':
                chi2_func = Chi2Functor_lc(BPL_lc,time,y,yerr_)
                kwdarg = dict(pedantic=True,print_level=2,F0=F0_guess,fix_F0=True,norm=norm_guess,fix_norm=False,limit_norm=(0.1,10),alpha1=-0.5,limit_alpha1=[-3,0],alpha2=0.5,limit_alpha2=[0,3],t1=t1_guess,fix_t1=False,limit_t1=[0,None],s=1,limit_s=[0.01,20])

            elif model == 'SPL':
                chi2_func = Chi2Functor_lc(SPL_lc,time,y,yerr_)
                kwdarg = dict(pedantic=True,print_level=2,F0=F0_guess,fix_F0=True,norm=norm_guess,fix_norm=False,limit_norm=(0.1,10),alpha=1,limit_alpha=[-10,10],t0=t1_guess,fix_t0=True,limit_t0=[0,None])
            #print (describe(chi2_func))
            else:
                sys.exit('Error: "%s" model for fitting the light curve unknown.\It should be either "BPL" or "SPL"' % model)

            m = Minuit(chi2_func,**kwdarg)

            m.set_strategy(1)
            #m.migrad(nsplit=1,precision=1e-10)
            d,l=m.migrad()
            #print (band)
            print ('Valid Minimum: %s ' % str(m.migrad_ok()))
            print ('Is the covariance matrix accurate: %s' % str(m.matrix_accurate()))


            grb_ref.append(grb_info['name'][mask][0])
            band_list.append(band_table['band'])
            telescope_list.append(band_table['telescope'])
            F0_list.append(m.values['F0'])
            norm_list.append(m.values['norm'])
            chi2_list.append(d.fval)
            if model == 'SPL':
                alpha_list.append(m.values['alpha'])
                t0_list.append(m.values['t0'])
            elif model == 'BPL':
                alpha1_list.append(m.values['alpha1'])
                alpha2_list.append(m.values['alpha2'])
                t1_list.append(m.values['t1'])
                s_list.append(m.values['s'])

        if method == 'best':
            # If few points take the parameters of the best fit. It assumes achromatic evolution
            best_fitted_band = obs_table['band'][np.argmax(obs_table['eff_wvl'])]

    #create astropy table as output
    if model == 'BPL': 
        lc_fit_params=Table([grb_ref,telescope_list,band_list,F0_list,norm_list,alpha1_list,alpha2_list,t1_list,s_list,chi2_list],names=['name','telescope','band','F0','norm','alpha1','alpha2','t1','s','chi2'])
    elif model == 'SPL':     
        lc_fit_params=Table([grb_ref,telescope_list,band_list,F0_list,norm_list,alpha_list,t0_list,chi2_list],names=['name','telescope','band','F0','norm','alpha','t0','chi2'])
    print (lc_fit_params)
    """
    if method == 'best':
        # If few points take the parameters of the best fit. It assumes achromatic evolution
        mask = lc_params['band'][np.argmax(lc_params['chi2'])]
    """


    return lc_fit_params

def extract_seds(observations, grb_info, plot=True, model='PL', method='ReddestBand', time_SED=1, output_dir='results/', filename_suffix=''):
    """ Extracts the SED at a given time for the given lightcurves
    """
    # Sort data by ascending eff. wavelength
    observations.sort(['Name','eff_wvl','time_since_burst'])
    #If data already in sed format
    mask_sed = grb_info['type'] == 'sed'
    if mask_sed.any():
       mask_sed2=np.array([False]*len(observations['Name']))
       for i in range(np.sum(mask_sed)):
           mask_sed2[mask_sed2==False]=observations['Name'][~mask_sed2]==grb_info['name'][mask_sed][i]
       seds=observations[mask_sed2].copy()

    #If data in light curve format
    mask_lc=grb_info['type'] == 'lc' 
    if mask_lc.any():
        mask_lc2=np.array([False]*len(observations['Name']))
        for i in range(np.sum(mask_lc)):
            mask_lc2[mask_lc2==False]=observations['Name'][~mask_lc2]==grb_info['name'][mask_lc][i]
    
        lc_fit_params= fit_lc(observations[mask_lc2],grb_info[mask_lc],model)
        #print (lc_fit_params)
        if plot: plot_lc_fit_check(observations[mask_lc2],grb_info[mask_lc],lc_fit_params,model,plot,output_dir=output_dir,filename_suffix=filename_suffix)
 
        name_sed=[]
        band_list=[]
        band_width_list=[]
        sys_response_list=[]
        wvl_eff=[]
        tel_name=[]
        sed_mag=[]
        sed_magerr=[]
        mag_ext_list=[]
        sed_flux=[]
        sed_fluxerr=[]
        time_sed_list=[]
        phot_sys=[]
        zp=[]
        zp2=[]
        detection=[]
        convert_dict={'photometry_system': 'AB'}
    
        for obs_table in observations[mask_lc2].group_by('Name').groups:

            #print (obs_table)
            #print (idx,obs_table[obs_table['band'] == 'H']['flux'],obs_table[obs_table['band'] == 'H']['flux'][idx],obs_table[obs_table['band'] == 'H']['Obs_time'],obs_table[obs_table['band'] == 'H']['Obs_time'][idx])
            #time_sed=obs_table[obs_table['band'] == 'H']['Obs_time'][-1]+obs_table['time_since_burst'][0] 
            #time_sed=obs_table[obs_table['band'] == 'H']['Obs_time'][idx] + obs_table['time_since_burst'][0]
            if method=='ReddestBand':
                #Find reddest band
                reddest_band = obs_table['band'][np.argmax(obs_table['eff_wvl'])]
                print ('reddest band: %s' % reddest_band) 

                # Maximum flux in reddest band in the observation set
                idx=np.argmax( obs_table[obs_table['band'] == reddest_band]['flux'])
                time_sed=obs_table[obs_table['band'] == reddest_band]['time_since_burst'][idx]

            elif method == 'fixed': 
                #Extract the sed at the given time
                time_sed=time_SED
 
            #print (time_sed)

            for tel in obs_table.group_by(['telescope','band']).groups.keys:
                #print (tel)
                # print (obs_table[obs_table['band'] == tel['band']])
                #Do not use bands with only one point Can be used if achromatic assumption
                if len(obs_table[obs_table['band'] == tel['band']]) <= 1: continue

                mask2 = (lc_fit_params['name'] == obs_table['Name'][0]) & (lc_fit_params['band'] == tel['band']) & (lc_fit_params['telescope'] == tel['telescope'])
                mask3 = (obs_table['band'] == tel['band']) & (obs_table['telescope'] == tel['telescope'])
    
                if model == 'BPL':
                    flux=BPL_lc(time_sed,float(lc_fit_params['F0'][mask2]),float(lc_fit_params['norm'][mask2]),float(lc_fit_params['alpha1'][mask2]),float(lc_fit_params['alpha2'][mask2]),float(lc_fit_params['t1'][mask2]),float(lc_fit_params['s'][mask2]))
                elif model == 'SPL':                
                    flux=SPL_lc(time_sed,float(lc_fit_params['F0'][mask2]),float(lc_fit_params['t0'][mask2]),float(lc_fit_params['norm'][mask2]),float(lc_fit_params['alpha'][mask2]))

                # Estimate the error with the closest data point
                idx=np.argmin((time_sed-obs_table['time_since_burst'][mask3])**2)
                fluxerr=obs_table['flux_err'][mask3][idx]

                name_sed.append(obs_table['Name'][0])
                band_list.append(tel['band'])
                band_width_list.append(obs_table['band_width'][mask3][0])
                sys_response_list.append(obs_table['sys_response'][mask3][0])
                wvl_eff.append(obs_table['eff_wvl'][mask3][0])
                sed_flux.append(flux)
                sed_fluxerr.append(fluxerr)
                time_sed_list.append(time_sed)
                sed_mag.append(Jy2Mag(convert_dict,flux*1e-6))
                sed_magerr.append(2.5/((flux)*np.log(10))*fluxerr)
                mag_ext_list.append(obs_table['ext_mag'][mask3][0])
                phot_sys.append(obs_table['phot_sys'][mask3][0])
                zp.append(obs_table['zeropoint'][mask3][0])
                zp2.append(obs_table['zp'][mask3][0])
                tel_name.append(obs_table['telescope'][mask3][0])
                detection.append(obs_table['detection'][mask3][idx])
         
        #create astropy table
        #seds_extracted = Table([name_sed,time_sed_list,band_list,wvl_eff,sed_mag,sed_magerr,sed_flux,sed_fluxerr,phot_sys,tel_name,detection,zp,band_width_list,sys_response_list,mag_ext_list],names=['Name','time_since_burst','band','eff_wvl','mag','mag_err','flux','flux_err','phot_sys','telescope','detection','zeropoint','band_width','sys_response','ext_mag']) 
        seds_extracted = Table([name_sed,time_sed_list,band_list,sed_mag,sed_magerr,zp2,phot_sys,detection,tel_name,wvl_eff,band_width_list,zp,sys_response_list,mag_ext_list,sed_flux,sed_fluxerr],names=['Name', 'time_since_burst', 'band', 'mag', 'mag_err', 'zp', 'phot_sys', 'detection', 'telescope', 'eff_wvl', 'band_width', 'zeropoint', 'sys_response', 'ext_mag', 'flux', 'flux_err']) 
        seds_extracted['time_since_burst'].unit='s'
        #seds['flux'].unit='microJy' 
        #seds['flux_err'].unit='microJy'     
        seds_extracted['eff_wvl'].unit='Angstrom'

        #dealing with non detection
        mask= seds_extracted['detection'] == -1
        if mask.any():
            seds_extracted['flux_err'][mask]=seds_extracted['flux'][mask]
            seds_extracted['flux'][mask]=seds_extracted['flux'][mask]
            seds_extracted['mag'][mask]=Jy2Mag(convert_dict,seds_extracted['flux'][mask]*1e-6)
            seds_extracted['mag_err'][mask]=seds_extracted['flux_err'][mask]*2.5/np.log(10)/seds_extracted['flux'][mask]

    if mask_sed.any() and mask_lc.any():
       seds=vstack([seds,seds_extracted],join_type='outer')
    elif mask_lc.any() and not mask_sed.any():
       seds=seds_extracted.copy()
    #print ("extracted seds")
    #print (seds)
    seds.sort(['Name','eff_wvl'])
    
    plot_sed(seds,grb_info,plot,model,output_dir=output_dir,filename_suffix=filename_suffix)
    return seds


def zeropoints(system_response,wavelength):
   """ Calculate the zeropoints for the given passbands """
   dwvl=np.gradient(wavelength)
   zp=2.5*np.log10(np.sum(system_response['sys_response']*dwvl,axis=1)) + 23.9
   return zp


def sumbandflux(flux,system_response,zeropoints,wavelength):
   """ Sum the flux over a filter band """
   #mag = -2.5*np.log10(np.trapz(flux*system_response['sys_response']/wavelength,wavelength) / np.trapz(3631*1e6*system_response['sys_response']/wavelength,wavelength))
   dwvl=np.gradient(wavelength)

   # Sum over the filter band
   mag=-2.5*np.log10(np.sum(flux*system_response*dwvl,axis=1))
   #Add zeropoints to convert into AB magnitudes
   #print (np.array(mag),np.array(zeropoints),np.array(mag)+np.array(zeropoints))
   mag += zeropoints

   #If the magnitude is infinite or nan, set it to 99
   mag[np.isinf(mag)]=99
   mag[np.isnan(mag)]=99

   return mag

def residuals(params,x,y,yerr,wavelength,F0,wvl0,system_response,zeropoints,ext_law,Host_dust,Host_gas,MW_dust,MW_gas,DLA,igm_att,kind='mag'):
   """ Calculate the residuals, observations - models """

   # Adapt the number of parameters in fonction of dust model
   if ext_law=='nodust': 
       z,beta,norm= params 
       Av=0
   else: z,beta,norm,Av= params
   #print (y)
   # Calculate the Flux in microJansky for the given set of parameters and a 
   flux_model = Flux_template(wavelength,F0,wvl0,norm,beta,z,Av,ext_law,Host_dust,Host_gas,MW_dust,MW_gas,DLA,igm_att)

   model = sumbandflux(flux_model,system_response,zeropoints,wavelength)

   # If comparing flux and not magnitudes
   if kind=='flux':
       model=mag2Jy({'photometry_system': 'AB'},model)*1e6
   
   return (y-model)/yerr


def lnprior(params,priors,ext_law):
   """ Set the allowed parameter range. Return the lnPrior """ 
  
   # Aapt number of parameters in fonction of chosen dust model
   if ext_law == 'nodust': z,beta,norm = params
   else: z,beta,norm,Av = params

   # So far only flat prior implemented
   # If the current parameter value is outside the allowed range, set to -inf, 
   # meaning that the probability to have this value is 0
   # Otherwise to 0, meaning probability to have this value == 1  
   if not priors['z'][0] < z < priors['z'][1]:
       return -np.inf
   if not priors['beta'][0] < beta < priors['beta'][1]:
       return -np.inf 
   if not priors['norm'][0] < norm < priors['norm'][1]:
       return -np.inf
   if ext_law != 'nodust':
       if not priors['Av'][0] < Av < priors['Av'][1]:
           return -np.inf

   return 0.0

def lnlike(params, x, y, yerr,detect,wavelength,F0,wvl0,system_response,zeropoints,ext_law,Host_dust,Host_gas,MW_dust,MW_gas,DLA,igm_att):
   """ Calculate the log likelihood. Return the lnLikelihood """
   
   # Calculate the residuals: (obs - model)/obs_err for each band     
   res=residuals(params,x,y,yerr,wavelength,F0,wvl0,system_response,zeropoints,ext_law,Host_dust,Host_gas,MW_dust,MW_gas,DLA,igm_att,kind='mag')

   # cumulative distribution function of the residuals
   residuals_cdf=0.5*(1+erf(res/np.sqrt(2)))
   # Survival function
   residuals_edf=1-residuals_cdf
   # residuals pdf
   residuals_pdf = -0.5*res**2

   # detect is 1 if detections and 0 if no detection
   lnlik=np.sum(detect*(residuals_pdf)+(1-detect)*(np.log(residuals_edf)))# +len(y)*np.log(2*np.pi*yerr**2))
   return  lnlik

def lnlik_C(yerr):
   """ constant term of the log likelihood expression """
   lnC=-0.5*len(y)*np.log(2*np.pi*yerr**2)
   return lnC

def lnprob(params, x, y, yerr,detect,wavelength,F0,wvl0,system_response,zeropoints,ext_law,Host_dust,Host_gas,MW_dust,MW_gas,DLA,igm_att,priors):
   """ Add lnPrior and lnLikelihood """

   # Get the lnPrior
   lp = lnprior(params,priors,ext_law)
   # Get the lnLikelihood
   lnlik=lnlike(params, x, y, yerr,detect,wavelength,F0,wvl0,system_response,zeropoints,ext_law,Host_dust,Host_gas,MW_dust,MW_gas,DLA,igm_att)

   # Check whether it is finite
   if not np.isfinite(lp):
       return -np.inf
   if not np.isfinite(lnlik):
       return -np.inf

   return lp + lnlik

def dof(params,y):
   """ Calculate the number of degrees of freedom """
   n=len(y)
   k=len(params)
   dof=n-k
   return dof

def Likelihood(yerr,lnlik):
   """ Compute the Likelihood from lnlik and lnlik_C """
   L=np.exp(-2*(lnlik_C(yerr)+lnlik))
   return L

def AIC(k,yerr,best_lnlik):
   """ AIC criteria """
   n=len(yerr)
   AIC = 2*k -2*(lnlik_C(yerr) + best_lnlik)
   return AIC

def AICc(k,yerr,best_lnlik):
   """  AICc criteria """
   n=len(yerr)
   AIC = AIC(k,yerr,best_lnlik)
   AICc=AIC+2*k*(k+1)/(n-k-1)
   return AICc


def mcmc(seds, grb_info,  wavelength, plot, sampler_type='ensemble',
         Nsteps1=300, Nsteps2=1000, nwalkers=30, nTemps=10, a=2,
         Nthreads=1, nburn=300, ext_law='smc', clean_data=False, plot_all=False, 
         plot_deleted=False, Host_dust=True, Host_gas=False, MW_dust=True,
         MW_gas=False, DLA=False, igm_att='Meiksin', output_dir='results/test/',
         filename_suffix='', std_gaussianBall = 1e-2,
         priors=dict(z=[0,11], Av=[0,2], beta=[0,2], norm=[0,10])):
   """ Compute the MCMC algorithm """

   results = []
 
   #if not os.path.exists(output_dir): os.makedirs(output_dir)
   

   # Check input parameters
   if Nsteps2 < nburn:
        print ('ERROR: Nsteps2 < nburn: there will be no values to estimate')
        sys.exit(1)

   # No cleaning implemented for PT sampler yet
   if sampler_type == 'pt':
        if clean_data == True: print ('clean_data sets to False. No clenaing implemented for PTsampler yet')
        clean_data = False
        plot_deleted = False
        
   # Adapt number of parameters in fonction of the selected dust model 
   if ext_law == 'nodust': ndim=3
   else: ndim = 4 

   if sampler_type == 'ensemble': nTemps = 1
   
   # Compute the initial values for the parameters
   # Initialise the ndim array "starting_guesses" with random values between 0 and 1
   starting_guesses = np.random.rand(nTemps, nwalkers, ndim)

   # Initial values for redshift taken between priors['z'][0] and priors['z'][1]
   starting_guesses[:, :, 0] *= (priors['z'][1]-priors['z'][0])
   starting_guesses[:, :, 0] += priors['z'][0]
   # Initial values for spectral slope taken between priors['beta'][0] and priors['beta'][1]
   starting_guesses[:, :, 1] *= (priors['beta'][1]-priors['beta'][0])  
   starting_guesses[:, :, 1] += priors['beta'][0]
   # Initial values for normalisation factor taken between priors['norm'][0] and priors['norm'][1]
   starting_guesses[:, :, 2] *= (priors['norm'][1]-priors['norm'][0])
   starting_guesses[:, :, 2] += priors['norm'][0]
   if ndim == 4:
       # Initial values for Av taken between priors['Av'][0] and priors['Av'][1]
       starting_guesses[:, :, 3] *= (priors['Av'][1]-priors['Av'][0]) 
       starting_guesses[:, :, 3] += priors['Av'][0]  

   # If sampler is Ensemble reshape the array containing the initial values
   if sampler_type == 'ensemble':
       starting_guesses=starting_guesses.reshape(nwalkers, ndim)

   list_notdetected=[]

   for counter,sed in enumerate(seds.group_by('Name').groups):
       sed.sort(['eff_wvl'])
       print ('\n\nFit {:d}/{:d} \t Object: {:s} \n'.format(counter+1,len(grb_info),sed['Name'][0]))
       #Check that there is at least a detection in one band
       if any(sed['detection']==1):
           mask = grb_info['name']==sed['Name'][0]
           z_sim=float(np.asscalar(grb_info['z'][mask]))
           Av_sim=float(np.asscalar(grb_info['Av_host'][mask]))
           if 'beta' in grb_info.colnames: beta_sim=float(np.asscalar(grb_info['beta'][mask]))
           else: beta_sim=0.
           print ('z_lit: {0:.2f}   Av_lit: {1:.2f}'.format(z_sim,Av_sim)) 

           #Normalisation values chosed to be the ones of the reddest band
           F0=sed['flux'][np.argmax(sed['eff_wvl'])]
           wvl0=sed['eff_wvl'][np.argmax(sed['eff_wvl'])]
           #print (wvl0,F0)
     
           #Substract the galctic extinction
           mag_corr=sed['mag']-sed['ext_mag']
     
           
           eff_wvl = np.array(sed['eff_wvl'])
           mag_err = np.array(sed['mag_err'])
           detection_flag = np.array(sed['detection'])
           sys_response = np.array(sed['sys_response'])
           zeropoints = np.array(sed['zeropoint'])
           mag_corr = np.array(mag_corr)

           # Set up the MCMC sampler
           if sampler_type == 'ensemble':
               sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, a=a, 
                                               threads=Nthreads, 
                          args = (eff_wvl, mag_corr, mag_err, detection_flag, wavelength, F0, wvl0, sys_response, zeropoints, ext_law, Host_dust, Host_gas, MW_dust, MW_gas, DLA, igm_att, priors))

           elif sampler_type == 'pt':
               sampler = emcee.PTSampler(nTemps, nwalkers, ndim, lnlike, lnprob, threads=Nthreads, a=a, loglargs = (eff_wvl, mag_corr, mag_err, detection_flag, wavelength, F0, wvl0, sys_response, zeropoints, ext_law, Host_dust, Host_gas, MW_dust, MW_gas, DLA, igm_att),logpargs = (eff_wvl, mag_corr, mag_err, detection_flag, wavelength, F0, wvl0, sys_response, zeropoints, ext_law, Host_dust, Host_gas, MW_dust, MW_gas, DLA, igm_att, priors))
               #for p, lnp, lnlik in sampler.sample(starting_guesses, iterations=Nsteps1):
               #                 pass
           
           # First run: burn-in
           pos=starting_guesses
           if Nsteps1>0:
               print("Running burn-in")
               pos, prob, state = sampler.run_mcmc(pos, Nsteps1)
               sampler.reset()

           # Second run: run used for the statisctics
           # Takes the values of the last steps of the burn-in run as initial values 
           if Nsteps2>0:
               print("Running production")
               if Nsteps1 > 0: 
                   print ('Nsteps1 > 0 --> Initial values are drawn from a Gaussian distribution with means equal to the values returning the best chi2 during first run and std of %.2e' % std_gaussianBall)
                   # Start from a gaussian centered on values returning the best chi2 for the first run
                   p = pos[np.unravel_index( np.nanargmax( prob ), prob.shape)]
                   if sampler_type == 'ensemble':
                       pos = [p + std_gaussianBall * np.random.randn(ndim) for i in range(nwalkers)]
                   elif sampler_type == 'pt':
                       pos = [[p + std_gaussianBall * np.random.randn(ndim) for i in range(nwalkers)] for t in range(nTemps)]
               sampler.run_mcmc(pos, Nsteps2)
              
           #Store the chains
           if sampler_type == 'ensemble': 
               samples_chain = sampler.chain[:, nburn:, :].copy()
               samples_lnproba = sampler.lnprobability[:, nburn:].copy()
           elif sampler_type == 'pt':
               samples_chain = sampler.chain[:, :, nburn:, :].copy()
               samples_lnproba = sampler.lnprobability[:, :, nburn:].copy()
           
           # Print the parameters for the best likelihood
           print ('\nBest fit:')
           mask_nan = np.isfinite(samples_lnproba)
           results_minL=return_bestlnproba(samples_lnproba,samples_chain,ndim)
           sum_proba=np.sum(np.exp(samples_lnproba[mask_nan]))
           mean_proba=np.mean(np.exp(samples_lnproba[mask_nan]))
           best_chi2=-2*samples_lnproba[np.unravel_index( np.nanargmax( samples_lnproba ), samples_lnproba.shape)]
           print ('\nMean Proba: %.2e' % mean_proba)
           print ('\nSum Proba: %.2e' % sum_proba)

           # Clean the chains. Normally no need to with this version              
           if clean_data: samples_corr, samples_del,index_removed = clean_chains(sampler, nburn, acceptance_frac_lim=0.15)
           else:
               samples_corr=samples_chain
               samples_del=None

           # Keep only data after discarding the "nburn" first steps
           if sampler_type == 'ensemble':
               data2keep=samples_corr[:, nburn:, :]
           elif sampler_type == 'pt':
               data2keep=samples_corr[:, :, nburn:, :]
  
           # If less than (nwalkers-5) chains remains after the "chains cleaning", do not compute statistics and ends here 
           if clean_data and (nwalkers - len(index_removed)) < 5:
               print ('WARNING: After cleaning the chains, only %d chains are left. A minimum of 5 walkers is required. Either increase the number of walkers or adapt the priors.' % len(samples_corr[:,0,0]))
               continue
           else:
               # Compute mean acceptance fraction
               mean_acceptance_fraction = np.mean(sampler.acceptance_fraction)
               print("Mean acceptance fraction: {0:.3f}"
                     .format(np.mean(sampler.acceptance_fraction)))
               # If too low decrease the "a" parameter

               # Create evolution plot
               if sampler_type == 'ensemble':
                   plot_mcmc_evolution(samples_corr, samples_del, nburn, ndim, ext_law, Av_sim, z_sim, sed['Name'][0], plot, plot_deleted, output_dir=output_dir, filename_suffix=filename_suffix, priors=priors)
               elif sampler_type == 'pt':
                   # Not implemented yet
                   pass 
    
               # Create the triangle plot 
               if ndim == 3: samplesTriangle=data2keep
               if ndim ==4:
                   if sampler_type == 'ensemble':
                       # Change orders to have norm at the bottom of triangle plot
                       samplesTriangle=data2keep.copy()
                       samplesTriangle[:,:,1]=data2keep[:, :, 3]
                       samplesTriangle[:,:,2]=data2keep[:, :, 1]
                       samplesTriangle[:,:,3]=data2keep[:, :, 2]
                   elif sampler_type == 'pt':
                       # Change orders to have norm at the bottom of triangle plot
                       samplesTriangle=data2keep[:, :, :, :].copy()
                       samplesTriangle[:,:,:,1]=data2keep[:, :, :, 3]
                       samplesTriangle[:,:,:,2]=data2keep[:, :, :, 1]
                       samplesTriangle[:,:,:,3]=data2keep[:, :, :, 2]

               plot_triangle(samplesTriangle.reshape((-1, ndim)), ndim, z_sim, ext_law, Av_sim, beta_sim, sed['Name'][0], plot, plot_deleted, filename_suffix=filename_suffix, output_dir=output_dir, priors=priors)
              
               # Compute statistics 
               if ndim == 3:
                   z_mcmc_68, beta_mcmc_68, norm_mcmc_68 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                                            zip(*np.percentile(data2keep.reshape((-1, ndim)), [16,50,84],
                                                                           axis=0)))
                   z_mcmc_95, beta_mcmc_95, norm_mcmc_95 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                                            zip(*np.percentile(data2keep.reshape((-1, ndim)), [2.5,50,97.5],
                                                                           axis=0)))
                   z_mcmc_99, beta_mcmc_99, norm_mcmc_99 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                                            zip(*np.percentile(data2keep.reshape((-1, ndim)), [0.15,50,99.85],
                                                                           axis=0)))

                   Av_mcmc_68=[0,0,0]
                   Av_mcmc_95=[0,0,0]
                   Av_mcmc_99=[0,0,0]

               elif ndim == 4:
                   z_mcmc_68, beta_mcmc_68, norm_mcmc_68, Av_mcmc_68 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(data2keep.reshape((-1, ndim)), [16,50,84],axis=0)))
                   z_mcmc_95, beta_mcmc_95, norm_mcmc_95, Av_mcmc_95 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(data2keep.reshape((-1, ndim)), [2.5,50,97.5],axis=0)))
                   z_mcmc_99, beta_mcmc_99, norm_mcmc_99, Av_mcmc_99 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(data2keep.reshape((-1, ndim)), [0.15,50,99.85],axis=0)))
     
               print ('\n68% - 1 sigma:')
               print ('z:{:.3f} +{:.3f} -{:.3f} \nAv:{:.3f} +{:.3f} -{:.3f} \nbeta: {:.3f} +{:.3f} -{:.3f} \nnorm: {:.3f} +{:.3f} -{:.3f}'.format(z_mcmc_68[0],z_mcmc_68[1],z_mcmc_68[2],Av_mcmc_68[0],Av_mcmc_68[1],Av_mcmc_68[2],beta_mcmc_68[0],beta_mcmc_68[1],beta_mcmc_68[2],norm_mcmc_68[0],norm_mcmc_68[1],norm_mcmc_68[2]))
               print ('\n95% - 2 sigmas:')
               print ('z:{:.3f} +{:.3f} -{:.3f} \nAv:{:.3f} +{:.3f} -{:.3f} \nbeta: {:.3f} +{:.3f} -{:.3f} \nnorm: {:.3f} +{:.3f} -{:.3f}'.format(z_mcmc_95[0],z_mcmc_95[1],z_mcmc_95[2],Av_mcmc_95[0],Av_mcmc_95[1],Av_mcmc_95[2],beta_mcmc_95[0],beta_mcmc_95[1],beta_mcmc_95[2],norm_mcmc_95[0],norm_mcmc_95[1],norm_mcmc_95[2]))
               print ('\n99% - 3 sigmas:')
               print ('z:{:.3f} +{:.3f} -{:.3f} \nAv:{:.3f} +{:.3f} -{:.3f} \nbeta: {:.3f} +{:.3f} -{:.3f} \nnorm: {:.3f} +{:.3f} -{:.3f}'.format(z_mcmc_99[0],z_mcmc_99[1],z_mcmc_99[2],Av_mcmc_99[0],Av_mcmc_99[1],Av_mcmc_99[2],beta_mcmc_99[0],beta_mcmc_99[1],beta_mcmc_99[2],norm_mcmc_99[0],norm_mcmc_99[1],norm_mcmc_99[2]))


               # Free memory
               del sampler

               #number of bands
               nb_bands=len(sed['band'])
               # Number of band with a signal
               nb_detected=sum(sed['detection'])
               # corresponding bands
               band_detected_list=sed[sed['detection']==1]['band']
               band_detected='' 
               for i in band_detected_list:
                   band_detected=band_detected+str(i)

               if ndim == 3: 
                    best_z, best_Av, best_slope, best_norm = results_minL[0], 0.0, results_minL[1], results_minL[2]
               elif ndim == 4: 
                    best_z, best_Av, best_slope, best_norm = results_minL[0], results_minL[3], results_minL[1], results_minL[2]

               #Save results
               results_current_run=[[sed['Name'][0]],[z_sim],[Av_sim],[ext_law],
                           [best_z], [best_Av], [best_slope], [best_norm],
                           [z_mcmc_68[0]],[z_mcmc_68[1]],[z_mcmc_68[2]],
                           [z_mcmc_95[0]],[z_mcmc_95[1]],[z_mcmc_95[2]],
                           [z_mcmc_99[0]],[z_mcmc_99[1]],[z_mcmc_99[2]],
                           [Av_mcmc_68[0]],[Av_mcmc_68[1]],[Av_mcmc_68[2]],
                           [Av_mcmc_95[0]],[Av_mcmc_95[1]],[Av_mcmc_95[2]],
                           [Av_mcmc_99[0]],[Av_mcmc_99[1]],[Av_mcmc_99[2]],
                           [beta_mcmc_68[0]],[beta_mcmc_68[1]],[beta_mcmc_68[2]],
                           [beta_mcmc_95[0]],[beta_mcmc_95[1]],[beta_mcmc_95[2]],
                           [beta_mcmc_99[0]],[beta_mcmc_99[1]],[beta_mcmc_99[2]],
                           [norm_mcmc_68[0]],[norm_mcmc_68[1]],[norm_mcmc_68[2]],
                           [norm_mcmc_95[0]],[norm_mcmc_95[1]],[norm_mcmc_95[2]],
                           [norm_mcmc_99[0]],[norm_mcmc_99[1]],[norm_mcmc_99[2]],
                           [best_chi2],[mean_proba],[sum_proba], [mean_acceptance_fraction],
                           [nb_bands],[nb_detected],[band_detected]]
               result_1_SED=Table(results_current_run,names = ['name', 'z_sim', 'Av_host_sim', 'ext_law', 'best_z', 'best_Av', 'best_slope', 'best_scaling', 'zphot_68', 'zphot_68_sup', 'zphot_68_inf', 'zphot_95', 'zphot_95_sup', 'zphot_95_inf', 'zphot_99', 'zphot_99_sup', 'zphot_99_inf', 'Av_68', 'Av_68_sup', 'Av_68_inf', 'Av_95', 'Av_95_sup', 'Av_95_inf', 'Av_99', 'Av_99_sup', 'Av_99_inf', 'beta_68', 'beta_68_sup', 'beta_68_inf', 'beta_95', 'beta_95_sup', 'beta_95_inf', 'beta_99', 'beta_99_sup', 'beta_99_inf', 'norm_68', 'norm_68_sup', 'norm_68_inf', 'norm_95', 'norm_95_sup', 'norm_95_inf', 'norm_99', 'norm_99_sup', 'norm_99_inf', 'best_chi2', 'mean_proba', 'sum_proba', 'mean_acc','nb_bands', 'nb_detection', 'band_detected'], dtype = ('S10', 'f8', 'f8', 'S10', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8','i2', 'i2', 'S10'))
               
               plot_mcmc_fit(result_1_SED,ndim,list(results_minL),sed,wavelength,samples_chain.reshape((-1, ndim)),plot_all,plot,ext_law,Host_dust,Host_gas,MW_dust,MW_gas,DLA,igm_att,output_dir=output_dir,filename_suffix=filename_suffix)
               # If detections, write a result file      
               ascii.write(result_1_SED,output_dir+str(sed['Name'][0])+'/best_fits_'+ext_law+filename_suffix+'.dat')

               results.append(result_1_SED)
       else: 
           #No detections
           print ('No detection in all bands for %s ' % sed['Name'][0])
           list_notdetected.append(sed['Name'][0])
   print ('\nList of GRB not detected: {}\n'.format(list_notdetected))
    
   #Write the grb params in an ascii file
   if list_notdetected:
       test_list=[]
       for name in list_notdetected:
           test_list.append(grb_info[grb_info['name']==name])
       ascii.write(np.array(list_notdetected),output_dir+'notdetected_'+ext_law+filename_suffix+'.dat')

   # If detections, write a result file      
   if results:
       results=vstack(results)
       ascii.write(results,output_dir+'best_fits_all_'+ext_law+filename_suffix+'.dat')

def clean_chains(sampler,nburn,acceptance_frac_lim=0.15):
   """ Clean the mcmc chains. So far just remove the chain with a low acceptance fraction and nan probability after burn-in phase. 
   """
   print_corr=True
   samples=sampler.chain
   lnproba=sampler.lnprobability

   index2remove=[]

   nwalkers=len(samples[:,0,0])

   for walker in range(nwalkers):

       # Remove all walker with low acceptance fraction
       if sampler.acceptance_fraction[walker] < acceptance_frac_lim:
           index2remove.append(walker)
           if print_corr: print ('Walker %d removed: mean acceptance fraction of %.2f < %.2f' % (walker, sampler.acceptance_fraction[walker], acceptance_frac_lim))
       else:
           # Searching for non finite probability after the burn-in phase
           mask = np.isfinite(lnproba[walker][nburn:]) == False
           if mask.any():
               index2remove.append(walker)
               if print_corr: print ('Walker %d removed: non finite probability found after burn-in phase' % walker)

   samples_corr=samples.copy()
   

   if not index2remove:
       print ('No walker removed for statistical analysis')
       samples_del=None
   else:
       #delete possible duplicates in index2remove
       index2remove=sorted(set(index2remove),reverse=True)
       print ("\n%d/%d walkers removed" % (len(index2remove),nwalkers))
       for i in index2remove:
           samples_corr=np.delete(samples_corr,i,0)

       samples_del=samples.copy()
       for i in range(nwalkers):
           i2=nwalkers-1-i
           if i2 not in index2remove:
               samples_del=np.delete(samples_del,i2,0)

   return samples_corr, samples_del, index2remove



def return_bestlnproba(lnproba,chain,ndim):
   """ Extract the values of the parameter for which the likelihood is min
   """
   #idx=np.ndarray.argmax(lnproba)
   idx = np.unravel_index( np.nanargmax( lnproba ), lnproba.shape)
   #print (lnproba)
   #print (idx, lnproba.shape)
   chi2=-2*lnproba[idx]
   #print ('\nbest fit:')
   if ndim==3:
       print ('z: {:.3f}  beta: {:.3f}  Norm: {:.3f}     chi2: {:.3f}'.format(chain[idx][0],chain[idx][1],chain[idx][2],chi2))
   elif ndim==4:
       print ('z: {:.3f}  Av: {:.3f}  beta: {:.3f}  Norm: {:.3f}     chi2: {:.3f}'.format(chain[idx][0],chain[idx][3],chain[idx][1],chain[idx][2],chi2))
   return chain[idx]


