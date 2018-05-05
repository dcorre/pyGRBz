# -*- coding: utf-8 -*-
import numpy as np
import os
from .utils import mag2Jy,convAB
from astropy.table import Table,Column
from astropy.io import ascii
from .extinction_correction import correct_MW_ext
from .io_grb import load_telescope_transmissions

def load_sys_response(data,wavelength,path='/home/dcorre/code/python_etc/grb_photoz/grb_photoz/'):
    """ Load the system throuput curves for each filter in the data

    Returns 
    -------
    sys_rep: astropy Table

    """
    dwvl=np.gradient(wavelength) #works for constant wvl_step

    sys_res=[]
    tel_name=[]
    tel_band=[]
    wvl_eff=[]
    width=[]
    zp=[]

    #Sort the telescope used
    for tel in data.group_by(['telescope','band']).groups.keys:
              #Import the filter throughput curve only once if filter used several times (for a light curve for instance)
              tel_name.append(tel['telescope'])
              tel_band.append(tel['band'])

              #Import the throughput curve
              filter_trans=load_telescope_transmissions({'telescope':tel['telescope'],'band':tel['band'],'path':path},wavelength)
              sys_res.append(filter_trans)

              #calculate the effective wavelength
              a=np.trapz(wavelength*filter_trans, wavelength)
              b=np.trapz(filter_trans, wavelength)
              wvl_eff.append(a/b)
              #print (a/b)

              #Calculate the width of the band
              mask = filter_trans > 0.05*max(filter_trans)
              width.append(wavelength[mask][-1] -  wavelength[mask][0])

              #Not sure it is used anymore ... to be checked. Formula also to check
              zp.append(2.5*np.log10(np.sum(filter_trans*dwvl,axis=0)) + 23.9)

         
    sys_res_table = Table([tel_name,tel_band,wvl_eff,width,sys_res,zp],names=['telescope','band','wvl_eff','band_width','sys_response','zeropoint'])
    #Sort the table by telescope names and ascending eff. wavelength
    sys_res_table.sort(['telescope','wvl_eff'])
    return sys_res_table 


def formatting_data(data,system_response,grb_info,wavelength,dustrecalib='yes',dustmapdir=os.getenv('GFTSIM_DIR')+'/los_extinction/los_extinction/galactic_dust_maps'):
    """ """
    # Add filter info to data (throughut curve,eff. wvl and width)
    col_band_width=Column(name='band_width',data=np.zeros(len(data)))
    col_band_effwvl=Column(name='eff_wvl',data=np.zeros(len(data)))
    col_band_zp=Column(name='zeropoint',data=np.zeros(len(data)))
    col_band_sysres=Column(name='sys_response',data=np.zeros((len(data),len(wavelength))))
    data.add_columns([col_band_effwvl,col_band_width,col_band_zp,col_band_sysres])

    for table in data.group_by(['telescope','band']).groups.keys:
         #print (table)
         mask1 = data['telescope'] == table['telescope']
         mask1[mask1==True]=data[mask1]['band'] == table['band']
         mask2 = system_response['telescope'] == table ['telescope']
         mask2[mask2==True] = system_response[mask2]['band'] == table ['band']
         #print (system_response[mask3][mask4]['sys_response'])
         #print (system_response[mask3][mask4]['sys_response'][0])

         width=[]
         effwvl=[]
         zp=[]
         sys_res=[]
         for i in range(np.sum(mask2)):
             width.append(system_response[mask2]['band_width'][0])
             effwvl.append(system_response[mask2]['wvl_eff'][0])
             zp.append(system_response[mask2]['zeropoint'][0])
             sys_res.append(system_response[mask2]['sys_response'][0])
         data['band_width'][mask1]=width
         data['eff_wvl'][mask1]=effwvl
         data['zeropoint'][mask1]=zp
         data['sys_response'][mask1]=sys_res

    #Convert vega magnitudes in AB if needed
    mask1=data['phot_sys'] == 'vega'
    if mask1.any():
         #print ('some vega')

         # If a Vega-AB correction is present in the file use this value

         if 'ABcorr' in data.colnames:
             mask2 = (data['phot_sys'] == 'vega') & (~data['ABcorr'].mask )
             if mask2.any():
                  #print ('AB corr')
                  for table in data[mask2]:
                      mask3 = mask2.copy()

                      mask3[mask3==True] = data[mask3]['Name'] == table['Name']
                      mask3[mask3==True] = data[mask3]['telescope'] == table['telescope']
                      mask3[mask3==True] = data[mask3]['band'] == table['band']
                      mask3[mask3==True] = data[mask3]['time_since_burst'] == table['time_since_burst']

                      newABmag=table['mag'] + table['ABcorr']
                      photsys='AB'
                      #substitute the vega magnitudes by AB ones
                      data['mag'][mask3]=newABmag
                      data['phot_sys'][mask3]=photsys

             # When no AB correction is given in input file, compute it
             mask2 = (data['phot_sys'] == 'vega') & (data['ABcorr'].mask )
             if mask2.any():
              
                  #print ('convAB')
                  for table in data[mask2]:
                      mask3 = mask2.copy()

                      mask3[mask3==True] = data[mask3]['Name'] == table['Name']
                      mask3[mask3==True] = data[mask3]['telescope'] == table['telescope']
                      mask3[mask3==True] = data[mask3]['band'] == table['band']
                      mask3[mask3==True] = data[mask3]['time_since_burst'] == table['time_since_burst']

                      newABmag=table['mag'] + convAB(wavelength,table['sys_response'])
                      photsys='AB'
                      #substitute the vega magnitudes by AB ones
                      data['mag'][mask3]=newABmag
                      data['phot_sys'][mask3]=photsys

         else:
              #print ('convAB')
              for table in data[mask1]:
                  mask3 = mask1.copy()

                  mask3[mask3==True] = data[mask3]['Name'] == table['Name']
                  mask3[mask3==True] = data[mask3]['telescope'] == table['telescope']
                  mask3[mask3==True] = data[mask3]['band'] == table['band']
                  mask3[mask3==True] = data[mask3]['time_since_burst'] == table['time_since_burst']

                  newABmag=table['mag'] + convAB(wavelength,table['sys_response'])
                  photsys='AB'
                  #substitute the vega magnitudes by AB ones
                  data['mag'][mask3]=newABmag
                  data['phot_sys'][mask3]=photsys
    
    # Correct for galactic extinction
    data=correct_MW_ext(data,grb_info,wavelength,dustmapdir=dustmapdir,recalibration=dustrecalib)

    #Add Flux to the seds 
    convert_dict={'photometry_system': 'AB'}

    flux=mag2Jy(convert_dict,data['mag']-data['ext_mag'])*1e6
    flux_err=np.array(abs(flux * -0.4 * np.log(10) * data['mag_err']))
    mask= data['detection'] == -1
    if mask.any():
         flux_err[mask]=flux[mask]/2.

    col_flux=Column(name='flux',data=flux,unit='microJy')
    col_flux_err=Column(name='flux_err',data=flux_err,unit='microJy')
 
    data.add_columns([col_flux,col_flux_err])
    
    data.sort(['Name','eff_wvl'])
    return data



