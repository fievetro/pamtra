import os
import pyPamtra
import xarray as xr
import numpy as np

# CAREFUL: if you don't restart the kernel after changing the pamtra_settings.py file, it will not take the new parameters into account
from settings import get_params

output_path= '/work/mh0731/m301196/ecomip/pamtra-curtains-july/'
radar = 'EarthCARE' # EarthCARE, HALO, Meteor
icon_version = 'new-rain' # orcestra, hackathon, new-rain
params= get_params(radar, icon_version) # in order to get the accurate parameters for the radar and icon version
        
# Define descriptorFiles 
if params['rain_N0']=='logn':
    descriptorFile = np.array([ # Can be modified if ICON parameters are changed
    ('cwc_q', 1.0,  1, -99.0,   -99.0, -99.0,  -99.0, -99.0,  3,  1,   'mono',           -99.0, -99.0, -99.0, -99.0, 2.0e-5,  -99.0, 'mie-sphere', 'corPowerLaw_24388657.6_2.0', -99.0),
    ('iwc_q', 0.2, -1, -99.0,   130.0,   3.0,  0.684,   2.0,  3,  1,   'mono_cosmo_ice', -99.0, -99.0, -99.0, -99.0, -99.0,  -99.0, 'ssrg-rt3_0.18_0.89_2.06_0.08', params['ice_velocity'], -99.0),
    ('rwc_q', 1.0,  1, -99.0,   -99.0, -99.0,  -99.0, -99.0,  3,  100, 'logn',            300, -99.0, np.log(1.4), -99.0, 1.0e-4, 5.0e-3, 'mie-sphere', params['rain_velocity'], -99.0),
    ('swc_q', 0.6, -1, -99.0,   0.069,   2.0, 0.3971,  1.88,  3,  100, 'exp_cosmo_snow', -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, 'ssrg-rt3_0.25_1.00_1.66_0.04', 'corPowerLaw_4.9_0.25', -99.0),
    ('gwc_q', 1.0, -1, -99.0,   169.6,   3.1,  -99.0, -99.0,  3,  100, 'exp',            -99.0, -99.0, 4.0e6, -99.0, -99.0, -99.0, 'mie-sphere', 'corPowerLaw_442.0_0.89', -99.0)
    ],)
else:
    descriptorFile = np.array([ # Can be modified if ICON parameters are changed
    ('cwc_q', 1.0,  1, -99.0,   -99.0, -99.0,  -99.0, -99.0,  3,  1,   'mono',           -99.0, -99.0, -99.0, -99.0,  2.0e-5,  -99.0, 'mie-sphere', 'corPowerLaw_24388657.6_2.0', -99.0),
    ('iwc_q', 0.2, -1, -99.0,   130.0,   3.0,  0.684,   2.0,  3,  1,   'mono_cosmo_ice', -99.0, -99.0, -99.0, -99.0,   -99.0,  -99.0, 'ssrg-rt3_0.18_0.89_2.06_0.08', params['ice_velocity'], -99.0),
    ('rwc_q', 1.0,  1, -99.0,   -99.0, -99.0,  -99.0, -99.0,  3,  100, 'exp',            -99.0, -99.0, params['rain_N0'], -99.0,  -99.0, -99.0, 'mie-sphere', params['rain_velocity'], -99.0),
    ('swc_q', 0.6, -1, -99.0,   0.069,   2.0, 0.3971,  1.88,  3,  100, 'exp_cosmo_snow', -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, 'ssrg-rt3_0.25_1.00_1.66_0.04', 'corPowerLaw_4.9_0.25', -99.0),
    ('gwc_q', 1.0, -1, -99.0,   169.6,   3.1,  -99.0, -99.0,  3,  100, 'exp',            -99.0, -99.0, 4.0e6, -99.0, -99.0, -99.0, 'mie-sphere', 'corPowerLaw_442.0_0.89', -99.0)
    ],)

files_icon = ["orcestra_ec-curtain_2024-08-11_15h51.nc", "orcestra_ec-curtain_2024-08-13_15h39.nc", "orcestra_ec-curtain_2024-08-16_16h13.nc",
              "orcestra_ec-curtain_2024-08-18_16h03.nc", "orcestra_ec-curtain_2024-08-22_15h41.nc", "orcestra_ec-curtain_2024-08-25_16h11.nc",
              "orcestra_ec-curtain_2024-08-27_16h01.nc", "orcestra_ec-curtain_2024-08-31_15h38.nc", "orcestra_ec-curtain_2024-09-03_16h08.nc",
              "orcestra_ec-curtain_2024-09-07_15h47.nc", "orcestra_ec-curtain_2024-09-09_17h05.nc", "orcestra_ec-curtain_2024-09-14_17h25.nc",
              "orcestra_ec-curtain_2024-09-16_17h15.nc", "orcestra_ec-curtain_2024-09-19_17h43.nc", "orcestra_ec-curtain_2024-09-23_17h22.nc",
              "orcestra_ec-curtain_2024-09-24_18h02.nc", "orcestra_ec-curtain_2024-09-26_17h51.nc", "orcestra_ec-curtain_2024-09-28_17h41.nc"]

for i in range(len(files_icon)): #len(files_icon) if you want to go through the whole list
    # Load the data
    data = xr.open_dataset(params['input_folder'] + files_icon[i]).transpose('time','track','height_full','height_half') # reorder dimensions
    data = data.isel(height_full=slice(33, 90), height_half=slice(33, 91)) # Crop the first 5 vertical levels to avoid issues with low pressure in pyPamtra

    # Get ICON time and date
    date_str = files_icon[i].split('_')[2][:10]
    time_str = files_icon[i].split('_')[3][:5]
    date_str = date_str + '_' + time_str

    # Get some data reshaped
    Ntimes = data.sizes['time']
    Npoints = data.sizes['track']
    Nhgt = data.sizes['height_half']
    Nhydro = len(descriptorFile)

    times = data.time.values.astype(np.float64)*1.0e-9 # to seconds
    times = np.tile(times[:, np.newaxis], (1, Npoints)) # this is model time, not track time
    lats = np.tile(data.track_lat.values, (Ntimes, 1))
    lons = np.tile(data.track_lon.values, (Ntimes, 1))

    zghalf = np.tile(data.zghalf.values[:, ::-1], (Ntimes, 1, 1))
    press = data.pfull.values[:,:,::-1]
    temp = data.ta.values[:,:,::-1]
    qv = data.qv.values[:,:,::-1]
    rh = 100*pyPamtra.meteoSI.q2rh(qv, temp, press)
    qc = data.qc.values[:,:,::-1]
    qi = data.qi.values[:,:,::-1]
    qr = data.qr.values[:,:,::-1]
    qs = data.qs.values[:,:,::-1]
    qg = data.qg.values[:,:,::-1]

    uv = np.hypot(data.ua, data.va).values[:,:,::-1]
    w = data.wa.values[:,:,::-1]
    w = 0.5*(w[:,:,:-1]+w[:,:,1:])

    pam = pyPamtra.pyPamtra()
    for hydro in descriptorFile:
        pam.df.addHydrometeor(hydro)

    pam.createProfile(hgt_lev=zghalf,
                    press=press,
                    temp=temp,
                    relhum=rh,
                    hydro_q=np.stack([qc, qi, qr, qs, qg], axis=-1),
                    wind_uv=uv,
                    wind_w=w,
                    timestamp=times,
                    lat=lats,
                    lon=lons,
                    )

    # SETTINGS
    pam.nmlSet["passive"] = False
    pam.nmlSet["active"] = True
    pam.nmlSet["emissivity"] = 93
    pam.nmlSet["radar_mode"] = 'moments'
    pam.set['verbose'] = 0 # set verbosity levels
    pam.set['pyVerbose'] = 0 # change to 0 if you do not want to see job progress number
    pam.p['turb_edr'][:] = 1.0e-4 # If you want to include estimated model turbulence it is possible
    pam.nmlSet['radar_airmotion'] = True
    pam.nmlSet['radar_airmotion_vmin'] = 0.0
    pam.nmlSet['radar_airmotion_model'] = 'constant'
    
    # Set specific radar parameters
    pam.nmlSet['radar_attenuation'] = params['radar_attenuation']
    pam.nmlSet['obs_height'] = params['obs_height']
    pam.nmlSet['radar_fwhr_beamwidth_deg'] = params['radar_fwhr_beamwidth_deg']
    pam.nmlSet['radar_integration_time'] = params['radar_integration_time']
    pam.nmlSet['radar_max_v'] = params['radar_max_v']
    pam.nmlSet['radar_min_v'] = params['radar_min_v']
    pam.nmlSet['radar_nfft'] = params['radar_nfft']
    pam.nmlSet['radar_pnoise0'] = params['radar_pnoise0']
    pam.nmlSet['radar_no_ave'] = params['radar_no_ave']
    pam.nmlSet['radar_peak_min_snr'] = params['radar_peak_min_snr']
    

    pam.runParallelPamtra(np.array([params['frequency']]), pp_deltaX=100, pp_deltaY=1, pp_deltaF=1, pp_local_workers=96)
    # output_folder = output_path + 'pamtra-' + params['model_name'] + '-' + params['radar_name'] + '/'
    # os.makedirs(output_folder, exist_ok=True)
    #pam.writeResultsToNetCDF(output_folder +'pamtra-'+ params['model_name'] + '-' + params['radar_name'] + '-' + date_str +'-non-att-no-obs-h.nc') # save output
    if params['rain_N0'] != 'logn':
        rain_N0_str = "{:.0e}".format(params['rain_N0']).replace("+0", "").replace(".e", "e")
    else:
        rain_N0_str = params['rain_N0']
    pam.writeResultsToNetCDF(output_path + 'pamtra-' + params['model_name'] + '-' + params['radar_name'] + '-' + date_str + '-' + rain_N0_str + '-' + params['rain_velocity'].split('_')[0]+ 'new-sigma-atlas-carre-bis.nc')  # save output

