
RADARS = {
    'EarthCARE': {
        'radar_name': 'ec',
        'frequency':94.0,
        'radar_attenuation': 'top-down',
        'obs_height':394000.0,
        'radar_fwhr_beamwidth_deg':0.3,#default value
        'radar_integration_time':0.095,
        'radar_max_v':15,# not real settings but otherwise it doesn't work
        'radar_min_v':-15,
        'radar_nfft':256,#default value
        'radar_pnoise0':-45.0,
        'radar_no_ave':150,#default value
        'radar_peak_min_snr':-10.0,
        
    },
    'HALO': {
        'radar_name': 'halo',
        'frequency':35.5,
        'radar_attenuation': 'top-down',
        'obs_height':15000.0,
        'radar_fwhr_beamwidth_deg':0.56,
        'radar_integration_time':1.024,
        'radar_max_v':15, # not real settings but otherwise it doesn't work
        'radar_min_v':-15,
        'radar_nfft':256,
        'radar_pnoise0':-62.0,
        'radar_no_ave':60,
        'radar_peak_min_snr':10.0,

    },
    'Meteor': {
        'radar_name': 'met',
        'frequency':94.0,
        'radar_attenuation': 'bottom-up',
        'obs_height':15000.0,
        'radar_fwhr_beamwidth_deg':0.5,
        'radar_integration_time':1.92,
        'radar_max_v':9.0,
        'radar_min_v':-9.0,
        'radar_nfft':256,
        'radar_pnoise0':-29.5,
        'radar_no_ave':17,
        'radar_peak_min_snr':-15.0,
        
    }
}

ICON_VERSIONS = {
    'orcestra': {
        'model_name': 'orc',
        'input_folder': '/work/mh0492/m301067/ec-curtains/data/curtains/build-master-031224/',
        'rain_N0': 8.0e6,
        'rain_velocity': 'corPowerLaw_115.7_0.444',
        'ice_velocity': 'corPowerLaw_30.606_0.5533'
    },
    'hackathon': {
        'model_name': 'hack',
        'input_folder': '/work/mh0492/m301067/ec-curtains/data/curtains/build-hackaton/',
        'rain_N0': 8.0e6,
        'rain_velocity': 'corPowerLaw_31.7_0.444',
        'ice_velocity': 'corPowerLaw_9.8_0.5533'
    },
    'new-rain': {
        'model_name': 'new',
        'input_folder': '/work/mh0492/m301067/ec-curtains/data/curtains/build-new-rain/',
        'rain_N0': 'logn',
        'rain_velocity': 'corAtlas_9.88_10.14_510.94', 
        #'rain_velocity': 'corAtlas_9.88_10.14_510.94', #for fit Atlas+square, 
        #'rain_velocity': 'corAtlas_9.2_9.2_600.0', #for regular Atlas non-zero (need to change back PaAMTRA souce code to use this velocity)
        'ice_velocity': 'corPowerLaw_9.8_0.5533'
    }
}

def get_params(radar, icon_version):
    try:
        params = {}
        params.update(RADARS[radar])
        params.update(ICON_VERSIONS[icon_version])
        return params
    except KeyError as e:
        raise ValueError(f"Unknown radar or ICON version: {e}")