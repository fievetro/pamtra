import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point, Polygon
from orcestra import get_flight_segments

# polygon of continents to exclude
polygon_america = Polygon([(-65, 11.5), (-61, 11.5), (-52, 6), (-47, 0.5), (-33, -5), (-65, -5)])
polygon_africa = Polygon([(-15, 25), (-18, 23), (-18, 12), (-15, 10)])
polygons = [polygon_america, polygon_africa]

def prepare_icon_dataset(file_icon, icon_path, itime, zz):
    ds_icon = xr.open_dataset(icon_path + file_icon)
    
    # Get exact time of the simulation
    icon_time = ds_icon['time'].values[itime]
    
    # Interpolate the data on a uniform mesh for more practical analysis
    zg = ds_icon['zg'].values[:, 0]
    ds_icon = ds_icon.assign_coords(height_full=zg)
    lat_icon = ds_icon['track_lat'][:]  
    lon_icon = ds_icon['track_lon'][:]
    
    ds_icon_interpolated = xr.Dataset()

    for var in ds_icon.data_vars:
        if 'height_full' in ds_icon[var].dims:
            #print(f"Interpolating variable: {var_name}")
            ds_icon_interpolated[var] = ds_icon[var].interp(height_full=zz, method='linear')
        else:
            ds_icon_interpolated[var]=ds_icon[var]

    # Extract the variables
    rho = ds_icon_interpolated['rho']  # Density
    qc = ds_icon_interpolated['qc']  # Cloud water content
    qi = ds_icon_interpolated['qi']  # Cloud ice content
    qs = ds_icon_interpolated['qs']  # Snow content
    qr = ds_icon_interpolated['qr']  # Rain content
    qg = ds_icon_interpolated['qg']  # Graupel content

    # Calculate total cloud condensate
    clwc_icon_data = (qc + qr)*1000*rho # TODO: check if this is correct
    clwc_icon_data = clwc_icon_data[itime,:,:].transpose()
    liq_water_path_icon = np.nansum(clwc_icon_data, axis=1) *100 #sum along the height dimension, *100 for vertical step of the interpolation
    ciwc_icon_data = (qi + qs + qg)*1000*rho
    ciwc_icon_data = ciwc_icon_data[itime,:,:].transpose()
    ice_water_path_icon = np.nansum(ciwc_icon_data, axis=1) *100
    cwc_icon_data = (qr)*1000*rho # (qc + qi + qs + qr + qg)*1000*rho # TODO: check if this is correct
    cwc_icon_data = cwc_icon_data[itime,:,:].transpose()
    
    # Transform everything into xr.DataArrays for uniformity with the other products
    liq_water_path_icon = xr.DataArray(liq_water_path_icon, dims=["track"], coords={"track": lat_icon}, name="liq_water_path_icon")
    ice_water_path_icon = xr.DataArray(ice_water_path_icon, dims=["track"], coords={"track": lat_icon}, name="ice_water_path_icon")
    clwc_icon_data = xr.DataArray(clwc_icon_data, dims=["track", "height_full"], coords={"track": lat_icon, "height_full": zz}, name="clwc_icon_data")
    ciwc_icon_data = xr.DataArray(ciwc_icon_data, dims=["track", "height_full"], coords={"track": lat_icon, "height_full": zz}, name="ciwc_icon_data")
    cwc_icon_data = xr.DataArray(cwc_icon_data, dims=["track", "height_full"], coords={"track": lat_icon, "height_full": zz},name="cwc_icon_data")
    
    # Limit coordinates to the area between 0 and 20 degrees of latitude
    liq_water_path_icon = liq_water_path_icon.where((lat_icon >= 0) & (lat_icon <= 20), np.nan)
    ice_water_path_icon = ice_water_path_icon.where((lat_icon >= 0) & (lat_icon <= 20), np.nan)
    clwc_icon_data = clwc_icon_data.where((lat_icon >= 0) & (lat_icon <= 20), np.nan)
    ciwc_icon_data = ciwc_icon_data.where((lat_icon >= 0) & (lat_icon <= 20), np.nan)
    cwc_icon_data = cwc_icon_data.where((lat_icon >= 0) & (lat_icon <= 20), np.nan)
    
    # To exclude data from the continents, create a 2D mask corresponding to the polygons defined at the beginning of this file
    points = [Point(lon, lat) for lon, lat in zip(lon_icon.values, lat_icon.values)]
    mask = np.array([any(poly.contains(point) for poly in polygons) for point in points])
    mask_1d = xr.DataArray(mask, dims=["track"], coords={"track": clwc_icon_data.track})
    mask_2d = xr.DataArray(np.repeat(mask[:, np.newaxis], clwc_icon_data.height_full.size, axis=1), dims=["track", "height_full"], coords={"track": clwc_icon_data.track, "height_full": clwc_icon_data.height_full})
    
    # replace data inside the continents by NaNs
    liq_water_path_icon = liq_water_path_icon.where(~mask_1d)
    ice_water_path_icon = ice_water_path_icon.where(~mask_1d)
    clwc_icon_data = clwc_icon_data.where(~mask_2d)
    ciwc_icon_data = ciwc_icon_data.where(~mask_2d)
    cwc_icon_data = cwc_icon_data.where(~mask_2d)
    return icon_time, liq_water_path_icon, ice_water_path_icon, lat_icon, lon_icon, clwc_icon_data, ciwc_icon_data, cwc_icon_data

def prepare_earthcare_dataset(file_earthcare, earthcare_path_CLP, zz):
    # Find time of EarthCARE flight
    first_ec_time = file_earthcare.split('T')[1][:4]
    second_ec_time = file_earthcare.split('T')[2][:4]
    first_ec_time = first_ec_time[:2]+":"+first_ec_time[2:4]
    second_ec_time = second_ec_time[:2]+":"+second_ec_time[2:4]
    
    # Open dataset
    file_earthcare_CLP = earthcare_path_CLP + file_earthcare
    ds_earthcare_CLP_geo = xr.open_dataset(file_earthcare_CLP, engine="h5netcdf", group="ScienceData/Geo")
    ds_earthcare_CLP_data = xr.open_dataset(file_earthcare_CLP, engine="h5netcdf", group="ScienceData/Data")

    # Interpolate the data on a uniform mesh for more practical analysis
    altitude_1d_CLP = ds_earthcare_CLP_geo["height"].mean(dim="phony_dim_0").values
    valid_mask_CLP = ~np.isnan(altitude_1d_CLP)
    altitude_clean_CLP = altitude_1d_CLP[valid_mask_CLP]
    ds_ec_interpolated_CLP = ds_earthcare_CLP_data.isel(phony_dim_1=np.where(valid_mask_CLP)[0])
    ds_ec_interpolated_CLP = ds_ec_interpolated_CLP.assign_coords(altitude=("phony_dim_1", altitude_clean_CLP))
    ds_ec_interpolated_CLP = ds_ec_interpolated_CLP.swap_dims({"phony_dim_1": "altitude"})

    ds_ec_interpolated2 = xr.Dataset()
    for var in ds_ec_interpolated_CLP.data_vars:
        if 'altitude' in ds_ec_interpolated_CLP[var].dims:
            ds_ec_interpolated2[var] = ds_ec_interpolated_CLP[var].interp(altitude=zz, method='nearest')
        else:
            ds_ec_interpolated2[var]=ds_ec_interpolated_CLP[var]

    # Extract the variables
    ref_ec = ds_ec_interpolated2['cloud_radar_reflectivity_10km']
    doppler_vel_ec = ds_ec_interpolated2['cloud_doppler_velocity_10km']
    
    liq_water_path_ec = ds_ec_interpolated2['cloud_water_path_10km']
    ice_water_path_ec = ds_ec_interpolated2['cloud_ice_water_path_10km']
    clwc_data = ds_ec_interpolated2['cloud_water_content_10km']
    ciwc_data = ds_ec_interpolated2['cloud_ice_content_10km']
    
    # Define and limit coordinates to the area between 0 and 20 degrees of latitude
    lat_CLP = ds_earthcare_CLP_geo['latitude'] 
    lon_CLP = ds_earthcare_CLP_geo['longitude'] 
    ref_ec = ref_ec.where((lat_CLP >= 0) & (lat_CLP <= 20), np.nan)
    doppler_vel_ec = doppler_vel_ec.where((lat_CLP >= 0) & (lat_CLP <= 20), np.nan)
    clwc_data = clwc_data.where((lat_CLP >= 0) & (lat_CLP <= 20), np.nan)
    ciwc_data = ciwc_data.where((lat_CLP >= 0) & (lat_CLP <= 20), np.nan)
    liq_water_path_ec = liq_water_path_ec.where((lat_CLP >= 0) & (lat_CLP <= 20), np.nan)
    ice_water_path_ec = ice_water_path_ec.where((lat_CLP >= 0) & (lat_CLP <= 20), np.nan)
    
    # Prepare the variables
    ref_ec = ref_ec.where(ref_ec >= -31)
    doppler_vel_ec = doppler_vel_ec.where(ref_ec >= -25)
    
    clwc_data = clwc_data.where(clwc_data >= 0)
    ciwc_data = ciwc_data.where(ciwc_data >= 0)
    cwc_ec_data = clwc_data.fillna(0) + ciwc_data.fillna(0)
    liq_water_path_ec = liq_water_path_ec.where(liq_water_path_ec >= 0, 0.0) 
    ice_water_path_ec = ice_water_path_ec.where(ice_water_path_ec >= 0, 0.0)

    # To exclude data from the continents, create a 2D mask corresponding to the polygons defined at the beginning of this file
    points = [Point(lon, lat) for lon, lat in zip(lon_CLP.values, lat_CLP.values)]
    mask = np.array([any(poly.contains(point) for poly in polygons) for point in points])
    mask_1d = xr.DataArray(mask, dims=["phony_dim_0"], coords={"phony_dim_0": ref_ec.phony_dim_0})
    mask_2d = xr.DataArray(np.repeat(mask[:, np.newaxis], ref_ec.altitude.size, axis=1), dims=("phony_dim_0", "altitude"), coords={"phony_dim_0": ref_ec.phony_dim_0, "altitude": ref_ec.altitude})
    
    # replace data inside the continents by NaNs
    ref_ec = ref_ec.where(~mask_2d)
    doppler_vel_ec = doppler_vel_ec.where(~mask_2d)
    liq_water_path_ec = liq_water_path_ec.where(~mask_1d)
    ice_water_path_ec = ice_water_path_ec.where(~mask_1d)
    clwc_data = clwc_data.where(~mask_2d)
    ciwc_data = ciwc_data.where(~mask_2d)
    cwc_ec_data = cwc_ec_data.where(~mask_2d)

    return first_ec_time, second_ec_time, liq_water_path_ec, ice_water_path_ec, lat_CLP, lon_CLP, clwc_data, ciwc_data, cwc_ec_data, ref_ec, doppler_vel_ec

def prepare_pamtra_dataset(file_pamtra, itime, zz, lat_min, lat_max):
    ds_pamtra = xr.open_dataset(file_pamtra)
    
    # Get and prepare values
    reflectivity_pam = ds_pamtra['Ze'][:,:,:,0,0,0]
    reflectivity_pam = reflectivity_pam[itime,:,:] # itime is the timestep
    reflectivity_pam = reflectivity_pam.where(reflectivity_pam >= -31)
    doppler_pam = ds_pamtra['Radar_MeanDopplerVel'][:,:,:,0,0,0]
    doppler_pam = doppler_pam[itime,:,:] # select timestep
    doppler_pam = doppler_pam.where(reflectivity_pam >= -25)
    
    height_pam = ds_pamtra['height'][itime,:,:].values
    height_pam_1d = np.nanmean(height_pam, axis=0)
    lat_pam = ds_pamtra['latitude'][itime,:].values
    lon_pam = ds_pamtra['longitude'][itime,:].values

    # Create new datasets for interpolation
    ds_ref_pam = xr.DataArray(
        reflectivity_pam,
        dims=("ray", "height"),
        coords={"height": ("height", height_pam_1d),
        "ray": np.arange(reflectivity_pam.shape[0]),
        "latitude": ("ray", lat_pam),
        "longitude": ("ray", lon_pam),},
        name="Reflectivity"
    )
    
    ds_dopp_pam = xr.DataArray(
        -doppler_pam,
        dims=("ray", "height"),
        coords={"height": ("height", height_pam_1d),
        "ray": np.arange(doppler_pam.shape[0]),
        "latitude": ("ray", lat_pam),
        "longitude": ("ray", lon_pam),},
        name="Doppler_velocity"
    )
    
    # Interpolation on height axis
    ref_pam_interp = ds_ref_pam.interp(height=zz, method='linear')
    dopp_pam_interp = ds_dopp_pam.interp(height=zz, method='linear')
    
    # Limit coordinates to the area between 0 and 20 degrees of latitude
    ref_pam_interp = ref_pam_interp.where((ref_pam_interp['latitude'] >= lat_min) & (ref_pam_interp['latitude'] <= lat_max), np.nan)
    dopp_pam_interp = dopp_pam_interp.where((ref_pam_interp['latitude'] >= lat_min) & (ref_pam_interp['latitude'] <= lat_max), np.nan)
    
    ref_pam_interp = ref_pam_interp.where(ref_pam_interp >= -31)
    dopp_pam_interp = dopp_pam_interp.where(ref_pam_interp >= -25)
    
    # To exclude data from the continents, create a 2D mask corresponding to the polygons defined at the beginning of this file
    points = [Point(lon, lat) for lon, lat in zip(lon_pam, lat_pam)]
    mask = np.array([any(poly.contains(point) for poly in polygons) for point in points])
    mask_2d = xr.DataArray(np.repeat(mask[:, np.newaxis], ref_pam_interp.height.size, axis=1), dims=("ray", "height"), coords={"ray": ref_pam_interp.ray, "height": ref_pam_interp.height})
    
    # replace data inside the continents by NaNs
    ref_pam_interp = ref_pam_interp.where(~mask_2d)
    dopp_pam_interp = dopp_pam_interp.where(~mask_2d)
    
    return ref_pam_interp, dopp_pam_interp, lat_pam, lon_pam

def prepare_jsim_dataset(file_jsim, zz):
    ds_jsim = xr.open_dataset(file_jsim)
    # Get and prepare values
    reflectivity_jsim = ds_jsim['dbz'][0,:,:,0]
    reflectivity_jsim = reflectivity_jsim.transpose("ny", "nz")
    reflectivity_jsim = reflectivity_jsim.where(reflectivity_jsim >= -31)
    doppler_jsim = ds_jsim['dvl'][0,:,:,0]
    doppler_jsim = doppler_jsim.transpose("ny", "nz")
    doppler_jsim = doppler_jsim.where(reflectivity_jsim >= -25)
    height_jsim_1d = ds_jsim['height'].values
    lat_jsim = ds_jsim['latitude'][:,0].values
    lon_jsim = ds_jsim['longitude'][:,0].values

    # Create new datasets for interpolation
    ds_ref_jsim = xr.DataArray(
        reflectivity_jsim,
        dims=("ny", "height"),
        coords={"height": ("height", height_jsim_1d),
        "ny": np.arange(reflectivity_jsim.shape[0]),
        "latitude": ("ny", lat_jsim),
        "longitude": ("ny", lon_jsim),},
        name="Reflectivity"
    )
    
    ds_dopp_jsim = xr.DataArray(
        doppler_jsim,
        dims=("ny", "height"),
        coords={"height": ("height", height_jsim_1d),
        "ny": np.arange(doppler_jsim.shape[0]),
        "latitude": ("ny", lat_jsim),
        "longitude": ("ny", lon_jsim),},
        name="Doppler_velocity"
    )
    
    # Interpolation on height axis
    ref_jsim_interp = ds_ref_jsim.interp(height=zz, method='linear')
    dopp_jsim_interp = ds_dopp_jsim.interp(height=zz, method='linear')
    
    ref_jsim_interp = ref_jsim_interp.where(ref_jsim_interp >= -31)
    dopp_jsim_interp = dopp_jsim_interp.where(ref_jsim_interp >= -25)
    
    # Limit coordinates to the area between 0 and 20 degrees of latitude
    ref_jsim_interp = ref_jsim_interp.where((ref_jsim_interp['latitude'] >= 0) & (ref_jsim_interp['latitude'] <= 20), np.nan)
    dopp_jsim_interp = dopp_jsim_interp.where((ref_jsim_interp['latitude'] >= 0) & (ref_jsim_interp['latitude'] <= 20), np.nan)
    
    # To exclude data from the continents, create a 2D mask corresponding to the polygons defined at the beginning of this file
    points = [Point(lon, lat) for lon, lat in zip(lon_jsim, lat_jsim)]
    mask = np.array([any(poly.contains(point) for poly in polygons) for point in points])
    mask_2d = xr.DataArray(np.repeat(mask[:, np.newaxis], ref_jsim_interp.height.size, axis=1), dims=("ny", "height"), coords={"ny": ref_jsim_interp.ny, "height": ref_jsim_interp.height})
    
    # replace data inside the continents by NaNs
    ref_jsim_interp = ref_jsim_interp.where(~mask_2d)
    dopp_jsim_interp = dopp_jsim_interp.where(~mask_2d)
    
    return ref_jsim_interp, dopp_jsim_interp, lat_jsim, lon_jsim

def prepare_halo_dataset(ds_halo, date_str, time_str, zz):    
    # Preparation to get only the segments following EarthCARE'track on the day date_str
    flight_id = "HALO-" + date_str[0:4] + date_str[5:7] + date_str[8:10] + "a"
    meta = get_flight_segments()
    segments = meta["HALO"][flight_id]["segments"]
    ec_segments = [s for s in segments if "ec_track" in s["kinds"]]
    
    # Create a mask and select only the segments following EarthCARE's track
    mask_ec = xr.zeros_like(ds_halo['time'], dtype=bool)
    for seg in ec_segments:
        start_time = np.datetime64(seg["start"])
        end_time   = np.datetime64(seg["end"])
        mask_ec = mask_ec | ((ds_halo['time'] >= start_time) & (ds_halo['time'] < end_time))
    
    # Get first and last times of these segments
    first_halo_time = f"{pd.to_datetime(ec_segments[0]['start']).hour}:{str(pd.to_datetime(ec_segments[0]['start']).minute).zfill(2)}"
    second_halo_time = f"{pd.to_datetime(ec_segments[-1]['end']).hour}:{str(pd.to_datetime(ec_segments[-1]['end']).minute).zfill(2)}"
    ds_halo_filtered = ds_halo.where(mask_ec, drop=True)
    
    # Find the latitude at overpass time
    target_time = np.datetime64(pd.Timestamp(date_str + ' ' + time_str))
    time_diff = np.abs(ds_halo_filtered['time'].values - target_time)
    min_idx = np.argmin(time_diff)
    overpass_lat = ds_halo_filtered['lat'].values[min_idx]
    
    # Interpolation on height axis
    ds_halo_interpolated = xr.Dataset()
    for var in ds_halo_filtered.data_vars:
        if 'height' in ds_halo_filtered[var].dims:
            ds_halo_interpolated[var] = ds_halo_filtered[var].interp(height=zz, method='linear')
        else:
            ds_halo_interpolated[var]=ds_halo_filtered[var]
        
    # Get the products
    ref_halo = ds_halo_interpolated['dBZg']
    ref_halo = ref_halo.where(ref_halo >= -31)
    lat_halo = ds_halo_interpolated['lat']
    lon_halo = ds_halo_interpolated['lon']
    
    # Limit coordinates to the area between 0 and 20 degrees of latitude
    ref_halo = ref_halo.where((lat_halo >= 0) & (lat_halo <= 20), np.nan)
        
    # To exclude data from the continents, create a 2D mask corresponding to the polygons defined at the beginning of this file
    points = [Point(lon, lat) for lon, lat in zip(lon_halo.values, lat_halo.values)]
    mask = np.array([any(poly.contains(point) for poly in polygons) for point in points])
    mask_2d = xr.DataArray(np.repeat(mask[:, np.newaxis], ref_halo.height.size, axis=1), dims=("time", "height"), coords={"time": ref_halo.time, "height": ref_halo.height})
    
    # replace data inside the continents by NaNs
    ref_halo = ref_halo.where(~mask_2d)
        
    return first_halo_time, second_halo_time, lat_halo, lon_halo, ref_halo, overpass_lat