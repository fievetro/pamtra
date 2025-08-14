'''
Use this script to plot day-to-day comparison between ICON and observations. 
'''

# Import modules and datasets
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14, 'font.family': 'TimesNewRoman', 'text.usetex': True})
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.5

from tools.file_names import *
from tools.prepare_dataset_functions import *
from tools.plot_functions import *

## What do you want to plot for the reflectivity? ##
radar = 'ec' #'ec' or 'halo'
orcestra = True #True or False, depending on if you want to see orcestra results or not
hackathon = True #True or False, depending on if you want to see hackathon results or not
new_rain = True #True or False, depending on if you want to see new-rain results or not
include_obs = True #True or False, depending on if you want to see the observation results or not

# Get HALO data
ds_halo = xr.open_dataset(halo_path, engine="zarr")

# Common altitude grid
zz = np.arange(25, 16026, 200)

for j in range(8,9): #  len(files_earthcare) if you want to loop through all the files
    print(f"Processing file {j+1}/{len(files_earthcare)}")
    itime_orc = 3 #overpass time
    itime_hack = 6 #overpass time
    
    # Find date and time of ICON for the plot
    file_icon = files_icon[j] 
    date_str = file_icon.split('_')[2][:10]
    time_str = file_icon.split('_')[3][:5]
    day = f"{date_str[8:10]}/{date_str[5:7]}/{date_str[0:4]}"
    hour = time_str[:2] + ':' + time_str[3:]
    
    # Get ICON values
    icon_time_orc, liq_water_path_icon_o, ice_water_path_icon_o, lat_icon_o, lon_icon_o, clwc_icon_data_o, ciwc_icon_data_o, cwc_icon_data_o = prepare_icon_dataset(files_icon[j], icon_path_orc, itime_orc, zz)
    icon_time_hack, liq_water_path_icon_h, ice_water_path_icon_h, lat_icon_h, lon_icon_h, clwc_icon_data_h, ciwc_icon_data_h, cwc_icon_data_h = prepare_icon_dataset(files_icon[j], icon_path_hack, itime_hack, zz)
    icon_time_new, liq_water_path_icon_n, ice_water_path_icon_n, lat_icon_n, lon_icon_n, clwc_icon_data_n, ciwc_icon_data_n, cwc_icon_data_n = prepare_icon_dataset(files_icon[j], icon_path_new, itime_hack, zz)

    # Get observations
    if radar == 'ec':
        first_ec_time, second_ec_time, liq_water_path_ec, ice_water_path_ec, lat_ec, lon_ec, clwc_data, ciwc_data, cwc_ec_data, ref_ec, dopp_ec = prepare_earthcare_dataset(files_earthcare[j], earthcare_path, zz)
    elif radar == 'halo':
        first_halo_time, second_halo_time, lat_halo, lon_halo, ref_halo, overpass_lat = prepare_halo_dataset(ds_halo, date_str, time_str, zz)
    
    # Get PAMTRA values
    file_pamtra_orc = pamtra_path + 'orc-' + radar + '-' + files_pamtra[j] + '-8e6-corPowerLaw.nc'
    file_pamtra_hack = pamtra_path + 'hack-' + radar + '-' + files_pamtra[j] + '-8e6-corPowerLaw.nc'
    file_pamtra_new = pamtra_path + 'new-' + radar + '-' + files_pamtra[j] + '-logn-corAtlasnew-sigma-atlas-carre.nc'
    
    if radar == 'ec':
        ref_pam_orc, dopp_pam_orc, lat_pam_orc, lon_pam_orc = prepare_pamtra_dataset(file_pamtra_orc, itime_orc, zz, float(np.nanmin(lat_ec)), float(np.nanmax(lat_ec)))
        ref_pam_hack, dopp_pam_hack, lat_pam_hack, lon_pam_hack = prepare_pamtra_dataset(file_pamtra_hack, itime_hack, zz, float(np.nanmin(lat_ec)), float(np.nanmax(lat_ec)))  
        ref_pam_new, dopp_pam_new, lat_pam_new, lon_pam_new = prepare_pamtra_dataset(file_pamtra_new, itime_hack, zz, float(np.nanmin(lat_ec)), float(np.nanmax(lat_ec)))  
    elif radar == 'halo':     
        ref_pam_orc, dopp_pam_orc, lat_pam_orc, lon_pam_orc = prepare_pamtra_dataset(file_pamtra_orc, itime_orc, zz, float(np.nanmin(lat_halo)), float(np.nanmax(lat_halo)))
        ref_pam_hack, dopp_pam_hack, lat_pam_hack, lon_pam_hack = prepare_pamtra_dataset(file_pamtra_hack, itime_hack, zz, float(np.nanmin(lat_halo)), float(np.nanmax(lat_halo))) 
        ref_pam_new, dopp_pam_new, lat_pam_new, lon_pam_new = prepare_pamtra_dataset(file_pamtra_new, itime_hack, zz, float(np.nanmin(lat_halo)), float(np.nanmax(lat_halo)))  
            
    # Prepare the scales
    vmin_r = -30
    vmax_r = 30
    colorbar_ticks_r = [vmin_r, 0, vmax_r]

    plotted_hour_orc = icon_time_orc.astype('datetime64[h]').item().hour
    plotted_minute_orc = icon_time_orc.astype('datetime64[m]').item().minute
    plotted_hour_hack = icon_time_hack.astype('datetime64[h]').item().hour
    plotted_minute_hack = icon_time_hack.astype('datetime64[m]').item().minute
    plotted_hour_new = icon_time_new.astype('datetime64[h]').item().hour
    plotted_minute_new = icon_time_new.astype('datetime64[m]').item().minute
    
    # # Plot the day-to-day comparison between the observations and ICON (reflectivity only)
    # Create a list of plots to do
    plots_to_make = []

    if orcestra:
        title_pam_orc = r'\textbf{ICON-orcestra via PAMTRA at ' + f'{plotted_hour_orc:02d}:{plotted_minute_orc:02d}' + '}'
        plots_to_make.append(("ref", lat_pam_orc, ref_pam_orc, title_pam_orc, None))

    if hackathon:
        title_pam_hack = r'\textbf{ICON-hackathon via PAMTRA at ' + f'{plotted_hour_hack:02d}:{plotted_minute_hack:02d}' + '}'
        plots_to_make.append(("ref", lat_pam_hack, ref_pam_hack, title_pam_hack, None))

    if new_rain:
        title_pam_new = r'\textbf{ICON-new via PAMTRA at ' + f'{plotted_hour_new:02d}:{plotted_minute_new:02d}' + '}'
        plots_to_make.append(("ref", lat_pam_new, ref_pam_new, title_pam_new, None))

    if include_obs:
        if radar == 'ec':
            title_obs = r'\textbf{EarthCARE at ' + f'{first_ec_time} - {second_ec_time}' + '}'
            plots_to_make.append(("ref", lat_ec, ref_ec, title_obs, None))
        elif radar == 'halo':
            title_obs = rf"\textbf{{HALO at {first_halo_time} - {second_halo_time}}}"
            plots_to_make.append(("ref", lat_halo, ref_halo, title_obs, None))

    if plots_to_make:
        plots_to_make = [
            (varname, lat, data, title, idx == len(plots_to_make)-1)
            for idx, (varname, lat, data, title, _) in enumerate(plots_to_make)
        ] # In order to add the x axis only for the last subplot

    # Plot with the good number of subplots
    fig, axs = plt.subplots(len(plots_to_make), 1, figsize=(10, 2.8*len(plots_to_make)))
    if len(plots_to_make) == 1:
        axs = [axs]

    for ax, (varname, lat, data, title, is_obs) in zip(axs, plots_to_make):
        contour = plot_contour(varname, ax, lat, zz, data, title, is_obs, True, vmin_r, vmax_r)
        if 'HALO' in title:
            ax.axvline(x=overpass_lat, color='black', linestyle='--', linewidth=2, label='Overpass')
            ax.text(overpass_lat, -1000, r'\textbf{overpass}', color='black', rotation=0, verticalalignment='top', horizontalalignment='center', fontsize=12)

    # Add colorbar
    cbar_ax1 = fig.add_axes([0.92, 0.11, 0.015, 0.76])
    cbar1 = fig.colorbar(contour, cax=cbar_ax1, ticks=colorbar_ticks_r)
    cbar1.set_label(r'\textbf{Reflectivity / dBZ}', color='black')
    cbar1.ax.set_yticklabels([f'\\textbf{{{tick}}}' for tick in colorbar_ticks_r])

    plt.subplots_adjust(left=0.07, right=0.9, top=0.87, hspace=0.3)
    #fig.suptitle(rf"\textbf{{{day}}}", fontsize=18, color = 'black', fontweight='bold', y=0.96)
    plt.show()
    
    # filename = f"/work/mh0731/m301196/ecomip/outputs/july-outputs/{radar}-comparisons/{date_str}_{time_str}.png"
    # plt.savefig(filename, bbox_inches='tight', facecolor=fig.get_facecolor())
    # plt.close(fig)