import numpy as np
import matplotlib.pyplot as plt
from pylab import *

plt.rcParams.update({'font.size': 14, 'font.family': 'Times New Roman', 'text.usetex': True})
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.5

from tools.file_names import *
from tools.prepare_dataset_functions import *
from tools.plot_functions import *

## What do you want to plot for the reflectivity? ##
radar = 'halo' #'ec' or 'halo'
orcestra = True #True or False, depending on if you want to see orcestra results or not
hackathon = True #True or False, depending on if you want to see hackathon results or not
new_rain = True #True or False, depending on if you want to see new-rain results or not
include_obs = True #True or False, depending on if you want to see the observation results or not
normalize='joint-hist' # 'joint-hist' or 'cfad'

# Get HALO data
ds_halo = xr.open_dataset(halo_path, engine="zarr")

# Common altitude grid
zz = np.arange(25, 16026, 200)

# Big lists to store the data when looping through the files
all_reflectivity_ec = []
all_ref_ec = []
all_doppler_vel_ec = []

all_ref_pam_orc= []
all_doppler_vel_pam_orc = []
all_ref_pam_hack= []
all_doppler_vel_pam_hack = []
all_ref_pam_new = []
all_doppler_vel_pam_new = []

all_ref_halo = []


for j in range(len(files_earthcare)): #  len(files_earthcare) if you want to loop through all the files
    print(f"Processing file {j+1}/{len(files_earthcare)}")
    itime_orc = 3
    itime_hack = 6
    
    # Find date and time of ICON for choosing the right day for HALO
    file_icon = files_icon[j] 
    date_str = file_icon.split('_')[2][:10]
    time_str = file_icon.split('_')[3][:5]
    
    if radar =='halo': 
        # Get HALO values
        first_halo_time, second_halo_time, lat_halo, lon_halo, ref_halo, overpass_lat = prepare_halo_dataset(ds_halo, date_str, time_str, zz)
    elif radar =='ec':   
        # Get EarthCARE values
        first_ec_time, second_ec_time, liq_water_path_ec, ice_water_path_ec, lat_ec, lon_ec, clwc_data, ciwc_data, cwc_ec_data, ref_ec, dopp_ec = prepare_earthcare_dataset(files_earthcare[j], earthcare_path, zz)

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

    all_ref_ec.append(ref_ec)
    all_doppler_vel_ec.append(dopp_ec)

    all_ref_pam_orc.append(ref_pam_orc)
    all_doppler_vel_pam_orc.append(dopp_pam_orc)
    all_ref_pam_hack.append(ref_pam_hack)
    all_doppler_vel_pam_hack.append(dopp_pam_hack)
    all_ref_pam_new.append(ref_pam_new)
    all_doppler_vel_pam_new.append(dopp_pam_new)
    all_ref_halo.append(ref_halo)

if radar =='ec': 
    ###  DOPPLER VELOCITIES ###
    # Get the mean and the standard deviation for every dataset

    # PAMTRA
    track_dim, height_dim = all_doppler_vel_pam_orc[0].dims
    profiles_pam_orc = [da.mean(dim=track_dim) for da in all_doppler_vel_pam_orc]
    stacked_pam_orc = xr.concat(profiles_pam_orc, dim="stack")
    mean_profile_pam_orc = stacked_pam_orc.mean(dim="stack")
    std_pam_orc  = stacked_pam_orc.std(dim="stack")

    track_dim, height_dim = all_doppler_vel_pam_hack[0].dims
    profiles_pam_hack = [da.mean(dim=track_dim) for da in all_doppler_vel_pam_hack]
    stacked_pam_hack = xr.concat(profiles_pam_hack, dim="stack")
    mean_profile_pam_hack = stacked_pam_hack.mean(dim="stack")
    std_pam_hack  = stacked_pam_hack.std(dim="stack")

    track_dim, height_dim = all_doppler_vel_pam_new[0].dims
    profiles_pam_new = [da.mean(dim=track_dim) for da in all_doppler_vel_pam_new]
    stacked_pam_new = xr.concat(profiles_pam_new, dim="stack")
    mean_profile_pam_new = stacked_pam_new.mean(dim="stack")
    std_pam_new  = stacked_pam_new.std(dim="stack")

    # EarthCARE
    track_dim, height_dim = all_doppler_vel_ec[0].dims
    profiles_ec = [da.mean(dim=track_dim) for da in all_doppler_vel_ec]
    stacked_ec = xr.concat(profiles_ec, dim="stack")
    mean_profile_ec = stacked_ec.mean(dim="stack")
    std_ec  = stacked_ec.std(dim="stack")

    # For uniformity between EarthCARE and PAMTRA, we delete the data under 500m of altitude
    mean_profile_pam_orc = mean_profile_pam_orc.where(mean_profile_pam_orc.height >= 500)
    mean_profile_pam_hack = mean_profile_pam_hack.where(mean_profile_pam_hack.height >= 500)
    mean_profile_pam_new = mean_profile_pam_new.where(mean_profile_pam_new.height >= 500)
    mean_profile_ec = mean_profile_ec.where(mean_profile_ec.altitude >= 500)

    std_pam_orc  = std_pam_orc.where(mean_profile_pam_orc.height >= 500)
    std_pam_hack  = std_pam_hack.where(mean_profile_pam_hack.height >= 500)
    std_pam_new  = std_pam_new.where(mean_profile_pam_new.height >= 500)
    std_ec  = std_ec.where(mean_profile_ec.altitude >= 500)

    # Plot
    plt.figure(figsize=(5, 6))
    plt.plot(mean_profile_pam_orc, mean_profile_pam_orc.height, label="ICON-orcestra", color = 'red')
    plt.fill_betweenx(mean_profile_pam_orc.height,
                    mean_profile_pam_orc - std_pam_orc,
                    mean_profile_pam_orc + std_pam_orc,
                    color="red", alpha=0.1)
    plt.plot(mean_profile_pam_hack, mean_profile_pam_hack.height, label="ICON-hackathon", color = 'maroon')
    plt.fill_betweenx(mean_profile_pam_hack.height,
                    mean_profile_pam_hack - std_pam_hack,
                    mean_profile_pam_hack + std_pam_hack,
                    color="maroon", alpha=0.1)
    plt.plot(mean_profile_pam_new, mean_profile_pam_new.height, label="ICON-new-rain", color='purple')
    plt.fill_betweenx(mean_profile_pam_new.height,
                    mean_profile_pam_new - std_pam_new,
                    mean_profile_pam_new + std_pam_new,
                    color="purple", alpha=0.1)
    plt.plot(mean_profile_ec, mean_profile_ec.altitude, label="EarthCARE", color="blue", linestyle='--')
    plt.fill_betweenx(mean_profile_ec.altitude,
                    mean_profile_ec - std_ec,
                    mean_profile_ec + std_ec,
                    color="blue", alpha=0.1)
    plt.xlabel(r"\textbf{Mean Doppler velocity / m.s$^{\mathbf{-1}}$}")
    plt.yticks([0, 5000, 10000, 16000], labels=[r'\textbf{0}', r'\textbf{5}', r'\textbf{10}', r'\textbf{16}'])
    plt.xticks([2, 0, -2, -4,-6], labels=[r'\textbf{2}', r'\textbf{0}', r'\textbf{-2}', r'\textbf{-4}', r'\textbf{-6}'])
    plt.xlim(-7,1)
    plt.ylabel(r"\textbf{Altitude / km}")
    plt.ylim(0, 16000)
    plt.legend(fontsize=14)
    plt.show()

### PREPARE THE REFLECTIVITY DATASETS ###

# Concatenate and flatten the lists to be able to plot them

# For EarthCARE
all_ref_ec = np.concatenate(all_ref_ec, axis=0)
ref_flat_ec = all_ref_ec.flatten()
all_doppler_vel_ec = np.concatenate(all_doppler_vel_ec, axis=0)
doppler_flat_ec = all_doppler_vel_ec.flatten()

height_flat_ec = np.tile(zz, all_ref_ec.shape[0])

# For ICON
all_ref_pam_orc = np.concatenate(all_ref_pam_orc, axis=0)
ref_flat_pam_orc = all_ref_pam_orc.flatten()
all_doppler_vel_pam_orc = np.concatenate(all_doppler_vel_pam_orc, axis=0)
doppler_flat_pam_orc = all_doppler_vel_pam_orc.flatten()
all_ref_pam_hack = np.concatenate(all_ref_pam_hack, axis=0)
ref_flat_pam_hack = all_ref_pam_hack.flatten()
all_doppler_vel_pam_hack = np.concatenate(all_doppler_vel_pam_hack, axis=0)
doppler_flat_pam_hack = all_doppler_vel_pam_hack.flatten()
all_ref_pam_new = np.concatenate(all_ref_pam_new, axis=0)
ref_flat_pam_new = all_ref_pam_new.flatten()
all_doppler_vel_pam_new = np.concatenate(all_doppler_vel_pam_new, axis=0)
doppler_flat_pam_new = all_doppler_vel_pam_new.flatten()

height_flat_icon = np.tile(zz, all_ref_pam_orc.shape[0])

# For HALO
all_ref_halo = np.concatenate(all_ref_halo, axis=0)
ref_flat_halo = all_ref_halo.flatten()

height_flat_halo = np.tile(zz, all_ref_halo.shape[0])

# For uniformity between EarthCARE and ICON, let's delete the values with altitude lower than 500m 
ref_flat_ec = np.where(height_flat_ec < 500, np.nan, ref_flat_ec)
doppler_flat_ec = np.where(height_flat_ec < 500, np.nan, doppler_flat_ec)

ref_flat_pam_orc = np.where(height_flat_icon < 500, np.nan, ref_flat_pam_orc)
doppler_flat_pam_orc = np.where(height_flat_icon < 500, np.nan, doppler_flat_pam_orc)
ref_flat_pam_hack = np.where(height_flat_icon < 500, np.nan, ref_flat_pam_hack)
doppler_flat_pam_hack = np.where(height_flat_icon < 500, np.nan, doppler_flat_pam_hack)
ref_flat_pam_new = np.where(height_flat_icon < 500, np.nan, ref_flat_pam_new)
doppler_flat_pam_new = np.where(height_flat_icon < 500, np.nan, doppler_flat_pam_new)

ref_flat_halo = np.where(height_flat_halo < 500, np.nan, ref_flat_halo)

### PLOT THE REFLECTIVITY HISTOGRAMS ###

# Function to prepare datasets list for plotting
def build_dataset_list(include_obs=True):
    """
    Returns a list of (label, ref_data, height_data) tuples
    depending on the orcestra/hackathon/new_rain flags.
    Observations are added at the end if include_obs is True.
    """
    datasets = []
    if orcestra:
        datasets.append((r"\textbf{ICON-orcestra}", ref_flat_pam_orc, height_flat_icon))
    if hackathon:
        datasets.append((r"\textbf{ICON-hackathon}", ref_flat_pam_hack, height_flat_icon))
    if new_rain:
        datasets.append((r"\textbf{ICON-new-rain}", ref_flat_pam_new, height_flat_icon))

    if include_obs:
        if radar == 'ec':
            datasets.append((r"\textbf{EarthCARE}", ref_flat_ec, height_flat_ec))
        else:
            datasets.append((r"\textbf{HALO}", ref_flat_halo, height_flat_halo))
    return datasets

# Main plots (CFAD or joint histograms)
datasets_to_plot = build_dataset_list(include_obs)

ncols = len(datasets_to_plot) + 1  # +1 for colorbar
fig = plt.figure(figsize=(2.8 * len(datasets_to_plot), 2.8))
gs = GridSpec(1, ncols, width_ratios=[1] * len(datasets_to_plot) + [0.04], figure=fig, wspace=0.2, hspace=0.3)

axes = [fig.add_subplot(gs[i]) for i in range(len(datasets_to_plot))]
cax = fig.add_subplot(gs[-1])

# Plot each dataset
for i, (label, ref_data, height_data) in enumerate(datasets_to_plot):
    first_plot = (i == 0)  # The altitude axis only appears for the first plot
    if normalize == 'cfad':
        ref = plot_cfad(radar, 'Reflectivity', ref_data, axes[i], height_data, label, first_plot, False, zz)
    elif normalize == 'joint-hist':
        ref = plot_joint_hist(radar, 'Reflectivity', ref_data, axes[i], height_data, label, first_plot, False, zz)

# Colorbar
cbar = fig.colorbar(ref, cax=cax, aspect=40, norm=ref.norm, cmap=ref.cmap)
if normalize == 'cfad':
    cbar.set_label("Probability density")
    cbar.set_ticks([0, 0.05, 0.1])
elif normalize == 'joint-hist':
    cbar.set_label("Probability density x$10^6$")
    cbar.set_ticks([0, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6])
    cbar.set_ticklabels([0, 1, 2, 3, 4, 5, 6, 7])

plt.show()


# Difference plots
# Build list without observations
datasets_for_diff = build_dataset_list(include_obs=False)

ncols = len(datasets_for_diff) + 1  # +1 for colorbar
fig = plt.figure(figsize=(3 * len(datasets_for_diff), 3))
gs = GridSpec(1, ncols, width_ratios=[1] * len(datasets_for_diff) + [0.04], figure=fig, wspace=0.2, hspace=0.3)

axes = [fig.add_subplot(gs[i]) for i in range(len(datasets_for_diff))]
cax = fig.add_subplot(gs[-1])

# Choose observation dataset
if radar == 'ec':
    obs_ref = ref_flat_ec
    obs_height = height_flat_ec
    suptitle_str = 'Difference with EarthCARE'
else:
    obs_ref = ref_flat_halo
    obs_height = height_flat_halo
    suptitle_str = 'Difference with HALO'

# Plot differences
for i, (label, ref_data, height_data) in enumerate(datasets_for_diff):
    first_plot = (i == 0) # The altitude axis only appears for the first plot
    if normalize == 'cfad':
        diff = plot_cfad_difference(radar, 'Reflectivity', ref_data, height_data, obs_ref, obs_height, axes[i], zz, label, first_plot)
    elif normalize == 'joint-hist':
        diff = plot_joint_hist_difference(radar, 'Reflectivity', ref_data, height_data, obs_ref, obs_height, axes[i], zz, label, first_plot)

# Colorbar for differences
cbar = fig.colorbar(diff, cax=cax, aspect=40, norm=diff.norm, cmap=diff.cmap)
if normalize == 'cfad':
    cbar.set_label("Probability density")
    cbar.set_ticks([-0.05, 0, 0.05])
elif normalize == 'joint-hist':
    cbar.set_label("Probability density x$10^6$")
    cbar.set_ticks([-6e-6, -4e-6, -2e-6, 0, 2e-6, 4e-6, 6e-6])
    cbar.set_ticklabels([-6, -4, -2, 0, 2, 4, 6])

#plt.suptitle(suptitle_str)
plt.show()