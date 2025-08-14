'''
This file groups the plotting functions used in other scripts. The most importants are:
    - plot_contour for the day-to-day comparisons
    - plot_joint_hist for joint histograms
    - plot_cfad for cfads
    - plot_joint_hist_difference for joint histogram difference between two datasets
    - plot_cfad_difference for cfad difference between two datasets
'''

import numpy as np
import matplotlib.pyplot as plt 
import xarray as xr
from matplotlib import colors

def plot_contour(type, ax, lat, height, product, title, xlabel, ylabel, vmin, vmax):
    '''
    To plot simple contours of a product (for example the reflectivity or doppler velocity), depending on altitude and latitude on a particular day.
    
    Inputs:
    - type can be 'ref' or 'dopp' for example, to determine the colormap.
    - ax is the ax to plot the contour on.
    - product should be 2D, whereas lat and height need to be 1D with sizes corresponding to the sizes of the two coordinates of the product. 
    The products I had were always reversed (coordinates in the wrong order), so I use .transpose() when plotting. 
    - title is the title of the plot.
    - xlabel and ylabel are boolean: True if a label is needed, False otherwise. Useful for plots with many subplots, to save some space.
    - vmin and vmax are the minimal and maximal values of the contour color levels.
    
    Output: the contour.
    '''
    
    # To change the colormap depending on the plotted product (diverging colormap such as 'bwr' for Doppler velocity for example).
    if type == 'ref':
        colormap = plt.cm.nipy_spectral
    else:
        colormap = 'bwr'
    
    contour = ax.contourf(
        lat, 
        height, 
        product.transpose(),
        levels=np.linspace(vmin, vmax, 100),  # Ensure consistent levels
        vmin=vmin,
        vmax=vmax,
        cmap=colormap
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([0, 5, 10, 15, 20], labels=[r'\textbf{0}', r'\textbf{5}', r'\textbf{10}', r'\textbf{15}', r'\textbf{20}'])  

    ax.set_yticks([0, 5000, 10000, 15000], labels=[r'\textbf{0}', r'\textbf{5}', r'\textbf{10}', r'\textbf{15}'])
    
    if xlabel:
        ax.set_xlabel(r'\textbf{Latitude / °N}')
    if ylabel:
        ax.set_ylabel(r'\textbf{Altitude / km}')
        
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 15000)
    return contour

def plot_joint_hist(radar, studied_product, product, ax, height, title, ylabel, bar, zz):
    if radar == 'halo':
        max_ref=40
    else:
        max_ref = 30
    if studied_product == 'Reflectivity':
        product_bins = np.linspace(-30, max_ref, 50)
    elif studied_product == 'Doppler velocity':
        product_bins = np.linspace(-10, 5, 50)
    elif studied_product == 'Cloud water content':
        product_bins = np.logspace(np.log10(1e-3), np.log10(3), 50)
    
    altitude_bins = zz

    hist, x_edges, y_edges = np.histogram2d(product, height, bins=[product_bins, altitude_bins], density=True)
    
    if studied_product == 'Cloud water content':
        # Use log scale for colormap
        # Create discrete colormap
        n_colors = 16
        bounds = np.logspace(np.log10(1e-5), np.log10(1e-1), n_colors + 1)
        cmap = plt.cm.nipy_spectral
        cmap_discrete = colors.ListedColormap(cmap(np.linspace(0, 1, n_colors)))
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=n_colors)
        # Plot the CFAD
        pcm = ax.pcolormesh(x_edges, y_edges, hist.T, cmap=cmap_discrete,norm=norm)        
        ax.set_xscale('log')
        ax.set_xlabel("Cloud water content / g.m$^{-3}$", fontsize=12, fontweight='bold')
        ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 3], labels=[r'\textbf{0.001}', r'\textbf{0.01}', r'\textbf{0.1}', r'\textbf{1}', r'\textbf{3}'])        
        cbar = plt.colorbar(pcm, ax=ax, aspect=40)
        cbar.ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1], labels=[r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
    elif studied_product == 'Reflectivity':
        # Create discrete colormap
        n_colors = 14
        bounds = np.linspace(0, 7e-6, n_colors + 1)
        cmap = plt.cm.nipy_spectral
        cmap_discrete = colors.ListedColormap(cmap(np.linspace(0, 1, n_colors)))
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=n_colors)
        # Plot the CFAD
        pcm = ax.pcolormesh(x_edges, y_edges, hist.T, cmap=cmap_discrete, norm=norm)
        ax.set_xlabel(r"\textbf{Reflectivity / dBZ}", fontsize=12)
        ax.set_xticks([-30, -10, 10, 30], labels=[r'\textbf{-30}', r'\textbf{-10}', r'\textbf{10}', r'\textbf{30}'])
        if bar:
            cbar = plt.colorbar(pcm, ax=ax, aspect=40)
            cbar.set_label(r'\textbf{Probability density x$\mathbf{10^{6}}$}', color ='black')
            cbar.set_ticks([0, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6], labels=[0,1,2,3,4,5,6,7])
    elif studied_product == 'Doppler velocity':
        # Create discrete colormap
        n_colors = 15
        bounds = np.linspace(0, 1.5e-4, n_colors + 1)
        cmap = plt.cm.nipy_spectral
        cmap_discrete = colors.ListedColormap(cmap(np.linspace(0, 1, n_colors)))
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=n_colors)
        # Plot the CFAD
        pcm = ax.pcolormesh(x_edges, y_edges, hist.T, cmap=cmap_discrete, norm=norm)
        ax.set_xlabel(r"\textbf{Doppler velocity / m.s$^{\mathbf{-1}}$}", fontsize=12)
        ax.set_xticks([-10, 0,5], labels=[r'\textbf{-10}', r'\textbf{0}', r'\textbf{5}'])
        cbar = plt.colorbar(pcm, ax=ax, aspect=40)
    
    if ylabel:
        ax.set_ylabel(r"\textbf{Altitude / km}", fontsize=12)
        ax.set_yticks([0, 5000, 10000, 16000], labels=[r'\textbf{0}', r'\textbf{5}', r'\textbf{10}', r'\textbf{16}'])
    else:
        ax.set_yticks([0, 5000, 10000, 16000], labels=[])
    ax.set_ylim(0, 16000)
    #ax.set_title(title)
    ax.text(0.98, 0.98, title, color='white', fontsize=14, ha='right', va='top', transform=ax.transAxes)

    return pcm

def plot_cfad(radar, studied_product, product, ax, height, title, ylabel, bar, zz):
    if radar == 'halo':
        max_ref=40
    else:
        max_ref = 30
    if studied_product == 'Reflectivity':
        product_bins = np.linspace(-30, max_ref, 50)
    elif studied_product == 'Doppler velocity':
        product_bins = np.linspace(-10, 5, 50)
    elif studied_product == 'Cloud water content':
        product_bins = np.logspace(np.log10(1e-3), np.log10(3), 50)
    
    altitude_bins = zz

    hist, x_edges, y_edges = np.histogram2d(product, height, bins=[product_bins, altitude_bins], density=False)
    hist = np.where(np.sum(hist, axis=0, keepdims=True) != 0, hist / np.sum(hist, axis=0, keepdims=True), 0)
    
    if studied_product == 'Cloud water content':
        # Use log scale for colormap
        # Create discrete colormap
        n_colors = 10
        bounds = np.linspace(0, 0.2, n_colors + 1)
        cmap = plt.cm.nipy_spectral
        cmap_discrete = colors.ListedColormap(cmap(np.linspace(0, 1, n_colors)))
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=n_colors)
        # Plot the CFAD
        pcm = ax.pcolormesh(x_edges, y_edges, hist.T, cmap=cmap_discrete, norm=norm)
        ax.set_xscale('log')
        ax.set_xlabel("Condensate content / g.kg$^{-1}$", fontsize=12, fontweight='bold')
        ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 3], labels=[r'\textbf{0.001}', r'\textbf{0.01}', r'\textbf{0.1}', r'\textbf{1}', r'\textbf{3}'])
        plt.colorbar(pcm, ax=ax, aspect=40, ticks=[0, 0.1, 0.2])
    elif studied_product == 'Reflectivity':
        # Create discrete colormap
        n_colors = 10
        bounds = np.linspace(0, 0.1, n_colors + 1)
        cmap = plt.cm.nipy_spectral
        cmap_discrete = colors.ListedColormap(cmap(np.linspace(0, 1, n_colors)))
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=n_colors)
        # Plot the CFAD
        pcm = ax.pcolormesh(x_edges, y_edges, hist.T, cmap=cmap_discrete, norm=norm)
        ax.set_xlabel(r"\textbf{Reflectivity / dBZ}", fontsize=12)
        ax.set_xticks([-30, -10, 10, 30], labels=[r'\textbf{-30}', r'\textbf{-10}', r'\textbf{10}', r'\textbf{30}'])
        if bar:
            cbar =  plt.colorbar(pcm, ax=ax, aspect=40, ticks=[0, 0.05, 0.1])
            cbar.set_label(r'\textbf{Probability density}', color ='black')
    elif studied_product == 'Doppler velocity':
        # Create discrete colormap
        n_colors = 15
        bounds = np.linspace(0, 0.2, n_colors + 1)
        cmap = plt.cm.nipy_spectral
        cmap_discrete = colors.ListedColormap(cmap(np.linspace(0, 1, n_colors)))
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=n_colors)
        # Plot the CFAD
        pcm = ax.pcolormesh(x_edges, y_edges, hist.T, cmap=cmap_discrete, norm=norm)
        ax.set_xlabel(r"\textbf{Doppler velocity / m.s$^{\mathbf{-1}}$}", fontsize=12)
        ax.set_xticks([-10, 0,5], labels=[r'\textbf{-10}', r'\textbf{0}', r'\textbf{5}'])
        cbar = plt.colorbar(pcm, ax=ax, aspect=40, ticks=[0, 0.1, 0.2])
    
    if ylabel:
        ax.set_ylabel(r"\textbf{Altitude / km}", fontsize=12)
        ax.set_yticks([0, 5000, 10000, 16000], labels=[r'\textbf{0}', r'\textbf{5}', r'\textbf{10}', r'\textbf{16}'])
    else:
        ax.set_yticks([0, 5000, 10000, 16000], labels=[])
    ax.set_ylim(0, 16000)
    #ax.set_title(title)
    ax.text(0.98, 0.98, title, color='white', fontsize=14, ha='right', va='top', transform=ax.transAxes)

    return pcm

def plot_joint_hist_difference(radar, studied_product, product1, height1, product2, height2, ax, zz, title, ylabel=True):
    if radar == 'halo':
        max_ref=40
    else:
        max_ref = 30
        
    if studied_product == 'Reflectivity':
        product_bins = np.linspace(-30, max_ref, 50)
    elif studied_product == 'Doppler velocity':
        product_bins = np.linspace(-10, 5, 50)
    elif studied_product == 'Cloud water content':
        product_bins = np.logspace(np.log10(1e-3), np.log10(3), 50)

    hist1, x_edges, y_edges = np.histogram2d(product1, height1, bins=[product_bins, zz], density=True)
    hist2, _, _ = np.histogram2d(product2, height2, bins=[product_bins, zz], density=True)
    
    # Difference between the two histograms
    diff = hist1 - hist2

    # Colormap divergente
    vmax = 2e-6 #6e-6
    pcm = ax.pcolormesh(x_edges, y_edges, diff.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    # Axes
    if studied_product == 'Reflectivity':
        ax.set_xlabel(r"\textbf{Reflectivity / dBZ}", fontsize=12, fontweight='bold')
        ax.set_xticks([-30, -10, 10, 30], labels=[r'\textbf{-30}', r'\textbf{-10}', r'\textbf{10}', r'\textbf{30}'])
    elif studied_product == 'Doppler velocity':
        ax.set_xlabel("Doppler velocity / m.s$^{-1}$", fontsize=12, fontweight='bold')
        ax.set_xticks([-10, 0, 5], labels=[r'\textbf{-10}', r'\textbf{0}', r'\textbf{5}'])
    elif studied_product == 'Cloud water content':
        ax.set_xscale('log')
        ax.set_xlabel("Cloud water content / g.m$^{-3}$", fontsize=12, fontweight='bold')
        ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 3], labels=[r'\textbf{0.001}', r'\textbf{0.01}', r'\textbf{0.1}', r'\textbf{1}', r'\textbf{3}'])

    if ylabel:
        ax.set_ylabel(r"\textbf{Altitude / km}", fontsize=12)
        ax.set_yticks([0, 5000, 10000, 16000], labels=[r'\textbf{0}', r'\textbf{5}', r'\textbf{10}', r'\textbf{16}'])
    else:
        ax.set_yticks([0, 5000, 10000, 16000], labels=[])
    ax.set_ylim(0, 16000)
    #ax.set_title(title)
    ax.text(0.98, 0.98, title, color='black', fontsize=14, ha='right', va='top', transform=ax.transAxes)


    return pcm

def plot_cfad_difference(radar, studied_product, product1, height1, product2, height2, ax, zz, title, ylabel=True):
    if radar == 'halo':
        max_ref=40
    else:
        max_ref = 30
        
    if studied_product == 'Reflectivity':
        product_bins = np.linspace(-30, max_ref, 50)
    elif studied_product == 'Doppler velocity':
        product_bins = np.linspace(-10, 5, 50)
    elif studied_product == 'Cloud water content':
        product_bins = np.logspace(np.log10(1e-3), np.log10(3), 50)

    hist1, x_edges, y_edges = np.histogram2d(product1, height1, bins=[product_bins, zz], density=False)
    hist2, _, _ = np.histogram2d(product2, height2, bins=[product_bins, zz], density=False)
    hist1 = np.where(np.sum(hist1, axis=0, keepdims=True) != 0, hist1 / np.sum(hist1, axis=0, keepdims=True), 0)
    hist2 = np.where(np.sum(hist2, axis=0, keepdims=True) != 0, hist2 / np.sum(hist2, axis=0, keepdims=True), 0)

    # Difference between the two histograms
    diff = hist1 - hist2

    # Colormap divergente
    vmax = 0.05 #np.max(np.abs(diff)) or 6e-6
    pcm = ax.pcolormesh(x_edges, y_edges, diff.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    #plt.colorbar(pcm, ax=ax, aspect=40, label="Density Difference")

    # Axes
    if studied_product == 'Reflectivity':
        ax.set_xlabel(r"\textbf{Reflectivity / dBZ}", fontsize=12, fontweight='bold')
        ax.set_xticks([-30, -10, 10, 30], labels=[r'\textbf{-30}', r'\textbf{-10}', r'\textbf{10}', r'\textbf{30}'])
    elif studied_product == 'Doppler velocity':
        ax.set_xlabel("Doppler velocity / m.s$^{-1}$", fontsize=12, fontweight='bold')
        ax.set_xticks([-10, 0, 5], labels=[r'\textbf{-10}', r'\textbf{0}', r'\textbf{5}'])
    elif studied_product == 'Cloud water content':
        ax.set_xscale('log')
        ax.set_xlabel("Cloud water content / g.m$^{-3}$", fontsize=12, fontweight='bold')
        ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 3], labels=[r'\textbf{0.001}', r'\textbf{0.01}', r'\textbf{0.1}', r'\textbf{1}', r'\textbf{3}'])

    if ylabel:
        ax.set_ylabel(r"\textbf{Altitude / km}", fontsize=12)
        ax.set_yticks([0, 5000, 10000, 16000], labels=[r'\textbf{0}', r'\textbf{5}', r'\textbf{10}', r'\textbf{16}'])
    else:
        ax.set_yticks([0, 5000, 10000, 16000], labels=[])
    ax.set_ylim(0, 16000)
    #ax.set_title(title)
    ax.text(0.98, 0.98, title, color='black', fontsize=14, ha='right', va='top', transform=ax.transAxes)


    return pcm

def hist_partial_cfad(studied_product, product, height, zz):
    if studied_product == 'Reflectivity':
        product_bins = np.linspace(-30, 30, 50)
    elif studied_product == 'Doppler velocity':
        product_bins = np.linspace(-10, 5, 50)
    elif studied_product == 'Cloud water content':
        product_bins = np.logspace(np.log10(1e-3), np.log10(3), 50)
    
    altitude_bins = zz

    hist, x_edges, y_edges = np.histogram2d(product, height, bins=[product_bins, altitude_bins], density=True)

    return hist, x_edges, y_edges

def smart_diff(a, b): # difference when there are nan values
    a_vals = a.values
    b_vals = b.values

    both_nan = np.isnan(a_vals) & np.isnan(b_vals)
    only_a_nan = np.isnan(a_vals) & ~np.isnan(b_vals)
    only_b_nan = ~np.isnan(a_vals) & np.isnan(b_vals)

    diff = np.empty_like(a_vals)
    diff[both_nan] = np.nan
    diff[only_a_nan] = -b_vals[only_a_nan]
    diff[only_b_nan] = a_vals[only_b_nan]
    diff[~np.isnan(a_vals) & ~np.isnan(b_vals)] = a_vals[~np.isnan(a_vals) & ~np.isnan(b_vals)] - b_vals[~np.isnan(a_vals) & ~np.isnan(b_vals)]

    return xr.DataArray(diff, coords=a.coords, dims=a.dims)

### TESTS 

def search_cloud_categories(ds_ref):
    track_dim, height_dim = ds_ref.dims
    
    # Inverse heights to search for reflectivity top height
    data_rev = ds_ref.isel({height_dim: slice(None, None, -1)})
    height_rev = ds_ref[height_dim][::-1]

    # Mask where reflectivity > -31 dBZ, index where the first valid value occurs, corresponding height
    mask = data_rev > -31
    first_valid_idx = mask.argmax(dim=height_dim)
    height_first = height_rev.isel({height_dim: first_valid_idx})

    # Check if a value >31 dBZ exists in the track
    has_valid = mask.any(dim=height_dim)

    # Initialize and fill table
    categories = np.full(ds_ref.sizes[track_dim], 'none', dtype=object)
    categories[np.where((has_valid) & (height_first > 10000))] = 'high'
    categories[np.where((has_valid) & (height_first <= 10000) & (height_first > 5000))] = 'middle'
    categories[np.where((has_valid) & (height_first <= 5000) & (height_first > 500))] = 'low'
    cat_expanded = xr.DataArray(
        np.repeat(categories[:, np.newaxis], 
        ds_ref.sizes[height_dim], axis=1), dims=[track_dim, height_dim],
        coords={track_dim: ds_ref[track_dim], height_dim: ds_ref[height_dim]}
)
    return cat_expanded

def east_or_west(ds_ref, lon):
    track_dim, height_dim = ds_ref.dims

    # Initialize and fill table
    categories = np.full(ds_ref.sizes[track_dim], 'east', dtype=object)
    categories[np.where(lon < -40)] = 'west'
    cat_expanded = xr.DataArray(
        np.repeat(categories[:, np.newaxis], 
        ds_ref.sizes[height_dim], axis=1), dims=[track_dim, height_dim],
        coords={track_dim: ds_ref[track_dim], height_dim: ds_ref[height_dim]}
)
    return cat_expanded

def plot_log_contour(ax, lat, height, product, title, xlabel, ylabel, vmin, vmax):
    '''
    Similar to plot_contour, with slight differences:
    - no input 'type'
    '''
    
    contour = ax.contourf(
        lat, 
        height, 
        product.transpose(),
        levels=np.logspace(np.log10(vmin), np.log10(vmax), 100),
        norm=colors.LogNorm(vmin=vmin, vmax=vmax),
        cmap=plt.cm.nipy_spectral
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([0, 5, 10, 15, 20], labels=[r'\textbf{0}', r'\textbf{5}', r'\textbf{10}', r'\textbf{15}', r'\textbf{20}'])  
    ax.set_yticks([0, 5000, 10000, 15000], labels=[r'\textbf{0}', r'\textbf{5}', r'\textbf{10}', r'\textbf{15}'])

    if xlabel:
        ax.set_xlabel(r'\textbf{Latitude / °N}')
    if ylabel:
        ax.set_ylabel(r'\textbf{Altitude / km}')
    
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 15000)  
    return contour
