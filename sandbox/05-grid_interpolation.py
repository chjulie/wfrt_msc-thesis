import xarray as xr
import numpy as np
import xwrf
import xgcm
import metpy.calc as mpcalc
from metpy.units import units

from pyresample import geometry, bilinear, kd_tree
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs
import cartopy.feature as cfeature

projection = ccrs.Stereographic(
    central_latitude=90.0,          # North Pole
    central_longitude=-90.0,        # STAND_LON
    true_scale_latitude=60.0        # Usually TRUELAT2 for polar stereo
)

def fix(lons):
    # Shift the longitudes from 0-360 to -180-180
    return np.where(lons > 180, lons - 360, lons)

def clip_coords(src_coords, tgt_coords, tolerance=0.03):
    
    tree_src = cKDTree(src_coords)
    indices = tree_src.query_ball_point(tgt_coords, r=tolerance)
    tgt_mask = np.array([len(idx) > 0 for idx in indices])

    tree_tgt = cKDTree(tgt_coords)
    indices = tree_tgt.query_ball_point(src_coords, r=tolerance)
    src_mask = np.array([len(idx) > 0 for idx in indices])

    return tgt_mask, src_mask

def scipy_resampling(
    resampling: str,
    src_coords: np.array,
    tgt_coords: np.array,
    data: xr.DataArray | np.ndarray ,
):
    rbf_interpolator = RBFInterpolator(
        y=src_coords,
        d=data,
        kernel=resampling,
    )
    resampled_data = rbf_interpolator(tgt_coords)

    return resampled_data

def pyresample_resampling(
    src_coords: np.array,
    tgt_coords: np.array,
    data: np.array,
):
    src_grid = geometry.SwathDefinition(lons=src_coords[:, 0], lats=src_coords[:, 1])
    tgt_grid = geometry.SwathDefinition(lons=tgt_coords[:, 0], lats=tgt_coords[:, 1])
    resampled_data = kd_tree.resample_nearest(
        source_geo_def=src_grid,
        data=data,
        target_geo_def=tgt_grid,
        radius_of_influence=50000,
    )
    return resampled_data

def rmse(array1: xr.DataArray | np.ndarray, array2: xr.DataArray | np.ndarray):

    assert array1.shape == array2.shape, "Both arrays must have the same shape"
    assert array1.ndim==1 and array2.ndim==1, "Both arrays must be one-dimesional"

    rmse = np.sqrt(np.power(array1-array2, 2)).mean()   # spatial average

    return rmse

if __name__ == "__main__":

    # Open datasets
    print('Starting script')
    wrf_ds = xr.open_dataset('data/wrf_data/wrfout_d02_processed_23040400.nc')
    print('Opened wrf')
    climatex_ds = xr.open_dataset('data/prediction_data/bris-lam-inference-20230101T12-20230102T12.nc')
    print('Opened climatex')

    src_coords = np.column_stack((wrf_ds.XLONG.values.flatten(), wrf_ds.XLAT.values.flatten())) # source coords is WAC00WG-01 XLONG/XLAT
    tgt_coords = np.column_stack((climatex_ds.longitude.values, climatex_ds.latitude.values))

    # Clip domains
    tgt_mask, src_mask = clip_coords(src_coords, tgt_coords)
    clipped_src_coords = src_coords[src_mask]
    clipped_tgt_coords = tgt_coords[tgt_mask]

    data_tgt = climatex_ds['2t'].sel(lead_time=0, initial_date='2023-01-01T12:00:00.000000000')[tgt_mask]
    data_src = wrf_ds['2t'].sel(XTIME='2023-04-04T06:00:00.000000000').values.flatten()[src_mask]

    print(f" - src: {clipped_src_coords.shape}")    
    print(f" - tgt: {clipped_tgt_coords.shape}")    # climatex
    print(f" - data src: {data_src.shape}")
    print(f" - data tgt: {data_tgt.values.shape}")

    # Perform interpolation
    print('nearest-neighbor resampling: ')
    resampled_data_NN = pyresample_resampling(src_coords=clipped_src_coords, tgt_coords=clipped_tgt_coords, data=data_src)
    print('linear resampling: ')
    resampled_data_linear = scipy_resampling(resampling="linear", src_coords=clipped_src_coords, tgt_coords=clipped_tgt_coords, data=data_src)
    print('Cubic resampling: ')
    resampled_data_cubic = scipy_resampling(resampling="cubic", src_coords=clipped_src_coords, tgt_coords=clipped_tgt_coords, data=data_src)

    # Plot results
    # print('Plotting results ... ')
    # fig, ax = plt.subplots(1,2,figsize=(15,10), subplot_kw={'projection': projection})
    # ax = ax.ravel()

    # # ORIGINAL CLIMATEX COORDS
    # ax[0].scatter(x=clipped_src_coords[:,0], y=clipped_src_coords[:,1], c=data_src, transform=ccrs.PlateCarree())
    # ax[0].set_title(f"Variable 2t - original CLIMATEX coordinates")

    # # RESAMPLED NEAREST-NEIGHBORS
    # ax[1].scatter(x=clipped_tgt_coords[:,0], y=clipped_tgt_coords[:,1], c=resampled_data_NN, transform=ccrs.PlateCarree())
    # ax[1].set_title(f"Variable 2t - Nearest-neighbor resampling")

    # plt.savefig('reports/plots/evaluation/NN_resampling_test.png', bbox_inches='tight', dpi=200)

    # Compute rmse:
    

    print('Plotting results ... ')
    fig, ax = plt.subplots(1,3,figsize=(20,10), subplot_kw={'projection': projection})
    ax = ax.ravel()

    # ORIGINAL CLIMATEX COORDS
    ax[0].scatter(x=src_coords[:,0], y=src_coords[:,1], c=data_src.values, transform=ccrs.PlateCarree())
    ax[0].set_title(f"Variable 2t - original CLIMATEX coordinates")

    # RESAMPLED LINEAR
    ax[1].scatter(x=clipped_tgt_coords[:,0], y=clipped_tgt_coords[:,1], c=resampled_data_linear.values, transform=ccrs.PlateCarree())
    ax[1].set_title(f"Variable 2t - linear resampling")

    # RESAMPLED CUBIC
    ax[2].scatter(x=clipped_tgt_coords[:,0], y=clipped_tgt_coords[:,1], c=resampled_data_cubic.values, transform=ccrs.PlateCarree())
    ax[2].set_title(f"Variable 2t - cubic resampling")

    plt.savefig('reports/plots/evaluation/resampling_test.png', bbox_inches='tight', dpi=200)
    print('Script finished succesfully !')