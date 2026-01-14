import numpy as np
from pyresample import geometry, bilinear, kd_tree
from scipy.interpolate import RBFInterpolator

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
