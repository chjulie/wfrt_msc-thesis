import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import xarray as xr
    import pandas as pd
    import os

    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import cKDTree
    return cKDTree, np, os, pd, xr


@app.cell
def _():
    obs_dir = '/scratch/juchar/eccc_data/'
    return (obs_dir,)


@app.cell
def _(xr):
    climatex_ds = xr.open_dataset('/scratch/juchar/prediction_data/bris-lam-inference-20230101T12-20230102T12.nc')
    wrf_ds = xr.open_dataset('../../data/wrf_data/wrfout_d02_processed_23012000.nc')
    return climatex_ds, wrf_ds


@app.cell
def _(climatex_ds, np, wrf_ds):
    src_coords = np.column_stack((wrf_ds.XLONG.values.flatten(), wrf_ds.XLAT.values.flatten())) # source coords is WAC00WG-01 XLONG/XLAT
    tgt_coords = np.column_stack((climatex_ds.longitude.values, climatex_ds.latitude.values))
    return src_coords, tgt_coords


@app.cell
def _(cKDTree, np):
    def clip_stations(station_coords, src_coords, tgt_coords, tolerance=0.03):
        tree_src = cKDTree(src_coords)
        tree_tgt = cKDTree(tgt_coords)

        src_near = np.array([len(idx) > 0 for idx in tree_src.query_ball_point(station_coords, r=tolerance)])
        tgt_near = np.array([len(idx) > 0 for idx in tree_tgt.query_ball_point(station_coords, r=tolerance)])

        return src_near & tgt_near
    return (clip_stations,)


@app.cell
def _(clip_stations, obs_dir, os, pd, src_coords, tgt_coords):
    for obs_file in os.listdir(obs_dir):
        if obs_file[:5] == 'eccc_':
            print(f" ⚡️ file : {obs_file}")
            obs_df = pd.read_csv(
                f"{obs_dir}/{obs_file}",
                converters={
                    "UTC_DATE": lambda x: pd.to_datetime(x, format="%Y-%m-%dT%H:%M:%S").tz_localize("utc")
                },
            )
            station_coords = obs_df[["x", "y"]].to_numpy()
            mask = clip_stations(station_coords=station_coords, src_coords=src_coords, tgt_coords=tgt_coords)

            clipped_stations = obs_df[mask].copy()
            clipped_stations.to_csv(f"{obs_dir}/clipped_{obs_file}")
    return


@app.cell
def _():


    return


if __name__ == "__main__":
    app.run()
