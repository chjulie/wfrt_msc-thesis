import argparse
import datetime
import pandas as pd
import numpy as np
from pyresample import geometry, bilinear, kd_tree

# from scipy.interpolate import RegularGridInterpolator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from utils import read_pkl
from data_constants import OBS_DATA_DIR, PRED_DATA_DIR


# def get_interpolated_value(lat, lon, interpolation_function):
#     interpolated_value = interpolation_function(np.array([[lat, lon]]))
#     return interpolated_value


if __name__ == "__main__":
    """
    Evaluates model prediction against observations

    """
    resampling = "nearest-neighbor"  # 'nearest-neighbor', 'bilinear'
    parser = argparse.ArgumentParser(description="Process WRFOUT data files.")
    parser.add_argument(
        "--datetime",
        type=str,
        required=True,
        help="Date and time in YYYY-mm-ddTHH:00:00 format. Must be < 1 month old. Supported times are 00, 06, 12, 18.",
    )
    parser.add_argument("--field", type=str, required=True, help="'temp'")
    args = parser.parse_args()

    field = args.field
    input_date = pd.to_datetime(args.datetime, format="%Y-%m-%dT%H:%M:%S").tz_localize(
        "utc"
    )
    # print(f"date: {type(input_date)}, {input_date}")

    obs_path = f"{OBS_DATA_DIR}/{args.datetime[:10].replace('-','')}_verif_eccc_obs.csv"
    pred_path = (
        f"{PRED_DATA_DIR}/{args.datetime[:13].replace('-','')}_regional_state.pkl"
    )

    # TODO:
    # - read both files
    # TODO: add checks that path exist
    p_date, p_latitudes, p_longitudes, p_fields = read_pkl(
        pred_path
    )  # 'p' prefix indicate 'predictions'. Lats and Lons are 1D, but not a regular grid (duplicated values).
    print(" > read prediction file")
    # plt.scatter(longitudes, p_latitudes, c=fields['2t'])
    # plt.show()
    # obs_df = pd.read_csv(obs_path, converters={'UTC_DATE': lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S")})
    obs_df = pd.read_csv(
        obs_path,
        converters={
            "UTC_DATE": lambda x: pd.to_datetime(
                x, format="%Y-%m-%dT%H:%M:%S"
            ).tz_localize("utc")
        },
    )
    print(" > read observations file")

    # print(obs_df.shape)
    # print(f'date: {type(obs_df.UTC_DATE[200])}, {obs_df.UTC_DATE[200]}')
    print(" ** obs date: ", obs_df.UTC_DATE[200])
    print(" ** pred date: ", p_date)
    print(" ** input date: ", input_date)
    # print(obs_df.UTC_DATE[200] - date)
    obs_df = obs_df[obs_df.UTC_DATE == input_date]
    print(" obs_df shape: ", obs_df.shape)
    s_longitudes = obs_df.x.values  # 's' prefix indicates 'station observation'
    s_latitudes = obs_df.y.values  # 's' prefix indicates 'station observation'
    print("\n")
    print(f" * s_lon: {s_longitudes.shape}")
    print(f" * s_lat: {s_latitudes.shape}")
    print(f" - unique stn_id: {np.count_nonzero(np.unique(obs_df['STN_ID']))}")

    # - get station localisation.

    # - y_pred: Interpolate prediction to station localisation (bilinear interpolation)
    # pyresample
    pred_grid = geometry.SwathDefinition(lons=p_longitudes, lats=p_latitudes)
    station_loc = geometry.SwathDefinition(lons=s_longitudes, lats=s_latitudes)
    # bilinear_resampler = bilinear.NumpyBilinearResampler(pred_grid, station_loc, 3000)
    # print(bilinear_resampler)
    # s_pred = bilinear_resampler.resample(p_fields[field])
    s_pred = kd_tree.resample_nearest(
        source_geo_def=pred_grid,
        data=p_fields[field],
        target_geo_def=station_loc,
        radius_of_influence=50000,
    )
    #     pred_grid, station_loc, data=p_fields[field], radius_of_influence=50000
    # )
    print("\n")
    print(f" - interpolated_values: {type(s_pred)}, {s_pred.shape}")

    ## Plot resampled data :
    fig, ax = plt.subplots(
        figsize=(6, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.scatter(s_longitudes, s_latitudes, c=s_pred, cmap="RdBu")

    triangulation = tri.Triangulation(p_longitudes, p_latitudes)
    contour = ax.tricontourf(
        triangulation, p_fields[field], levels=20, cmap="RdBu", alpha=0.5
    )  # transform=ccrs.PlateCarree()

    plt.show()
    # # - y_obs: get observation value
    # => df.columns : STN_ID, LAT, LON, Y_PRED, Y_OBS
    # - compare y_pred and y_obs
    # => df.columns : STN_ID, LAT, LON, Y_PRED, Y_OBS, ERR_0, ..., ERR_N
    # - save error metric to data/error_data

    # for now:
    # timestamps: only 1
    # stations: all
    # error metric: MSE
    print(" > Program finished successfully !")
