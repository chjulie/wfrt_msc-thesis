import argparse
import datetime
import pandas as pd
import numpy as np
from pyresample import geometry, bilinear

# from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from utils import read_pkl
from data_constants import OBS_DATA_DIR, PRED_DATA_DIR


def get_interpolated_value(lat, lon, interpolation_function):
    interpolated_value = interpolation_function(np.array([[lat, lon]]))
    return interpolated_value


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
    input_date = pd.to_datetime(args.datetime, format="%Y-%m-%dT%H:%M:%S")
    print(f"date: {type(input_date)}, {input_date}")

    obs_path = f"{OBS_DATA_DIR}/{args.datetime[:10].replace('-','')}_verif_eccc_obs.csv"
    pred_path = (
        f"{PRED_DATA_DIR}/{args.datetime[:10].replace('-','')}_regional_state.pkl"
    )

    # TODO:
    # - read both files
    # TODO: add checks that path exist
    p_date, p_latitudes, p_longitudes, p_fields = read_pkl(
        pred_path
    )  # 'p' prefix indicate 'predictions'. Lats and Lons are 1D, but not a regular grid (duplicated values).
    print(
        f"0: {np.count_nonzero(p_latitudes[p_latitudes == p_latitudes[0]])}, -1: {np.count_nonzero(p_latitudes[p_latitudes == p_latitudes[-1]])}"
    )
    print(f" - lats.shape: {p_latitudes.shape}")
    print(f" - lons.shape: {p_longitudes.shape}")
    print(
        " - np.unique(lats, return_counts=True): ",
        np.unique(p_latitudes, return_counts=True),
    )
    print(f" - field: {p_fields['2t'].shape}")

    # plt.scatter(longitudes, p_latitudes, c=fields['2t'])
    # plt.show()
    # obs_df = pd.read_csv(obs_path, converters={'UTC_DATE': lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S")})
    # obs_df = pd.read_csv(obs_path, converters={'UTC_DATE': lambda x: pd.to_datetime(x, format="%Y-%m-%dT%H:%M:%S").tz_localize('utc')})
    # print(obs_df.shape)
    # print(f'date: {type(obs_df.UTC_DATE[200])}, {obs_df.UTC_DATE[200]}')
    # print(' ** obs date: ', obs_df.UTC_DATE[200].tzinfo)
    # print(' ** pred date: ', date.tzinfo)
    # print(obs_df.UTC_DATE[200] - date)
    # obs_df = obs_df[obs_df.UTC_DATE == date]
    # print(obs_df.shape)
    s_longitudes = obs_df.x.values  # 's' prefix indicates 'station observation'
    s_latitudes = obs_df.y.values  # 's' prefix indicates 'station observation'
    # - get station localisation.

    # - y_pred: Interpolate prediction to station localisation (bilinear interpolation)
    # pyresample
    pred_grid = geometry.SwathDefinition(lons=p_longitudes, lats=p_latitudes)
    station_loc = geometry.SwathDefinition(lons=s_longitudes, lats=s_latitudes)
    interpolated_values = bilinear.NumpyBilinearResampler(
        pred_grid, station_loc, 3000
    ).resample(p_fields[field])
    print(
        f" - interpolated_values: {type(interpolated_values)}, {interpolated_values.shape}"
    )
    # - y_obs: get observation value
    # => df.columns : STN_ID, LAT, LON, Y_PRED, Y_OBS
    # - compare y_pred and y_obs
    # => df.columns : STN_ID, LAT, LON, Y_PRED, Y_OBS, ERR_0, ..., ERR_N
    # - save error metric to data/error_data

    # for now:
    # timestamps: only 1
    # stations: all
    # error metric: MSE
    print(" > Program finished successfully !")
