import argparse
import datetime
import pandas as pd
import numpy as np
from pyresample import geometry, bilinear, kd_tree

from scipy.interpolate import RBFInterpolator

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from utils import read_pkl
from data_constants import OBS_DATA_DIR, PRED_DATA_DIR, ERROR_DATA_DIR

FIELDS = ["2t"]
RESAMPLING_METHODS = ["nearest-neighbor", "linear", "cubic"]
METRICS = ["rmse"]


def pyresample_resampling(
    src_coords: np.array,
    tgt_coords: np.array,
    data: np.array,
):
    pred_grid = geometry.SwathDefinition(lons=src_coords[:, 0], lats=src_coords[:, 1])
    station_loc = geometry.SwathDefinition(lons=tgt_coords[:, 0], lats=tgt_coords[:, 1])
    station_prediction = kd_tree.resample_nearest(
        source_geo_def=pred_grid,
        data=data,
        target_geo_def=station_loc,
        radius_of_influence=50000,
    )
    return station_prediction


def scipy_resampling(
    resampling: str,
    src_coords: np.array,
    tgt_coords: np.array,
    data: np.array,
):
    rbf_interpolator = RBFInterpolator(
        y=src_coords,
        d=data,
        kernel=resampling,
    )
    station_prediction = rbf_interpolator(tgt_coords)

    return station_prediction


def get_station_prediction(
    resampling: str,
    src_coords: np.array,
    tgt_coords: np.array,
    data: np.array,
):
    if resampling == "nearest-neighbor":
        station_prediction = pyresample_resampling(src_coords, tgt_coords, data)
    elif (resampling == "linear") or (resampling == "cubic"):
        station_prediction = scipy_resampling(resampling, src_coords, tgt_coords, data)
    else:
        raise NotImplementedError

    return station_prediction


def plot_interpolation(
    src_coords: np.array,
    src_data: np.array,
    tgt_coords: np.array,
    tgt_pred: np.array,
    specs: dict,
):

    fig, ax = plt.subplots(
        figsize=(6, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.scatter(tgt_coords[:, 0], tgt_coords[:, 1], c=tgt_pred, cmap="RdBu")

    triangulation = tri.Triangulation(src_coords[:, 0], src_coords[:, 1])
    contour = ax.tricontourf(
        triangulation, src_data, levels=20, cmap="RdBu", alpha=0.5
    )  # transform=ccrs.PlateCarree()
    ax.set_title(f"field: {specs["field"]}, resampling method: {specs["resampling"]}")

    plt.show()


if __name__ == "__main__":
    """
    Evaluates model prediction against observations

    """
    metric_functions = {"rmse": lambda x, y: (x - y) ** 2}

    parser = argparse.ArgumentParser(description="Process WRFOUT data files.")
    parser.add_argument(
        "--datetime",
        type=str,
        required=True,
        help="Date and time in YYYY-mm-ddTHH:00:00 format. Must be < 1 month old. Supported times are 00, 06, 12, 18.",
    )
    args = parser.parse_args()

    input_date = pd.to_datetime(args.datetime, format="%Y-%m-%dT%H:%M:%S").tz_localize(
        "utc"
    )

    obs_path = f"{OBS_DATA_DIR}/{args.datetime[:10].replace('-','')}_verif_eccc_obs.csv"
    pred_path = (
        f"{PRED_DATA_DIR}/{args.datetime[:13].replace('-','')}_regional_state.pkl"
    )

    # TODO: add checks that path exist
    p_date, p_latitudes, p_longitudes, p_fields = read_pkl(
        pred_path
    )  # 'p' prefix indicate 'predictions'. Lats and Lons are 1D, but not a regular grid (duplicated values).
    print(" > read prediction file")

    obs_df = pd.read_csv(
        obs_path,
        converters={
            "UTC_DATE": lambda x: pd.to_datetime(
                x, format="%Y-%m-%dT%H:%M:%S"
            ).tz_localize("utc")
        },
    )
    obs_df = obs_df.rename(columns={"TEMP": "obs_2t"})
    print(" > read observations file")

    # Get station localisation.
    obs_df = obs_df[obs_df.UTC_DATE == p_date]
    s_longitudes = obs_df.x.values  # 's' prefix indicates 'station observation'
    s_latitudes = obs_df.y.values  # 's' prefix indicates 'station observation'

    p_coords = np.column_stack((p_longitudes, p_latitudes))
    s_coords = np.column_stack((s_longitudes, s_latitudes))

    # Resample data
    for field in FIELDS:
        print(field)
        for resampling in RESAMPLING_METHODS:
            print(" ", resampling)
            s_pred = get_station_prediction(
                resampling=resampling,
                src_coords=p_coords,
                tgt_coords=s_coords,
                data=p_fields[field],
            )
            obs_df.loc[:, f"s_pred_{field}_{resampling}"] = np.squeeze(s_pred)
            plot_interpolation(
                src_coords=p_coords,
                src_data=p_fields[field],
                tgt_coords=s_coords,
                tgt_pred=s_pred,
                specs={"field": field, "resampling": resampling},
            )
            for metric in METRICS:
                print("  ", metric)
                fun = metric_functions.get(metric)
                obs_df.loc[:, f"{metric}_{field}_{resampling}"] = fun(
                    obs_df[f"s_pred_{field}_{resampling}"], obs_df[f"obs_{field}"]
                )

    obs_df.to_csv(
        f"{ERROR_DATA_DIR}/{args.datetime[:10].replace('-','')}_station-interp.csv"
    )

    # for now:
    # timestamps: only 1
    # stations: all
    # error metric: MSE
    print(" > Program finished successfully !")
