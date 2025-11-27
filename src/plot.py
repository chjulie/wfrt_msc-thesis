"""
Generate error plots

"""

import argparse
import datetime
import pandas as pd
import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from data_constants import OBS_DATA_DIR, PRED_DATA_DIR, ERROR_DATA_DIR

FIELDS = ["2t"]
RESAMPLING_METHODS = ["nearest-neighbor", "linear", "cubic"]

if __name__ == "__main__":

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

    df = pd.read_csv(
        f"{ERROR_DATA_DIR}/{args.datetime[:10].replace('-','')}_station-interp.csv"
    )
    s_longitudes = df.x.values  # 's' prefix indicates 'station observation'
    s_latitudes = df.y.values  # 's' prefix indicates 'station observation'

    # Plot resampled data :
    for field in FIELDS:
        for resampling in RESAMPLING_METHODS:
            fig, ax = plt.subplots(
                figsize=(6, 6), subplot_kw={"projection": ccrs.PlateCarree()}
            )
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            ax.scatter(
                s_longitudes,
                s_latitudes,
                c=df[f"s_pred_{field}_{resampling}"],
                cmap="RdBu",
            )
            ax.set_title(f"field: {field}, resampling method: {resampling}")

            plt.show()
