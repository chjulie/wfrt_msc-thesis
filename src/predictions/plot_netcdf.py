import argparse
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import pandas as pd
import warnings

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from src.data_constants import (
    DOMAIN_MINX,
    DOMAIN_MAXX,
    DOMAIN_MINY,
    DOMAIN_MAXY,
    PRED_DATA_DIR,
    PRED_PLOT_DIR,
    TEST_YEAR,
)

PROJECTION = ccrs.PlateCarree()  # ccrs.Stereographic(
#     central_longitude=-90.0, central_latitude=90.0, true_scale_latitude=60.0
# )


def parse_time(time_str):
    """
    Parses lead time input string and returns datetime.timedelta object

    """
    numeric_part = int("".join(filter(str.isdigit, time_str)))
    unit = time_str[-1].lower()
    unit_mapping = {"h": "hours", "d": "days"}
    if unit not in unit_mapping:
        raise ValueError(f"Unsupported unit: '{unit}'. Use 'h', 'm', 's', or 'd'.")
    delta = timedelta(**{unit_mapping[unit]: numeric_part})

    # check that it is a multiple of 6 hours
    delta_hours = delta.total_seconds() / 3600
    if delta_hours % 6 != 0:
        raise ValueError(
            f"Delta must be a multiple of 6 hours. Got {delta_hours} hours."
        )

    return delta_hours


def read_netcdf(path, start_date, end_date, lead_time):

    with xr.open_dataset(path, engine="netcdf4") as ds:

        times = np.array(
            [pd.Timestamp(x).to_pydatetime() for x in ds.initial_date.values]
        )

        mask = (times >= start_date) & (times <= end_date)

        for t in times[mask]:
            ground_truth_t = t + timedelta(hours=lead_time)
            yield ds.sel(initial_date=t, lead_time=lead_time).compute(), ds.sel(
                initial_date=ground_truth_t, lead_time=0
            ).compute()


def plot_tri(ax, triangulation, values):

    ax.set_extent([-145, -110, 44, 65], crs=PROJECTION)

    # contours and grid line
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    contour = ax.tricontourf(
        triangulation, values  # , transform=PROJECTION,# , rasterized=True
    )
    return contour


def mask_small_triangles(triangulation, longitude, latitude, masking_threshold=1e-6):

    triangles = triangulation.triangles
    x = longitude[triangles]
    y = latitude[triangles]

    # Compute triangle area (shoelace formula)
    area = 0.5 * np.abs(
        x[:, 0] * (y[:, 1] - y[:, 2])
        + x[:, 1] * (y[:, 2] - y[:, 0])
        + x[:, 2] * (y[:, 0] - y[:, 1])
    )

    # Mask very small triangles
    return triangulation.set_mask(area < masking_threshold)


def fix(lons):
    # Shift the longitudes from 0-360 to -180-180
    return np.where(lons > 180, lons - 360, lons)


if __name__ == "__main__":
    print("Starting script!")
    EXPERIENCE = "bris-lam"
    FORECAST_DATA_PATH = f"{PRED_DATA_DIR}/bris-lam-inference-20230101T12-20230102T12.nc"  # f"data/prediction_data/{EXPERIENCE}_inference_{TEST_YEAR}"  # f"../date/prediction_data/{EXPERIENCE}_inference_2023"
    VARIABLES_TO_PLOT = ["2t", "sp", "z_850", "z_500"]
    SUBSAMPLE = 1000

    parser = argparse.ArgumentParser(description="Inference with anemoi model")
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Date and time in YYYY-mm-ddTHH:00:00 format. Supported times are 00, 06, 12, 18.",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=True,
        help="Date and time in YYYY-mm-ddTHH:00:00 format. Supported times are 00, 06, 12, 18.",
    )
    parser.add_argument(
        "--lead_time",
        type=str,
        required=False,
        default="6h",
        help="Forecast lead time in the format Xh (hours) or Xd (days). Must be a multiple of 6.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="bris-lam",
        help="'global', 'climatex', 'bris'",
    )

    args = parser.parse_args()
    parsed_lead_time = parse_time(args.lead_time)

    start_date = datetime.strptime(args.start_date, "%Y-%m-%dT%H:%M:%S")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%dT%H:%M:%S")

    count = 0
    subsampling_idx = None

    for pred_data, gt_data in read_netcdf(
        FORECAST_DATA_PATH,
        start_date=start_date,
        end_date=end_date,
        lead_time=parsed_lead_time,
    ):
        fig, ax = plt.subplots(
            len(VARIABLES_TO_PLOT),
            2,
            figsize=(6, 6),
            subplot_kw={"projection": PROJECTION},
        )

        timestamp = pd.to_datetime(str(gt_data.initial_date.values)).strftime(
            "%Y%m%dT%H"
        )
        print(f" ☔️ Timestamp: {timestamp}")

        if count == 0:
            subsampling_idx = np.arange(0, len(gt_data.longitude.values), SUBSAMPLE)
            lons = fix(gt_data.longitude.values[subsampling_idx])
            lats = gt_data.latitude.values[subsampling_idx]

            projection = PROJECTION
            xy = projection.transform_points(PROJECTION, lons, lats)

            print(f" * lons: min: {lons.min()}, max: {lons.max()}")
            print(f" * lats: min: {lats.min()}, max: {lats.max()}")

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*triangulation.*", category=UserWarning
                )
                triangulation = tri.Triangulation(
                    lons,
                    lats,
                    # xy[:, 0],
                    # xy[:, 1],
                )
                print(" 📐 Computed triangulation ")

        count += 1

        for j, var in enumerate(VARIABLES_TO_PLOT):
            print(f"  - variable {j+1}/{len(VARIABLES_TO_PLOT)}: {var}")

            pred_values = pred_data[var].values
            gt_values = gt_data[var].values

            gt_mappable = plot_tri(
                ax[j, 0],
                triangulation=triangulation,
                values=gt_values[subsampling_idx],
            )
            pred_mappable = plot_tri(
                ax[j, 1],
                triangulation=triangulation,
                values=pred_values[subsampling_idx],
            )

            if j == 0:
                ax[j, 0].set_title("Ground truth")
                ax[j, 1].set_title("Prediction")

        plt.show()
        plt.savefig(f"{PRED_PLOT_DIR}/{EXPERIENCE}-{timestamp}.png", dpi=100)
        print(f" ✅ Image saved at {PRED_PLOT_DIR}/{EXPERIENCE}-{timestamp}.png")

    print("Program finished successfully!")
